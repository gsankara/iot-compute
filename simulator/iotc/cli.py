"""Command line interface: ``iotc <command>``."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


# --------------------------------------------------------------------------
def _cmd_plugins(args) -> int:
    from . import registry, sim  # noqa: F401  (import populates the registry)

    print(registry.describe())
    return 0


def _cmd_devices(args) -> int:
    from .devices import describe_library

    print(describe_library())
    return 0


# --------------------------------------------------------------------------
def _cmd_link(args) -> int:
    """A standalone link-budget calculator.

    Answers the first question anyone asks -- "how far can these things talk in
    this stuff" -- without needing a config file.
    """
    from .registry import create
    from .sim.links import LinkGeometry

    medium = create("medium", args.medium)
    channel = create("channel", "fso", turbulence=args.turbulence,
                     receiver=args.receiver, modulation=args.modulation,
                     coding=args.coding)
    geo = LinkGeometry(
        distance=args.distance,
        wavelength=args.wavelength * 1e-9,
        tx_power=args.tx_power * 1e-3,
        divergence=args.divergence * 1e-3,
        tx_aperture_radius=args.tx_aperture * 1e-3,
        rx_aperture_radius=args.rx_aperture * 1e-3,
        rx_fov=args.rx_fov * 1e-3,
        tx_misalignment=args.misalignment * 1e-3,
    )
    budget = channel.evaluate(geo, medium, None, stochastic=False,
                              line_rate=args.rate * 1e9)
    print(f"medium     : {medium.summary()}")
    print(f"extinction : {medium.extinction(geo.wavelength):.5g} /m "
          f"at {args.wavelength:.0f} nm")
    print(f"spot radius: {budget.extras.get('spot_radius', float('nan')) * 1e3:.2f} mm")
    print(f"budget     : {budget.report()}")
    print(f"max range  : {channel.max_range(geo, medium, line_rate=args.rate * 1e9):.2f} m")

    if args.sweep:
        print()
        print("  range [m]   rx power [W]   SNR [dB]        BER      PER")
        for d in _logspace(0.1, max(args.distance * 4, 10.0), 14):
            geo.distance = d
            b = channel.evaluate(geo, medium, None, stochastic=False,
                                 line_rate=args.rate * 1e9)
            print(f"  {d:9.3f}   {b.rx_power:12.3e}   {b.snr_db:8.2f}  "
                  f"{b.ber:9.2e}  {b.per:7.4f}")
    return 0


def _logspace(lo, hi, n):
    return [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)]


# --------------------------------------------------------------------------
def _cmd_sim(args) -> int:
    from . import config as cfgmod
    from .sim.trace import Trace

    cfg = cfgmod.load_file(args.config)
    trace = Trace(sample_every=args.sample_every)
    world = cfgmod.build_world(cfg, trace=trace)
    sim = cfgmod.build_simulation(cfg, world)

    print(cfgmod.dump_summary(world, cfg))
    print()
    print(cfgmod.reach_report(world))

    if args.check:
        return 0

    if args.verify_surrogate:
        rc = _verify_surrogate(world)
        if rc:
            return rc

    until = args.until if args.until is not None else float(
        (cfg.get("run", {}) or {}).get("until", 60.0)
    )
    if args.steps is not None:
        until = world.time + args.steps * sim.dt

    if args.connect:
        _connect_neighbours(sim, world)

    print()
    print(f"running to t = {until:g} s (dt = {sim.dt:g} s) ...")
    sim.run(until)

    if args.traffic:
        from .bench.sim_bench import _offer_traffic

        _offer_traffic(sim, world, args.traffic)
    sim.drain(1.0)

    print()
    print(json.dumps(sim.summary(), indent=2, default=float))
    if trace.counters:
        print()
        print("trace:")
        print(trace.report())

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        trace.to_csv(out / "events.csv")
        trace.series_to_csv(out / "series.csv")
        (out / "summary.json").write_text(
            json.dumps(sim.summary(), indent=2, default=float), encoding="utf-8"
        )
        (out / "devices.json").write_text(
            json.dumps([d.snapshot() for d in world.devices], indent=2, default=float),
            encoding="utf-8",
        )
        print(f"\nwrote {out}/events.csv, series.csv, summary.json, devices.json")
    return 0


def _connect_neighbours(sim, world) -> None:
    """Aim every device at its nearest reachable peers, to exercise the links."""
    from .sim.links.base import LinkGeometry

    established = 0
    for i, dev in enumerate(world.devices):
        o = dev.cls.optical
        geo = LinkGeometry(
            distance=1.0, wavelength=o.wavelength, tx_power=dev.tx_power,
            divergence=o.divergence, tx_aperture_radius=o.tx_aperture_radius,
            rx_aperture_radius=o.rx_aperture_radius, rx_fov=o.rx_fov,
        )
        reach = world.channel.max_range(geo, world.medium, line_rate=o.line_rate)
        for j in world.neighbours(i, reach):
            if dev.free_transceiver(world.time) is None:
                break
            if dev.transceiver_for(j) is not None:
                continue
            if world.devices[j].free_transceiver(world.time) is None:
                continue
            if sim.establish(i, j).get("ok"):
                established += 1
    print(f"\nestablished {established} optical links")


def _verify_surrogate(world) -> int:
    from .sim.links.base import LinkGeometry

    channel = world.channel
    if not hasattr(channel, "verify"):
        print("\nchannel is not a surrogate; nothing to verify", file=sys.stderr)
        return 0
    dev = world.devices[0]
    o = dev.cls.optical
    geo = LinkGeometry(
        distance=1.0, wavelength=o.wavelength, tx_power=dev.tx_power,
        divergence=o.divergence, tx_aperture_radius=o.tx_aperture_radius,
        rx_aperture_radius=o.rx_aperture_radius, rx_fov=o.rx_fov,
    )
    stats = channel.verify(geo, world.medium, n=400)
    print()
    print("surrogate check against the exact model:")
    print(f"  samples {stats['samples']}  max |error| {stats['max_abs_db']:.3f} dB  "
          f"rms {stats['rms_db']:.3f} dB  bias {stats['bias_db']:+.3f} dB")
    if stats["max_abs_db"] > 1.0:
        print("  ERROR: surrogate deviates by more than 1 dB", file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------------------
def _cmd_bench(args) -> int:
    from .bench import run_sweep

    return run_sweep(args)


# --------------------------------------------------------------------------
def _cmd_graph(args) -> int:
    """Build a task graph and say what it demands of the hardware."""
    from .bench import ir_bench
    from .compiler import build_workload

    params = {}
    for item in args.set or []:
        key, _, value = item.partition("=")
        if not _:
            print(f"--set needs key=value, got {item!r}", file=sys.stderr)
            return 2
        try:
            params[key] = json.loads(value)
        except json.JSONDecodeError:
            params[key] = value

    if args.dump:
        graph = build_workload(args.workloads[0], **params)
        graph.to_json(args.dump)
        print(graph.summary())
        return 0

    per = {w: params for w in (args.workloads or [])}
    rows = ir_bench.run(
        args.workloads or None,
        tuple(args.classes or ir_bench.DEFAULT_CLASSES),
        params=per,
        out=args.out,
    )
    for cls in sorted({r["device_class"] for r in rows}):
        print(ir_bench.report(rows, cls))
        print()
    print(ir_bench.verdict(rows))
    return 0


# --------------------------------------------------------------------------
def _cmd_tile(args) -> int:
    """Tile workloads onto a device class and report what the cut costs."""
    from .bench import tile_bench

    rows = tile_bench.run(
        args.workloads or None,
        tuple(args.classes or tile_bench.DEFAULT_CLASSES),
        tuple(args.tiler or tile_bench.DEFAULT_MODELS),
        out=args.out,
    )
    for cls in sorted({r["device_class"] for r in rows}):
        print(tile_bench.report(rows, cls))
        print()
    print(tile_bench.verdict(rows))
    return 0


# --------------------------------------------------------------------------
def _cmd_partition(args) -> int:
    """Group tasks onto devices, then measure how the partitioners scale."""
    from .bench import partition_bench

    methods = tuple(args.method or partition_bench.DEFAULT_METHODS)
    quality = partition_bench.run(
        args.workloads or None, args.device_class, methods, out=args.out
    )
    print(partition_bench.report(quality))
    print()

    scale = []
    if not args.no_scaling:
        scale = partition_bench.scaling(
            methods, device_class=args.device_class, out=args.out
        )
        print(partition_bench.scaling_report(scale))
        print()
    print(partition_bench.verdict(quality, scale))
    return 0


# --------------------------------------------------------------------------
def _cmd_place(args) -> int:
    """Place devices in space and report what the formation costs to fly."""
    from .bench import place_bench

    print(place_bench.budget_report())
    print()
    rows = place_bench.run(
        args.workloads or None,
        tuple(args.platform or place_bench.PLATFORMS),
        tuple(args.placer or place_bench.DEFAULT_PLACERS),
        out=args.out,
    )
    for platform in sorted({r["platform"] for r in rows}):
        print(place_bench.report(rows, platform))
        print()
    print(place_bench.verdict(rows))
    return 0


# --------------------------------------------------------------------------
def _cmd_motion(args) -> int:
    """Plan the manoeuvre and report what keeping the links up costs."""
    from .bench import motion_bench

    rows = motion_bench.run(
        args.workloads or None,
        tuple(args.platform or ("float",)),
        tuple(args.planner or motion_bench.DEFAULT_PLANNERS),
        out=args.out,
    )
    print(motion_bench.report(rows))
    print()
    print(motion_bench.verdict(rows))
    return 0


# --------------------------------------------------------------------------
def _cmd_beam(args) -> int:
    """Assign beams and report what the gimbals can actually point at."""
    from .bench import beam_bench

    rows = beam_bench.run(
        args.workloads or None,
        tuple(args.platform or beam_bench.PLATFORMS),
        tuple(args.assigner or beam_bench.DEFAULT_ASSIGNERS),
        out=args.out,
    )
    print(beam_bench.report(rows))
    print()
    print(beam_bench.verdict(rows))
    return 0


# --------------------------------------------------------------------------
def _cmd_schedule(args) -> int:
    """Build the slot schedule and report what bounds its efficiency."""
    from .bench import schedule_bench

    rows = schedule_bench.run(
        args.workloads or None,
        tuple(args.classes or schedule_bench.CLASSES),
        tuple(args.scheduler or schedule_bench.DEFAULT_SCHEDULERS),
        out=args.out,
    )
    print(schedule_bench.report(rows))
    print()
    print(schedule_bench.verdict(rows))
    return 0


# --------------------------------------------------------------------------
def _cmd_energy(args) -> int:
    """Price the compiled schedule against what the devices can harvest."""
    from .bench import energy_bench

    rows = energy_bench.run(
        args.workloads or None,
        tuple(args.platform or energy_bench.PLATFORMS),
        tuple(args.policy or energy_bench.DEFAULT_POLICIES),
        out=args.out,
    )
    print(energy_bench.report(rows))
    print()
    print(energy_bench.verdict(rows))
    return 0


# --------------------------------------------------------------------------
def _cmd_resilience(args) -> int:
    """Measure what redundancy buys and what the tracker costs."""
    from pathlib import Path as _Path

    from .bench import resilience_bench as rb
    from .bench.sim_bench import write_csv

    rows = rb.run(args.workloads or None,
                  tuple(args.policy or rb.DEFAULT_POLICIES), out=args.out)
    tracker = rb.tracker_rows()
    print(rb.report(rows))
    print()
    print(rb.tracker_report(tracker))
    print()
    print(rb.verdict(rows, tracker))
    if args.out:
        write_csv(tracker, _Path(args.out) / "tracker.csv")
    return 0


# --------------------------------------------------------------------------
def _cmd_codegen(args) -> int:
    """Emit the bytecode, size it against flash, and check it on the VM."""
    from .bench import codegen_bench

    rows = codegen_bench.run(
        args.workloads or None,
        tuple(args.classes or ("mote-c",)),
        tuple(args.encoding or codegen_bench.DEFAULT_ENCODINGS),
        out=args.out,
    )
    print(codegen_bench.report(rows))
    print()
    print(codegen_bench.verdict(rows))
    return 0


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="iotc", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("plugins", help="list every registered plugin").set_defaults(
        func=_cmd_plugins
    )
    sub.add_parser("devices", help="list the stock device classes").set_defaults(
        func=_cmd_devices
    )

    link = sub.add_parser("link", help="standalone link-budget calculator")
    link.add_argument("--medium", default="water")
    link.add_argument("--wavelength", type=float, default=470.0, help="nm")
    link.add_argument("--distance", type=float, default=10.0, help="m")
    link.add_argument("--tx-power", type=float, default=1.0, help="mW")
    link.add_argument("--divergence", type=float, default=5.0, help="mrad half-angle")
    link.add_argument("--tx-aperture", type=float, default=1.5, help="mm")
    link.add_argument("--rx-aperture", type=float, default=2.5, help="mm")
    link.add_argument("--rx-fov", type=float, default=15.0, help="mrad")
    link.add_argument("--misalignment", type=float, default=0.0, help="mrad")
    link.add_argument("--rate", type=float, default=1.0, help="Gb/s")
    link.add_argument("--turbulence", default="none")
    link.add_argument("--receiver", default="apd")
    link.add_argument("--modulation", default="ook")
    link.add_argument("--coding", default="rs")
    link.add_argument("--sweep", action="store_true", help="tabulate against range")
    link.set_defaults(func=_cmd_link)

    sim = sub.add_parser("sim", help="run a simulation from a config")
    sim.add_argument("config")
    sim.add_argument("--until", type=float, default=None, help="simulated seconds")
    sim.add_argument("--steps", type=int, default=None)
    sim.add_argument("--out", default=None, help="directory for CSV and JSON output")
    sim.add_argument("--check", action="store_true",
                     help="build and report, but do not run")
    sim.add_argument("--connect", action="store_true",
                     help="aim every device at its nearest reachable peers")
    sim.add_argument("--verify-surrogate", action="store_true",
                     help="check a surrogate channel against the exact model")
    sim.add_argument("--traffic", type=int, default=0,
                     help="offer N rounds of traffic across every live link")
    sim.add_argument("--sample-every", type=float, default=0.0,
                     help="minimum interval between time-series samples")
    sim.set_defaults(func=_cmd_sim)

    bench = sub.add_parser("bench", help="run an evaluation sweep")
    bench.add_argument("config")
    bench.add_argument("--out", default="results")
    bench.add_argument("--seeds", type=int, default=None,
                       help="override the number of seeds")
    bench.add_argument("--plot", action="store_true", help="also write figures")
    bench.set_defaults(func=_cmd_bench)

    graph = sub.add_parser(
        "graph", help="build a workload task graph and characterize it (B0)"
    )
    graph.add_argument("workloads", nargs="*",
                       help="workload names; default is every registered one")
    graph.add_argument("--set", action="append", metavar="KEY=VALUE",
                       help="workload parameter override (repeatable)")
    graph.add_argument("--class", dest="classes", action="append",
                       help="device class to score against (repeatable)")
    graph.add_argument("--out", default=None, help="directory for CSV and JSON output")
    graph.add_argument("--dump", default=None, metavar="PATH",
                       help="write the first workload's task graph as JSON and stop")
    graph.set_defaults(func=_cmd_graph)

    tile = sub.add_parser(
        "tile", help="split workloads to fit a device class and score the cut (B1)"
    )
    tile.add_argument("workloads", nargs="*",
                      help="workload names; default is every registered one")
    tile.add_argument("--class", dest="classes", action="append",
                      help="device class to tile onto (repeatable)")
    tile.add_argument("--tiler", action="append",
                      help="tiler plugin to compare (repeatable)")
    tile.add_argument("--out", default=None, help="directory for CSV and JSON output")
    tile.set_defaults(func=_cmd_tile)

    part = sub.add_parser(
        "partition", help="group tasks onto devices and score the cut (B2)"
    )
    part.add_argument("workloads", nargs="*")
    part.add_argument("--class", dest="device_class", default="mote-c")
    part.add_argument("--method", action="append",
                      help="partitioner to compare (repeatable)")
    part.add_argument("--out", default=None, help="directory for CSV and JSON output")
    part.add_argument("--no-scaling", action="store_true",
                      help="skip the runtime ladder")
    part.set_defaults(func=_cmd_partition)

    pl = sub.add_parser(
        "place", help="project the embedding onto physical coordinates (B3)"
    )
    pl.add_argument("workloads", nargs="*")
    pl.add_argument("--platform", action="append",
                    help="float, drone or tether (repeatable)")
    pl.add_argument("--placer", action="append", help="placer to compare (repeatable)")
    pl.add_argument("--out", default=None, help="directory for CSV and JSON output")
    pl.set_defaults(func=_cmd_place)

    mo = sub.add_parser(
        "motion", help="fly the swarm to its placement without dropping links (B4)"
    )
    mo.add_argument("workloads", nargs="*")
    mo.add_argument("--platform", action="append", help="float or drone (repeatable)")
    mo.add_argument("--planner", action="append", help="planner to compare (repeatable)")
    mo.add_argument("--out", default=None, help="directory for CSV and JSON output")
    mo.set_defaults(func=_cmd_motion)

    bm = sub.add_parser(
        "beam", help="assign links to transceiver rounds and check aimability (B5)"
    )
    bm.add_argument("workloads", nargs="*")
    bm.add_argument("--platform", action="append", help="float, drone or relay")
    bm.add_argument("--assigner", action="append", help="assigner to compare")
    bm.add_argument("--out", default=None, help="directory for CSV and JSON output")
    bm.set_defaults(func=_cmd_beam)

    sc = sub.add_parser(
        "schedule", help="time the beam rounds and amortise the retarget (B6)"
    )
    sc.add_argument("workloads", nargs="*")
    sc.add_argument("--class", dest="classes", action="append")
    sc.add_argument("--scheduler", action="append")
    sc.add_argument("--out", default=None, help="directory for CSV and JSON output")
    sc.set_defaults(func=_cmd_schedule)

    en = sub.add_parser(
        "energy", help="price the schedule against harvest and storage (B7)"
    )
    en.add_argument("workloads", nargs="*")
    en.add_argument("--platform", action="append")
    en.add_argument("--policy", action="append")
    en.add_argument("--out", default=None, help="directory for CSV and JSON output")
    en.set_defaults(func=_cmd_energy)

    rs = sub.add_parser(
        "resilience", help="replication, failover and the pointing tracker (B8)"
    )
    rs.add_argument("workloads", nargs="*")
    rs.add_argument("--policy", action="append")
    rs.add_argument("--out", default=None, help="directory for CSV and JSON output")
    rs.set_defaults(func=_cmd_resilience)

    cg = sub.add_parser(
        "codegen", help="emit per-device bytecode and run it on the VM (B9)"
    )
    cg.add_argument("workloads", nargs="*")
    cg.add_argument("--class", dest="classes", action="append")
    cg.add_argument("--encoding", action="append")
    cg.add_argument("--out", default=None, help="directory for CSV and JSON output")
    cg.set_defaults(func=_cmd_codegen)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

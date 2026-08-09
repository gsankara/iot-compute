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

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

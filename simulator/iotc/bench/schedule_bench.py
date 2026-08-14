"""B6 evaluation -- can the aim be amortised, and what stops it.

Phase A put link efficiency under 1%: with 8 kbit packets, over 99% of the time
a link is up is spent slewing rather than sending.  B6 has one lever -- stay
aimed and send more per aim -- and this bench measures how far that lever goes
before something else stops it.

The answer is the pass's most important output, and it is not a schedule.
Amortising a half-second retarget against microseconds of wire time needs a
batch of order a million supersteps; a mote has kilobytes of RAM.  So the bench
reports both the batch the amortisation *wanted* and the batch it was *granted*,
and the gap is a statement about the hardware rather than the compiler.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..compiler import build_workload
from ..compiler.beam import assign_beams, refine_positions
from ..compiler.partition import partition
from ..compiler.place import place
from ..compiler.schedule import build_schedule
from ..devices import MOTE_C, MOTE_M, RELAY, aquatic
from ..registry import create
from .sim_bench import write_csv

DEFAULT_SCHEDULERS = ("single", "amortised", "max_batch")
CLASSES = {"mote-c": MOTE_C, "mote-m": MOTE_M, "relay": RELAY}
QUALITY_PARAMS = {
    "stencil3d": dict(nx=128, ny=128, nz=128, tx=8, ty=8, tz=8),
    "spmv": dict(n=1 << 16, parts=256),
}


def evaluate(devices, assign, pos, cls, model, workload) -> dict:
    s = build_schedule(devices, assign, pos, cls, model=model)
    return {
        "workload": workload,
        "device_class": cls.name,
        "scheduler": model,
        "rounds": assign.rounds,
        "batch": s.batch,
        "batch_wanted": s.batch_wanted,
        "starved": s.starved,
        "period_s": s.period,
        "wire_s": s.wire_time,
        "retarget_s": s.retarget_time,
        "guard_s": s.guard_time,
        "efficiency": s.efficiency,
        "throughput_bps": s.throughput,
        "latency_s": s.latency,
        "buffer_kib": s.buffer_bytes / 1024,
        "buffer_wanted_mib": s.buffer_wanted / (1024 * 1024),
        "buffer_shortfall": s.buffer_shortfall,
        "ram_kib": cls.compute.ram_bytes / 1024,
        "fits": s.fits(),
    }


def run(workloads=None, classes=tuple(CLASSES), schedulers=DEFAULT_SCHEDULERS,
        params: dict | None = None, out: str | Path | None = None) -> list[dict]:
    params = params if params is not None else QUALITY_PARAMS
    names = sorted(workloads or params)
    water = create("medium", "water")
    rows = []
    for name in names:
        graph = build_workload(name, **params.get(name, {}))
        devices = partition(graph, MOTE_C, seed=0).coarsened()
        for cls_name in classes:
            cls = aquatic(CLASSES[cls_name])
            pos = refine_positions(devices, place(devices, cls, water, seed=0).positions, cls)
            assign = assign_beams(devices, pos, cls)
            for model in schedulers:
                rows.append(evaluate(devices, assign, pos, cls, model, name))
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        write_csv(rows, out / "schedule.csv")
        with open(out / "schedule.json", "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
    return rows


def report(rows: list[dict]) -> str:
    head = (
        f"{'workload':11s} {'class':10s} {'scheduler':11s} {'batch':>8s} {'wanted':>9s} "
        f"{'period s':>9s} {'efficiency':>11s} {'kB/s':>10s} {'buffer KiB':>11s}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['workload']:11s} {r['device_class']:10s} {r['scheduler']:11s} "
            f"{r['batch']:8d} {r['batch_wanted']:9d} {r['period_s']:9.2f} "
            f"{r['efficiency']:10.4%} {r['throughput_bps'] / 1e3:10.2f} "
            f"{r['buffer_kib']:11.1f}"
        )
    return "\n".join(lines)


def verdict(rows: list[dict]) -> str:
    lines = []
    naive = [r for r in rows if r["scheduler"] == "single"]
    smart = [r for r in rows if r["scheduler"] == "amortised"]

    # Report per (workload, class).  Averaging across workloads would be
    # meaningless here: payload per link differs by orders of magnitude between
    # a stencil halo and an SpMV block exchange, so a mean efficiency describes
    # no configuration that exists.
    for r in naive:
        key = (r["workload"], r["device_class"])
        match = [x for x in smart if (x["workload"], x["device_class"]) == key]
        if not match:
            continue
        s = match[0]
        lines.append(
            f"{r['workload']} on {r['device_class']}: unbatched {r['efficiency']:.4%} "
            f"-> amortised {s['efficiency']:.4%} "
            f"({s['efficiency'] / max(r['efficiency'], 1e-12):.0f}x), batch {s['batch']}"
        )
    starved = [r for r in smart if r["starved"]]
    if starved:
        lines.append(
            f"but {len(starved)} of {len(smart)} schedules are RAM-starved: the batch "
            f"needed is {np.mean([r['batch_wanted'] / max(r['batch'], 1) for r in starved]):.0f}x "
            f"larger than the buffer allows"
        )
        lines.append(
            f"reaching the target would need {np.mean([r['buffer_wanted_mib'] for r in starved]):.0f} MiB "
            f"of buffer against {np.mean([r['ram_kib'] for r in starved]) / 1024:.1f} MiB of RAM "
            f"({np.mean([r['buffer_shortfall'] for r in starved]):.0f}x short) -- the fabric stays "
            f"aim-dominated because the part is too small, not because the schedule is poor"
        )
    # Only mote-c and mote-m isolate RAM: they share a gimbal and a head count.
    # The relay differs in transceivers *and* step size too, so including it
    # would make this look non-monotonic and prove nothing.
    for workload in sorted({r["workload"] for r in smart}):
        family = sorted(
            (r["ram_kib"], r["efficiency"], r["batch"])
            for r in smart
            if r["workload"] == workload and r["device_class"].startswith("mote-")
        )
        if len(family) > 1:
            lines.append(
                f"{workload}, holding the gimbal fixed and varying only RAM: "
                + ", ".join(f"{k / 1024:.1f} MiB batch {b} -> {v:.3%}" for k, v, b in family)
            )
    lines.append(
        "the relay is not in that comparison: it has six heads and a finer step, "
        "so it differs in more than buffer"
    )
    lines.append(
        f"every schedule fits the buffer it was given: {all(r['fits'] for r in rows)}"
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="iotc schedule", description=__doc__)
    p.add_argument("workloads", nargs="*")
    p.add_argument("--class", dest="classes", action="append", choices=sorted(CLASSES))
    p.add_argument("--scheduler", action="append")
    p.add_argument("--out")
    args = p.parse_args(argv)

    rows = run(args.workloads or None, tuple(args.classes or CLASSES),
               tuple(args.scheduler or DEFAULT_SCHEDULERS), out=args.out)
    print(report(rows))
    print()
    print(verdict(rows))
    return 0

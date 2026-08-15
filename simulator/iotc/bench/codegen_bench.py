"""B9 evaluation -- how big is the image, and does it do what was planned.

Two questions, and the second is not a quality metric but a correctness gate.

``does the image fit flash?``
    B1 asked this of RAM and it decided the tiling.  Here it decides the
    *encoding*.  A run is thousands of frames long, so an unrolled program grows
    without bound while a looped one does not -- which makes the choice a
    compiler decision rather than a serialization detail.

``does the VM deliver what the schedule promised?``
    The bytecode is only a program if something executes it.  ``iotc/sim/node.py``
    runs each device's image and the bench checks the bytes it moves against the
    slots B6 planned, and that every SEND has a matching RECV -- a link needs
    both gimbals, so a one-way edge is not physically representable.

The comparators are ``flat`` (fixed-width, unrolled -- the obvious first
implementation) and ``packed`` (varints, still unrolled), which separate the
cost of the encoding from the cost of not looping.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..compiler import build_workload
from ..compiler.beam import assign_beams, refine_positions
from ..compiler.codegen import emit, fits_flash, program_bytes
from ..compiler.partition import partition
from ..compiler.place import place
from ..compiler.resilience import plan_resilience
from ..compiler.schedule import build_schedule
from ..devices import MOTE_C, MOTE_M, aquatic
from ..registry import create
from ..sim.node import delivered_bytes, run_swarm
from .sim_bench import write_csv

DEFAULT_ENCODINGS = ("flat", "packed", "looped")
CLASSES = {"mote-c": MOTE_C, "mote-m": MOTE_M}
REPEATS = (100, 1_000, 10_000, 100_000)
QUALITY_PARAMS = {
    "stencil3d": dict(nx=128, ny=128, nz=128, tx=8, ty=8, tz=8),
    "spmv": dict(n=1 << 16, parts=256),
}


def compile_all(workload: str, params: dict, cls_name: str, repeat: int):
    cls = aquatic(CLASSES[cls_name])
    water = create("medium", "water")
    graph = build_workload(workload, **params)
    devices = partition(graph, MOTE_C, seed=0).coarsened()
    pos = refine_positions(devices, place(devices, cls, water, seed=0).positions, cls)
    assign = assign_beams(devices, pos, cls)
    sched = build_schedule(devices, assign, pos, cls, model="amortised")
    replicas = plan_resilience(devices, max(1, devices.n_tasks // 5),
                               model="traffic", seed=0).replicated
    manifest, programs = emit(devices, sched, assign, pos, cls,
                              start=np.zeros_like(pos), replicated=replicas,
                              repeat=repeat)
    return devices, sched, cls, manifest, programs


def evaluate(workload, params, cls_name, repeat, encoding) -> dict:
    devices, sched, cls, manifest, programs = compile_all(workload, params, cls_name, repeat)
    sizes = [program_bytes(p, encoding) for p in programs]
    states = run_swarm(programs, transceivers=cls.optical.n_transceivers, repeat=1)
    tx, rx = delivered_bytes(states)
    planned = sum(s.payload for s in sched.slots)
    return {
        "workload": workload,
        "device_class": cls.name,
        "encoding": encoding,
        "repeat": repeat,
        "devices": len(programs),
        "instructions_stored": programs[0].instructions,
        "instructions_executed": programs[0].executed,
        "total_kib": sum(sizes) / 1024,
        "worst_device_kib": max(sizes) / 1024,
        "flash_kib": cls.compute.flash_bytes / 1024,
        "fits_flash": all(fits_flash(p, cls, encoding) for p in programs),
        "vm_tx_bytes": tx,
        "vm_rx_bytes": rx,
        "planned_bytes": planned,
        "delivers_plan": bool(tx == rx and abs(tx - planned) <= max(1.0, planned * 1e-9)),
    }


def run(workloads=None, classes=("mote-c",), encodings=DEFAULT_ENCODINGS,
        repeats=REPEATS, params: dict | None = None,
        out: str | Path | None = None) -> list[dict]:
    params = params if params is not None else QUALITY_PARAMS
    names = sorted(workloads or params)
    rows = [
        evaluate(name, params.get(name, {}), cls_name, repeat, encoding)
        for name in names
        for cls_name in classes
        for repeat in repeats
        for encoding in encodings
    ]
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        write_csv(rows, out / "codegen.csv")
        with open(out / "codegen.json", "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
    return rows


def report(rows: list[dict]) -> str:
    head = (
        f"{'workload':10s} {'class':10s} {'encoding':9s} {'repeat':>8s} "
        f"{'stored':>7s} {'executed':>10s} {'total KiB':>10s} {'worst KiB':>10s} "
        f"{'flash':>5s} {'plan':>5s}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['workload']:10s} {r['device_class']:10s} {r['encoding']:9s} "
            f"{r['repeat']:8d} {r['instructions_stored']:7d} "
            f"{r['instructions_executed']:10d} {r['total_kib']:10.1f} "
            f"{r['worst_device_kib']:10.1f} "
            f"{'ok' if r['fits_flash'] else 'OVER':>5s} "
            f"{'ok' if r['delivers_plan'] else 'BAD':>5s}"
        )
    return "\n".join(lines)


def verdict(rows: list[dict]) -> str:
    lines = []
    longest = max(r["repeat"] for r in rows)
    at_long = [r for r in rows if r["repeat"] == longest]
    by = {}
    for r in at_long:
        by.setdefault(r["encoding"], []).append(r["total_kib"])
    if "flat" in by and "looped" in by:
        flat, looped = np.mean(by["flat"]), np.mean(by["looped"])
        packed = np.mean(by.get("packed", [flat]))
        lines.append(
            f"at {longest} frames: flat {flat:.0f} KiB, packed {packed:.0f} KiB, "
            f"looped {looped:.1f} KiB -- varints save {flat / packed:.1f}x, "
            f"looping saves a further {packed / max(looped, 1e-9):.0f}x"
        )

    overflow = sorted({(r["encoding"], r["repeat"]) for r in rows if not r["fits_flash"]})
    if overflow:
        first = min(r for enc, r in overflow if enc == "flat") if any(
            enc == "flat" for enc, _ in overflow) else None
        lines.append(
            "images that do not fit flash: "
            + ", ".join(f"{enc} at {rep} frames" for enc, rep in overflow)
            + (f" -- unrolling overflows a mote from {first} frames on" if first else "")
        )
    else:
        lines.append("every image fits flash at every run length tested")

    looped_sizes = {r["repeat"]: r["total_kib"] for r in rows if r["encoding"] == "looped"}
    if len(looped_sizes) > 1:
        spread = max(looped_sizes.values()) / max(min(looped_sizes.values()), 1e-9)
        lines.append(
            f"the looped image is independent of run length ({spread:.2f}x across a "
            f"{max(looped_sizes) // min(looped_sizes)}x range of frame counts) -- "
            f"the others are linear in it"
        )

    bad = [r for r in rows if not r["delivers_plan"]]
    lines.append(
        f"VM round trip: {len(rows) - len(bad)} of {len(rows)} images deliver exactly "
        f"the bytes B6 planned, with every SEND matched by a RECV"
        + ("" if not bad else f" -- {len(bad)} DO NOT")
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="iotc codegen", description=__doc__)
    p.add_argument("workloads", nargs="*")
    p.add_argument("--class", dest="classes", action="append", choices=sorted(CLASSES))
    p.add_argument("--encoding", action="append")
    p.add_argument("--out")
    args = p.parse_args(argv)

    rows = run(args.workloads or None, tuple(args.classes or ("mote-c",)),
               tuple(args.encoding or DEFAULT_ENCODINGS), out=args.out)
    print(report(rows))
    print()
    print(verdict(rows))
    return 0

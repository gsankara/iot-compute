"""B5 evaluation -- how many rounds the fabric needs, and what it can aim at.

Three things are measured, and the first is a cross-pass result rather than a
property of this pass alone.

``rejected``
    Share of links whose bearing falls outside the gimbal's sweep limits.  These
    are impossible, not weak.  On a float this is where B3's energy-optimal
    placement is found to be unusable, and where the refinement iteration earns
    its place.

``rounds`` and ``first-round traffic``
    A device has two heads and wants up to thirteen neighbours, so the demand
    graph is realised as a *sequence* of b-matchings.  Each extra round is
    another retarget for everyone in it, and Phase A priced a retarget at ~0.5 s
    against microseconds of transmission -- so what matters is not the round
    count alone but how much of the payload the opening round carries.

``pointing loss``
    Quantization residual turned into dB against a Gaussian beam.

The comparator is ``lightest_first``: the same b-matching with the edge order
reversed.  It isolates the ordering from the matching, which is the only part
this pass actually chooses.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..compiler import build_workload
from ..compiler.beam import assign_beams, refine_positions
from ..compiler.partition import partition
from ..compiler.place import place
from ..devices import MOTE_C, RELAY, STANDARD_GIMBAL, aerial, aquatic, tethered
from ..units import deg
from ..registry import create
from .sim_bench import write_csv

DEFAULT_ASSIGNERS = ("heaviest_first", "lightest_first", "arbitrary")

def _hemispherical(cls):
    """``cls`` with its gimbal's elevation limit lifted to +/-90 deg, nothing else."""
    grid = replace(STANDARD_GIMBAL.grid, el_limits=(-90 * deg, 90 * deg))
    gimbal = replace(cls.optical.gimbal, grid=grid)
    return cls.variant(f"{cls.name}-hemi", optical=replace(cls.optical, gimbal=gimbal))


PLATFORMS = {
    "float": (lambda: aquatic(MOTE_C), "water"),
    "drone": (lambda: aerial(MOTE_C), "air"),
    "relay": (lambda: aquatic(RELAY), "water"),
    # B3 gives this platform a *slab* (anisotropy 0.06) where the float gets a
    # column (231), and a column is what B5 found 96.5% unaimable.  ``aquatic``
    # wraps first: the modifier order decides whether the head is on blue-green
    # or still on 850 nm, and the wrong order silently reports a 1.75 m reach.
    "tether": (lambda: tethered(aquatic(MOTE_C)), "water"),
    # A single-variable control.  Identical to ``float`` -- same placement, same
    # blue-green optics, same step, slew, settle and energy -- except the
    # gimbal's elevation limit is +/-90 deg instead of +/-75.  FINE_GIMBAL is
    # deliberately not used: it also changes the step, the slew rate and the
    # retarget energy, so it could not say which one removed the rejections.
    "float-hemi": (lambda: _hemispherical(aquatic(MOTE_C)), "water"),
}

QUALITY_PARAMS = {
    "stencil3d": dict(nx=128, ny=128, nz=128, tx=8, ty=8, tz=8),
    "stencil2d": dict(nx=2048, ny=2048, tx=16, ty=16),
    "spmv": dict(n=1 << 16, parts=256),
}


def evaluate(devices, positions, cls, model, workload, platform, refined) -> dict:
    a = assign_beams(devices, positions, cls, model=model)
    demand = devices.degree().max(initial=0)
    return {
        "workload": workload,
        "platform": platform,
        "assigner": model,
        "refined": refined,
        "devices": devices.n_tasks,
        "links": devices.n_edges,
        "transceivers": cls.optical.n_transceivers,
        "max_demand_degree": int(demand),
        "oversubscription": float(demand) / max(cls.optical.n_transceivers, 1),
        "rounds": a.rounds,
        "first_round_traffic": a.first_round_traffic,
        "rejected_fraction": a.rejected_fraction,
        "feasible": a.feasible,
        "mean_loss_db": a.mean_loss_db,
        "worst_loss_db": a.worst_loss_db,
    }


def run(
    workloads=None,
    platforms=tuple(PLATFORMS),
    assigners=DEFAULT_ASSIGNERS,
    params: dict | None = None,
    out: str | Path | None = None,
) -> list[dict]:
    params = params if params is not None else QUALITY_PARAMS
    names = sorted(workloads or params)
    rows = []
    for name in names:
        graph = build_workload(name, **params.get(name, {}))
        devices = partition(graph, MOTE_C, seed=0).coarsened()
        for platform in platforms:
            factory, medium_name = PLATFORMS[platform]
            cls, medium = factory(), create("medium", medium_name)
            raw = place(devices, cls, medium, seed=0).positions
            fixed = refine_positions(devices, raw, cls)
            for model in assigners:
                rows.append(evaluate(devices, raw, cls, model, name, platform, False))
                rows.append(evaluate(devices, fixed, cls, model, name, platform, True))
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        write_csv(rows, out / "beam.csv")
        with open(out / "beam.json", "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
    return rows


def report(rows: list[dict]) -> str:
    head = (
        f"{'workload':11s} {'platform':10s} {'assigner':15s} {'B3 fix':>7s} {'tx':>3s} "
        f"{'over':>5s} {'rounds':>7s} {'1st traffic':>12s} {'rejected':>9s} {'loss dB':>8s}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['workload']:11s} {r['platform']:10s} {r['assigner']:15s} "
            f"{'yes' if r['refined'] else 'no':>7s} {r['transceivers']:3d} "
            f"{r['oversubscription']:5.1f} {r['rounds']:7d} "
            f"{r['first_round_traffic']:11.1%} {r['rejected_fraction']:9.1%} "
            f"{r['mean_loss_db']:8.2f}"
        )
    return "\n".join(lines)


def verdict(rows: list[dict]) -> str:
    lines = []
    over = np.mean([r["oversubscription"] for r in rows])
    lines.append(
        f"demand exceeds transceivers by {over:.1f}x on average -- the graph cannot "
        f"be a topology, only a sequence of them"
    )

    for platform in sorted({r["platform"] for r in rows}):
        raw = [r for r in rows if r["platform"] == platform and not r["refined"]]
        fixed = [r for r in rows if r["platform"] == platform and r["refined"]]
        if not raw or not fixed:
            continue
        lines.append(
            f"{platform}: B3's placement leaves {np.mean([r['rejected_fraction'] for r in raw]):.1%} "
            f"of links unaimable; after the refinement iteration, "
            f"{np.mean([r['rejected_fraction'] for r in fixed]):.1%}"
        )

    heavy = [r for r in rows if r["assigner"] == "heaviest_first" and r["refined"]]
    light = [r for r in rows if r["assigner"] == "lightest_first" and r["refined"]]
    if heavy and light:
        lines.append(
            f"ordering the b-matching by payload puts "
            f"{np.mean([r['first_round_traffic'] for r in heavy]):.1%} of the traffic in "
            f"the opening round against {np.mean([r['first_round_traffic'] for r in light]):.1%} "
            f"reversed -- same rounds, different bytes"
        )

    by_tx: dict[int, list[int]] = {}
    for r in rows:
        if r["refined"]:
            by_tx.setdefault(r["transceivers"], []).append(r["rounds"])
    if len(by_tx) > 1:
        lines.append(
            "rounds against heads: "
            + ", ".join(f"{k} heads {np.mean(v):.1f} rounds" for k, v in sorted(by_tx.items()))
        )
    infeasible = [r for r in rows if not r["feasible"]]
    lines.append(
        f"assignments exceeding a device's head count: {len(infeasible)} of {len(rows)}"
        + (" -- every assignment is realisable" if not infeasible else "")
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="iotc beam", description=__doc__)
    p.add_argument("workloads", nargs="*")
    p.add_argument("--platform", action="append", choices=sorted(PLATFORMS))
    p.add_argument("--assigner", action="append")
    p.add_argument("--out")
    args = p.parse_args(argv)

    rows = run(
        args.workloads or None,
        tuple(args.platform or PLATFORMS),
        tuple(args.assigner or DEFAULT_ASSIGNERS),
        out=args.out,
    )
    print(report(rows))
    print()
    print(verdict(rows))
    return 0

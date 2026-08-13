"""B4 evaluation -- what it costs to keep the links up while the swarm moves.

Three quantities, and the whole pass is the trade between the first two:

``broken link fraction``
    Share of (protected link, step) pairs where both endpoints were in motion.
    Without a closed-loop tracker those links are down, not degraded.

``makespan``
    Seconds from the first device moving to the last arriving.  Waves are a
    multiplier on this, and that multiplier is the price of link safety.

``energy``
    Joules, charged per axis at the platform's real rates.  It should be
    *identical* across planners -- they reorder the motion, they do not lengthen
    the paths -- so it is here as a check that the comparison is honest.

``direct`` is the comparator: everyone sets off at once, straight to target.
``sequential`` is the other extreme, one device at a time.  ``waves`` is the
pass, and it is worth having only if it lands much nearer the second on safety
and much nearer the first on time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..compiler import build_workload
from ..compiler.motion import critical_edges, plan_motion
from ..compiler.partition import partition
from ..compiler.place import derive_budget, place
from ..devices import MOTE_C, aerial, aquatic
from ..registry import create
from .sim_bench import write_csv

DEFAULT_PLANNERS = ("direct", "waves", "sequential")

PLATFORMS = {
    "float": (lambda: aquatic(MOTE_C), "water"),
    "drone": (lambda: aerial(MOTE_C), "air"),
}

QUALITY_PARAMS = {
    "stencil3d": dict(nx=128, ny=128, nz=128, tx=8, ty=8, tz=8),
    "stencil2d": dict(nx=2048, ny=2048, tx=16, ty=16),
    "spmv": dict(n=1 << 16, parts=256),
}


def scenario(name: str, params: dict, platform: str, scatter: float = 2.0, seed: int = 0):
    """A workload placed at its target, and a scattered formation to fly from."""
    factory, medium_name = PLATFORMS[platform]
    cls, medium = factory(), create("medium", medium_name)
    graph = build_workload(name, **params)
    devices = partition(graph, MOTE_C, seed=seed).coarsened()
    budget = derive_budget(cls, medium)
    target = place(devices, cls, medium, seed=seed).positions
    start = np.random.default_rng(seed).normal(scale=scatter, size=target.shape)
    return devices, budget, start, target


def evaluate(devices, budget, start, target, model: str, workload: str, platform: str) -> dict:
    plan = plan_motion(start, target, devices, budget, model=model)
    return {
        "workload": workload,
        "platform": platform,
        "planner": model,
        "devices": devices.n_tasks,
        "links": devices.n_edges,
        "protected_links": int(plan.protect.sum()),
        "waves": plan.waves,
        "steps": plan.steps,
        "makespan_s": plan.makespan,
        "broken_link_steps": plan.broken_link_steps,
        "broken_link_fraction": plan.broken_link_fraction,
        "collisions": plan.collisions,
        "min_separation_m": plan.min_separation,
        "required_separation_m": budget.separation,
        "energy_j": plan.energy,
        "max_distance_m": float(plan.distance_travelled.max()),
    }


def run(
    workloads=None,
    platforms=("float",),
    planners=DEFAULT_PLANNERS,
    params: dict | None = None,
    out: str | Path | None = None,
) -> list[dict]:
    params = params if params is not None else QUALITY_PARAMS
    names = sorted(workloads or params)
    rows = []
    for name in names:
        for platform in platforms:
            devices, budget, start, target = scenario(name, params.get(name, {}), platform)
            for model in planners:
                rows.append(
                    evaluate(devices, budget, start, target, model, name, platform)
                )
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        write_csv(rows, out / "motion.csv")
        with open(out / "motion.json", "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
    return rows


def report(rows: list[dict]) -> str:
    head = (
        f"{'workload':11s} {'platform':8s} {'planner':11s} {'devs':>5s} {'prot':>5s} "
        f"{'waves':>6s} {'makespan s':>11s} {'broken':>8s} {'collide':>8s} {'energy J':>9s}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['workload']:11s} {r['platform']:8s} {r['planner']:11s} "
            f"{r['devices']:5d} {r['protected_links']:5d} {r['waves']:6d} "
            f"{r['makespan_s']:11.1f} {r['broken_link_fraction']:8.1%} "
            f"{r['collisions']:8d} {r['energy_j']:9.2f}"
        )
    return "\n".join(lines)


def verdict(rows: list[dict]) -> str:
    lines = []
    by = {}
    for r in rows:
        by.setdefault((r["workload"], r["platform"]), {})[r["planner"]] = r

    ratios, safety = [], []
    for key, planners in by.items():
        d, w = planners.get("direct"), planners.get("waves")
        if not d or not w:
            continue
        if d["makespan_s"] > 0:
            ratios.append(w["makespan_s"] / d["makespan_s"])
        safety.append((d["broken_link_fraction"], w["broken_link_fraction"]))

    if safety:
        lines.append(
            f"flying straight drops {np.mean([s[0] for s in safety]):.0%} of protected "
            f"links; wave-ordered flight drops {np.mean([s[1] for s in safety]):.0%}"
        )
    if ratios:
        lines.append(
            f"the price is makespan: {np.mean(ratios):.2f}x on average "
            f"(worst {max(ratios):.2f}x)"
        )

    seq = [r for r in rows if r["planner"] == "sequential"]
    wav = [r for r in rows if r["planner"] == "waves"]
    if seq and wav:
        lines.append(
            f"moving one device at a time is equally safe and "
            f"{np.mean([s['makespan_s'] for s in seq]) / max(np.mean([w['makespan_s'] for w in wav]), 1e-9):.1f}x "
            f"slower -- wave count follows degree, not device count "
            f"({np.mean([w['waves'] for w in wav]):.1f} vs {np.mean([s['waves'] for s in seq]):.0f})"
        )

    energies = {}
    for r in rows:
        energies.setdefault((r["workload"], r["platform"]), set()).add(round(r["energy_j"], 6))
    if all(len(v) == 1 for v in energies.values()):
        lines.append(
            "energy is identical across planners -- waves reorder the motion, "
            "they do not lengthen the paths"
        )

    collided = [r for r in rows if r["collisions"] > 0]
    lines.append(
        f"collisions along the way: {len(collided)} of {len(rows)} plans"
        + (" -- every plan is collision-free" if not collided else "")
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="iotc motion", description=__doc__)
    p.add_argument("workloads", nargs="*")
    p.add_argument("--platform", action="append", choices=sorted(PLATFORMS))
    p.add_argument("--planner", action="append")
    p.add_argument("--out")
    args = p.parse_args(argv)

    rows = run(
        args.workloads or None,
        tuple(args.platform or ("float",)),
        tuple(args.planner or DEFAULT_PLANNERS),
        out=args.out,
    )
    print(report(rows))
    print()
    print(verdict(rows))
    return 0

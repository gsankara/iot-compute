"""B3 evaluation -- what the projection costs to fly, and whether it can talk.

A placement is scored on three things that do not reduce to one another:

``motion energy``
    Joules to fly the formation from a common origin, charged per axis at that
    platform's real rate.  This is where the anisotropy shows up, because a
    float pays nothing for depth and a great deal for lateral travel.

``unreachable links``
    Edges longer than the optical budget can close.  Not a quality measure -- a
    single one is a disconnected machine -- so the placers are all held to zero
    and the comparison happens on energy.

``weighted link length``
    Mean link length weighted by payload.  Shorter is better for margin: Phase A
    measured that a 0.04 m/s drift destroys an open-loop link, so a formation
    sitting at the edge of its reach is fragile even when it closes today.

The comparator that matters is ``velocity``.  It is the *pre-correction* design
-- scale each axis by how fast the body moves along it -- and running it beside
``anisotropic`` puts a number on the Phase A finding that z is slow but free
while xy is fast but rationed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..compiler import build_workload
from ..compiler.partition import partition
from ..compiler.place import derive_budget, place
from ..devices import MOTE_C, aerial, aquatic, tethered
from ..registry import create
from .sim_bench import write_csv

DEFAULT_PLACERS = ("anisotropic", "velocity", "isotropic")

#: The three platforms, each with the medium it actually lives in.  Running all
#: three is the point: their anisotropies differ in *direction*, not just
#: degree, so a placer tuned to one must be shown against the others.
PLATFORMS = {
    "float": (lambda: aquatic(MOTE_C), "water"),
    "drone": (lambda: aerial(MOTE_C), "air"),
    "tether": (lambda: tethered(MOTE_C), "water"),
}

QUALITY_PARAMS = {
    "stencil3d": dict(nx=128, ny=128, nz=128, tx=8, ty=8, tz=8),
    "stencil2d": dict(nx=2048, ny=2048, tx=16, ty=16),
    "spmv": dict(n=1 << 16, parts=256),
    "maxclique": dict(n=1 << 13, parts=128),
}


def evaluate(devices, device_class, medium, model: str, platform: str) -> dict:
    result = place(devices, device_class, medium, model=model, seed=0)
    extent = result.extent
    return {
        "workload": devices.name.split("[")[0].split("/")[0],
        "platform": platform,
        "placer": model,
        "devices": devices.n_tasks,
        "links": devices.n_edges,
        "anisotropy": result.budget.anisotropy,
        "max_link_m": result.budget.max_link,
        "extent_x_m": float(extent[0]),
        "extent_y_m": float(extent[1]),
        "extent_z_m": float(extent[2]),
        "weighted_link_m": result.weighted_link_length,
        "longest_link_m": float(result.link_lengths.max()) if devices.n_edges else 0.0,
        "unreachable": result.unreachable,
        "unreachable_fraction": result.unreachable_fraction,
        "motion_energy_j": result.motion_energy,
        "min_separation_m": result.min_separation,
    }


def run(
    workloads=None,
    platforms=tuple(PLATFORMS),
    placers=DEFAULT_PLACERS,
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
            for model in placers:
                rows.append(evaluate(devices, cls, medium, model, platform))

    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        write_csv(rows, out / "placement.csv")
        with open(out / "placement.json", "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
    return rows


def budget_report() -> str:
    """What each platform can actually reach.  The input to every projection."""
    lines = [
        f"{'platform':9s} {'xy reach':>10s} {'z reach':>10s} {'anisotropy':>11s} "
        f"{'optical':>9s} {'xy J/m':>9s} {'z J/m':>9s}",
        "-" * 72,
    ]
    for name, (factory, medium_name) in PLATFORMS.items():
        b = derive_budget(factory(), create("medium", medium_name))
        lines.append(
            f"{name:9s} {b.xy_reach:10.2f} {b.z_reach:10.2f} {b.anisotropy:11.2f} "
            f"{b.max_link:9.2f} {b.xy_cost_per_m:9.3f} {b.z_cost_per_m:9.3f}"
        )
    return "\n".join(lines)


def report(rows: list[dict], platform: str | None = None) -> str:
    shown = [r for r in rows if platform in (None, r["platform"])]
    head = (
        f"{'workload':11s} {'platform':8s} {'placer':12s} {'extent x/y/z (m)':>26s} "
        f"{'link m':>8s} {'unreach':>8s} {'energy J':>10s}"
    )
    lines = [head, "-" * len(head)]
    for r in shown:
        extent = f"{r['extent_x_m']:7.1f} {r['extent_y_m']:7.1f} {r['extent_z_m']:9.1f}"
        lines.append(
            f"{r['workload']:11s} {r['platform']:8s} {r['placer']:12s} {extent:>26s} "
            f"{r['weighted_link_m']:8.2f} {r['unreachable_fraction']:7.1%} "
            f"{r['motion_energy_j']:10.2f}"
        )
    return "\n".join(lines)


def verdict(rows: list[dict]) -> str:
    lines = []
    broken = [r for r in rows if r["unreachable"] > 0]
    lines.append(
        f"links beyond optical reach: {len(broken)} of {len(rows)} placements"
        + (" -- every placement is connected" if not broken else "")
    )

    for platform in sorted({r["platform"] for r in rows}):
        rows_p = [r for r in rows if r["platform"] == platform]
        by = {}
        for r in rows_p:
            by.setdefault(r["placer"], []).append(r["motion_energy_j"])
        totals = {k: float(np.sum(v)) for k, v in by.items()}
        if not totals or max(totals.values()) <= 0:
            lines.append(f"{platform}: motion is free on this platform, energy cannot rank it")
            continue
        best = min(totals, key=totals.get)
        naive = totals.get("velocity")
        line = f"{platform}: cheapest placer is {best} ({totals[best]:.2f} J total)"
        if naive and totals[best] > 0 and best != "velocity":
            line += f", {naive / totals[best]:.2f}x cheaper than the velocity reading"
        elif best == "velocity":
            line += " -- the velocity reading is not penalised on this platform"
        lines.append(line)

    aniso = {r["platform"]: r["anisotropy"] for r in rows}
    lines.append(
        "platform anisotropy (z reach / xy reach): "
        + ", ".join(f"{p} {a:.2f}" for p, a in sorted(aniso.items()))
        + " -- the axes do not favour the same direction on every platform"
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="iotc place", description=__doc__)
    p.add_argument("workloads", nargs="*")
    p.add_argument("--platform", action="append", choices=sorted(PLATFORMS))
    p.add_argument("--placer", action="append")
    p.add_argument("--out")
    args = p.parse_args(argv)

    print(budget_report())
    print()
    rows = run(
        args.workloads or None,
        tuple(args.platform or PLATFORMS),
        tuple(args.placer or DEFAULT_PLACERS),
        out=args.out,
    )
    for platform in sorted({r["platform"] for r in rows}):
        print(report(rows, platform))
        print()
    print(verdict(rows))
    return 0

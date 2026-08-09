"""B1 evaluation -- what a tiling costs, and whether the shape of the cut matters.

B1 has a pass/fail condition (does every task fit?) and a quality metric (how
many bytes did making it fit cost?).  A pass that only reported the first would
hide the interesting half: *every* tiler can make a workload fit by subdividing
far enough, and they differ by multiples in what that does to the traffic.

So the bench runs each workload against each device class under each tiler and
reports the pair together.  The comparator is ``slab`` -- cut along one axis,
the tiling you get from treating "make it fit" as a division problem.
``geometric`` is the pass proper.  The gap between them is the pass's value.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..compiler import build_workload
from ..compiler.tile import tile
from ..devices import get_class
from ..registry import available
from ..units import KiB, MiB
from .sim_bench import write_csv

DEFAULT_CLASSES = ("mote-c", "mote-m")
DEFAULT_MODELS = ("geometric", "planar", "slab")

#: Instance sizes that actually overflow the parts under test.
#:
#: A tiling bench run on problems that already fit measures nothing -- every
#: tiler scores identically because none of them cuts anything.  These are the
#: default sizes so the pass is evaluated where it does work; pass ``params`` to
#: override.  Each is a problem a single mote-c plainly cannot hold: a 256^3
#: double-buffered grid is 16 MiB against 512 KiB of RAM.
STRESS_PARAMS = {
    "stencil3d": dict(nx=256, ny=256, nz=256, tx=2, ty=2, tz=2),
    "stencil2d": dict(nx=4096, ny=4096, tx=2, ty=2),
    "gemm": dict(n=4096, p=2),
    "fft": dict(n=1 << 22, p=4),
    "allreduce": dict(count=1 << 20, p=8),
    "spmv": dict(n=1 << 18, parts=8),
    "bfs": dict(n=1 << 18, parts=8),
    "maxclique": dict(n=1 << 15, parts=8),
    "mis": dict(n=1 << 15, parts=8),
    "coloring": dict(n=1 << 15, parts=8),
    "kmedian": dict(n=1 << 15, parts=8),
    "maxcut": dict(n=1 << 15, parts=8),
}


def evaluate(graph, device_class, model: str) -> dict:
    """One row: tile ``graph`` for ``device_class`` with ``model`` and score it."""
    result = tile(graph, device_class, model=model)
    tiled = result.graph
    ram = float(device_class.compute.ram_bytes)

    return {
        "workload": graph.name.split("[")[0],
        "device_class": device_class.name,
        "tiler": model,
        "tasks_before": graph.n_tasks,
        "tasks_after": tiled.n_tasks,
        "edges_before": graph.n_edges,
        "edges_after": tiled.n_edges,
        "max_factor": result.max_factor,
        "cut_shape": "x".join(str(f) for f in result.plans[0].factors) if result.plans else "-",
        "rounds": result.rounds,
        "converged": result.converged,
        "fits": bool(tiled.fits(device_class).all()),
        "reason": result.reason,
        "halos_added": result.added_traffic_halos,
        "ws_max_kib": float(tiled.working_set.max() / KiB),
        "ram_kib": ram / KiB,
        "mib_before": graph.total_bytes / MiB,
        "mib_after": tiled.total_bytes / MiB,
        "traffic_ratio": (
            tiled.total_bytes / graph.total_bytes if graph.total_bytes > 0 else 1.0
        ),
        "ai_before": graph.arithmetic_intensity,
        "ai_after": tiled.arithmetic_intensity,
        "degree_max_after": int(tiled.degree().max()) if tiled.n_edges else 0,
        "aim_fraction_after": tiled.aim_fraction(device_class),
    }


def run(
    workloads=None,
    classes=DEFAULT_CLASSES,
    models=DEFAULT_MODELS,
    params: dict | None = None,
    out: str | Path | None = None,
) -> list[dict]:
    names = sorted(workloads or available("workload"))
    params = params if params is not None else STRESS_PARAMS
    rows = []
    for name in names:
        graph = build_workload(name, **params.get(name, {}))
        for cls_name in classes:
            for model in models:
                rows.append(evaluate(graph, get_class(cls_name), model))

    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        write_csv(rows, out / "tiling.csv")
        with open(out / "tiling.json", "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
    return rows


def report(rows: list[dict], device_class: str | None = None) -> str:
    shown = [r for r in rows if device_class in (None, r["device_class"])]
    head = (
        f"{'workload':11s} {'class':7s} {'tiler':10s} {'tasks':>12s} {'cut':>10s} "
        f"{'halos':>6s} {'MiB':>10s} {'x traffic':>9s} {'AI':>8s} {'fit':>4s}"
    )
    lines = [head, "-" * len(head)]
    for r in shown:
        arrow = f"{r['tasks_before']}->{r['tasks_after']}"
        fit = "ok" if r["fits"] else "NO"
        lines.append(
            f"{r['workload']:11s} {r['device_class']:7s} {r['tiler']:10s} {arrow:>12s} "
            f"{r['cut_shape']:>10s} {r['halos_added']:6d} {r['mib_after']:10.1f} "
            f"{r['traffic_ratio']:9.2f} {r['ai_after']:8.2f} {fit:>4s}"
        )
    return "\n".join(lines)


def verdict(rows: list[dict]) -> str:
    """State the comparison the bench exists to make."""
    lines = []
    needed = [r for r in rows if r["max_factor"] > 1]
    if not needed:
        return "no workload needed tiling on any class tested"

    geo = {(r["workload"], r["device_class"]): r for r in needed if r["tiler"] == "geometric"}
    slab = {(r["workload"], r["device_class"]): r for r in needed if r["tiler"] == "slab"}
    shared = sorted(set(geo) & set(slab))

    ratios, task_ratios = [], []
    for key in shared:
        g, s = geo[key], slab[key]
        if g["mib_after"] > 0:
            ratios.append(s["mib_after"] / g["mib_after"])
            task_ratios.append(s["tasks_after"] / max(g["tasks_after"], 1))

    if ratios:
        lines.append(
            f"geometric vs slab over {len(ratios)} tiled cases: "
            f"{np.mean(ratios):.2f}x less traffic on average "
            f"(best {max(ratios):.2f}x), {np.mean(task_ratios):.2f}x fewer pieces"
        )

    infeasible = sorted({
        (r["workload"], r["device_class"]) for r in rows if not r["converged"]
    })
    if infeasible:
        lines.append(
            "cannot be tiled onto the part at all (halo exceeds RAM): "
            + ", ".join(f"{w} on {c}" for w, c in infeasible)
        )

    worst = max(needed, key=lambda r: r["traffic_ratio"])
    lines.append(
        f"worst traffic amplification: {worst['workload']} under {worst['tiler']} "
        f"on {worst['device_class']}, {worst['traffic_ratio']:.1f}x"
    )
    still_aim = [r for r in rows if r["fits"] and r["aim_fraction_after"] > 0.9]
    lines.append(
        f"{len(still_aim)} of {len([r for r in rows if r['fits']])} fitted tilings are "
        f"still aim-dominated -- tiling does not rescue B6"
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="iotc tile", description=__doc__)
    p.add_argument("workloads", nargs="*")
    p.add_argument("--class", dest="classes", action="append")
    p.add_argument("--tiler", action="append")
    p.add_argument("--out")
    args = p.parse_args(argv)

    rows = run(
        args.workloads or None,
        tuple(args.classes or DEFAULT_CLASSES),
        tuple(args.tiler or DEFAULT_MODELS),
        out=args.out,
    )
    for cls in sorted({r["device_class"] for r in rows}):
        print(report(rows, cls))
        print()
    print(verdict(rows))
    return 0

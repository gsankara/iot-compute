"""B2 evaluation -- cut quality, capacity feasibility, and the scaling study.

The roadmap sets B2 an acceptance criterion that is not about quality at all:
*it must demonstrate near-linear runtime.*  So this bench has two halves and the
second is the one that decides the pass.

**Quality**, at fixed device count: cut payload as a fraction of total traffic,
the balance of the heaviest group, and whether every group actually fits.  The
comparators are `block` (contiguous ids), `spectral` (recursive Fiedler
bisection) and `kernighan_lin`.

**Scaling**: the same partitioner run over a geometric ladder of problem sizes,
with a log-log slope fitted to the wall-clock.  A slope near 1 is near-linear;
spectral and Kernighan-Lin are carried precisely because they are not, and the
number puts a figure on it rather than an assertion.

A note on reading the quality table.  `block` is *not* a weak baseline on the
lattice workloads: the frontends number their tiles in lattice order, so
contiguous id ranges are already coherent 3D blocks.  Beating it there means
beating a partition that was handed the geometry for free.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..compiler import build_workload
from ..compiler.partition import minimum_devices, partition
from ..devices import get_class
from ..units import MiB
from .sim_bench import write_csv

DEFAULT_METHODS = ("block", "fmbm", "spectral", "kernighan_lin")

#: Instances with many tasks per device, which is where grouping has anything to
#: decide.  A graph whose tasks each fill a device leaves B2 no freedom -- k is
#: forced to n and every edge is cut whatever the partitioner does.
QUALITY_PARAMS = {
    "stencil3d": dict(nx=128, ny=128, nz=128, tx=8, ty=8, tz=8),
    "stencil2d": dict(nx=2048, ny=2048, tx=16, ty=16),
    "gemm": dict(n=1024, p=8),
    "spmv": dict(n=1 << 16, parts=256),
    "bfs": dict(n=1 << 16, parts=256),
    "maxclique": dict(n=1 << 13, parts=128),
    "mis": dict(n=1 << 13, parts=128),
    "maxcut": dict(n=1 << 13, parts=128),
}

#: Cube sides for the scaling ladder, giving 512 / 1728 / 4096 / 8000 tasks.
#: Small instances are dominated by fixed costs -- FastMap pays for kappa passes
#: whatever the size -- so a ladder that stops early measures overhead and calls
#: it complexity.  These reach far enough for the asymptotics to show.
SCALING_SIDES = (8, 12, 16, 20)


def evaluate(graph, device_class, method: str, seed: int = 0) -> dict:
    result = partition(graph, device_class, model=method, seed=seed)
    return {
        "workload": graph.name.split("[")[0],
        "device_class": device_class.name,
        "method": method,
        "tasks": graph.n_tasks,
        "edges": graph.n_edges,
        "k": result.k,
        "groups_used": result.occupied,
        "min_devices": minimum_devices(graph, device_class),
        "cut_fraction": result.cut_fraction,
        "cut_mib": result.cut_bytes / MiB,
        "balance": result.balance,
        "feasible": result.feasible,
        "seconds": result.seconds,
    }


def run(
    workloads=None,
    device_class: str = "mote-c",
    methods=DEFAULT_METHODS,
    params: dict | None = None,
    seed: int = 0,
    out: str | Path | None = None,
) -> list[dict]:
    params = params if params is not None else QUALITY_PARAMS
    names = sorted(workloads or params)
    cls = get_class(device_class)
    rows = [
        evaluate(build_workload(name, **params.get(name, {})), cls, method, seed)
        for name in names
        for method in methods
    ]
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        write_csv(rows, out / "partition.csv")
        with open(out / "partition.json", "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
    return rows


def scaling(
    methods=DEFAULT_METHODS,
    sides=SCALING_SIDES,
    device_class: str = "mote-c",
    seed: int = 0,
    out: str | Path | None = None,
) -> list[dict]:
    """Time each partitioner over a ladder of problem sizes.

    The workload is a 3D stencil refined in place, so task count grows while the
    structure stays comparable -- the cost difference measured is the
    partitioner's, not the instance's.

    Each method is run once untimed before the ladder starts.  Without that,
    the first measurement carries import, JIT and allocator warm-up, and on a
    short ladder a single inflated first point is enough to invert the fitted
    slope -- which is exactly the number this bench exists to report.
    """
    cls = get_class(device_class)
    rows = []

    warm = build_workload("stencil3d", nx=32, ny=32, nz=32, tx=4, ty=4, tz=4)
    for method in methods:
        partition(warm, cls, model=method, seed=seed)

    for side in sides:
        graph = build_workload(
            "stencil3d", nx=8 * side, ny=8 * side, nz=8 * side,
            tx=side, ty=side, tz=side,
        )
        for method in methods:
            start = time.perf_counter()
            result = partition(graph, cls, model=method, seed=seed)
            rows.append({
                "method": method,
                "side": side,
                "tasks": graph.n_tasks,
                "edges": graph.n_edges,
                "k": result.k,
                "seconds": time.perf_counter() - start,
                "cut_fraction": result.cut_fraction,
            })
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        write_csv(rows, out / "scaling.csv")
        with open(out / "scaling.json", "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
    return rows


def fitted_exponent(rows: list[dict], method: str) -> float:
    """Slope of log(time) against log(tasks) -- the empirical complexity."""
    pts = [(r["tasks"], r["seconds"]) for r in rows if r["method"] == method]
    pts = [(n, t) for n, t in pts if n > 0 and t > 0]
    if len(pts) < 2:
        return float("nan")
    x = np.log(np.array([n for n, _ in pts], dtype=float))
    y = np.log(np.array([t for _, t in pts], dtype=float))
    return float(np.polyfit(x, y, 1)[0])


def report(rows: list[dict]) -> str:
    head = (
        f"{'workload':11s} {'method':14s} {'tasks':>6s} {'k':>5s} {'cut':>7s} "
        f"{'cut MiB':>9s} {'balance':>7s} {'fits':>5s} {'ms':>9s}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['workload']:11s} {r['method']:14s} {r['tasks']:6d} {r['k']:5d} "
            f"{r['cut_fraction']:7.3f} {r['cut_mib']:9.2f} {r['balance']:7.2f} "
            f"{'ok' if r['feasible'] else 'NO':>5s} {r['seconds'] * 1e3:9.1f}"
        )
    return "\n".join(lines)


def scaling_report(rows: list[dict]) -> str:
    methods = sorted({r["method"] for r in rows})
    sizes = sorted({r["tasks"] for r in rows})
    head = f"{'method':14s} " + "".join(f"{n:>10d}" for n in sizes) + f"{'exponent':>10s}"
    lines = [head, "-" * len(head)]
    for m in methods:
        by_size = {r["tasks"]: r["seconds"] for r in rows if r["method"] == m}
        cells = "".join(f"{by_size.get(n, float('nan')) * 1e3:10.1f}" for n in sizes)
        lines.append(f"{m:14s} {cells}{fitted_exponent(rows, m):10.2f}")
    lines.append("")
    lines.append("wall-clock in ms; exponent is the log-log slope against task count")
    return "\n".join(lines)


def verdict(quality: list[dict], scale: list[dict]) -> str:
    lines = []
    exps = {m: fitted_exponent(scale, m) for m in sorted({r["method"] for r in scale})}
    fmbm = exps.get("fmbm", float("nan"))
    lines.append(
        "empirical complexity: "
        + ", ".join(f"{m} n^{e:.2f}" for m, e in sorted(exps.items()))
    )
    if fmbm == fmbm:  # not NaN
        slower = [m for m, e in exps.items() if m != "fmbm" and e > fmbm + 0.3]
        lines.append(
            f"fmbm is near-linear at n^{fmbm:.2f}"
            + (f"; {', '.join(slower)} grow strictly faster" if slower else "")
        )

    by_wl: dict[str, dict[str, dict]] = {}
    for r in quality:
        by_wl.setdefault(r["workload"], {})[r["method"]] = r
    wins = [w for w, m in by_wl.items()
            if "fmbm" in m and "block" in m
            and m["fmbm"]["cut_fraction"] < m["block"]["cut_fraction"]]
    losses = [w for w, m in by_wl.items()
              if "fmbm" in m and "block" in m
              and m["fmbm"]["cut_fraction"] > m["block"]["cut_fraction"]]
    lines.append(
        f"cut vs the block baseline: fmbm wins on {len(wins)} of {len(by_wl)} "
        f"({', '.join(sorted(wins)) or 'none'})"
        + (f"; loses on {', '.join(sorted(losses))}" if losses else "")
    )

    infeasible = sorted({
        (r["method"], r["workload"]) for r in quality if not r["feasible"]
    })
    if infeasible:
        by_method: dict[str, list[str]] = {}
        for m, w in infeasible:
            by_method.setdefault(m, []).append(w)
        lines.append(
            "produced groups that do not fit the device: "
            + "; ".join(f"{m} on {len(w)} workloads" for m, w in sorted(by_method.items()))
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="iotc partition", description=__doc__)
    p.add_argument("workloads", nargs="*")
    p.add_argument("--class", dest="device_class", default="mote-c")
    p.add_argument("--method", action="append")
    p.add_argument("--out")
    p.add_argument("--no-scaling", action="store_true")
    args = p.parse_args(argv)

    methods = tuple(args.method or DEFAULT_METHODS)
    quality = run(args.workloads or None, args.device_class, methods, out=args.out)
    print(report(quality))
    print()

    scale = []
    if not args.no_scaling:
        scale = scaling(methods, device_class=args.device_class, out=args.out)
        print(scaling_report(scale))
        print()
    print(verdict(quality, scale))
    return 0

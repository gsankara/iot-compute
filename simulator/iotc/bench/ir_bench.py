"""B0 evaluation -- workload characterization against real device classes.

Every compiler pass ships a bench reporting its own quality metric.  B0 builds
no schedule and makes no choice, so its metric is not "how good is the output"
but **"what does this problem demand, and can this machine supply it?"**  Three
numbers answer that, and each one determines which later pass has work to do:

``fit``
    Fraction of tasks whose working set fits the class's RAM.  Below 1.0, B1
    (Tile) must split further before anything else can proceed.

``arithmetic intensity``
    FLOP per communicated byte.  Sets whether the problem is compute- or
    communication-bound before any placement is attempted.

``aim fraction``
    Share of link time spent slewing gimbals rather than sending.  Phase A
    measured aiming at ~0.5 s against microseconds of transmission, so this is
    near 1.0 for most workloads -- and where it is, B6's whole job is to
    amortise retargeting.

The output is a table where each row is a (workload, device class) pair, which
is the form B1 and B3 consume when deciding a tiling target.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..compiler import build_workload
from ..devices import get_class
from ..registry import available
from ..units import KiB, MiB
from .sim_bench import write_csv

#: The classes a workload is scored against by default: the small compute part,
#: the memory-rich part, and the relay.  Between them they bracket what the
#: heterogeneous swarm can actually offer a task.
DEFAULT_CLASSES = ("mote-c", "mote-m", "relay")


def characterize(graph, device_class) -> dict:
    """One row: what ``graph`` demands of ``device_class``."""
    fits = graph.fits(device_class)
    ws = graph.working_set
    compute = graph.compute_time(device_class)
    wire = graph.transfer_time(device_class)
    ram = float(device_class.compute.ram_bytes)

    return {
        "workload": graph.name.split("[")[0],
        "instance": graph.name,
        "device_class": device_class.name,
        "tasks": graph.n_tasks,
        "edges": graph.n_edges,
        "iterations": graph.iterations,
        "degree_mean": float(graph.degree().mean()),
        "degree_max": int(graph.degree().max()) if graph.n_edges else 0,
        "ws_max_kib": float(ws.max() / KiB),
        "ram_kib": ram / KiB,
        "ws_over_ram": float(ws.max() / ram),
        "fit_fraction": float(fits.mean()),
        "tiles_needed": int(np.ceil(ws.max() / ram)),
        "gflop_total": graph.total_flop / 1e9,
        "mib_total": graph.total_bytes / MiB,
        "arithmetic_intensity": graph.arithmetic_intensity,
        "compute_time_s": float(compute.sum()) * graph.iterations,
        "wire_time_s": float(wire.sum()) * graph.iterations,
        "aim_fraction": graph.aim_fraction(device_class),
        # Transceivers are the hard limit on how many neighbours a task may
        # keep at once; above it B5 has to time-share a beam.
        "transceivers": device_class.optical.n_transceivers,
        "degree_over_transceivers": (
            float(graph.degree().max()) / device_class.optical.n_transceivers
            if graph.n_edges else 0.0
        ),
    }


def run(
    workloads=None,
    classes=DEFAULT_CLASSES,
    params: dict | None = None,
    out: str | Path | None = None,
) -> list[dict]:
    """Characterize every workload against every class and report."""
    names = sorted(workloads or available("workload"))
    params = params or {}
    rows = []
    for name in names:
        graph = build_workload(name, **params.get(name, {}))
        for cls_name in classes:
            rows.append(characterize(graph, get_class(cls_name)))

    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        write_csv(rows, out / "characterization.csv")
        with open(out / "characterization.json", "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
    return rows


def report(rows: list[dict], device_class: str | None = None) -> str:
    """Console table.  Restrict to one class to keep it readable."""
    shown = [r for r in rows if device_class in (None, r["device_class"])]
    head = (
        f"{'workload':11s} {'class':9s} {'tasks':>6s} {'edges':>6s} {'deg':>4s} "
        f"{'ws/RAM':>7s} {'fit':>5s} {'tiles':>5s} {'AI':>9s} {'aim':>6s} {'deg/tx':>6s}"
    )
    lines = [head, "-" * len(head)]
    for r in shown:
        lines.append(
            f"{r['workload']:11s} {r['device_class']:9s} {r['tasks']:6d} {r['edges']:6d} "
            f"{r['degree_max']:4d} {r['ws_over_ram']:7.2f} {r['fit_fraction']:5.2f} "
            f"{r['tiles_needed']:5d} {r['arithmetic_intensity']:9.2f} "
            f"{r['aim_fraction']:6.3f} {r['degree_over_transceivers']:6.1f}"
        )
    return "\n".join(lines)


def verdict(rows: list[dict]) -> str:
    """What the numbers imply for the passes that follow.

    Stated as consequences rather than statistics: a bench that reports only
    numbers leaves the reader to guess which pass has to act on them.
    """
    lines = []
    by_workload: dict[str, list[dict]] = {}
    for r in rows:
        by_workload.setdefault(r["workload"], []).append(r)

    needs_tiling = sorted(
        w for w, rs in by_workload.items() if any(r["fit_fraction"] < 1.0 for r in rs)
    )
    aim_bound = sorted(
        w for w, rs in by_workload.items()
        if np.mean([r["aim_fraction"] for r in rs]) > 0.9
    )
    over_degree = sorted(
        w for w, rs in by_workload.items()
        if max(r["degree_over_transceivers"] for r in rs) > 1.0
    )

    lines.append(f"B1 must tile ({len(needs_tiling)}): {', '.join(needs_tiling) or 'none'}")
    lines.append(f"B6 must amortise aiming ({len(aim_bound)}): {', '.join(aim_bound) or 'none'}")
    lines.append(
        f"B5 must time-share beams ({len(over_degree)}): {', '.join(over_degree) or 'none'}"
    )
    intensities = {w: rs[0]["arithmetic_intensity"] for w, rs in by_workload.items()}
    best = max(intensities, key=intensities.get)
    worst = min(intensities, key=intensities.get)
    lines.append(
        f"intensity spans {intensities[worst]:.2f} ({worst}) to "
        f"{intensities[best]:.1f} ({best}) FLOP/B"
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="iotc graph", description=__doc__)
    p.add_argument("workloads", nargs="*", help="workload names (default: all)")
    p.add_argument("--class", dest="classes", action="append",
                   help="device class to score against (repeatable)")
    p.add_argument("--out", help="write characterization.csv/.json here")
    args = p.parse_args(argv)

    rows = run(args.workloads or None, tuple(args.classes or DEFAULT_CLASSES), out=args.out)
    print(report(rows))
    print()
    print(verdict(rows))
    return 0

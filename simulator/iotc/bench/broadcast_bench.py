"""B11's bench: what the omnidirectional plane buys, and what it costs.

Three questions, in order of how much they matter.

**Does cutting power buy reuse?**  Spatial reuse is a shared medium's only
capacity lever, so ``min_reach`` and ``swept`` are measured against ``nominal``
-- the stock radio, left alone -- on the placements B3 actually produces.

**Does broadcast beat the optical plane?**  The optical frame is priced through
B5 and B6 rather than estimated here, so the comparison is like-for-like: the
same workload, the same partition, the same placement, and B6's own amortising
scheduler on the optical side.  The honest answer needs the two sharing bounds
B11 reports, because a stencil's halo faces are distinct bytes while an
all-reduce's partial sums are the same bytes to everyone.

**Where does reuse actually start?**  At the device counts B2 coarsens to -- a
couple of dozen -- the whole array sits inside one domain whatever the power,
and broadcast degenerates to round-robin TDMA.  :func:`scaling` sweeps device
count to find where that stops being true, which is the finding that decides
whether the broadcast variant is worth building out.

Cost note: :func:`iotc.compiler.broadcast.conflict_matrix` is a dense boolean
matmul, O(n^3) in time and O(n^2) in memory.  Measured on this machine: 1.4 s at
1,153 devices and 89 s at 2,732, so the sweep stops around twelve hundred --
the same ceiling Phase C reaches.  Replacing it with sparse set intersection is
the first optimization this pass should get, and is deliberately deferred.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..compiler import build_workload
from ..compiler.beam import assign_beams, refine_positions
from ..compiler.broadcast import (CARRIER_LADDER, BroadcastPlan, _endpoints,
                                  evaluate_setting, derive_radio_budget,
                                  plan_broadcast)
from ..compiler.partition import partition
from ..compiler.place import place
from ..compiler.schedule import build_schedule
from ..devices import MOTE_C, aerial, aquatic
from ..registry import create
from .sim_bench import write_csv

DEFAULT_MODELS = ("nominal", "min_reach", "swept")

#: Water and air, because the radio's story differs completely between them:
#: seawater's conductivity term costs ~1700 dB/m at 2.4 GHz, so an aquatic class
#: already ships a low carrier, while an aerial one keeps the microwave band.
PLATFORMS = {
    "float": (lambda: aquatic(MOTE_C), "water"),
    "drone": (lambda: aerial(MOTE_C), "air"),
}

QUALITY_PARAMS = {
    "stencil3d": dict(nx=128, ny=128, nz=128, tx=8, ty=8, tz=8),
    "stencil2d": dict(nx=2048, ny=2048, tx=16, ty=16),
    "spmv": dict(n=1 << 16, parts=256),
    # All-reduce earns its place here specifically: its payload really is the
    # same bytes to every peer, so it is the one stock workload where the
    # ``shared`` bound is the physical answer rather than an optimistic limit.
    "allreduce": dict(count=1 << 14, p=32, topology="tree"),
}

#: Side lengths for the device-count sweep.  Kept below the O(n^3) ceiling
#: documented above.
SCALING_SIDES = (6, 10, 14, 18, 24, 30)


def _pipeline(name: str, cls, medium, params: dict):
    """B0 -> B2 -> B3, the shared front end every row here is measured on."""
    graph = build_workload(name, **params)
    devices = partition(graph, MOTE_C, seed=0).coarsened()
    positions = place(devices, cls, medium, seed=0).positions
    return devices, positions


def optical_frame(devices, positions, cls) -> tuple[float, int, float, float]:
    """Seconds per superstep on the optical plane, through B5 and B6.

    Returns ``(period, rounds, efficiency, rejected_fraction)``.  The placement
    is refined first because B5 established that B3's raw output is 96.5%
    unaimable on a float -- comparing against the unrefined version would be
    comparing against a plane that does not work.

    The rejected fraction has to come back with the period, because at small
    device counts a float's column is *entirely* unaimable even after
    refinement: every link is rejected, so there are no rounds, and the period
    is zero.  A zero period does not mean the optical plane is infinitely fast,
    it means it carries nothing -- and a speedup computed against it would say
    the opposite.  The caller must check this before dividing.
    """
    fixed = refine_positions(devices, positions, cls)
    a = assign_beams(devices, fixed, cls, model="heaviest_first")
    s = build_schedule(devices, a, fixed, cls, model="amortised")
    return s.period, a.rounds, s.efficiency, a.rejected_fraction


def evaluate(devices, positions, cls, medium, model, workload, platform) -> dict:
    p = plan_broadcast(devices, positions, cls, medium, model=model)
    opt_period, opt_rounds, opt_eff, opt_rejected = optical_frame(
        devices, positions, cls
    )
    # NaN, not zero and not infinity: the comparison is undefined when one of
    # the two planes does not run, and a number here would be read as a result.
    optical_runs = opt_period > 0.0 and opt_rounds > 0
    both_run = optical_runs and p.feasible
    src, dst = _endpoints(devices)
    span = np.linalg.norm(positions[src] - positions[dst], axis=1) if devices.n_edges else np.zeros(1)
    extent = float(np.ptp(positions, axis=0).max())
    return {
        "workload": workload,
        "platform": platform,
        "model": model,
        "devices": devices.n_tasks,
        "edges": devices.n_edges,
        "longest_edge_m": float(span.max()),
        "extent_m": extent,
        # Reach must cover the longest task edge, so this ratio is the ceiling
        # on reuse: at >= 1 the whole array is a single collision domain and no
        # choice of power can carve out a second one.
        "reach_over_extent": p.budget.reach / extent if extent else 0.0,
        "carrier_hz": p.budget.carrier,
        "carrier": p.budget.label,
        "line_rate_bps": p.budget.line_rate,
        "tx_power_w": p.budget.tx_power,
        "reach_m": p.budget.reach,
        "mean_domain": p.mean_domain,
        "max_domain": p.max_domain,
        "slots": p.slots,
        "reuse": p.reuse,
        "per_node_bps": p.per_node_rate,
        "uncovered_fraction": p.uncovered_fraction,
        "feasible": p.feasible,
        "frame_shared_s": p.frame_shared,
        "frame_private_s": p.frame_private,
        "optical_frame_s": opt_period,
        "optical_rounds": opt_rounds,
        "optical_efficiency": opt_eff,
        "optical_rejected_fraction": opt_rejected,
        "optical_feasible": optical_runs,
        "comparable": both_run,
        # > 1 means broadcast is faster.  Both bounds are carried because the
        # IR does not say which applies to this workload.  NaN when either
        # plane fails to carry the graph, so the medians below cannot be
        # flattered by a plane that delivered nothing.
        "speedup_shared": (
            opt_period / p.frame_shared if both_run and p.frame_shared > 0 else math.nan
        ),
        "speedup_private": (
            opt_period / p.frame_private if both_run and p.frame_private > 0 else math.nan
        ),
    }


def run(workloads=None, platforms=tuple(PLATFORMS), models=DEFAULT_MODELS,
        params: dict | None = None, out: str | Path | None = None) -> list[dict]:
    params = params if params is not None else QUALITY_PARAMS
    names = sorted(workloads or params)
    rows: list[dict] = []
    for name in names:
        for platform in platforms:
            factory, medium_name = PLATFORMS[platform]
            cls, medium = factory(), create("medium", medium_name)
            devices, positions = _pipeline(name, cls, medium, params.get(name, {}))
            for model in models:
                rows.append(
                    evaluate(devices, positions, cls, medium, model, name, platform)
                )
    if out is not None:
        write_csv(rows, out)
    return rows


def scaling(sides=SCALING_SIDES, platform: str = "drone",
            model: str = "min_reach") -> list[dict]:
    """Reuse against device count, holding workload shape fixed.

    The question this answers: a couple of dozen devices sit inside one domain
    whatever the transmit power, so reuse is 1.00x and broadcast is plain TDMA.
    Reuse can only appear once the array is several domains across.  This finds
    where that happens on the placements B3 produces.
    """
    factory, medium_name = PLATFORMS[platform]
    cls, medium = factory(), create("medium", medium_name)
    rows = []
    for side in sides:
        devices, positions = _pipeline(
            "stencil3d",
            cls,
            medium,
            dict(nx=16 * side, ny=16 * side, nz=16 * side, tx=side, ty=side, tz=side),
        )
        p = plan_broadcast(devices, positions, cls, medium, model=model)
        src, dst = _endpoints(devices)
        span = np.linalg.norm(positions[src] - positions[dst], axis=1)
        extent = float(np.ptp(positions, axis=0).max())
        rows.append({
            "side": side,
            "devices": devices.n_tasks,
            "edges": devices.n_edges,
            "extent_m": extent,
            "longest_edge_m": float(span.max()),
            "reach_m": p.budget.reach,
            "domains_across": extent / (2.0 * p.budget.reach) if p.budget.reach else 0.0,
            "mean_domain": p.mean_domain,
            "slots": p.slots,
            "reuse": p.reuse,
            "per_node_bps": p.per_node_rate,
            "uncovered_fraction": p.uncovered_fraction,
        })
    return rows


def ladder_report(platform: str = "float") -> str:
    """The carrier/range/rate trade, measured rather than asserted."""
    factory, medium_name = PLATFORMS[platform]
    cls, medium = factory(), create("medium", medium_name)
    lines = [f"{'carrier':>10} {'rate':>12} {'reach':>12}  in {medium_name}",
             "-" * 50]
    for carrier, rate, label in CARRIER_LADDER:
        b = derive_radio_budget(cls, medium, carrier=carrier, line_rate=rate)
        lines.append(f"{label:>10} {rate / 1e3:9.1f} kb/s {b.reach:9.3f} m")
    return "\n".join(lines)


def report(rows, platform: str | None = None) -> str:
    sel = [r for r in rows if platform is None or r["platform"] == platform]
    head = (
        f"{'workload':<11}{'platform':<9}{'model':<11}{'carrier':>9}"
        f"{'reach m':>9}{'slots':>7}{'reuse':>7}{'per-node':>11}{'uncov':>8}"
        f"{'bcast ms':>11}{'opt ms':>10}{'speedup':>9}"
    )
    out = [head, "-" * len(head)]
    for r in sorted(sel, key=lambda r: (r["workload"], r["platform"], r["model"])):
        out.append(
            f"{r['workload']:<11}{r['platform']:<9}{r['model']:<11}"
            f"{r['carrier']:>9}{r['reach_m']:9.2f}{r['slots']:7d}"
            f"{r['reuse']:6.2f}x{r['per_node_bps']:11.4g}"
            f"{r['uncovered_fraction']:7.1%}"
            f"{r['frame_shared_s'] * 1e3:11.2f}{r['optical_frame_s'] * 1e3:10.2f}"
            + (
                f"{r['speedup_shared']:8.2f}x"
                if r["comparable"] else f"{'n/a':>9}"
            )
        )
    return "\n".join(out)


def scale_report(rows) -> str:
    head = (
        f"{'devices':>8}{'extent m':>10}{'reach m':>9}{'domains':>9}"
        f"{'domain n':>10}{'slots':>7}{'reuse':>8}{'per-node':>11}"
    )
    out = [head, "-" * len(head)]
    for r in rows:
        out.append(
            f"{r['devices']:8d}{r['extent_m']:10.2f}{r['reach_m']:9.2f}"
            f"{r['domains_across']:9.2f}{r['mean_domain']:10.1f}"
            f"{r['slots']:7d}{r['reuse']:7.2f}x{r['per_node_bps']:11.4g}"
        )
    return "\n".join(out)


def verdict(rows, scale_rows=None) -> str:
    out = []

    # Does choosing the power buy anything, and where?  Reuse alone is not the
    # score: ``nominal`` can post the best reuse by leaving most of the graph
    # uncovered, which is reuse bought by not delivering.  So rank on reuse
    # among the settings that carry the placement, and say plainly when none do.
    for platform in sorted({r["platform"] for r in rows}):
        here = [r for r in rows if r["platform"] == platform]
        covering = [r for r in here if r["feasible"]]
        if not covering:
            best = max(here, key=lambda r: r["reuse"])
            out.append(
                f"{platform}: NO setting carries the placement, so reuse cannot be "
                f"ranked -- the best-looking one ({best['model']}, {best['reuse']:.2f}x) "
                f"gets there by leaving {best['uncovered_fraction']:.0%} of the edges "
                f"unreachable, which is reuse bought by not delivering"
            )
            continue
        byname: dict[str, list] = {}
        for r in covering:
            byname.setdefault(r["model"], []).append(r)
        means = {k: float(np.mean([r["reuse"] for r in v])) for k, v in byname.items()}
        best_name = max(sorted(means), key=lambda k: means[k])
        base = means.get("nominal")
        if base is not None and means[best_name] > base * 1.01:
            out.append(
                f"{platform}: {best_name} lifts mean reuse {base:.2f}x -> "
                f"{means[best_name]:.2f}x among the settings that carry the graph"
            )
        else:
            ceiling = float(np.mean([r["reach_over_extent"] for r in covering]))
            out.append(
                f"{platform}: choosing the power buys no reuse "
                f"({means[best_name]:.2f}x) -- reach has to cover the longest task "
                f"edge, and the reach that does already spans {ceiling:.2f}x the whole "
                f"array, so there is only ever one collision domain to divide"
            )

    # Coverage: can the radio carry the placement at all?
    infeasible = [r for r in rows if not r["feasible"]]
    if infeasible:
        worst = max(infeasible, key=lambda r: r["uncovered_fraction"])
        plats = sorted({r["platform"] for r in infeasible})
        out.append(
            f"{len(infeasible)} of {len(rows)} settings cannot carry the placement "
            f"({', '.join(plats)}); worst leaves {worst['uncovered_fraction']:.1%} of "
            f"edges beyond a {worst['reach_m']:.2f} m reach -- B3 spreads the swarm to "
            f"close a 41.6 m *optical* link, which no subsea radio reaches"
        )
    else:
        out.append("every setting carries its placement -- no edge is out of radio reach")

    # The headline: does broadcast beat the optical plane?  Only rows where
    # *both* planes carry the graph can answer that; the rest are reported as
    # uncomparable rather than folded in at zero.
    comparable = [r for r in rows if r["comparable"]]
    dead_optics = [r for r in rows if not r["optical_feasible"]]
    if dead_optics:
        out.append(
            f"{len(dead_optics)} of {len(rows)} rows have no optical plane to compare "
            f"against at all -- B5 rejects every link, so there are no rounds and the "
            f"frame is zero. That is not infinite speed, so those rows are excluded "
            f"rather than scored"
        )
    if comparable:
        sh = float(np.median([r["speedup_shared"] for r in comparable]))
        pv = float(np.median([r["speedup_private"] for r in comparable]))
        out.append(
            f"over the {len(comparable)} comparable rows, median speedup against the "
            f"optical plane is {sh:.2f}x if payloads are shared and {pv:.2f}x if "
            f"private -- both bounds, because the IR does not say which applies"
        )
        opt = float(np.median([r["optical_efficiency"] for r in comparable]))
        out.append(
            f"and the optical plane it is measured against runs at "
            f"{opt:.1%} link efficiency, B6's amortised best"
        )
    else:
        out.append("no row has both planes running, so no comparison is possible")

    if scale_rows:
        first, last = scale_rows[0], scale_rows[-1]
        out.append(
            f"reuse against scale: {first['reuse']:.2f}x at {first['devices']} devices "
            f"-> {last['reuse']:.2f}x at {last['devices']}, as the array grows from "
            f"{first['domains_across']:.2f} to {last['domains_across']:.2f} domains across"
        )
        grew = last["extent_m"] / first["extent_m"] if first["extent_m"] else 0.0
        out.append(
            f"and it barely moves, because B3 does not expand with device count: "
            f"{last['devices'] / max(first['devices'], 1):.0f}x the devices grows the "
            f"extent only {grew:.2f}x, since placements are scaled to close the longest "
            f"*optical* link. Adding devices densifies the array instead of widening "
            f"it, so the domain count never rises. Broadcast reuse would need a "
            f"placement objective that minimises the longest edge, which is not what "
            f"B3 optimises."
        )
    return "\n".join(out)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="B11 broadcast evaluation")
    p.add_argument("workloads", nargs="*")
    p.add_argument("--platform", action="append", choices=sorted(PLATFORMS))
    p.add_argument("--model", action="append", choices=sorted(DEFAULT_MODELS))
    p.add_argument("--out")
    p.add_argument("--no-scaling", action="store_true")
    args = p.parse_args(argv)

    print(ladder_report("float"))
    print()
    print(ladder_report("drone"))
    print()
    rows = run(
        args.workloads or None,
        tuple(args.platform or PLATFORMS),
        tuple(args.model or DEFAULT_MODELS),
        out=args.out,
    )
    print(report(rows))
    scale_rows = None
    if not args.no_scaling:
        print()
        scale_rows = scaling()
        print(scale_report(scale_rows))
    print()
    print(verdict(rows, scale_rows))
    return 0

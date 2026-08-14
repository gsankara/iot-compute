"""B8 evaluation -- what redundancy buys, and whether the clever choice pays.

Three questions.

``does replication help, and how much?``
    Kill a fraction of the swarm at random, many times, and measure the payload
    still deliverable.  Averaged over trials, because a single draw of which
    devices die says nothing.

``is choosing *which* devices to replicate worth anything?``
    ``uniform`` replicates an arbitrary set; ``traffic`` replicates the busiest;
    ``centrality`` replicates by sampled betweenness, which is the expensive
    option and the one the roadmap names.  The bench reports the spread between
    them against the trial-to-trial noise, so "better" has to clear the error
    bar rather than merely win a row.

``how often must a link re-aim?``
    Phase A leaves the closed-loop tracker out and measured what that costs: a
    0.04 m/sqrt(s) drift destroys every link, and open-loop tolerance is
    centimetres.  Drift is a random walk, so the interval is quadratic in
    tolerance -- and the frame B6 produced is long enough that the answer is
    "many times per frame", which is a real charge against the schedule.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..compiler import build_workload
from ..compiler.beam import assign_beams, refine_positions
from ..compiler.partition import partition
from ..compiler.place import place
from ..compiler.resilience import FailoverIndex, TrackerCost, plan_resilience, reaim_interval
from ..compiler.schedule import build_schedule
from ..devices import MOTE_C, aquatic
from ..registry import create
from .sim_bench import write_csv

DEFAULT_POLICIES = ("none", "uniform", "traffic", "centrality")
KILL_FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.5)
TRIALS = 60
QUALITY_PARAMS = {
    "stencil3d": dict(nx=128, ny=128, nz=128, tx=8, ty=8, tz=8),
    "spmv": dict(n=1 << 16, parts=256),
}


def survival(result, n: int, fraction: float, trials: int = TRIALS) -> tuple[float, float]:
    """Mean and standard error of surviving traffic under random failure."""
    vals = []
    for trial in range(trials):
        dead = np.zeros(n, dtype=bool)
        k = max(1, int(round(n * fraction)))
        dead[np.random.default_rng(trial).choice(n, size=k, replace=False)] = True
        vals.append(result.surviving_traffic(dead))
    arr = np.asarray(vals)
    return float(arr.mean()), float(arr.std(ddof=1) / math.sqrt(len(arr)))


def run(workloads=None, policies=DEFAULT_POLICIES, fractions=KILL_FRACTIONS,
        params: dict | None = None, out: str | Path | None = None) -> list[dict]:
    params = params if params is not None else QUALITY_PARAMS
    names = sorted(workloads or params)
    rows = []
    for name in names:
        graph = build_workload(name, **params.get(name, {}))
        devices = partition(graph, MOTE_C, seed=0).coarsened()
        n = devices.n_tasks
        budget = max(1, n // 5)
        for policy in policies:
            r = plan_resilience(devices, budget if policy != "none" else 0,
                                model=policy, seed=0)
            for frac in fractions:
                mean, err = survival(r, n, frac)
                rows.append({
                    "workload": name,
                    "policy": policy,
                    "devices": n,
                    "replicated": r.budget,
                    "kill_fraction": frac,
                    "surviving_traffic": mean,
                    "stderr": err,
                    "degree_mean": float(devices.degree().mean()),
                    "degree_max": int(devices.degree().max()),
                })
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        write_csv(rows, out / "resilience.csv")
        with open(out / "resilience.json", "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
    return rows


def tracker_rows(workload: str = "stencil3d", params: dict | None = None) -> list[dict]:
    """What closing the pointing loop costs, against the frame B6 produced."""
    params = (params or QUALITY_PARAMS).get(workload, {})
    cls = aquatic(MOTE_C)
    water = create("medium", "water")
    graph = build_workload(workload, **params)
    devices = partition(graph, MOTE_C, seed=0).coarsened()
    pos = refine_positions(devices, place(devices, cls, water, seed=0).positions, cls)
    assign = assign_beams(devices, pos, cls)
    sched = build_schedule(devices, assign, pos, cls, model="amortised")

    rows = []
    for drift in (0.0, 0.01, 0.04, 0.1):
        for tolerance in (0.02, 0.05):
            interval = reaim_interval(tolerance, drift)
            cost = TrackerCost(
                interval=interval, frame=sched.period,
                energy_per_reaim=cls.optical.gimbal.retarget_energy(10),
                tracked_links=int((assign.round_of >= 0).sum()),
            )
            rows.append({
                "workload": workload,
                "drift_m_per_sqrt_s": drift,
                "tolerance_m": tolerance,
                "interval_s": interval,
                "frame_s": sched.period,
                "reaims_per_frame": cost.reaims_per_frame,
                "tracked_links": cost.tracked_links,
                "energy_per_frame_mj": cost.energy_per_frame * 1e3,
                "needed": cost.needed,
            })
    return rows


def report(rows: list[dict]) -> str:
    head = (
        f"{'workload':10s} {'policy':11s} {'devs':>5s} {'repl':>5s} "
        + " ".join(f"{f:>7.0%}" for f in sorted({r['kill_fraction'] for r in rows}))
    )
    lines = [head, "-" * len(head)]
    fracs = sorted({r["kill_fraction"] for r in rows})
    seen = []
    for r in rows:
        key = (r["workload"], r["policy"])
        if key in seen:
            continue
        seen.append(key)
        by = {x["kill_fraction"]: x["surviving_traffic"] for x in rows
              if (x["workload"], x["policy"]) == key}
        cells = " ".join(f"{by.get(f, float('nan')):7.3f}" for f in fracs)
        lines.append(
            f"{r['workload']:10s} {r['policy']:11s} {r['devices']:5d} "
            f"{r['replicated']:5d} {cells}"
        )
    return "\n".join(lines)


def tracker_report(rows: list[dict]) -> str:
    head = (
        f"{'drift m/sqrt(s)':>15s} {'tol m':>7s} {'interval s':>11s} {'frame s':>9s} "
        f"{'re-aims/frame':>14s} {'energy mJ':>10s}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        interval = "inf" if not math.isfinite(r["interval_s"]) else f"{r['interval_s']:.3f}"
        lines.append(
            f"{r['drift_m_per_sqrt_s']:15.3f} {r['tolerance_m']:7.2f} {interval:>11s} "
            f"{r['frame_s']:9.2f} {r['reaims_per_frame']:14.1f} "
            f"{r['energy_per_frame_mj']:10.3f}"
        )
    return "\n".join(lines)


def verdict(rows: list[dict], tracker: list[dict]) -> str:
    lines = []
    workloads = sorted({r["workload"] for r in rows})

    for w in workloads:
        mid = [r for r in rows if r["workload"] == w and r["kill_fraction"] == 0.3]
        by = {r["policy"]: r for r in mid}
        if "none" in by and "traffic" in by:
            lines.append(
                f"{w}: at 30% device loss, no redundancy keeps "
                f"{by['none']['surviving_traffic']:.1%} of traffic; replicating a fifth of "
                f"the swarm by traffic keeps {by['traffic']['surviving_traffic']:.1%}"
            )
        if "uniform" in by and "traffic" in by:
            gap = by["traffic"]["surviving_traffic"] - by["uniform"]["surviving_traffic"]
            err = math.hypot(by["traffic"]["stderr"], by["uniform"]["stderr"])
            lines.append(
                f"{w}: choosing by importance beats choosing at random by "
                f"{gap:+.3f} +/- {err:.3f} -- {'significant' if abs(gap) > 2 * err else 'NOT significant'}"
            )
        if "centrality" in by and "traffic" in by:
            gap = by["centrality"]["surviving_traffic"] - by["traffic"]["surviving_traffic"]
            err = math.hypot(by["centrality"]["stderr"], by["traffic"]["stderr"])
            lines.append(
                f"{w}: sampled betweenness over the cheap traffic proxy: "
                f"{gap:+.3f} +/- {err:.3f} -- "
                f"{'significant' if abs(gap) > 2 * err else 'NOT significant, so the proxy is enough'}"
            )

    dense = [r for r in rows if r["degree_mean"] >= r["devices"] - 1.5]
    if dense:
        lines.append(
            "note: B2 leaves device graphs dense -- "
            + ", ".join(
                f"{w} at mean degree {next(r['degree_mean'] for r in rows if r['workload'] == w):.1f}"
                f"/{next(r['devices'] for r in rows if r['workload'] == w) - 1}"
                for w in sorted({r["workload"] for r in dense})
            )
            + " -- so there are few structural bottlenecks for centrality to find"
        )

    needed = [t for t in tracker if t["needed"]]
    if needed:
        worst = max(needed, key=lambda t: t["reaims_per_frame"])
        lines.append(
            f"pointing tracker: at Phase A's 0.04 m/sqrt(s) drift and 2 cm tolerance a link "
            f"survives {reaim_interval(0.02, 0.04):.2f} s, against a "
            f"{worst['frame_s']:.2f} s frame -- "
            f"{next(t['reaims_per_frame'] for t in tracker if t['drift_m_per_sqrt_s'] == 0.04 and t['tolerance_m'] == 0.02):.0f} "
            f"re-aims per frame just to hold the links open"
        )
        lines.append(
            f"that is {len(needed)} of {len(tracker)} drift/tolerance combinations needing "
            f"a tracker at all; a still medium needs none"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="iotc resilience", description=__doc__)
    p.add_argument("workloads", nargs="*")
    p.add_argument("--policy", action="append")
    p.add_argument("--out")
    args = p.parse_args(argv)

    rows = run(args.workloads or None, tuple(args.policy or DEFAULT_POLICIES),
               out=args.out)
    tracker = tracker_rows()
    print(report(rows))
    print()
    print(tracker_report(tracker))
    print()
    print(verdict(rows, tracker))

    if args.out:
        out = Path(args.out)
        write_csv(tracker, out / "tracker.csv")
    return 0

"""B7 evaluation -- what the schedule costs to power, and what that forces.

Two questions, and the second is the one that changes a deployment.

``where does the energy go?``
    Six terms, each read from the ``DeviceClass``.  Everything in this project
    so far has been about the fabric being *aim-dominated*, so the expectation
    going in is that retargeting dominates the draw as it dominates the time.
    It does not, and that is worth knowing before anyone optimises the wrong
    thing.

``what duty cycle can the harvest sustain?``
    A frame the device cannot power is not a schedule.  The pass pads with
    sleep until the average balances and reports the resulting duty cycle -- or
    reports that no duty cycle is sufficient, which is a verdict on the part.

The comparator is ``ignore``: run B6's frame as written and see what happens.
That is what every pass before B7 implicitly assumed, so the gap between it and
``duty_cycle`` is precisely the cost of having ignored energy.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..compiler import build_workload
from ..compiler.beam import assign_beams, refine_positions
from ..compiler.energy import assess
from ..compiler.partition import partition
from ..compiler.place import place
from ..compiler.schedule import build_schedule
from ..devices import MOTE_C, MOTE_M, aerial, aquatic
from ..registry import create
from .sim_bench import write_csv

DEFAULT_POLICIES = ("ignore", "duty_cycle", "half_duty")
PLATFORMS = {
    "float": (lambda: aquatic(MOTE_C), "water"),
    "float-m": (lambda: aquatic(MOTE_M), "water"),
    "drone": (lambda: aerial(MOTE_C), "air"),
}
QUALITY_PARAMS = {
    "stencil3d": dict(nx=128, ny=128, nz=128, tx=8, ty=8, tz=8),
    "spmv": dict(n=1 << 16, parts=256),
}


def evaluate(sched, devices, pos, cls, medium, policy, workload, platform) -> dict:
    b = assess(sched, devices, pos, cls, medium, policy=policy)
    row = {
        "workload": workload,
        "platform": platform,
        "device_class": cls.name,
        "policy": policy,
        "batch": sched.batch,
        "frame_s": b.frame,
        "period_s": b.period if math.isfinite(b.period) else float("inf"),
        "harvest_mw": b.harvest_w * 1e3,
        "draw_mj": b.draw_j * 1e3,
        "harvest_mj": b.harvest_j * 1e3 if math.isfinite(b.harvest_j) else float("inf"),
        "duty_cycle": b.duty_cycle,
        "sustainable": b.sustainable,
        "endurance_s": b.endurance,
        "dominant": b.dominant,
        "dominant_share": b.dominant_share,
    }
    row.update({f"{k}_mj": v * 1e3 for k, v in b.terms.items()})
    return row


def run(workloads=None, platforms=tuple(PLATFORMS), policies=DEFAULT_POLICIES,
        params: dict | None = None, out: str | Path | None = None) -> list[dict]:
    params = params if params is not None else QUALITY_PARAMS
    names = sorted(workloads or params)
    rows = []
    for name in names:
        graph = build_workload(name, **params.get(name, {}))
        devices = partition(graph, MOTE_C, seed=0).coarsened()
        for platform in platforms:
            factory, medium_name = PLATFORMS[platform]
            cls, medium = factory(), create("medium", medium_name)
            pos = refine_positions(devices, place(devices, cls, medium, seed=0).positions, cls)
            sched = build_schedule(devices, assign_beams(devices, pos, cls), pos, cls,
                                   model="amortised")
            for policy in policies:
                rows.append(evaluate(sched, devices, pos, cls, medium, policy,
                                     name, platform))
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        write_csv(rows, out / "energy.csv")
        with open(out / "energy.json", "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1, default=str)
    return rows


def report(rows: list[dict]) -> str:
    head = (
        f"{'workload':10s} {'platform':9s} {'policy':11s} {'batch':>6s} "
        f"{'draw mJ':>9s} {'harv mW':>8s} {'duty':>9s} {'dominant':>10s} {'share':>7s} {'ok':>4s}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        duty = "0.000%" if r["duty_cycle"] == 0 else f"{r['duty_cycle']:.3%}"
        lines.append(
            f"{r['workload']:10s} {r['platform']:9s} {r['policy']:11s} {r['batch']:6d} "
            f"{r['draw_mj']:9.2f} {r['harvest_mw']:8.1f} {duty:>9s} "
            f"{r['dominant']:>10s} {r['dominant_share']:7.1%} "
            f"{'yes' if r['sustainable'] else 'NO':>4s}"
        )
    return "\n".join(lines)


def verdict(rows: list[dict]) -> str:
    lines = []
    ignored = [r for r in rows if r["policy"] == "ignore"]

    # Which term actually dominates, counted rather than assumed.
    tally: dict[str, int] = {}
    for r in ignored:
        tally[r["dominant"]] = tally.get(r["dominant"], 0) + 1
    lines.append(
        "dominant energy term: "
        + ", ".join(f"{k} in {v}/{len(ignored)} cases" for k, v in sorted(tally.items()))
    )
    retarget_share = np.mean([r["retarget_mj"] / max(r["draw_mj"], 1e-12) for r in ignored])
    lines.append(
        f"retargeting is {retarget_share:.1%} of the draw on average -- the fabric is "
        f"aim-dominated in TIME but not in energy, which is not the same statement"
    )

    draining = [r for r in ignored if not r["sustainable"]]
    lines.append(
        f"running B6's frame as written: {len(draining)} of {len(ignored)} configurations "
        f"drain their store"
        + (f" (worst endurance {min(r['endurance_s'] for r in draining):.0f} s)" if draining else "")
    )

    for r in rows:
        if r["policy"] == "duty_cycle" and r["duty_cycle"] < 0.999:
            lines.append(
                f"{r['workload']} on {r['platform']} must drop to a "
                f"{r['duty_cycle']:.1%} duty cycle to be sustainable "
                f"(batch {r['batch']}, {r['dominant']} dominates at {r['dominant_share']:.0%})"
            )

    half = [r for r in rows if r["policy"] == "half_duty" and not r["sustainable"]]
    if half:
        lines.append(
            f"a fixed 50% duty rule still leaves {len(half)} configurations draining -- "
            f"the sleep has to be computed, not chosen"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="iotc energy", description=__doc__)
    p.add_argument("workloads", nargs="*")
    p.add_argument("--platform", action="append", choices=sorted(PLATFORMS))
    p.add_argument("--policy", action="append")
    p.add_argument("--out")
    args = p.parse_args(argv)

    rows = run(args.workloads or None, tuple(args.platform or PLATFORMS),
               tuple(args.policy or DEFAULT_POLICIES), out=args.out)
    print(report(rows))
    print()
    print(verdict(rows))
    return 0

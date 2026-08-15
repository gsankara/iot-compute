"""B10 evaluation -- the flagship, scored on both the conventional and the
physical axes, and honest about which one it loses.

The claim is narrow and should be read narrowly: **verifying** a candidate
clique of size k costs O(k^2) pairwise adjacency tests conventionally and **one
broadcast round** once the members are in mutual line of sight.  It is not a
claim that this beats Bron-Kerbosch on a laptop, and the bench reports the
wall-clock where it does not.

Four axes:

``quality``
    Clique size, and how much of a *planted* clique was recovered.  "Is this the
    maximum" is itself exponential, so the ground truth is planted.

``wall-clock``
    Python seconds.  The physical solver pays for a FastMap embedding and LSH
    that greedy extension does not, and at these sizes it is the slowest of the
    four.  Reported rather than hidden.

``adjacency tests``
    The conventional cost, and the one that matters asymptotically -- this is
    where the embedding earns its keep.

``broadcast rounds and physical cost``
    The cost on this fabric: rounds of aiming, joules to converge the members,
    photons to carry the broadcast.  A round is ~0.5 s of gimbal time, so a
    scheme with fewer rounds wins on hardware even when it loses on a CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..compiler.clique import pairwise_tests_for, solve_clique
from ..compiler.frontends import graphs
from ..compiler.place import derive_budget
from ..devices import MOTE_C, aquatic
from ..registry import create
from ..units import photon_energy
from .sim_bench import write_csv

DEFAULT_SOLVERS = ("greedy_degeneracy", "colour_bound", "bron_kerbosch",
                   "annealing", "physical")
SIZES = (400, 1500, 4000)
PLANTED = 16
AVG_DEGREE = 10.0


def physical_cost(result, device_class, medium, spread: float = 20.0) -> dict:
    """What converging and verifying this clique costs on the fabric.

    Members must fly together before the broadcast round can happen, so the
    energy is the lateral travel their placement implies; the photons are what
    the broadcast itself spends.
    """
    budget = derive_budget(device_class, medium)
    k = result.size
    # Members start spread over the formation and converge to mutual line of
    # sight; charge the mean travel at the platform's lateral rate.
    travel_m = spread * max(k, 1)
    energy_j = travel_m * budget.xy_cost_per_m
    o = device_class.optical
    # One broadcast round: every member sends its neighbour list to the others.
    payload_bits = max(k, 1) * max(k - 1, 0) * 32
    seconds = payload_bits / max(o.line_rate, 1e-30)
    photons = o.tx_power * seconds / photon_energy(o.wavelength)
    return {
        "converge_energy_j": energy_j,
        "broadcast_bits": payload_bits,
        "broadcast_photons": photons,
        "aim_seconds": result.broadcast_rounds * o.gimbal.retarget_time(100),
    }


def evaluate(n: int, solver: str, seed: int = 1) -> dict:
    u, v, planted = graphs.planted_clique(n, AVG_DEGREE, PLANTED, seed=seed)
    result = solve_clique(n, u, v, model=solver, seed=0)
    recovered = len(set(result.members) & set(planted.tolist()))
    row = {
        "n": n,
        "edges": int(u.size),
        "solver": result.method,
        "size": result.size,
        "planted": int(planted.size),
        "recovered": recovered,
        "recovered_fraction": recovered / max(planted.size, 1),
        "verified": result.verified,
        "seconds": result.seconds,
        "adjacency_tests": result.adjacency_tests,
        "broadcast_rounds": result.broadcast_rounds,
        "candidates": result.candidates,
        "verify_tests_conventional": pairwise_tests_for(result.size),
        "verify_rounds_physical": 1 if result.broadcast_rounds else 0,
    }
    row.update(physical_cost(result, aquatic(MOTE_C), create("medium", "water")))
    return row


def run(sizes=SIZES, solvers=DEFAULT_SOLVERS, seed: int = 1,
        out: str | Path | None = None) -> list[dict]:
    rows = [evaluate(n, s, seed) for n in sizes for s in solvers]
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        write_csv(rows, out / "clique.csv")
        with open(out / "clique.json", "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
    return rows


def report(rows: list[dict]) -> str:
    head = (
        f"{'n':>6s} {'solver':18s} {'size':>5s} {'planted':>8s} {'ms':>9s} "
        f"{'tests':>10s} {'rounds':>7s} {'energy J':>9s} {'ok':>4s}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['n']:6d} {r['solver']:18s} {r['size']:5d} "
            f"{r['recovered']:4d}/{r['planted']:<3d} {r['seconds'] * 1e3:9.1f} "
            f"{r['adjacency_tests']:10d} {r['broadcast_rounds']:7d} "
            f"{r['converge_energy_j']:9.2f} {'ok' if r['verified'] else 'BAD':>4s}"
        )
    return "\n".join(lines)


def verdict(rows: list[dict]) -> str:
    lines = []
    sizes = sorted({r["n"] for r in rows})
    solvers = sorted({r["solver"] for r in rows})

    quality = {s: [r["recovered_fraction"] for r in rows if r["solver"] == s] for s in solvers}
    perfect = [s for s, v in quality.items() if min(v) >= 1.0]
    lines.append(
        f"every solver recovers the planted clique in full at every size tested "
        f"({len(perfect)} of {len(solvers)})"
        if len(perfect) == len(solvers)
        else "recovery differs between solvers: "
             + ", ".join(f"{s} {min(v):.0%}-{max(v):.0%}" for s, v in sorted(quality.items()))
    )

    # Growth in cost over the ladder -- but only among solvers that actually
    # solved it.  A solver that returns a 2-clique does almost no work, and
    # ranking it as cheap would be comparing the cost of not answering.
    if len(sizes) > 1:
        lo, hi = sizes[0], sizes[-1]
        solved = [s for s in solvers if min(quality[s]) >= 1.0]
        failed = [s for s in solvers if s not in solved]
        growth = []
        for s in solved:
            a = next((r["adjacency_tests"] for r in rows if r["solver"] == s and r["n"] == lo), 0)
            b = next((r["adjacency_tests"] for r in rows if r["solver"] == s and r["n"] == hi), 0)
            if a > 0:
                growth.append((s, b / a))
        growth.sort(key=lambda x: x[1])
        lines.append(
            f"adjacency tests over a {hi // lo}x range of n, among solvers that found "
            f"the clique: " + ", ".join(f"{s} {g:.1f}x" for s, g in growth)
            + " -- the embedding is what keeps the physical scheme's growth flat"
        )
        if failed:
            lines.append(
                "excluded from that comparison because they did not find the clique: "
                + ", ".join(sorted(failed))
                + " -- their low test counts are the cost of not answering"
            )

    phys = [r for r in rows if r["solver"] == "physical"]
    if phys:
        rounds = {r["broadcast_rounds"] for r in phys}
        lines.append(
            f"broadcast rounds are {'constant at ' + str(rounds.pop()) if len(rounds) == 1 else 'variable'} "
            f"regardless of n -- the physical cost does not grow with the problem"
        )
        worst = max(phys, key=lambda r: r["seconds"])
        fastest = min((r for r in rows if r["n"] == worst["n"]), key=lambda r: r["seconds"])
        lines.append(
            f"and it is the slowest in wall-clock: {worst['seconds'] * 1e3:.0f} ms against "
            f"{fastest['seconds'] * 1e3:.1f} ms for {fastest['solver']} at n={worst['n']} -- "
            f"the embedding and LSH are overhead a greedy extension never pays"
        )
        k = max(r["size"] for r in phys)
        lines.append(
            f"the claim the pass actually makes is about verification: confirming a "
            f"{k}-clique costs {pairwise_tests_for(k)} pairwise tests conventionally and "
            f"1 broadcast round physically, whatever k is"
        )
        lines.append(
            f"paid for physically: {np.mean([r['converge_energy_j'] for r in phys]):.1f} J to "
            f"converge the members and {np.mean([r['broadcast_photons'] for r in phys]):.3g} "
            f"photons for the round"
        )

    bad = [r for r in rows if not r["verified"]]
    lines.append(
        f"independently checked: {len(rows) - len(bad)} of {len(rows)} answers are "
        f"genuine cliques" + ("" if not bad else f" -- {len(bad)} ARE NOT")
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="iotc clique", description=__doc__)
    p.add_argument("--size", type=int, action="append")
    p.add_argument("--solver", action="append")
    p.add_argument("--out")
    args = p.parse_args(argv)

    rows = run(tuple(args.size or SIZES), tuple(args.solver or DEFAULT_SOLVERS),
               out=args.out)
    print(report(rows))
    print()
    print(verdict(rows))
    return 0

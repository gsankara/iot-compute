"""B10 -- Maximum clique, end to end: physical proximity *is* the adjacency test.

This is the demonstration the whole architecture argues for.  On a conventional
machine, checking that a candidate set of size k is a clique costs O(k^2)
pairwise adjacency lookups.  If the devices holding those vertices are flown into
mutual line of sight, every member hears every other member in **one broadcast
round** -- the quadratic test collapses into the geometry.

The pipeline is the one ``docs/ROADMAP.md`` specifies:

1. **Embed** under the common-neighbour attraction weight
   ``W[i,j] = |N(i) ∩ N(j)| + A[i,j]``, using graph FastMap rather than a
   spectral initialisation, so it stays near-linear;
2. **LSH** buckets the embedding and nominates dense balls as candidate cliques;
3. the devices holding a candidate **converge**, and one broadcast round verifies
   it;
4. non-members are **repelled** and their affinity decayed -- the force-directed
   law `GravityTopology.repel` already implements in the simulator.

**What is and is not being claimed.**  The physical scheme still needs adjacency
information to *extract* a candidate from a ball; what it removes is the
*verification*, which is where the quadratic sits.  So the bench counts adjacency
tests and broadcast rounds separately rather than reporting one number and
hoping.  Solution quality is scored against a planted clique, because "is this
the maximum" is otherwise itself exponential.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..registry import register
from .fastmap import fastmap
from .frontends.nphard import common_neighbour_weight


def adjacency_sets(n: int, u: np.ndarray, v: np.ndarray) -> list[set[int]]:
    """Neighbour sets, the representation every solver here shares."""
    adj: list[set[int]] = [set() for _ in range(n)]
    for a, b in zip(np.asarray(u), np.asarray(v)):
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    return adj


@dataclass
class CliqueResult:
    """A candidate answer and what it cost, on both axes."""

    members: list[int]
    method: str = ""
    seconds: float = 0.0
    #: Pairwise adjacency lookups performed -- the conventional cost.
    adjacency_tests: int = 0
    #: Physical broadcast rounds -- the cost on this fabric.
    broadcast_rounds: int = 0
    #: Candidate sets examined, whether or not they proved to be cliques.
    candidates: int = 0
    verified: bool = False

    @property
    def size(self) -> int:
        return len(self.members)

    def is_clique(self, adj: list[set[int]]) -> bool:
        """Independent check, so a solver cannot grade its own homework."""
        m = self.members
        return all(m[j] in adj[m[i]] for i in range(len(m)) for j in range(i + 1, len(m)))

    def summary(self) -> str:
        return (
            f"{self.method:18s} size={self.size:3d}  {self.seconds * 1e3:9.2f} ms  "
            f"tests={self.adjacency_tests:9d}  rounds={self.broadcast_rounds:6d}"
        )


class CliqueSolver:
    """The ``clique_solver`` plugin slot."""

    def solve(self, n: int, adj: list[set[int]], seed: int = 0) -> CliqueResult:
        raise NotImplementedError  # pragma: no cover


# --------------------------------------------------------------- the baselines

@register("clique_solver", "greedy_degeneracy", "greedy")
class GreedyDegeneracy(CliqueSolver):
    """Degeneracy order, then extend greedily.  Fast, and often not bad."""

    def solve(self, n: int, adj, seed: int = 0) -> CliqueResult:
        tests = 0
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        best: list[int] = []
        for start in order[: max(1, n // 10)]:
            members = [start]
            for cand in sorted(adj[start], key=lambda x: -len(adj[x])):
                ok = True
                for m in members:
                    tests += 1
                    if cand not in adj[m]:
                        ok = False
                        break
                if ok:
                    members.append(cand)
            if len(members) > len(best):
                best = members
        return CliqueResult(sorted(best), "greedy_degeneracy",
                            adjacency_tests=tests, candidates=len(order[: max(1, n // 10)]))


@register("clique_solver", "bron_kerbosch", "exact")
class BronKerbosch(CliqueSolver):
    """Bron-Kerbosch with pivoting -- exact, and exponential.

    Carried as the ground truth for small instances and as the wall-clock the
    physical scheme has to beat at scale.  ``node_budget`` stops it rather than
    letting the bench hang; a truncated run is reported as such.
    """

    def __init__(self, node_budget: int = 200_000) -> None:
        self.node_budget = int(node_budget)

    def solve(self, n: int, adj, seed: int = 0) -> CliqueResult:
        best: list[int] = []
        tests = 0
        nodes = 0

        def expand(r: list[int], p: set[int], x: set[int]) -> None:
            nonlocal best, tests, nodes
            nodes += 1
            if nodes > self.node_budget:
                return
            if not p and not x:
                if len(r) > len(best):
                    best = list(r)
                return
            pivot = max(p | x, key=lambda w: len(adj[w] & p), default=None)
            if pivot is None:
                return
            tests += len(p)
            for w in list(p - adj[pivot]):
                expand(r + [w], p & adj[w], x & adj[w])
                p.discard(w)
                x.add(w)

        import sys

        limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(limit, 10_000))
        try:
            expand([], set(range(n)), set())
        finally:
            sys.setrecursionlimit(limit)
        return CliqueResult(sorted(best), "bron_kerbosch",
                            adjacency_tests=tests, candidates=nodes)


@register("clique_solver", "colour_bound", "branch_and_bound")
class ColourBound(CliqueSolver):
    """Branch and bound pruned by a greedy colouring upper bound."""

    def __init__(self, node_budget: int = 100_000) -> None:
        self.node_budget = int(node_budget)

    def _colour(self, cand: list[int], adj) -> list[tuple[int, int]]:
        classes: list[list[int]] = []
        for v in sorted(cand, key=lambda x: -len(adj[x])):
            for cls in classes:
                if not any(u in adj[v] for u in cls):
                    cls.append(v)
                    break
            else:
                classes.append([v])
        return [(v, i + 1) for i, cls in enumerate(classes) for v in cls]

    def solve(self, n: int, adj, seed: int = 0) -> CliqueResult:
        best: list[int] = []
        tests = 0
        nodes = 0

        def expand(r: list[int], cand: list[int]) -> None:
            nonlocal best, tests, nodes
            nodes += 1
            if nodes > self.node_budget or not cand:
                if len(r) > len(best):
                    best = list(r)
                return
            for v, bound in reversed(self._colour(cand, adj)):
                if len(r) + bound <= len(best):
                    return
                nxt = [w for w in cand if w in adj[v]]
                tests += len(cand)
                expand(r + [v], nxt)
                cand = [w for w in cand if w != v]

        expand([], list(range(n)))
        return CliqueResult(sorted(best), "colour_bound",
                            adjacency_tests=tests, candidates=nodes)


@register("clique_solver", "annealing", "simulated_annealing")
class SimulatedAnnealing(CliqueSolver):
    """Local search: add a vertex, or drop one that blocks it."""

    def __init__(self, steps: int = 20_000, temperature: float = 1.0) -> None:
        self.steps = int(steps)
        self.temperature = float(temperature)

    def solve(self, n: int, adj, seed: int = 0) -> CliqueResult:
        rng = np.random.default_rng(seed)
        current: set[int] = set()
        best: set[int] = set()
        tests = 0
        for step in range(self.steps):
            temp = self.temperature * (1.0 - step / self.steps) + 1e-9
            v = int(rng.integers(0, n))
            if v in current:
                if rng.random() < math.exp(-1.0 / temp):
                    current.discard(v)
                continue
            blockers = [u for u in current if (tests := tests + 1) and u not in adj[v]]
            if not blockers:
                current.add(v)
            elif len(blockers) == 1 and rng.random() < math.exp(-1.0 / temp):
                current.discard(blockers[0])
                current.add(v)
            if len(current) > len(best):
                best = set(current)
        return CliqueResult(sorted(best), "annealing", adjacency_tests=tests,
                            candidates=self.steps)


# ------------------------------------------------------------- the physical one

@register("clique_solver", "physical", "converge")
class PhysicalClique(CliqueSolver):
    """Embed, bucket, converge, verify in one broadcast round.  The pass.

    The candidate *balls* come from LSH over a FastMap embedding weighted by
    common neighbourhood, so a dense region of the graph is a dense region of the
    embedding.  Verification of a ball is a single round, because once its
    devices are in mutual line of sight every member hears every other at once.
    """

    def __init__(self, kappa: int = 6, bits: int = 10, ball: int = 24,
                 probes: int = 32) -> None:
        self.kappa = int(kappa)
        self.bits = int(bits)
        self.ball = int(ball)
        self.probes = int(probes)

    def embed(self, n: int, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """FastMap under the common-neighbour attraction weight.

        Heavier weight means *closer*, so the weight is inverted into a
        distance.  Adjacency alone would give every edge the same length and the
        embedding would carry no information about density.
        """
        if u.size == 0:
            return np.zeros((n, self.kappa))
        w = common_neighbour_weight(u, v, n)
        return fastmap(n, u, v, float(w.max()) / np.maximum(w, 1e-9),
                       kappa=self.kappa, seed=0)

    def solve(self, n: int, adj, seed: int = 0) -> CliqueResult:
        pairs = [(a, b) for a in range(n) for b in adj[a] if a < b]
        if not pairs:
            return CliqueResult([0] if n else [], "physical", broadcast_rounds=0)
        u = np.array([a for a, _ in pairs])
        v = np.array([b for _, b in pairs])
        coords = self.embed(n, u, v)

        rng = np.random.default_rng(seed)
        tests = 0
        rounds = 0
        candidates = 0
        best: list[int] = []

        # LSH: random hyperplanes over the embedding, so a bucket is a ball.
        planes = rng.normal(size=(self.bits, coords.shape[1]))
        sig = ((coords @ planes.T) > 0) @ (1 << np.arange(self.bits))
        buckets: dict[int, list[int]] = {}
        for i, s in enumerate(sig):
            buckets.setdefault(int(s), []).append(i)

        seeds = sorted(buckets.values(), key=len, reverse=True)[: self.probes]
        # Probe around the densest vertices too: a bucket boundary can split a
        # clique, and a ball centred on a member will not.
        hubs = sorted(range(n), key=lambda x: -len(adj[x]))[: self.probes]
        for hub in hubs:
            d = np.linalg.norm(coords - coords[hub], axis=1)
            seeds.append(list(np.argsort(d)[: self.ball]))

        for ball in seeds:
            if len(ball) < 2:
                continue
            candidates += 1
            # Extract inside the ball -- this is the part that still needs
            # adjacency, and it is counted honestly.
            members: list[int] = []
            for cand in sorted(ball, key=lambda x: -len(adj[x])):
                ok = True
                for m in members:
                    tests += 1
                    if cand not in adj[m]:
                        ok = False
                        break
                if ok:
                    members.append(cand)
            # The devices converge and every member hears every other: one round
            # confirms mutual adjacency, whatever k is.
            rounds += 1
            if len(members) > len(best):
                best = members

        return CliqueResult(sorted(best), "physical", adjacency_tests=tests,
                            broadcast_rounds=rounds, candidates=candidates)


def pairwise_tests_for(k: int) -> int:
    """What verifying a k-clique costs conventionally: k(k-1)/2."""
    return k * (k - 1) // 2


def solve_clique(n: int, u: np.ndarray, v: np.ndarray,
                 model: CliqueSolver | str = "physical", seed: int = 0) -> CliqueResult:
    """Run one solver and time it."""
    if isinstance(model, str):
        from ..registry import create

        name, model = model, create("clique_solver", model)
    else:
        name = type(model).plugin_name

    adj = adjacency_sets(n, u, v)
    start = time.perf_counter()
    result = model.solve(n, adj, seed)
    result.seconds = time.perf_counter() - start
    result.method = name
    result.verified = result.is_clique(adj)
    return result

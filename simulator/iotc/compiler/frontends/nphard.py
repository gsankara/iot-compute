"""The NP-hard workload family.

Maximum clique, maximum independent set, graph colouring, capacitated k-median
and max-cut.  All five are search problems over a graph, and all five reduce on
this substrate to the same physical question: *which vertices should be near
each other, and how do we test a candidate set cheaply?*

Maximum clique is the flagship (roadmap B10) because it is the case where the
answer is most starkly physical.  Verifying that a candidate set of size k is a
clique costs O(k^2) pairwise adjacency tests on a conventional machine.  If the
devices holding those vertices are flown into mutual line of sight, every member
hears every other member in **one broadcast round** -- the O(k^2) test collapses
into the geometry.  That is the claim the whole architecture exists to make, so
the IR carries what B3 needs to attempt it: edges weighted by common
neighbourhood, not merely by adjacency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...registry import register
from ..ir import TaskGraph
from . import graphs
from .base import Workload
from .sparse import _cut_edges


def common_neighbour_weight(
    u: np.ndarray, v: np.ndarray, n: int
) -> np.ndarray:
    """``|N(i) ∩ N(j)| + A[i,j]`` for each edge, the clique attraction weight.

    Two vertices that share many neighbours are likely to sit in a clique
    together, and this weight says so far more sharply than adjacency alone --
    which matters because B3 embeds *distances*, and adjacency on its own gives
    every edge the same length.  Under this weight a dense region contracts and
    LSH can nominate it as a candidate ball.
    """
    if u.size == 0:
        return np.zeros(0, dtype=float)
    order = np.argsort(np.concatenate([u, v]), kind="stable")
    flat_src = np.concatenate([u, v])[order]
    flat_dst = np.concatenate([v, u])[order]
    starts = np.searchsorted(flat_src, np.arange(n + 1))
    w = np.empty(u.size, dtype=float)
    for e in range(u.size):
        i, j = int(u[e]), int(v[e])
        ni = flat_dst[starts[i]:starts[i + 1]]
        nj = flat_dst[starts[j]:starts[j + 1]]
        w[e] = np.intersect1d(ni, nj, assume_unique=False).size + 1.0
    return w


def _block_aggregate(
    u: np.ndarray, v: np.ndarray, weight: np.ndarray, part: np.ndarray, parts: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum an edge weight over the block pairs it connects."""
    pu, pv = part[u], part[v]
    cross = pu != pv
    lo = np.minimum(pu[cross], pv[cross])
    hi = np.maximum(pu[cross], pv[cross])
    w = weight[cross]
    if lo.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, float)
    key = lo * parts + hi
    uniq, inv = np.unique(key, return_inverse=True)
    summed = np.zeros(uniq.size, dtype=float)
    np.add.at(summed, inv, w)
    return uniq // parts, uniq % parts, summed


@dataclass(frozen=True)
class _GraphProblem(Workload):
    """Shared parameters for the graph-search family."""

    n: int = 1024
    parts: int = 16
    graph: str = "erdos_renyi"
    avg_degree: float = 12.0
    iterations: int = 20
    seed: int = 0

    def _instance(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        u, v = graphs.build(
            self.graph, self.n, self.seed, avg_degree=self.avg_degree, m_attach=3
        )
        return u, v, graphs.block_partition(self.n, self.parts)

    def _degrees(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        deg = np.zeros(self.n, dtype=float)
        if u.size:
            np.add.at(deg, u, 1.0)
            np.add.at(deg, v, 1.0)
        return deg


@register("workload", "maxclique", "clique")
@dataclass(frozen=True)
class MaxClique(_GraphProblem):
    """Maximum clique -- the B10 flagship.

    Edge payload is the **common-neighbour attraction weight**, so a placement
    that minimises weighted communication distance is automatically a placement
    that pulls likely-clique vertices together.  ``clique_size`` plants a known
    answer when ``graph="planted_clique"``, which is what makes the end-to-end
    evaluation checkable rather than merely plausible.
    """

    graph: str = "planted_clique"
    clique_size: int = 16
    #: A candidate-set broadcast carries vertex ids.
    bytes_per_vertex: int = 4

    def _instance(self):
        if self.graph == "planted_clique":
            u, v, _ = graphs.planted_clique(
                self.n, self.avg_degree, self.clique_size, self.seed
            )
            return u, v, graphs.block_partition(self.n, self.parts)
        return super()._instance()

    def planted(self) -> np.ndarray:
        """The ground-truth clique members, for scoring B10's answer."""
        if self.graph != "planted_clique":
            return np.zeros(0, dtype=np.int64)
        _, _, members = graphs.planted_clique(
            self.n, self.avg_degree, self.clique_size, self.seed
        )
        return members

    def build(self) -> TaskGraph:
        u, v, part = self._instance()
        deg = self._degrees(u, v)
        counts = np.bincount(part, minlength=self.parts).astype(float)
        block_deg = np.zeros(self.parts, dtype=float)
        np.add.at(block_deg, part, deg)

        # Per round a block extends its candidate sets: for each local vertex,
        # intersect its neighbourhood with the current candidate set.
        flop = block_deg * 2.0 + counts
        ws = counts * 4 + block_deg * 4  # vertex ids + adjacency

        w = common_neighbour_weight(u, v, self.n)
        bu, bv, wsum = _block_aggregate(u, v, w, part, self.parts)
        pay = wsum * self.bytes_per_vertex

        return TaskGraph.from_arrays(
            flop, ws, bu, bv, pay,
            name=f"maxclique[{self.graph},n={self.n},p={self.parts}]",
            iterations=self.iterations,
            kind="clique",
            label=[f"block{i}" for i in range(self.parts)],
        )


@register("workload", "mis")
@dataclass(frozen=True)
class MaxIndependentSet(_GraphProblem):
    """Maximum independent set by Luby-style rounds.

    The structural opposite of clique: members must be mutually *non*-adjacent,
    so the placement objective inverts and the swarm wants its candidates spread
    out.  Keeping both in the family stops B3 from being tuned to contraction
    alone.
    """

    def build(self) -> TaskGraph:
        u, v, part = self._instance()
        deg = self._degrees(u, v)
        counts = np.bincount(part, minlength=self.parts).astype(float)
        block_deg = np.zeros(self.parts, dtype=float)
        np.add.at(block_deg, part, deg)

        # Each round: draw a priority per live vertex, compare against neighbours.
        flop = counts + block_deg
        ws = counts * (4 + 4 + 1) + block_deg * 4  # id, priority, state, adjacency
        bu, bv, ncut = _cut_edges(u, v, part, self.parts)
        pay = ncut * (4 + 4)  # vertex id and its priority

        return TaskGraph.from_arrays(
            flop, ws, bu, bv, pay,
            name=f"mis[{self.graph},n={self.n},p={self.parts}]",
            iterations=self.iterations,
            kind="mis",
            label=[f"block{i}" for i in range(self.parts)],
        )


@register("workload", "coloring", "colouring")
@dataclass(frozen=True)
class GraphColouring(_GraphProblem):
    """Distributed greedy colouring.

    Each round a vertex proposes the smallest colour not taken by a neighbour
    and the conflicts are resolved by id.  Traffic is a colour per cut edge --
    tiny payloads on many edges, which is the aim-dominated regime by
    construction and therefore a useful stress case for B6.
    """

    #: One byte holds up to 256 colours, which is ample for these instances.
    bytes_per_colour: int = 1

    def build(self) -> TaskGraph:
        u, v, part = self._instance()
        deg = self._degrees(u, v)
        counts = np.bincount(part, minlength=self.parts).astype(float)
        block_deg = np.zeros(self.parts, dtype=float)
        np.add.at(block_deg, part, deg)

        flop = block_deg * 2.0  # scan neighbour colours, pick the mex
        ws = counts * (4 + self.bytes_per_colour) + block_deg * 4
        bu, bv, ncut = _cut_edges(u, v, part, self.parts)
        pay = ncut * (4 + self.bytes_per_colour)

        return TaskGraph.from_arrays(
            flop, ws, bu, bv, pay,
            name=f"coloring[{self.graph},n={self.n},p={self.parts}]",
            iterations=self.iterations,
            kind="coloring",
            label=[f"block{i}" for i in range(self.parts)],
        )


@register("workload", "kmedian")
@dataclass(frozen=True)
class CapacitatedKMedian(_GraphProblem):
    """Capacitated vertex k-median -- Lloyd-style reassignment rounds.

    This is the aggregator-selection problem from the roadmap wearing its own
    hat: choose ``k`` centres under a per-centre capacity, assign every vertex
    to one.  Capacity is the whole difficulty, and it is not an abstraction
    here -- it is the device's RAM, so the same constraint appears again in B1
    and B7.
    """

    k: int = 32
    capacity: int = 64

    def build(self) -> TaskGraph:
        u, v, part = self._instance()
        counts = np.bincount(part, minlength=self.parts).astype(float)
        deg = self._degrees(u, v)
        block_deg = np.zeros(self.parts, dtype=float)
        np.add.at(block_deg, part, deg)

        # Per round: distance to every candidate centre, then a capacity check.
        flop = counts * self.k * 2.0 + counts
        ws = counts * (4 + 4) + self.k * (4 + 4)  # assignment + centre table
        bu, bv, ncut = _cut_edges(u, v, part, self.parts)
        # Blocks trade centre positions and the overflow they cannot absorb.
        pay = ncut * 4 + self.k * 8
        if bu.size == 0 and self.parts > 1:
            # A disconnected instance still has to agree on the centre table.
            bu = np.arange(self.parts - 1)
            bv = bu + 1
            pay = np.full(self.parts - 1, float(self.k * 8))

        return TaskGraph.from_arrays(
            flop, ws, bu, bv, pay,
            name=f"kmedian[k={self.k},cap={self.capacity},n={self.n},p={self.parts}]",
            iterations=self.iterations,
            kind="kmedian",
            label=[f"block{i}" for i in range(self.parts)],
        )


@register("workload", "maxcut")
@dataclass(frozen=True)
class MaxCut(_GraphProblem):
    """Max-cut by local search.

    Each round every vertex evaluates the gain of switching sides and the
    positive-gain ones flip.  The interesting property for this fabric is that
    traffic is proportional to the *cut*, so the workload gets cheaper to
    communicate exactly as it converges -- the one member of the family whose
    schedule could safely be relaxed over time.
    """

    def build(self) -> TaskGraph:
        u, v, part = self._instance()
        deg = self._degrees(u, v)
        counts = np.bincount(part, minlength=self.parts).astype(float)
        block_deg = np.zeros(self.parts, dtype=float)
        np.add.at(block_deg, part, deg)

        flop = block_deg * 2.0 + counts
        ws = counts * (4 + 1) + block_deg * 4  # id + side bit + adjacency
        bu, bv, ncut = _cut_edges(u, v, part, self.parts)
        pay = ncut * (4 + 1)

        return TaskGraph.from_arrays(
            flop, ws, bu, bv, pay,
            name=f"maxcut[{self.graph},n={self.n},p={self.parts}]",
            iterations=self.iterations,
            kind="maxcut",
            label=[f"block{i}" for i in range(self.parts)],
        )

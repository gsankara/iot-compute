"""Irregular workloads: SpMV and BFS.

Both take a problem graph, cut it into vertex blocks, and make each block a
task.  An edge between two tasks exists exactly when the partition cuts edges
between their blocks, and its payload is the boundary traffic that cut implies.

That makes these the frontends where **the partition is visible in the IR**:
change the vertex-to-block map and the task graph's edge set changes with it.
Everything B2 does is measured on this surface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...registry import register
from ..ir import TaskGraph
from . import graphs
from .base import Workload


def _cut_edges(
    u: np.ndarray, v: np.ndarray, part: np.ndarray, parts: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate cut edges into (block, block, count) triples."""
    pu, pv = part[u], part[v]
    cross = pu != pv
    lo = np.minimum(pu[cross], pv[cross])
    hi = np.maximum(pu[cross], pv[cross])
    if lo.size == 0:
        return (np.zeros(0, np.int64),) * 2 + (np.zeros(0, float),)
    key = lo * parts + hi
    uniq, counts = np.unique(key, return_counts=True)
    return uniq // parts, uniq % parts, counts.astype(float)


@register("workload", "spmv")
@dataclass(frozen=True)
class SpMV(Workload):
    """Repeated sparse matrix-vector product ``y = A x`` over ``parts`` tasks.

    The canonical low-intensity kernel: two FLOP per nonzero and a vector
    element to fetch for every off-block column.  If a workload is going to be
    aim-dominated on this fabric, it is this one, and the IR should say so
    before any placement work is wasted on it.
    """

    n: int = 4096
    parts: int = 16
    graph: str = "erdos_renyi"
    avg_degree: float = 8.0
    iterations: int = 50
    seed: int = 0

    def build(self) -> TaskGraph:
        u, v = graphs.build(self.graph, self.n, self.seed, avg_degree=self.avg_degree)
        part = graphs.block_partition(self.n, self.parts)
        counts = np.bincount(part, minlength=self.parts).astype(float)

        # Nonzeros per block: the diagonal plus both endpoints of every incident
        # edge, symmetric storage.
        nnz = counts.copy()
        if u.size:
            np.add.at(nnz, part[u], 1.0)
            np.add.at(nnz, part[v], 1.0)

        flop = 2.0 * nnz                       # one multiply-add per nonzero
        ws = (nnz * (self.dtype_bytes + 4)     # value + column index
              + counts * 2 * self.dtype_bytes  # x and y segments
              ) * 1.0

        bu, bv, ncut = _cut_edges(u, v, part, self.parts)
        pay = ncut * self.dtype_bytes          # one x element per cut edge

        return TaskGraph.from_arrays(
            flop, ws, bu, bv, pay,
            name=f"spmv[{self.graph},n={self.n},p={self.parts}]",
            iterations=self.iterations,
            kind="spmv",
            label=[f"block{i}" for i in range(self.parts)],
        )


@register("workload", "bfs")
@dataclass(frozen=True)
class BFS(Workload):
    """Level-synchronous breadth-first search over ``parts`` tasks.

    One superstep is one BFS level: expand the local frontier, then exchange
    the vertices discovered that belong elsewhere.  Traffic is bursty and
    data-dependent, so ``frontier_fraction`` models the fraction of a block's
    vertices active in an average level rather than pretending it is known.

    The reason this workload matters here is that its traffic *changes shape*
    between supersteps while the beam assignment cannot.  B5 has to commit to a
    topology that serves every level tolerably rather than any level well.
    """

    n: int = 4096
    parts: int = 16
    graph: str = "power_law"
    avg_degree: float = 8.0
    frontier_fraction: float = 0.15
    #: A level-synchronous BFS runs for roughly the graph's diameter.
    iterations: int = 12
    seed: int = 0
    #: A discovered-vertex message is an id, not a value.
    bytes_per_vertex: int = 4

    def build(self) -> TaskGraph:
        u, v = graphs.build(
            self.graph, self.n, self.seed, avg_degree=self.avg_degree, m_attach=3
        )
        part = graphs.block_partition(self.n, self.parts)
        counts = np.bincount(part, minlength=self.parts).astype(float)

        deg = np.zeros(self.n, dtype=float)
        if u.size:
            np.add.at(deg, u, 1.0)
            np.add.at(deg, v, 1.0)
        block_deg = np.zeros(self.parts, dtype=float)
        np.add.at(block_deg, part, deg)

        # Per level: touch the frontier's incident edges, one compare each.
        flop = block_deg * self.frontier_fraction
        ws = (counts * (4 + 1)                 # parent id + visited flag
              + block_deg * 4)                 # adjacency indices
        bu, bv, ncut = _cut_edges(u, v, part, self.parts)
        pay = ncut * self.frontier_fraction * self.bytes_per_vertex

        return TaskGraph.from_arrays(
            flop, ws, bu, bv, pay,
            name=f"bfs[{self.graph},n={self.n},p={self.parts}]",
            iterations=self.iterations,
            kind="bfs",
            label=[f"block{i}" for i in range(self.parts)],
        )

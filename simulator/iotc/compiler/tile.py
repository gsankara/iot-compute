"""B1 -- Tile: split tasks until their working set fits the target's RAM.

The stopping condition is :meth:`TaskGraph.fits`, read straight from the
``DeviceClass`` so the compiler cannot drift from what the simulator believes a
node holds.  Everything interesting is in *how* the split is made.

**The shape of the cut is the whole cost.**  Model a task as a d-dimensional
block of ``V`` bytes cut into ``f_1 x ... x f_d`` pieces.  A cut perpendicular to
axis ``i`` creates ``(f_i - 1) * prod_{j != i} f_j`` internal faces, and each is
a fraction ``1 / prod_{j != i} f_j`` of the task's original boundary face.  The
added traffic telescopes:

    internal bytes = halo * sum_i (f_i - 1)

So cutting into ``k`` pieces along **one** axis costs ``(k - 1) * halo``, while
cutting a cube into ``k`` balanced pieces costs ``d * (k^(1/d) - 1) * halo``.  At
k = 64 in three dimensions that is 63 halos against 9 -- a **7x** difference in
added traffic for exactly the same fit.  Choosing the factorization is therefore
not bookkeeping, it is the pass.

**The halo is measured, not invented.**  ``halo`` for a task is the mean payload
on its existing edges.  For a stencil an external edge *is* a face exchange, so
this is exact rather than a fitted constant; for a graph workload it is the
right order of magnitude and is declared as an assumption rather than hidden.

**Halo counts against RAM.**  A piece must buffer the ghost data it exchanges,
so its working set is ``V/k`` plus its incident payload.  That makes the fit
condition a fixed point -- splitting shrinks the bulk but adds halo -- so
:func:`tile` iterates to convergence rather than dividing once.  It does
converge: both terms fall as the factors grow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..registry import register
from .ir import TaskGraph

#: How many axes a task can be cut along, by task kind.  A stencil is genuinely
#: geometric; a graph-workload block has no interior structure to exploit, so it
#: can only be sliced, and it pays the 1D price.  Anything unlisted is treated as
#: 1D, which is the conservative choice -- it over-estimates the traffic a split
#: costs rather than under-estimating it.
KIND_DIMENSION = {
    "stencil3d": 3,
    "stencil2d": 2,
    "stencil": 3,
    "gemm": 2,
    "fft": 1,
    "collective": 1,
    "spmv": 1,
    "bfs": 1,
    "clique": 1,
    "mis": 1,
    "coloring": 1,
    "kmedian": 1,
    "maxcut": 1,
    "compute": 1,
}


def balanced_factors(k: int, d: int) -> list[int]:
    """Factor ``k`` pieces across ``d`` axes as evenly as possible.

    Returns factors whose product is **at least** ``k``.  Balance is not a
    stylistic preference: added traffic is ``sum_i (f_i - 1)`` at fixed product,
    which the AM-GM inequality minimises exactly when the factors are equal.
    """
    if d <= 1:
        return [max(1, int(k))]
    if k <= 1:
        return [1] * d
    side = max(1, int(math.floor(k ** (1.0 / d))))
    factors = [side] * d
    # Raise individual axes until the product covers k, keeping them balanced.
    i = 0
    while math.prod(factors) < k:
        factors[i % d] += 1
        i += 1
    return factors


@dataclass(frozen=True)
class SplitPlan:
    """How one task becomes several."""

    factors: list[int]
    #: Lattice coordinate of each piece, in factor order.
    coords: list[tuple[int, ...]]
    #: ``(local_i, local_j, payload)`` for faces internal to this task.
    internal: list[tuple[int, int, float]]

    @property
    def pieces(self) -> int:
        return len(self.coords)

    @property
    def added_traffic_halos(self) -> int:
        """``sum_i (f_i - 1)`` -- added internal traffic in units of one halo."""
        return sum(f - 1 for f in self.factors)


class TileModel:
    """How a task of a given kind may be cut.  The ``tiler`` plugin slot."""

    def dimension(self, kind: str) -> int:  # pragma: no cover - trivial
        raise NotImplementedError

    def plan(self, kind: str, pieces: int, halo: float) -> SplitPlan:
        """Lay ``pieces`` sub-tasks out and enumerate the faces between them."""
        d = self.dimension(kind)
        factors = balanced_factors(pieces, d)
        coords = [
            tuple(np.unravel_index(i, factors)) for i in range(math.prod(factors))
        ]
        index = {c: i for i, c in enumerate(coords)}

        internal: list[tuple[int, int, float]] = []
        for c, i in index.items():
            for axis in range(len(factors)):
                nxt = list(c)
                nxt[axis] += 1
                if nxt[axis] >= factors[axis]:
                    continue
                j = index[tuple(nxt)]
                # This face is 1 / (product of the other factors) of the
                # original boundary face.
                other = math.prod(f for a, f in enumerate(factors) if a != axis)
                internal.append((i, j, halo / max(other, 1)))
        return SplitPlan(factors=factors, coords=coords, internal=internal)


@register("tiler", "geometric", "block")
class GeometricTiler(TileModel):
    """Cut along every axis the task's kind allows -- the default and the best."""

    def __init__(self, dimensions: dict | None = None) -> None:
        self.dimensions = dict(KIND_DIMENSION)
        if dimensions:
            self.dimensions.update(dimensions)

    def dimension(self, kind: str) -> int:
        return int(self.dimensions.get(kind, 1))


@register("tiler", "slab", "linear")
class SlabTiler(TileModel):
    """Cut along one axis only.

    The naive tiler, kept as the comparator B1 has to beat.  It is what you get
    if you treat "make it fit" as a division problem rather than a geometry
    problem, and the bench measures exactly what that costs.
    """

    def dimension(self, kind: str) -> int:
        return 1


@register("tiler", "planar")
class PlanarTiler(TileModel):
    """Cut along at most two axes -- the intermediate case."""

    def dimension(self, kind: str) -> int:
        return min(2, int(KIND_DIMENSION.get(kind, 1)))


@dataclass
class TileResult:
    """The tiled graph plus the provenance every later pass needs."""

    graph: TaskGraph
    #: Original task id for each task of :attr:`graph`.
    origin: np.ndarray
    #: Pieces each original task was split into.
    factor: np.ndarray
    #: Per-original-task factorization, for reporting.
    plans: list[SplitPlan] = field(default_factory=list)
    rounds: int = 0
    converged: bool = True
    #: Why tiling stopped, when it did not converge.
    reason: str = ""

    @property
    def split_tasks(self) -> int:
        return int((self.factor > 1).sum())

    @property
    def max_factor(self) -> int:
        return int(self.factor.max()) if self.factor.size else 1

    @property
    def added_traffic_halos(self) -> int:
        return sum(p.added_traffic_halos for p in self.plans)

    def summary(self) -> str:
        shapes = {}
        for p in self.plans:
            if p.pieces > 1:
                shapes["x".join(str(f) for f in p.factors)] = (
                    shapes.get("x".join(str(f) for f in p.factors), 0) + 1
                )
        shape_text = ", ".join(f"{k}({v})" for k, v in sorted(shapes.items())) or "none"
        return (
            f"{self.graph.name:28s} {self.graph.n_tasks:6d} tasks "
            f"({self.split_tasks} split, max {self.max_factor}x)  "
            f"rounds={self.rounds}  cuts={shape_text}"
        )


def _halos(graph: TaskGraph) -> np.ndarray:
    """Mean incident payload per task -- the size of one boundary exchange."""
    deg = graph.degree()
    traffic = graph.traffic_per_task()
    return np.where(deg > 0, traffic / np.maximum(deg, 1), 0.0)


def tile(
    graph: TaskGraph,
    device_class,
    model: TileModel | str = "geometric",
    max_rounds: int = 12,
    max_pieces: int = 1 << 16,
    stall_tolerance: float = 0.01,
) -> TileResult:
    """Split every task until it fits ``device_class``'s RAM, halo included.

    Iterates because halo counts against the working set: a piece that fits on
    bulk alone may not fit once it has to buffer its ghost faces.  Each round
    raises the piece count of any task that still overflows and rebuilds.

    Not every workload can be made to fit, and the pass says so rather than
    subdividing forever.  When further splitting stops shrinking the largest
    piece by more than ``stall_tolerance``, the halo has hit its floor -- the
    boundary exchange alone exceeds RAM -- and the result comes back with
    ``converged=False`` and a ``reason``.  That is a real answer: it means this
    workload needs a bigger part, not a finer tiling.
    """
    if isinstance(model, str):
        from ..registry import create

        model = create("tiler", model)

    ram = float(device_class.compute.ram_bytes)
    if ram <= 0:
        raise ValueError(f"device class {device_class.name!r} declares no RAM")

    n = graph.n_tasks
    # The frontends fold the ghost ring into working_set, so the halo has to be
    # taken back out before it can be recomputed for the new geometry --
    # otherwise every round would charge for boundary data twice.  A task whose
    # bulk vanishes under this is genuinely halo-dominated, and the fit result
    # should say so rather than be propped up.
    bulk = np.maximum(graph.working_set - graph.traffic_per_task(), 1.0)
    halo = _halos(graph)
    kinds = graph.kinds
    src, dst = graph.edges[:, 0], graph.edges[:, 1]
    payload = graph.payload

    # Seed the piece count from bulk alone; the loop corrects for halo.
    pieces = np.maximum(1, np.ceil(bulk / ram)).astype(np.int64)

    result_graph = None
    plans: list[SplitPlan] = []
    origin = np.arange(n)
    rounds = 0
    converged = False
    reason = ""
    previous_worst = math.inf
    counts = np.ones(n, dtype=np.int64)

    for rounds in range(1, max_rounds + 1):
        plans = [model.plan(kinds[t], int(pieces[t]), float(halo[t])) for t in range(n)]
        counts = np.array([p.pieces for p in plans], dtype=np.int64)
        offset = np.concatenate([[0], np.cumsum(counts)])
        total = int(offset[-1])

        new_src: list[int] = []
        new_dst: list[int] = []
        new_pay: list[float] = []

        # Faces internal to a split task.
        for t, plan in enumerate(plans):
            base = int(offset[t])
            for i, j, pay in plan.internal:
                new_src.append(base + i)
                new_dst.append(base + j)
                new_pay.append(pay)

        # Original edges, spread over a round-robin matching of the pieces so
        # that total external traffic is conserved exactly.
        for e in range(src.size):
            u, v = int(src[e]), int(dst[e])
            ku, kv = int(counts[u]), int(counts[v])
            m = max(ku, kv)
            share = float(payload[e]) / m
            for i in range(m):
                new_src.append(int(offset[u]) + i % ku)
                new_dst.append(int(offset[v]) + i % kv)
                new_pay.append(share)

        origin = np.repeat(np.arange(n), counts)
        new_bulk = bulk[origin] / counts[origin]

        # Halo counts against RAM, so the working set is bulk plus whatever this
        # piece has to buffer.  Build once to total the incident payload.
        probe = TaskGraph.from_arrays(
            graph.flop[origin] / counts[origin],
            np.maximum(new_bulk, 1.0),
            new_src, new_dst, new_pay,
            name=graph.name,
            iterations=graph.iterations,
            kind=[kinds[t] for t in origin],
        )
        working = new_bulk + probe.traffic_per_task()

        result_graph = TaskGraph.from_arrays(
            graph.flop[origin] / counts[origin],
            working,
            np.asarray(new_src), np.asarray(new_dst), np.asarray(new_pay),
            name=f"{graph.name}/tiled",
            iterations=graph.iterations,
            kind=[kinds[t] for t in origin],
            label=[f"t{t}.{i}" for t, c in enumerate(counts) for i in range(c)],
        )

        over = working > ram
        if not over.any():
            converged = True
            break

        worst = float(working.max())
        if worst >= previous_worst * (1.0 - stall_tolerance):
            # Splitting further is no longer shrinking the largest piece.  The
            # halo has hit its floor: along an axis the model cannot subdivide,
            # a piece buffers a whole boundary exchange however small its bulk
            # gets.  Grinding on would multiply tasks without ever fitting.
            reason = (
                f"halo floor reached: largest piece is {worst / ram:.2f}x RAM and "
                f"stopped shrinking after {rounds} rounds; the exchange itself does "
                f"not fit this class"
            )
            break
        previous_worst = worst

        if counts.sum() >= max_pieces:
            reason = f"piece budget exhausted ({max_pieces}) before fitting"
            break

        # Raise only the tasks that still overflow, by the factor they overflow
        # by, so the loop cannot oscillate.
        need = np.zeros(n, dtype=np.int64)
        np.maximum.at(need, origin[over], np.ceil(working[over] / ram).astype(np.int64))
        pieces = np.where(need > 0, np.maximum(pieces + 1, pieces * np.maximum(need, 1)), pieces)
    else:
        reason = f"did not fit within max_rounds={max_rounds}"

    return TileResult(
        graph=result_graph,
        origin=origin,
        factor=np.array([p.pieces for p in plans], dtype=np.int64),
        plans=plans,
        rounds=rounds,
        converged=converged,
        reason="" if converged else reason,
    )

"""B2 -- Partition: group tasks into per-device sets.

B1 made each task small enough to fit a device.  B2 does the opposite arithmetic:
it packs tasks *back together* so that one device carries as many as its RAM
allows, because every task boundary that survives partitioning becomes a beam
that has to be aimed.

Two things are being optimised at once, and they pull against each other.

**Cut payload** is the objective.  Bytes crossing a group boundary become
optical traffic; bytes inside a group are a memory copy and cost nothing on this
fabric.  Minimising the cut is minimising the thing the whole machine is bad at.

**Capacity** is the constraint.  A group's working set must fit the device's RAM,
which is the same predicate B1 used, read from the same ``DeviceClass``.

The default partitioner is ``fmbm``: embed with graph FastMap, then cluster in
the embedded space.  It is the only one of the four that is near-linear, and the
bench measures that rather than asserting it -- Kernighan-Lin and spectral are
carried specifically as the comparators that do not scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..registry import register
from .fastmap import embed_graph
from .ir import TaskGraph


@dataclass
class PartitionResult:
    """Group labels plus everything needed to score them."""

    labels: np.ndarray
    k: int
    graph: TaskGraph
    device_class: object
    method: str = ""
    seconds: float = 0.0

    @property
    def group_working_set(self) -> np.ndarray:
        """Total working set per group."""
        out = np.zeros(self.k, dtype=float)
        np.add.at(out, self.labels, self.graph.working_set)
        return out

    @property
    def cut_bytes(self) -> float:
        """Payload crossing a group boundary, per superstep."""
        e = self.graph.edges
        if e.size == 0:
            return 0.0
        cross = self.labels[e[:, 0]] != self.labels[e[:, 1]]
        return float(self.graph.payload[cross].sum())

    @property
    def cut_fraction(self) -> float:
        total = float(self.graph.payload.sum())
        return self.cut_bytes / total if total > 0 else 0.0

    @property
    def balance(self) -> float:
        """Heaviest group's working set over the mean.  1.0 is perfect."""
        ws = self.group_working_set
        mean = ws.mean() if ws.size else 0.0
        return float(ws.max() / mean) if mean > 0 else 1.0

    @property
    def feasible(self) -> bool:
        """Does every group fit the device it is destined for?"""
        ram = float(self.device_class.compute.ram_bytes)
        return bool((self.group_working_set <= ram).all())

    @property
    def occupied(self) -> int:
        return int(np.unique(self.labels).size)

    def coarsened(self) -> TaskGraph:
        """One task per group -- what B3 places and B5 assigns beams to."""
        k = self.k
        flop = np.zeros(k)
        ws = np.zeros(k)
        np.add.at(flop, self.labels, self.graph.flop)
        np.add.at(ws, self.labels, self.graph.working_set)

        e = self.graph.edges
        if e.size:
            gu, gv = self.labels[e[:, 0]], self.labels[e[:, 1]]
            cross = gu != gv
            gu, gv, pay = gu[cross], gv[cross], self.graph.payload[cross]
        else:
            gu = gv = np.zeros(0, dtype=np.int64)
            pay = np.zeros(0)

        return TaskGraph.from_arrays(
            flop, np.maximum(ws, 1.0), gu, gv, pay,
            name=f"{self.graph.name}/part[{self.method}]",
            iterations=self.graph.iterations,
            kind="group",
            label=[f"dev{i}" for i in range(k)],
        )

    def summary(self) -> str:
        return (
            f"{self.method:14s} k={self.k:5d} used={self.occupied:5d}  "
            f"cut={self.cut_fraction:6.3f}  balance={self.balance:5.2f}  "
            f"feasible={'yes' if self.feasible else 'NO ':3s}  {self.seconds * 1e3:8.1f} ms"
        )


# --------------------------------------------------------------------------
class Partitioner:
    """The ``partitioner`` plugin slot."""

    def assign(self, graph: TaskGraph, k: int, seed: int = 0) -> np.ndarray:
        raise NotImplementedError  # pragma: no cover


def kd_order(coords: np.ndarray) -> np.ndarray:
    """Order points so that neighbours in the sequence are near in space.

    Recursive median bisection on the widest axis -- a KD traversal.  Two
    properties matter here and neither is available from clustering.

    It is ``O(n log n)`` **independent of k**.  A capacitated k-means costs
    ``O(n k)`` per round, and since the device count grows with the problem,
    that is quadratic in disguise; it would sink the near-linear claim that is
    B2's acceptance criterion.

    And it turns partitioning into *cutting a sequence*, so a group is a
    contiguous run and capacity can be honoured by construction rather than by
    a repair pass that undoes the locality the embedding just found.
    """
    n = coords.shape[0]
    order = np.arange(n)
    if n <= 1 or coords.shape[1] == 0:
        return order

    out = np.empty(n, dtype=np.int64)
    cursor = 0
    stack = [order]
    # Depth-first so that finished runs are emitted in spatial order.
    while stack:
        block = stack.pop()
        if block.size <= 1:
            out[cursor:cursor + block.size] = block
            cursor += block.size
            continue
        pts = coords[block]
        spread = pts.max(axis=0) - pts.min(axis=0)
        axis = int(np.argmax(spread))
        if spread[axis] <= 0:
            out[cursor:cursor + block.size] = block
            cursor += block.size
            continue
        mid = block.size // 2
        part = np.argpartition(pts[:, axis], mid)
        left, right = block[part[:mid]], block[part[mid:]]
        stack.append(right)
        stack.append(left)
    return out


def _chunk_by_weight(order: np.ndarray, weights: np.ndarray, k: int) -> np.ndarray:
    """Cut an ordered sequence into ``k`` contiguous runs of equal weight.

    Equal weight rather than equal count, because the constraint downstream is
    bytes of RAM and tasks are not uniform.
    """
    n = order.size
    labels = np.zeros(n, dtype=np.int64)
    if k <= 1:
        return labels
    w = weights[order]
    cumulative = np.cumsum(w)
    total = cumulative[-1]
    if total <= 0:
        labels[order] = np.arange(n) * k // n
        return labels
    # Group g takes weight up to (g+1)/k of the total.
    edges = np.searchsorted(cumulative, total * np.arange(1, k) / k, side="left")
    assigned = np.zeros(n, dtype=np.int64)
    start = 0
    for g, stop in enumerate(np.append(edges, n)):
        stop = max(stop, start)
        assigned[start:stop] = g
        start = stop
    assigned[start:] = k - 1
    labels[order] = assigned
    return labels


def refine_boundary(
    graph: TaskGraph,
    labels: np.ndarray,
    capacity: float,
    sweeps: int = 4,
) -> np.ndarray:
    """Move boundary tasks to a neighbouring group when it shrinks the cut.

    Only tasks with at least one edge leaving their group are candidates, and a
    move is taken only if it strictly reduces cut payload and the destination
    still fits.  Each sweep touches every edge a constant number of times, so
    this is ``O(sweeps * m)`` -- it sharpens the cut without costing the
    near-linearity that is the whole reason for choosing FastMap.

    This is what separates "embed and chop" from a partitioner.  The embedding
    decides the global shape; the refinement fixes the local mistakes an
    ordering necessarily makes when it flattens several dimensions into one.
    """
    labels = np.array(labels, dtype=np.int64, copy=True)
    e, pay = graph.edges, graph.payload
    if e.size == 0:
        return labels

    n = graph.n_tasks
    neighbours: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for idx in range(e.shape[0]):
        u, v, w = int(e[idx, 0]), int(e[idx, 1]), float(pay[idx])
        neighbours[u].append((v, w))
        neighbours[v].append((u, w))

    load = np.zeros(int(labels.max()) + 1, dtype=float)
    np.add.at(load, labels, graph.working_set)
    ws = graph.working_set

    for _ in range(sweeps):
        moved = 0
        for v in range(n):
            home = int(labels[v])
            pull: dict[int, float] = {}
            stay = 0.0
            for nb, w in neighbours[v]:
                g = int(labels[nb])
                if g == home:
                    stay += w
                else:
                    pull[g] = pull.get(g, 0.0) + w
            if not pull:
                continue
            best = max(pull, key=pull.get)
            # Gain is what stops crossing minus what starts crossing.
            if pull[best] - stay <= 0:
                continue
            if load[best] + ws[v] > capacity:
                continue
            load[home] -= ws[v]
            load[best] += ws[v]
            labels[v] = best
            moved += 1
        if moved == 0:
            break
    return labels


@register("partitioner", "fmbm", "fastmap")
class FastMapPartitioner(Partitioner):
    """Embed with graph FastMap, then cluster under capacity.  The default.

    Near-linear by construction: the embedding is ``O(kappa * (m + n log n))``
    and the clustering is ``O(n k)`` per round at fixed ``kappa``.  Nothing here
    touches an adjacency matrix or an eigendecomposition, which is exactly why
    it survives at the scale Phase C targets.
    """

    def __init__(self, kappa: int = 4, sweeps: int = 4) -> None:
        self.kappa = int(kappa)
        self.sweeps = int(sweeps)

    def assign(self, graph: TaskGraph, k: int, seed: int = 0) -> np.ndarray:
        coords = embed_graph(graph, kappa=self.kappa, seed=seed)
        labels = _chunk_by_weight(kd_order(coords), graph.working_set, k)
        if self.sweeps <= 0 or k <= 1:
            return labels
        # Chunking splits evenly, so allow the same slack the caller's device
        # count already implies rather than freezing the initial balance.
        capacity = float(graph.working_set.sum()) / k * 1.10
        return refine_boundary(graph, labels, capacity, self.sweeps)


@register("partitioner", "block", "naive")
class BlockPartitioner(Partitioner):
    """Contiguous id ranges -- the baseline every other method must beat.

    Deliberately structure-blind.  On a banded matrix it is already close to
    optimal, which is the point: a partitioner has to prove it wins where
    structure is absent, not where the numbering handed it over.
    """

    def assign(self, graph: TaskGraph, k: int, seed: int = 0) -> np.ndarray:
        n = graph.n_tasks
        sizes = np.full(k, n // k, dtype=np.int64)
        sizes[: n % k] += 1
        return np.repeat(np.arange(k), sizes)[:n]


@register("partitioner", "spectral")
class SpectralPartitioner(Partitioner):
    """Recursive bisection on the Fiedler vector.

    Carried as a quality yardstick, not as a candidate.  It needs an
    eigendecomposition of the Laplacian, so its cost grows far faster than
    linearly and it is the comparator the runtime study is aimed at.
    """

    def assign(self, graph: TaskGraph, k: int, seed: int = 0) -> np.ndarray:
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import eigsh
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "the spectral partitioner needs scipy; install the 'full' extra"
            ) from exc

        n = graph.n_tasks
        labels = np.zeros(n, dtype=np.int64)
        e, pay = graph.edges, graph.payload

        def bisect(members: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            m = members.size
            if m < 2:
                return members, members[:0]
            index = {int(v): i for i, v in enumerate(members)}
            mask = np.isin(e[:, 0], members) & np.isin(e[:, 1], members)
            if not mask.any():
                return members[: m // 2], members[m // 2:]
            rows = np.array([index[int(x)] for x in e[mask, 0]])
            cols = np.array([index[int(x)] for x in e[mask, 1]])
            w = pay[mask]
            adj = csr_matrix((np.concatenate([w, w]),
                              (np.concatenate([rows, cols]),
                               np.concatenate([cols, rows]))), shape=(m, m))
            deg = np.asarray(adj.sum(axis=1)).ravel()
            lap = csr_matrix(np.diag(deg)) - adj
            try:
                # eigsh draws its Krylov start vector at random unless told
                # otherwise, which would make the same graph partition
                # differently on two runs.  Determinism is a correctness
                # requirement here, so the start vector is pinned.
                v0 = np.full(m, 1.0 / math.sqrt(m))
                _, vec = eigsh(lap, k=min(2, m - 1), which="SM", v0=v0, maxiter=5000)
                fiedler = vec[:, -1]
            except Exception:  # pragma: no cover - numerical fallback
                fiedler = np.arange(m, dtype=float)
            order = np.argsort(fiedler)
            return members[order[: m // 2]], members[order[m // 2:]]

        groups = [np.arange(n)]
        while len(groups) < k:
            groups.sort(key=lambda g: -g.size)
            head = groups.pop(0)
            left, right = bisect(head)
            if left.size == 0 or right.size == 0:
                groups.append(head)
                break
            groups.extend([left, right])
        for i, g in enumerate(groups):
            labels[g] = i
        return labels


@register("partitioner", "kernighan_lin", "kl")
class KernighanLinPartitioner(Partitioner):
    """Block start, then Kernighan-Lin swap refinement between group pairs.

    The classical quality reference.  Each refinement pass scores every
    candidate swap between a pair of groups, so it is quadratic in group size --
    excellent cuts on small inputs, and the clearest illustration of why the
    scale this project targets needs something else.
    """

    def __init__(self, passes: int = 2, max_pair_size: int = 512) -> None:
        self.passes = int(passes)
        self.max_pair_size = int(max_pair_size)

    def assign(self, graph: TaskGraph, k: int, seed: int = 0) -> np.ndarray:
        labels = BlockPartitioner().assign(graph, k, seed)
        adj = {}
        e, pay = graph.edges, graph.payload
        for idx in range(e.shape[0]):
            u, v, w = int(e[idx, 0]), int(e[idx, 1]), float(pay[idx])
            adj.setdefault(u, []).append((v, w))
            adj.setdefault(v, []).append((u, w))

        def external_minus_internal(v: int, home: int, other: int) -> float:
            ext = inn = 0.0
            for nb, w in adj.get(v, ()):
                if labels[nb] == other:
                    ext += w
                elif labels[nb] == home:
                    inn += w
            return ext - inn

        rng = np.random.default_rng(seed)
        for _ in range(self.passes):
            for a in range(k):
                b = int(rng.integers(0, k))
                if b == a:
                    continue
                ga = np.flatnonzero(labels == a)
                gb = np.flatnonzero(labels == b)
                if ga.size == 0 or gb.size == 0:
                    continue
                if ga.size > self.max_pair_size:
                    ga = rng.choice(ga, self.max_pair_size, replace=False)
                if gb.size > self.max_pair_size:
                    gb = rng.choice(gb, self.max_pair_size, replace=False)
                da = {int(v): external_minus_internal(int(v), a, b) for v in ga}
                db = {int(v): external_minus_internal(int(v), b, a) for v in gb}
                for u in sorted(da, key=lambda x: -da[x])[: max(1, len(da) // 4)]:
                    best_v, best_gain = None, 0.0
                    for v in gb:
                        v = int(v)
                        gain = da[u] + db[v]
                        if gain > best_gain:
                            best_gain, best_v = gain, v
                    if best_v is not None:
                        labels[u], labels[best_v] = b, a
                        da[u] = -math.inf
                        db[best_v] = -math.inf
        return labels


# --------------------------------------------------------------------------
def minimum_devices(graph: TaskGraph, device_class) -> int:
    """Fewest devices whose RAM could hold this graph at all."""
    ram = float(device_class.compute.ram_bytes)
    if ram <= 0:
        raise ValueError(f"device class {device_class.name!r} declares no RAM")
    return max(1, int(math.ceil(float(graph.working_set.sum()) / ram)))


def partition(
    graph: TaskGraph,
    device_class,
    k: int | None = None,
    model: Partitioner | str = "fmbm",
    seed: int = 0,
    headroom: float = 1.15,
) -> PartitionResult:
    """Group ``graph``'s tasks onto devices, minimising the cut.

    ``k`` defaults to the fewest devices that could hold the working set, times
    ``headroom``.  Packing to exactly the minimum leaves no slack for the
    capacity sweep and reliably produces one overfull group; the headroom buys
    feasibility for a few percent more devices, which on a swarm of thousands is
    the cheap side of that trade.
    """
    import time

    if isinstance(model, str):
        from ..registry import create

        name, model = model, create("partitioner", model)
    else:
        name = type(model).plugin_name

    if k is None:
        k = max(1, int(math.ceil(minimum_devices(graph, device_class) * headroom)))
    k = min(int(k), graph.n_tasks)

    start = time.perf_counter()
    labels = np.asarray(model.assign(graph, k, seed), dtype=np.int64)
    seconds = time.perf_counter() - start

    if labels.shape != (graph.n_tasks,):
        raise ValueError(f"partitioner {name!r} returned {labels.shape}, expected ({graph.n_tasks},)")
    if labels.min() < 0 or labels.max() >= k:
        raise ValueError(f"partitioner {name!r} produced labels outside [0, {k})")

    return PartitionResult(
        labels=labels, k=k, graph=graph, device_class=device_class,
        method=name, seconds=seconds,
    )

"""Graph FastMap -- embed a graph into R^kappa preserving shortest-path distance.

Li & Kumar, *Revisiting FastMap: New Applications* (arXiv:2503.11908), Alg. 2.
This is the spine of the compiler: B2 partitions in the embedded space and B3
places from it, and both have to run at thousands of tasks, so the embedding has
to be near-linear rather than merely polynomial.

The method takes ``kappa`` passes.  Each pass picks a far-apart pivot pair
``(a, b)`` by double sweep, runs one shortest-path computation from each, and
reads off a coordinate by the triangle projection

    p(v) = (d(a,v) + d(a,b) - d(b,v)) / 2

Then it *removes* the distance already explained by that coordinate, so the next
pass works on what is left.  Doing that exactly would mean recomputing an
all-pairs residual metric; FastMap's trick is to push the correction down onto
the edge weights,

    w'(u,v) = sqrt(max(w(u,v)^2 - (p(u) - p(v))^2, 0))

which keeps every pass at two shortest-path runs.  Total cost is
``O(kappa * (m + n log n))`` -- near-linear, with kappa a small constant.

**Distance here is communication distance, not adjacency.**  Tasks that
exchange more bytes must end up closer, so an edge's weight is the *reciprocal*
of its payload.  Embedding that metric is what makes "physically near" and
"talks a lot" the same statement, which is the whole premise of the substrate.
"""

from __future__ import annotations

import heapq
import math

import numpy as np


def build_csr(
    n: int, src: np.ndarray, dst: np.ndarray, weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Symmetric CSR adjacency.  Edges are undirected, so each is stored twice."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    weight = np.asarray(weight, dtype=float)
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    w = np.concatenate([weight, weight])
    order = np.argsort(u, kind="stable")
    u, v, w = u[order], v[order], w[order]
    indptr = np.searchsorted(u, np.arange(n + 1))
    return indptr, v, w


def dijkstra(
    indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, source: int, n: int
) -> np.ndarray:
    """Single-source shortest paths.  Returns ``inf`` for unreachable vertices."""
    dist = np.full(n, math.inf)
    dist[source] = 0.0
    heap = [(0.0, int(source))]
    seen = np.zeros(n, dtype=bool)
    while heap:
        d, u = heapq.heappop(heap)
        if seen[u]:
            continue
        seen[u] = True
        for k in range(indptr[u], indptr[u + 1]):
            v = int(indices[k])
            if seen[v]:
                continue
            nd = d + data[k]
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def _finite_max(a: np.ndarray) -> float:
    finite = a[np.isfinite(a)]
    return float(finite.max()) if finite.size else 1.0


def _repair(dist: np.ndarray) -> np.ndarray:
    """Give unreachable vertices a large finite distance.

    A disconnected task graph is pathological rather than impossible -- a fault
    campaign can produce one -- and the embedding should degrade into "very far
    apart" rather than propagate NaN into every downstream pass.
    """
    out = np.array(dist, dtype=float)
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = _finite_max(out) * 2.0 + 1.0
    return out


def _double_sweep(
    indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, n: int, start: int
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Find a far-apart pivot pair, and return the distance arrays with it.

    Two sweeps: the vertex farthest from an arbitrary start, then the vertex
    farthest from *that*.  It is the standard cheap approximation to a diameter
    pair, and it costs the same two shortest-path runs the coordinate needs
    anyway -- so the pivots are free.
    """
    d0 = _repair(dijkstra(indptr, indices, data, start, n))
    a = int(np.argmax(d0))
    da = _repair(dijkstra(indptr, indices, data, a, n))
    b = int(np.argmax(da))
    db = _repair(dijkstra(indptr, indices, data, b, n))
    return a, b, da, db


def fastmap(
    n: int,
    src: np.ndarray,
    dst: np.ndarray,
    weight: np.ndarray,
    kappa: int = 4,
    seed: int = 0,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Embed into ``R^kappa``.  Returns an ``(n, kappa)`` coordinate array.

    ``weight`` is the *distance* along each edge, so callers wanting "heavy
    traffic means close" must invert their payloads before calling.
    :func:`traffic_distance` does that.
    """
    if n <= 0:
        return np.zeros((0, max(kappa, 0)))
    kappa = max(int(kappa), 0)
    coords = np.zeros((n, kappa), dtype=float)
    if kappa == 0 or np.asarray(src).size == 0:
        return coords

    rng = np.random.default_rng(seed)
    indptr, indices, original = build_csr(n, src, dst, weight)
    original = original.astype(float)
    data = original.copy()
    head = np.repeat(np.arange(n), np.diff(indptr))

    for axis in range(kappa):
        a, b, da, db = _double_sweep(indptr, indices, data, n, int(rng.integers(0, n)))
        d_ab = float(da[b])

        if d_ab <= epsilon:
            # The residual metric has collapsed.  On a lattice this happens
            # immediately and is not a defect of the graph: shortest-path
            # distance there is L1, so *every* edge contributes equally to the
            # first diagonal coordinate and the Pythagorean update zeroes them
            # all at once.  Taking that at face value would return a
            # one-dimensional embedding of a three-dimensional mesh.
            #
            # Recover by drawing a fresh pivot pair from the *original* metric
            # and orthogonalising the new coordinate against the axes already
            # found.  That still costs two shortest-path runs, so the pass stays
            # near-linear, and it yields kappa genuinely distinct directions
            # instead of one direction and kappa-1 zeros.
            data = original.copy()
            a, b, da, db = _double_sweep(indptr, indices, data, n, int(rng.integers(0, n)))
            d_ab = float(da[b])
            if d_ab <= epsilon:
                break
            p = (da + d_ab - db) / 2.0
            for prior in range(axis):
                q = coords[:, prior]
                norm = float(q @ q)
                if norm > epsilon:
                    p = p - (float(p @ q) / norm) * q
            if float(np.linalg.norm(p)) <= epsilon:
                break
        else:
            p = (da + d_ab - db) / 2.0

        coords[:, axis] = p

        # Push the explained distance down onto the edge weights so the next
        # axis sees only the residual.  This is what keeps the pass count, and
        # therefore the total cost, linear in the number of edges.
        delta = p[head] - p[indices]
        data = np.sqrt(np.maximum(data**2 - delta**2, 0.0))

    return coords


def traffic_distance(payload: np.ndarray, floor: float = 1e-9) -> np.ndarray:
    """Turn per-edge payload into a distance: heavy traffic means close.

    Normalised by the heaviest edge so the unit is "how much further apart than
    the chattiest pair", which keeps the embedding scale-free -- the same
    workload at twice the message size embeds identically.
    """
    payload = np.asarray(payload, dtype=float)
    if payload.size == 0:
        return payload
    peak = float(payload.max())
    if peak <= 0:
        return np.ones_like(payload)
    return peak / np.maximum(payload, peak * floor)


def embed_graph(graph, kappa: int = 4, seed: int = 0) -> np.ndarray:
    """Embed a :class:`~iotc.compiler.ir.TaskGraph` by communication distance."""
    edges = graph.edges
    if graph.n_edges == 0:
        return np.zeros((graph.n_tasks, kappa))
    return fastmap(
        graph.n_tasks,
        edges[:, 0],
        edges[:, 1],
        traffic_distance(graph.payload),
        kappa=kappa,
        seed=seed,
    )

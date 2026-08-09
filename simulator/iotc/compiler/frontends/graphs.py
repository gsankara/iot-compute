"""Seeded problem-graph generators.

The sparse and NP-hard frontends both need an input graph before they can build
a task graph, and the two must agree on what "the same instance" means.  Every
generator here takes a seed and returns a symmetric edge list in canonical
``(u < v)`` order, sorted -- so a given ``(kind, n, seed)`` is one fixed
instance on every machine and every run.
"""

from __future__ import annotations

import numpy as np


def _canonical(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate, drop self-loops, and sort into canonical order."""
    lo, hi = np.minimum(u, v), np.maximum(u, v)
    keep = lo != hi
    lo, hi = lo[keep].astype(np.int64), hi[keep].astype(np.int64)
    if lo.size == 0:
        return lo, hi
    pairs = np.unique(np.column_stack([lo, hi]), axis=0)
    return pairs[:, 0], pairs[:, 1]


def erdos_renyi(n: int, avg_degree: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """G(n, p) with ``p`` chosen to hit ``avg_degree`` in expectation.

    Sampled by drawing a binomial edge count and then picking that many pairs,
    which is O(m) rather than O(n^2) -- necessary because these frontends are
    meant to be exercised at thousands of vertices.
    """
    rng = np.random.default_rng(seed)
    p = min(max(avg_degree / max(n - 1, 1), 0.0), 1.0)
    max_edges = n * (n - 1) // 2
    m = int(rng.binomial(max_edges, p))
    if m == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    # Oversample to absorb the collisions the dedupe will remove.
    draw = min(int(m * 1.15) + 16, max_edges)
    u = rng.integers(0, n, size=draw)
    v = rng.integers(0, n, size=draw)
    u, v = _canonical(u, v)
    # _canonical sorts, so trimming to m must sample rather than truncate --
    # taking a prefix would hand every edge to the lowest-numbered vertices.
    if u.size > m:
        pick = np.sort(rng.choice(u.size, size=m, replace=False))
        u, v = u[pick], v[pick]
    return u, v


def planted_clique(
    n: int, avg_degree: float, clique_size: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """An Erdos-Renyi graph with a known clique embedded in it.

    Returns ``(u, v, members)``.  The planted set is the ground truth for B10:
    it is the only instance family where "did the swarm find the right answer"
    has an answer that does not itself require an exponential search.
    """
    rng = np.random.default_rng(seed)
    u, v = erdos_renyi(n, avg_degree, seed=seed)
    members = np.sort(rng.choice(n, size=min(clique_size, n), replace=False))
    cu, cv = np.triu_indices(members.size, k=1)
    u = np.concatenate([u, members[cu]])
    v = np.concatenate([v, members[cv]])
    u, v = _canonical(u, v)
    return u, v, members


def power_law(n: int, m_attach: int = 3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Preferential attachment (Barabasi-Albert).

    Included because its degree distribution is the adversarial case for
    partitioning and for beam assignment alike: a handful of tasks want far more
    neighbours than any device has transceivers, so B5 is forced to route rather
    than match, and B8's centrality-driven replication has something to bite on.
    """
    rng = np.random.default_rng(seed)
    m_attach = max(1, min(m_attach, max(n - 1, 1)))
    targets = list(range(m_attach))
    repeated: list[int] = list(targets)
    us, vs = [], []
    for node in range(m_attach, n):
        chosen = set()
        while len(chosen) < m_attach:
            chosen.add(int(repeated[rng.integers(0, len(repeated))]))
        for t in chosen:
            us.append(node); vs.append(t)
            repeated.extend((node, t))
    if not us:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return _canonical(np.asarray(us), np.asarray(vs))


def banded(n: int, bandwidth: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """A banded matrix's sparsity pattern -- the easy, local, PDE-like case."""
    us, vs = [], []
    for d in range(1, bandwidth + 1):
        i = np.arange(0, n - d)
        us.append(i); vs.append(i + d)
    if not us:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return _canonical(np.concatenate(us), np.concatenate(vs))


def build(kind: str, n: int, seed: int = 0, **kw):
    """Dispatch by name, so a config can name an instance family as a string."""
    if kind == "erdos_renyi":
        return erdos_renyi(n, kw.get("avg_degree", 8.0), seed)
    if kind == "power_law":
        return power_law(n, int(kw.get("m_attach", 3)), seed)
    if kind == "banded":
        return banded(n, int(kw.get("bandwidth", 3)))
    if kind == "planted_clique":
        u, v, _ = planted_clique(n, kw.get("avg_degree", 8.0), int(kw.get("clique_size", 12)), seed)
        return u, v
    raise ValueError(f"unknown graph kind {kind!r}")


def block_partition(n: int, parts: int) -> np.ndarray:
    """Contiguous vertex-to-block map -- the naive partition B2 must beat.

    Deliberately dumb.  It is the baseline every partitioner is scored against,
    and on a banded matrix it is already near-optimal, which is the point: B2
    has to show it wins where structure is *absent*, not where it is handed over.
    """
    sizes = np.full(parts, n // parts, dtype=np.int64)
    sizes[: n % parts] += 1
    return np.repeat(np.arange(parts), sizes)

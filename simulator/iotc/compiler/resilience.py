"""B8 -- Resilience: survive the faults Phase A can already inject.

Three mechanisms, and they answer three different failures.

**Replication chosen by centrality.**  Devices die.  A swarm with no redundancy
loses whatever the dead device held, but it also loses everything that was
*routed through* it -- and those are not the same set.  Betweenness centrality
finds the tasks whose loss disconnects the most traffic, and replicating the top
few is worth far more than replicating the same number at random.  That claim is
the pass's quality metric, and ``uniform`` is kept beside ``centrality`` to test
it rather than assert it.

**Backup beam targets via LSH.**  When a neighbour dies, the survivor needs
somewhere else to point, quickly, without a global search.  The FastMap
embedding B2 and B3 already built is the index: hash the embedded coordinates
into buckets and a failover candidate is a bucket lookup.  ``docs/ROADMAP.md``
lists this as the Ch.1 FastMap+LSH application, and it costs nothing extra
because the embedding exists anyway.

**A closed-loop pointing tracker.**  This is the gap Phase A left on purpose
(``docs/PHYSICS.md`` §7): pointing is open loop, which is why the drift fault is
so destructive -- a 0.04 m/√s current destroys every link, and open-loop
tolerance is centimetres.  The tracker belongs to the compiled runtime rather
than the simulator, so it lands here: compute how often a link must re-aim to
stay inside tolerance, and charge the schedule for those slots.

Erasure coding and checkpoint intervals are named in the roadmap and are *not*
implemented here; they are recorded as remaining work rather than quietly
dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..registry import register
from .fastmap import build_csr, embed_graph
from .ir import TaskGraph


# ------------------------------------------------------------------ centrality

def betweenness(graph: TaskGraph, samples: int = 64, seed: int = 0) -> np.ndarray:
    """Approximate betweenness by Brandes from a sample of sources.

    Exact betweenness is ``O(n·m)`` and this project has a standing near-linear
    constraint, so the sources are sampled: ``O(samples·(n + m))`` with the
    ranking converging long before the absolute values do.  Ranking is all the
    pass needs -- it replicates the top K, not a numeric threshold.
    """
    n = graph.n_tasks
    if n == 0 or graph.n_edges == 0:
        return np.zeros(n)

    e = graph.edges
    indptr, indices, _ = build_csr(n, e[:, 0], e[:, 1], np.ones(e.shape[0]))
    rng = np.random.default_rng(seed)
    sources = rng.choice(n, size=min(samples, n), replace=False)

    score = np.zeros(n)
    for s in sources:
        # Brandes' single-source pass on an unweighted graph.
        sigma = np.zeros(n)
        sigma[s] = 1.0
        dist = np.full(n, -1, dtype=np.int64)
        dist[s] = 0
        order: list[int] = []
        preds: list[list[int]] = [[] for _ in range(n)]
        queue = [int(s)]
        while queue:
            v = queue.pop(0)
            order.append(v)
            for k in range(indptr[v], indptr[v + 1]):
                w = int(indices[k])
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    preds[w].append(v)
        delta = np.zeros(n)
        for w in reversed(order):
            for v in preds[w]:
                if sigma[w] > 0:
                    delta[v] += sigma[v] / sigma[w] * (1.0 + delta[w])
            if w != s:
                score[w] += delta[w]
    return score * (n / max(len(sources), 1))


def traffic_weight(graph: TaskGraph) -> np.ndarray:
    """Bytes each device would strand if it vanished."""
    return graph.traffic_per_task()


# ----------------------------------------------------------------- replication

class ReplicationPolicy:
    """The ``resilience`` plugin slot: which tasks get a standby copy."""

    def choose(self, graph: TaskGraph, budget: int, seed: int = 0) -> np.ndarray:
        raise NotImplementedError  # pragma: no cover


@register("resilience", "centrality", "betweenness")
class CentralityReplication(ReplicationPolicy):
    """Replicate the tasks whose loss disconnects the most traffic.  The pass."""

    def __init__(self, samples: int = 64) -> None:
        self.samples = int(samples)

    def choose(self, graph: TaskGraph, budget: int, seed: int = 0) -> np.ndarray:
        score = betweenness(graph, self.samples, seed)
        return np.argsort(-score, kind="stable")[:budget]


@register("resilience", "traffic")
class TrafficReplication(ReplicationPolicy):
    """Replicate the busiest devices.

    A cheaper proxy for centrality -- degree-weighted rather than path-weighted.
    Carried because it is what most people would reach for, and because the gap
    between it and betweenness is the value of the harder computation.
    """

    def choose(self, graph: TaskGraph, budget: int, seed: int = 0) -> np.ndarray:
        return np.argsort(-traffic_weight(graph), kind="stable")[:budget]


@register("resilience", "uniform", "random")
class UniformReplication(ReplicationPolicy):
    """Replicate an arbitrary set.  The comparator."""

    def choose(self, graph: TaskGraph, budget: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.choice(graph.n_tasks, size=min(budget, graph.n_tasks), replace=False)


@register("resilience", "none")
class NoReplication(ReplicationPolicy):
    """No redundancy at all -- the floor every other policy has to beat."""

    def choose(self, graph: TaskGraph, budget: int, seed: int = 0) -> np.ndarray:
        return np.zeros(0, dtype=np.int64)


# -------------------------------------------------------------- LSH failover

class FailoverIndex:
    """Bucketed nearest-healthy-device lookup over the FastMap embedding.

    Random-hyperplane LSH on coordinates the compiler already computed, so the
    index is free.  A query walks its own bucket first and widens only if that
    bucket is entirely dead, which keeps the common case constant-time instead
    of a scan over the swarm.
    """

    def __init__(self, coords: np.ndarray, bits: int = 8, seed: int = 0) -> None:
        self.coords = np.asarray(coords, dtype=float)
        rng = np.random.default_rng(seed)
        dim = self.coords.shape[1] if self.coords.ndim == 2 else 1
        self.planes = rng.normal(size=(bits, dim))
        self.buckets: dict[int, list[int]] = {}
        for i, sig in enumerate(self._signatures(self.coords)):
            self.buckets.setdefault(int(sig), []).append(i)

    def _signatures(self, coords: np.ndarray) -> np.ndarray:
        bits = (coords @ self.planes.T) > 0
        weights = 1 << np.arange(bits.shape[1])
        return (bits * weights).sum(axis=1)

    def nearest_healthy(self, device: int, alive: np.ndarray) -> int:
        """Closest live device to ``device`` in the embedding, or -1 if none."""
        alive = np.asarray(alive, dtype=bool)
        sig = int(self._signatures(self.coords[device:device + 1])[0])
        candidates = [i for i in self.buckets.get(sig, []) if alive[i] and i != device]
        if not candidates:
            candidates = [i for i in range(self.coords.shape[0])
                          if alive[i] and i != device]
        if not candidates:
            return -1
        d = np.linalg.norm(self.coords[candidates] - self.coords[device], axis=1)
        return int(candidates[int(np.argmin(d))])

    def bucket_sizes(self) -> list[int]:
        return sorted((len(v) for v in self.buckets.values()), reverse=True)


# ------------------------------------------------------- the pointing tracker

def reaim_interval(tolerance: float, drift_rate: float) -> float:
    """Seconds a link stays inside ``tolerance`` under a random-walk drift.

    Phase A models drift as a random walk, so displacement grows as
    ``rate * sqrt(t)`` -- the fault is specified in m/sqrt(s).  Inverting gives
    the longest a link may be left alone:

        t = (tolerance / rate)^2

    The square is why this matters: halving the tolerance quarters the interval,
    so a tighter beam does not merely cost pointing loss, it costs *slots*.
    """
    if drift_rate <= 0:
        return math.inf
    return (tolerance / drift_rate) ** 2


@dataclass
class TrackerCost:
    """What closing the pointing loop costs a schedule."""

    interval: float
    frame: float
    energy_per_reaim: float
    #: Links that must be tracked -- the ones open long enough to drift.
    tracked_links: int = 0

    @property
    def reaims_per_frame(self) -> float:
        if not math.isfinite(self.interval) or self.interval <= 0:
            return 0.0
        return self.frame / self.interval

    @property
    def energy_per_frame(self) -> float:
        return self.reaims_per_frame * self.energy_per_reaim * self.tracked_links

    @property
    def needed(self) -> bool:
        """Does the frame outlast the drift tolerance?"""
        return self.reaims_per_frame > 1.0


# ---------------------------------------------------------------- assessment

@dataclass
class ResilienceResult:
    """Which tasks were replicated, and what that buys under failure."""

    replicated: np.ndarray
    graph: TaskGraph
    method: str = ""
    coords: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    @property
    def budget(self) -> int:
        return int(self.replicated.size)

    def surviving_traffic(self, dead: np.ndarray) -> float:
        """Share of payload still deliverable after ``dead`` devices fail.

        An edge survives if both endpoints are alive *or* replicated -- a
        replica stands in for its original.  Traffic is the unit rather than
        task count because a swarm that keeps every small task and loses the
        hub has not survived anything.
        """
        dead = np.asarray(dead, dtype=bool)
        covered = ~dead
        covered[self.replicated] = True
        e = self.graph.edges
        if e.size == 0:
            return 1.0
        live = covered[e[:, 0]] & covered[e[:, 1]]
        total = float(self.graph.payload.sum())
        return float(self.graph.payload[live].sum()) / total if total > 0 else 1.0

    def connected_traffic(self, dead: np.ndarray) -> float:
        """Share of payload left inside the largest surviving component.

        The measure :meth:`surviving_traffic` misses.  A superstep is a
        collective: tasks that survive but can no longer reach each other have
        not survived in any useful sense.  This counts only traffic whose
        endpoints are still mutually reachable through live devices, which is
        exactly what betweenness centrality is a proxy for -- and the reason a
        policy chosen by betweenness has to be scored on connectivity rather
        than on direct edge survival, or it is being judged on someone else's
        objective.
        """
        dead = np.asarray(dead, dtype=bool)
        covered = ~dead
        covered[self.replicated] = True
        e = self.graph.edges
        if e.size == 0:
            return 1.0

        alive_edges = covered[e[:, 0]] & covered[e[:, 1]]
        sub = self.graph.subgraph(covered)
        if sub.n_tasks == 0:
            return 0.0
        labels = sub.connected_components()
        sizes = np.bincount(labels)
        biggest = int(np.argmax(sizes))
        keep = labels == biggest

        # Map surviving edges back through the subgraph's renumbering.
        remap = np.full(self.graph.n_tasks, -1, dtype=np.int64)
        remap[covered] = np.arange(int(covered.sum()))
        src, dst = remap[e[alive_edges, 0]], remap[e[alive_edges, 1]]
        inside = keep[src] & keep[dst]
        total = float(self.graph.payload.sum())
        delivered = float(self.graph.payload[alive_edges][inside].sum())
        return delivered / total if total > 0 else 1.0

    def summary(self) -> str:
        return f"{self.method:12s} replicated={self.budget:3d}"


def plan_resilience(
    graph: TaskGraph,
    budget: int,
    model: ReplicationPolicy | str = "centrality",
    seed: int = 0,
    kappa: int = 4,
) -> ResilienceResult:
    """Choose replicas and build the failover index over the same embedding."""
    if isinstance(model, str):
        from ..registry import create

        name, model = model, create("resilience", model)
    else:
        name = type(model).plugin_name

    chosen = np.asarray(model.choose(graph, budget, seed), dtype=np.int64)
    coords = embed_graph(graph, kappa=kappa, seed=seed)
    return ResilienceResult(replicated=chosen, graph=graph, method=name, coords=coords)

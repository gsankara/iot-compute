"""B0 -- the compiler's intermediate representation.

A :class:`TaskGraph` is what every later pass consumes: *tasks* that carry work
(FLOPs) and state (working-set bytes), and *edges* that carry traffic (payload
bytes per superstep).

Three modelling decisions are worth stating up front, because the rest of the
compiler is built on them.

**It is a communication graph, not a dependence DAG.**  One graph describes one
*superstep*: every task is ready, they compute, they exchange payloads with
their neighbours, and the step repeats ``iterations`` times.  That is the right
shape for this substrate.  A dependence DAG would ask the swarm to re-aim
between every producer and consumer, and Phase A measured retargeting at ~0.5 s
against microseconds of transmission -- a fabric that re-plans its topology per
edge would spend all of its time pointing.  Supersteps let B5/B6 hold a beam
assignment still and amortise it, which is the entire scheduling strategy.

**Edges are undirected and stored canonically** (``src < dst``).  A link on this
fabric exists only when *both* gimbals are aimed at each other, so a one-way
edge is not physically representable; making that an invariant of the IR means
no later pass can invent one.

**The numeric fields are arrays, not objects.**  Placement has to run over
thousands of tasks in near-linear time (see ``docs/ROADMAP.md``), so the source
of truth is a handful of contiguous ``numpy`` arrays and :class:`Task` is a
read-only view constructed on demand.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from ..units import KiB, MiB


@dataclass(frozen=True)
class Task:
    """A read-only view of one task in a :class:`TaskGraph`."""

    id: int
    flop: float
    working_set: float
    kind: str
    label: str

    def __str__(self) -> str:
        return (
            f"task {self.id:5d} {self.kind:10s} "
            f"{self.flop / 1e6:9.3f} MFLOP  {self.working_set / KiB:8.1f} KiB"
            f"{'  ' + self.label if self.label else ''}"
        )


class TaskGraphError(ValueError):
    """Raised when a graph violates an IR invariant."""


class TaskGraph:
    """Tasks with work and state, joined by edges with payload.

    Build one incrementally::

        g = TaskGraph("jacobi")
        a = g.add_task(flop=1e6, working_set=64 * KiB, kind="stencil")
        b = g.add_task(flop=1e6, working_set=64 * KiB, kind="stencil")
        g.add_edge(a, b, payload=4 * KiB)
        g.validate()

    or hand it arrays directly with :meth:`from_arrays` when a frontend can
    generate them in bulk, which is the fast path every stock frontend takes.
    """

    __slots__ = (
        "name",
        "iterations",
        "_flop",
        "_working_set",
        "_kind",
        "_label",
        "_src",
        "_dst",
        "_payload",
        "_n",
        "_finalized",
    )

    def __init__(self, name: str = "graph", iterations: int = 1) -> None:
        if iterations < 1:
            raise TaskGraphError(f"iterations must be >= 1, got {iterations}")
        self.name = name
        #: How many times the superstep repeats.  Traffic and FLOPs scale with
        #: it; the working set and the topology do not.
        self.iterations = int(iterations)
        self._flop: list[float] | np.ndarray = []
        self._working_set: list[float] | np.ndarray = []
        self._kind: list[str] = []
        self._label: list[str] = []
        self._src: list[int] | np.ndarray = []
        self._dst: list[int] | np.ndarray = []
        self._payload: list[float] | np.ndarray = []
        self._n = 0
        self._finalized = False

    # ------------------------------------------------------------------ build

    def add_task(
        self,
        flop: float,
        working_set: float,
        kind: str = "compute",
        label: str = "",
    ) -> int:
        """Append one task and return its id."""
        self._ensure_mutable()
        if flop < 0 or working_set < 0:
            raise TaskGraphError(f"negative task cost: flop={flop}, working_set={working_set}")
        self._flop.append(float(flop))  # type: ignore[union-attr]
        self._working_set.append(float(working_set))  # type: ignore[union-attr]
        self._kind.append(kind)
        self._label.append(label)
        self._n += 1
        return self._n - 1

    def add_edge(self, u: int, v: int, payload: float) -> None:
        """Join two tasks with ``payload`` bytes exchanged per superstep.

        Edges are undirected; ``(u, v)`` and ``(v, u)`` are the same edge and are
        stored canonically.  Duplicates are *not* merged here -- :meth:`finalize`
        coalesces them by summing payload, so a frontend may emit the same pair
        from two different reasons without bookkeeping.
        """
        self._ensure_mutable()
        if u == v:
            raise TaskGraphError(f"self-loop on task {u}: a device cannot aim at itself")
        if payload < 0:
            raise TaskGraphError(f"negative payload on edge ({u}, {v}): {payload}")
        lo, hi = (u, v) if u < v else (v, u)
        self._src.append(int(lo))  # type: ignore[union-attr]
        self._dst.append(int(hi))  # type: ignore[union-attr]
        self._payload.append(float(payload))  # type: ignore[union-attr]

    def _ensure_mutable(self) -> None:
        if self._finalized:
            raise TaskGraphError("graph is finalized; build a new one or use .copy()")

    @classmethod
    def from_arrays(
        cls,
        flop: Sequence[float] | np.ndarray,
        working_set: Sequence[float] | np.ndarray,
        src: Sequence[int] | np.ndarray,
        dst: Sequence[int] | np.ndarray,
        payload: Sequence[float] | np.ndarray,
        *,
        name: str = "graph",
        iterations: int = 1,
        kind: str | Sequence[str] = "compute",
        label: Sequence[str] | None = None,
    ) -> "TaskGraph":
        """Bulk constructor -- the path every stock frontend uses."""
        g = cls(name=name, iterations=iterations)
        flop = np.asarray(flop, dtype=float)
        working_set = np.asarray(working_set, dtype=float)
        if flop.shape != working_set.shape:
            raise TaskGraphError(
                f"flop {flop.shape} and working_set {working_set.shape} disagree"
            )
        n = int(flop.size)
        g._flop = flop
        g._working_set = working_set
        g._kind = [kind] * n if isinstance(kind, str) else list(kind)
        g._label = list(label) if label is not None else [""] * n
        g._n = n

        src = np.asarray(src, dtype=np.int64).ravel()
        dst = np.asarray(dst, dtype=np.int64).ravel()
        payload = np.asarray(payload, dtype=float).ravel()
        lo = np.minimum(src, dst)
        hi = np.maximum(src, dst)
        g._src, g._dst, g._payload = lo, hi, payload
        return g.finalize()

    def finalize(self) -> "TaskGraph":
        """Freeze into arrays, coalesce duplicate edges, and validate."""
        if not self._finalized:
            self._flop = np.asarray(self._flop, dtype=float)
            self._working_set = np.asarray(self._working_set, dtype=float)
            src = np.asarray(self._src, dtype=np.int64).ravel()
            dst = np.asarray(self._dst, dtype=np.int64).ravel()
            payload = np.asarray(self._payload, dtype=float).ravel()
            if src.size:
                # Coalesce (u, v) duplicates by summing payload.  Sorting on a
                # single packed key keeps this O(m log m) rather than hashing
                # pairs, which matters at 10k tasks.
                key = src * (self._n + 1) + dst
                order = np.argsort(key, kind="stable")
                key, src, dst, payload = key[order], src[order], dst[order], payload[order]
                first = np.empty(key.size, dtype=bool)
                first[0] = True
                np.not_equal(key[1:], key[:-1], out=first[1:])
                group = np.cumsum(first) - 1
                merged = np.zeros(int(group[-1]) + 1, dtype=float)
                np.add.at(merged, group, payload)
                src, dst, payload = src[first], dst[first], merged
            self._src, self._dst, self._payload = src, dst, payload
            self._finalized = True
        self.validate()
        return self

    # ------------------------------------------------------------- properties

    @property
    def n_tasks(self) -> int:
        return self._n

    @property
    def n_edges(self) -> int:
        return int(np.asarray(self._src).size)

    @property
    def flop(self) -> np.ndarray:
        return np.asarray(self._flop, dtype=float)

    @property
    def working_set(self) -> np.ndarray:
        return np.asarray(self._working_set, dtype=float)

    @property
    def payload(self) -> np.ndarray:
        return np.asarray(self._payload, dtype=float)

    @property
    def edges(self) -> np.ndarray:
        """``(m, 2)`` array of canonical task-id pairs."""
        return np.column_stack([np.asarray(self._src), np.asarray(self._dst)])

    @property
    def kinds(self) -> list[str]:
        return list(self._kind)

    def task(self, i: int) -> Task:
        return Task(
            id=int(i),
            flop=float(self.flop[i]),
            working_set=float(self.working_set[i]),
            kind=self._kind[i],
            label=self._label[i],
        )

    def __iter__(self) -> Iterator[Task]:
        return (self.task(i) for i in range(self._n))

    def __len__(self) -> int:
        return self._n

    def __repr__(self) -> str:
        return (
            f"<TaskGraph {self.name!r} tasks={self.n_tasks} edges={self.n_edges} "
            f"iterations={self.iterations}>"
        )

    # ------------------------------------------------------------- structure

    def degree(self) -> np.ndarray:
        """Number of distinct neighbours per task."""
        deg = np.zeros(self._n, dtype=np.int64)
        if self.n_edges:
            np.add.at(deg, np.asarray(self._src), 1)
            np.add.at(deg, np.asarray(self._dst), 1)
        return deg

    def traffic_per_task(self) -> np.ndarray:
        """Bytes each task sends+receives per superstep, summed over its edges."""
        out = np.zeros(self._n, dtype=float)
        if self.n_edges:
            np.add.at(out, np.asarray(self._src), self.payload)
            np.add.at(out, np.asarray(self._dst), self.payload)
        return out

    def neighbours(self, i: int) -> np.ndarray:
        src, dst = np.asarray(self._src), np.asarray(self._dst)
        return np.concatenate([dst[src == i], src[dst == i]])

    def adjacency(self) -> np.ndarray:
        """Dense symmetric payload matrix.  For small graphs and tests only."""
        a = np.zeros((self._n, self._n), dtype=float)
        if self.n_edges:
            src, dst = np.asarray(self._src), np.asarray(self._dst)
            a[src, dst] = self.payload
            a[dst, src] = self.payload
        return a

    def connected_components(self) -> np.ndarray:
        """Component label per task, via weighted union-find (near-linear)."""
        parent = np.arange(self._n)

        def find(x: int) -> int:
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:  # path compression
                parent[x], x = root, parent[x]
            return root

        for u, v in zip(np.asarray(self._src), np.asarray(self._dst)):
            ru, rv = find(int(u)), find(int(v))
            if ru != rv:
                parent[max(ru, rv)] = min(ru, rv)
        roots = np.array([find(i) for i in range(self._n)])
        _, labels = np.unique(roots, return_inverse=True)
        return labels

    # ----------------------------------------------------------- aggregates

    @property
    def total_flop(self) -> float:
        """FLOPs over the whole run, all tasks, all iterations."""
        return float(self.flop.sum()) * self.iterations

    @property
    def total_bytes(self) -> float:
        """Payload bytes moved over the whole run."""
        return float(self.payload.sum()) * self.iterations

    @property
    def arithmetic_intensity(self) -> float:
        """FLOP per byte communicated.

        The single most predictive number about whether a workload belongs on
        this fabric.  Phase A measured aiming at ~0.5 s against microseconds of
        transmission, so a low-intensity workload does not merely communicate a
        lot -- it re-aims a lot, and the gimbals become the machine.
        """
        b = self.total_bytes
        return math.inf if b == 0 else self.total_flop / b

    def summary(self) -> str:
        deg = self.degree()
        ws = self.working_set
        return (
            f"{self.name:16s} tasks={self.n_tasks:6d} edges={self.n_edges:7d} "
            f"iters={self.iterations:4d}  "
            f"deg {deg.min() if self._n else 0:3d}/{deg.mean() if self._n else 0:6.2f}"
            f"/{deg.max() if self._n else 0:3d}  "
            f"ws {ws.min() / KiB if self._n else 0:7.1f}-{ws.max() / KiB if self._n else 0:7.1f} KiB  "
            f"{self.total_flop / 1e9:9.3f} GFLOP  {self.total_bytes / MiB:9.3f} MiB  "
            f"AI {self.arithmetic_intensity:8.2f} FLOP/B"
        )

    # ------------------------------------------------------------- machine fit

    def fits(self, device_class) -> np.ndarray:
        """Boolean mask of tasks whose working set fits the class's RAM.

        This is the predicate B1 (Tile) exists to satisfy.  It reads
        ``DeviceClass`` directly rather than taking a byte count, so the
        compiler cannot drift from what the simulator thinks a node holds.
        """
        return self.working_set <= float(device_class.compute.ram_bytes)

    def compute_time(self, device_class) -> np.ndarray:
        """Per-task compute time [s] on ``device_class``, one superstep."""
        return self.flop / max(device_class.compute.flops, 1e-30)

    def transfer_time(self, device_class) -> np.ndarray:
        """Per-edge on-the-wire time [s], ignoring aiming."""
        return self.payload * 8.0 / max(device_class.optical.line_rate, 1e-30)

    def aim_fraction(self, device_class, retarget_steps: int = 100) -> float:
        """Share of link time spent aiming rather than sending, per superstep.

        Reproduces the aim-dominated finding at IR level: if this is close to
        1.0, B6 must amortise retargeting or the workload will not run, and no
        amount of payload tuning in B0/B1 will save it.
        """
        aim = device_class.optical.gimbal.retarget_time(retarget_steps)
        wire = float(self.transfer_time(device_class).sum())
        total_aim = aim * self.n_edges
        denom = total_aim + wire
        return 0.0 if denom == 0 else total_aim / denom

    # ------------------------------------------------------------- transforms

    def subgraph(self, keep: np.ndarray) -> "TaskGraph":
        """Induced subgraph on the tasks selected by boolean mask ``keep``."""
        keep = np.asarray(keep, dtype=bool)
        if keep.shape != (self._n,):
            raise TaskGraphError(f"mask shape {keep.shape} != ({self._n},)")
        remap = np.full(self._n, -1, dtype=np.int64)
        remap[keep] = np.arange(int(keep.sum()))
        src, dst = np.asarray(self._src), np.asarray(self._dst)
        alive = keep[src] & keep[dst]
        return TaskGraph.from_arrays(
            self.flop[keep],
            self.working_set[keep],
            remap[src[alive]],
            remap[dst[alive]],
            self.payload[alive],
            name=f"{self.name}/sub",
            iterations=self.iterations,
            kind=[k for k, on in zip(self._kind, keep) if on],
            label=[s for s, on in zip(self._label, keep) if on],
        )

    def copy(self) -> "TaskGraph":
        return TaskGraph.from_arrays(
            self.flop.copy(),
            self.working_set.copy(),
            np.asarray(self._src).copy(),
            np.asarray(self._dst).copy(),
            self.payload.copy(),
            name=self.name,
            iterations=self.iterations,
            kind=list(self._kind),
            label=list(self._label),
        )

    # ---------------------------------------------------------------- validate

    def validate(self) -> "TaskGraph":
        """Check every IR invariant.  Raises :class:`TaskGraphError`."""
        n = self._n
        flop, ws = self.flop, self.working_set
        if flop.shape != (n,) or ws.shape != (n,):
            raise TaskGraphError(f"task arrays disagree with n_tasks={n}")
        if len(self._kind) != n or len(self._label) != n:
            raise TaskGraphError(f"kind/label length != n_tasks={n}")
        if not np.all(np.isfinite(flop)) or not np.all(np.isfinite(ws)):
            raise TaskGraphError("non-finite task cost")
        if np.any(flop < 0) or np.any(ws < 0):
            raise TaskGraphError("negative task cost")

        src, dst, pay = np.asarray(self._src), np.asarray(self._dst), self.payload
        if not (src.shape == dst.shape == pay.shape):
            raise TaskGraphError("edge arrays disagree in length")
        if src.size:
            if src.min() < 0 or dst.max() >= n:
                raise TaskGraphError("edge references a task id out of range")
            if np.any(src == dst):
                raise TaskGraphError("self-loop: a device cannot aim at itself")
            if np.any(src > dst):
                raise TaskGraphError("edge not in canonical (src < dst) order")
            if not np.all(np.isfinite(pay)) or np.any(pay < 0):
                raise TaskGraphError("negative or non-finite payload")
            key = src * (n + 1) + dst
            if np.unique(key).size != key.size:
                raise TaskGraphError("duplicate edge survived coalescing")
        return self

    # ----------------------------------------------------------- serialization

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "flop": self.flop.tolist(),
            "working_set": self.working_set.tolist(),
            "kind": list(self._kind),
            "label": list(self._label),
            "src": np.asarray(self._src).tolist(),
            "dst": np.asarray(self._dst).tolist(),
            "payload": self.payload.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TaskGraph":
        return cls.from_arrays(
            d["flop"],
            d["working_set"],
            d["src"],
            d["dst"],
            d["payload"],
            name=d.get("name", "graph"),
            iterations=int(d.get("iterations", 1)),
            kind=d.get("kind", "compute"),
            label=d.get("label"),
        )

    def to_json(self, path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh)

    @classmethod
    def from_json(cls, path) -> "TaskGraph":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

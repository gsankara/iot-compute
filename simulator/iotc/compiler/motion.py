"""B4 -- Motion: fly the swarm from where it is to where B3 wants it.

The naive plan is to interpolate: every device sets off toward its target at
once, along a straight line.  It is the shortest path and the fastest makespan,
and on this substrate it destroys the machine while it runs.

**A link needs both gimbals aimed, and there is no tracker.**  Phase A leaves the
closed-loop pointing tracker out on purpose (``docs/PHYSICS.md`` §7) and measured
what that costs: open-loop pointing tolerates *centimetres*, and a 0.04 m/s
current is enough to break every link.  So a link whose two endpoints are both
in motion is not a degraded link, it is a dropped one.  If one end holds still,
the moving end has a fixed target to re-aim against and the link survives.

That turns trajectory planning into a graph-colouring problem.  Colour the
critical links so that no two adjacent devices move in the same wave; then every
critical link has a stationary endpoint throughout, at the cost of a longer
manoeuvre.  :class:`WavePlanner` does that and :class:`DirectPlanner` is kept
beside it to measure what interpolating would have cost.

**Bodies have volume.**  Two devices whose straight lines cross will collide even
if their endpoints are properly separated, and the placement's separation
guarantee says nothing about the journey.  Every plan is checked along its whole
path, not only at its destination.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..registry import register
from .ir import TaskGraph
from .place import MotionBudget


def critical_edges(graph: TaskGraph, fraction: float = 0.5) -> np.ndarray:
    """Boolean mask of the edges worth protecting in flight.

    Not every link matters equally: payload is heavily skewed, so protecting the
    heaviest ``fraction`` of the traffic costs far fewer waves than protecting
    every edge.  Returns the smallest set of edges carrying that share.
    """
    pay = graph.payload
    if pay.size == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(-pay)
    cumulative = np.cumsum(pay[order])
    total = cumulative[-1]
    keep_count = int(np.searchsorted(cumulative, total * fraction)) + 1
    mask = np.zeros(pay.size, dtype=bool)
    mask[order[:keep_count]] = True
    return mask


def wave_colouring(graph: TaskGraph, protect: np.ndarray) -> np.ndarray:
    """Assign each device a wave so no protected link moves both its ends.

    Greedy colouring in descending degree order over the protected subgraph.
    Greedy is not optimal, but it needs at most ``max_degree + 1`` colours and
    runs in ``O(m)`` -- and the wave count is a *makespan multiplier*, so an
    exact chromatic number would buy at most one wave for a great deal of time.
    """
    n = graph.n_tasks
    e = graph.edges
    neighbours: list[list[int]] = [[] for _ in range(n)]
    for idx in np.flatnonzero(protect):
        u, v = int(e[idx, 0]), int(e[idx, 1])
        neighbours[u].append(v)
        neighbours[v].append(u)

    colour = np.full(n, -1, dtype=np.int64)
    for v in sorted(range(n), key=lambda x: -len(neighbours[x])):
        taken = {colour[nb] for nb in neighbours[v] if colour[nb] >= 0}
        c = 0
        while c in taken:
            c += 1
        colour[v] = c
    return colour


@dataclass
class MotionPlan:
    """Waypoints over time, plus the evidence the manoeuvre is survivable."""

    #: ``(steps + 1, n, 3)`` positions.  Index 0 is the start formation.
    path: np.ndarray
    graph: TaskGraph
    budget: MotionBudget
    protect: np.ndarray
    method: str = ""
    waves: int = 1
    step_seconds: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def steps(self) -> int:
        return self.path.shape[0] - 1

    @property
    def makespan(self) -> float:
        """Seconds from the first device moving to the last arriving."""
        return float(self.step_seconds.sum())

    def _lengths(self, frame: np.ndarray) -> np.ndarray:
        e = self.graph.edges
        if e.size == 0:
            return np.zeros(0)
        return np.linalg.norm(frame[e[:, 0]] - frame[e[:, 1]], axis=1)

    @property
    def broken_link_steps(self) -> int:
        """Count of (protected edge, step) pairs where both ends were moving.

        The pointing failure, not the range failure: two moving gimbals cannot
        hold a beam on each other without a tracker, however close they are.
        """
        if not self.protect.any():
            return 0
        e = self.graph.edges
        moving = np.linalg.norm(np.diff(self.path, axis=0), axis=2) > 1e-9
        both = moving[:, e[self.protect, 0]] & moving[:, e[self.protect, 1]]
        return int(both.sum())

    @property
    def broken_link_fraction(self) -> float:
        total = self.protect.sum() * max(self.steps, 1)
        return self.broken_link_steps / total if total else 0.0

    @property
    def out_of_reach_steps(self) -> int:
        """Steps at which some edge is stretched beyond the optical budget."""
        if self.graph.n_edges == 0:
            return 0
        return int(sum(
            (self._lengths(frame) > self.budget.max_link).any()
            for frame in self.path
        ))

    @property
    def min_separation(self) -> float:
        """Closest approach of any two bodies at any point on the journey."""
        best = math.inf
        for frame in self.path:
            if frame.shape[0] < 2:
                return math.inf
            d = np.linalg.norm(frame[:, None, :] - frame[None, :, :], axis=2)
            np.fill_diagonal(d, np.inf)
            best = min(best, float(d.min()))
        return best

    @property
    def collisions(self) -> int:
        """Times two bodies come closer than they can physically be."""
        count = 0
        for frame in self.path:
            if frame.shape[0] < 2:
                continue
            d = np.linalg.norm(frame[:, None, :] - frame[None, :, :], axis=2)
            np.fill_diagonal(d, np.inf)
            count += int((d < self.budget.separation * 0.999).sum() // 2)
        return count

    @property
    def distance_travelled(self) -> np.ndarray:
        """Path length per device -- longer than the straight line if it detours."""
        return np.linalg.norm(np.diff(self.path, axis=0), axis=2).sum(axis=0)

    @property
    def energy(self) -> float:
        """Joules for the whole manoeuvre, charged per axis at platform rates."""
        deltas = np.abs(np.diff(self.path, axis=0))
        xy = np.linalg.norm(deltas[:, :, :2], axis=2).sum()
        z = deltas[:, :, 2].sum()
        return float(xy * self.budget.xy_cost_per_m + z * self.budget.z_cost_per_m)

    def summary(self) -> str:
        return (
            f"{self.method:12s} waves={self.waves:3d} steps={self.steps:4d}  "
            f"makespan {self.makespan:8.1f} s  broken {self.broken_link_fraction:6.1%}  "
            f"collisions {self.collisions:4d}  energy {self.energy:8.2f} J"
        )


class Planner:
    """The ``planner`` plugin slot."""

    def waves_for(self, graph: TaskGraph, protect: np.ndarray) -> np.ndarray:
        raise NotImplementedError  # pragma: no cover


@register("planner", "direct", "interpolate")
class DirectPlanner(Planner):
    """Everything moves at once, straight to its target.

    Shortest paths and the fastest makespan.  Kept as the comparator because it
    is what any reasonable person writes first, and because the bench needs a
    number for what it costs in dropped links.
    """

    def waves_for(self, graph: TaskGraph, protect: np.ndarray) -> np.ndarray:
        return np.zeros(graph.n_tasks, dtype=np.int64)


@register("planner", "waves", "coloured")
class WavePlanner(Planner):
    """Move in colour classes so every protected link keeps one end still.

    The pass.  Costs a factor of the wave count in makespan and buys back the
    links that would otherwise drop for the whole manoeuvre.
    """

    def waves_for(self, graph: TaskGraph, protect: np.ndarray) -> np.ndarray:
        return wave_colouring(graph, protect)


@register("planner", "sequential")
class SequentialPlanner(Planner):
    """One device at a time.  Nothing ever breaks; nothing ever finishes.

    The other extreme, and the reason wave colouring is worth the trouble: it
    achieves the same link safety in a number of waves set by degree rather than
    by device count.
    """

    def waves_for(self, graph: TaskGraph, protect: np.ndarray) -> np.ndarray:
        return np.arange(graph.n_tasks, dtype=np.int64)


def _leg_duration(delta: np.ndarray, budget: MotionBudget) -> float:
    """Seconds for the slowest device in a wave to cover its displacement."""
    if delta.size == 0:
        return 0.0
    lateral = np.linalg.norm(delta[:, :2], axis=1)
    vertical = np.abs(delta[:, 2])
    t_xy = lateral / budget.xy_speed if budget.xy_speed > 0 else np.zeros_like(lateral)
    t_z = vertical / budget.z_speed if budget.z_speed > 0 else np.zeros_like(vertical)
    # The axes run concurrently, so a leg takes as long as its slower axis.
    return float(np.maximum(t_xy, t_z).max(initial=0.0))


def plan_motion(
    start: np.ndarray,
    target: np.ndarray,
    graph: TaskGraph,
    budget: MotionBudget,
    model: Planner | str = "waves",
    protect_fraction: float = 0.5,
    steps_per_wave: int = 8,
) -> MotionPlan:
    """Fly from ``start`` to ``target``, one wave at a time."""
    if isinstance(model, str):
        from ..registry import create

        name, model = model, create("planner", model)
    else:
        name = type(model).plugin_name

    start = np.asarray(start, dtype=float)
    target = np.asarray(target, dtype=float)
    if start.shape != target.shape:
        raise ValueError(f"start {start.shape} and target {target.shape} disagree")

    protect = critical_edges(graph, protect_fraction)
    colour = np.asarray(model.waves_for(graph, protect), dtype=np.int64)
    order = sorted(set(colour.tolist()))

    frames = [start.copy()]
    durations: list[float] = []
    current = start.copy()

    for wave in order:
        movers = colour == wave
        delta = target[movers] - current[movers]
        leg = _leg_duration(delta, budget)
        for s in range(1, steps_per_wave + 1):
            frame = current.copy()
            frame[movers] = current[movers] + delta * (s / steps_per_wave)
            frames.append(frame)
            durations.append(leg / steps_per_wave)
        current = frames[-1].copy()

    return MotionPlan(
        path=np.array(frames),
        graph=graph,
        budget=budget,
        protect=protect,
        method=name,
        waves=len(order),
        step_seconds=np.array(durations),
    )

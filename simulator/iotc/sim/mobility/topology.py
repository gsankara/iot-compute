"""Topology dynamics: what the swarm is trying to do with its geometry.

This is the second pluggable axis.  Kinematics answers "how does this body
move"; topology dynamics answers "where should it be going", and the two are
deliberately independent -- the same flocking law runs on drones and on floats,
and the float simply takes longer to get there.

The interesting plugin is :class:`GravityDynamics`.  It is the attract/repel law
from ``cgkar_gravity.py`` lifted out of an abstract embedding space and into the
physical one, so that "these vertices are mutually adjacent" is expressed by the
devices actually flying together.  Two things change when you do that, and both
are modelled here: bodies have volume and cannot occupy the same point, and they
cannot teleport, so convergence is bounded by terminal velocity rather than by
iteration count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ...geom import fibonacci_sphere
from ...registry import register


@dataclass
class TopologyContext:
    """Everything a topology-dynamics plugin is allowed to look at."""

    positions: np.ndarray  # (N, 3)
    velocities: np.ndarray  # (N, 3)
    time: float = 0.0
    #: Per-device goal supplied by the compiler, or NaN where there is none.
    goals: np.ndarray | None = None
    #: Symmetric non-negative affinity between devices; drives cohesion.
    affinity: np.ndarray | None = None
    #: Pairs that currently carry an optical link and should not be broken.
    links: list[tuple[int, int]] = field(default_factory=list)
    #: Devices that are dead and should be treated as obstacles, not partners.
    alive: np.ndarray | None = None
    radii: np.ndarray | None = None
    bounds: tuple[np.ndarray, np.ndarray] | None = None
    rng: object | None = None

    @property
    def n(self) -> int:
        return len(self.positions)


def _clip_to_bounds(targets: np.ndarray, ctx: TopologyContext) -> np.ndarray:
    if ctx.bounds is None:
        return targets
    lo, hi = ctx.bounds
    return np.clip(targets, lo, hi)


def _separation_push(positions: np.ndarray, radii: np.ndarray, margin: float = 1.5) -> np.ndarray:
    """Pairwise push that keeps physical bodies from overlapping.

    Every dynamics below adds this, because an embedding algorithm is free to
    put two points on top of each other and a swarm is not.

    Exactly coincident bodies are the case that has to be handled rather than
    divided by: a placement pass that assigns two tasks the same coordinates is
    not a rare accident, it is what happens whenever an embedding degenerates.
    Those pairs are separated along a deterministic direction, so the recovery
    is reproducible instead of depending on floating-point noise.
    """
    n = len(positions)
    if n < 2:
        return np.zeros_like(positions)

    delta = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(dist, np.inf)

    # How badly each pair overlaps -- computed from the *true* distance, so a
    # coincident pair gets the full separation, not none of it.
    min_sep = margin * (radii[:, None] + radii[None, :])
    overlap = np.maximum(min_sep - dist, 0.0)

    eps = 1e-9
    safe = np.isfinite(dist) & (dist > eps)
    direction = np.divide(delta, np.where(safe, dist, 1.0)[:, :, None],
                          out=np.zeros_like(delta), where=safe[:, :, None])

    # Coincident pairs have no direction of their own, so give them a
    # deterministic one rather than dividing by zero or leaving them stacked.
    coincident = (~safe) & np.isfinite(dist)
    if coincident.any():
        tie = fibonacci_sphere(n)
        tie_delta = tie[:, None, :] - tie[None, :, :]
        tie_norm = np.linalg.norm(tie_delta, axis=2)
        tie_dir = np.divide(tie_delta, np.where(tie_norm > eps, tie_norm, 1.0)[:, :, None],
                            out=np.zeros_like(tie_delta),
                            where=(tie_norm > eps)[:, :, None])
        direction = np.where(coincident[:, :, None], tie_dir, direction)

    return np.sum(direction * overlap[:, :, None], axis=1)


class TopologyDynamics:
    """Base class: map the current world state to a desired position per device."""

    def reset(self) -> None:
        """Clear any internal state between runs."""

    def targets(self, ctx: TopologyContext) -> np.ndarray:
        raise NotImplementedError

    def converged(self, ctx: TopologyContext, tol: float = 0.05) -> bool:
        """Whether every device is within ``tol`` of where it wants to be."""
        t = self.targets(ctx)
        return bool(np.all(np.linalg.norm(t - ctx.positions, axis=1) <= tol))


# --------------------------------------------------------------------------
@register("topology", "static", "hold")
class StaticTopology(TopologyDynamics):
    """Hold station wherever the devices already are."""

    def targets(self, ctx: TopologyContext) -> np.ndarray:
        return ctx.positions.copy()


# --------------------------------------------------------------------------
@register("topology", "gradient", "goal", "compiled")
class GradientTopology(TopologyDynamics):
    """Fly to the coordinates the compiler chose.

    The normal operating mode once a program has been compiled: placement has
    already decided where every task belongs, and the swarm's job is to realize
    that geometry.  Devices with no goal hold station and get pushed apart.
    """

    def __init__(self, separation_margin: float = 1.5) -> None:
        self.separation_margin = float(separation_margin)

    def targets(self, ctx: TopologyContext) -> np.ndarray:
        out = ctx.positions.copy()
        if ctx.goals is not None:
            valid = np.isfinite(ctx.goals).all(axis=1)
            out[valid] = ctx.goals[valid]
        if ctx.radii is not None:
            out = out + _separation_push(out, ctx.radii, self.separation_margin)
        return _clip_to_bounds(out, ctx)


# --------------------------------------------------------------------------
@register("topology", "flock", "boids")
class FlockTopology(TopologyDynamics):
    """Reynolds flocking with an added link-preservation term.

    Plain boids will happily let two nodes drift past the range at which their
    beam closes.  Here every currently established link contributes an
    attractive term that grows once the pair passes ``link_range``, so the group
    can translate as a unit *without dropping its interconnect* -- which is the
    behaviour the brief asks for and the reason this is not stock flocking.
    """

    def __init__(
        self,
        cohesion: float = 0.35,
        separation: float = 1.2,
        alignment: float = 0.25,
        link_keep: float = 1.6,
        link_range: float = 20.0,
        neighbour_radius: float = 12.0,
        horizon: float = 1.0,
        drift: tuple[float, float, float] = (0.0, 0.0, 0.0),
        separation_margin: float = 2.0,
    ) -> None:
        self.cohesion = float(cohesion)
        self.separation = float(separation)
        self.alignment = float(alignment)
        self.link_keep = float(link_keep)
        self.link_range = float(link_range)
        self.neighbour_radius = float(neighbour_radius)
        self.horizon = float(horizon)
        self.drift = np.asarray(drift, dtype=float)
        self.separation_margin = float(separation_margin)

    def targets(self, ctx: TopologyContext) -> np.ndarray:
        p, v = ctx.positions, ctx.velocities
        n = len(p)
        if n == 0:
            return p.copy()

        delta = p[:, None, :] - p[None, :, :]
        dist = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(dist, np.inf)
        near = dist < self.neighbour_radius

        steer = np.zeros_like(p)

        # Cohesion: move toward the centroid of visible neighbours.
        counts = near.sum(axis=1)
        has = counts > 0
        if has.any():
            centroid = (near[:, :, None] * p[None, :, :]).sum(axis=1)
            centroid[has] /= counts[has, None]
            steer[has] += self.cohesion * (centroid[has] - p[has])

        # Alignment: match neighbour velocity.
        if has.any():
            mean_v = (near[:, :, None] * v[None, :, :]).sum(axis=1)
            mean_v[has] /= counts[has, None]
            steer[has] += self.alignment * (mean_v[has] - v[has])

        # Separation: strongest at short range.
        radii = ctx.radii if ctx.radii is not None else np.full(n, 0.05)
        steer += self.separation * _separation_push(p, radii, self.separation_margin)

        # Link preservation: hold on to the beams we already have.
        for i, j in ctx.links:
            d = float(dist[i, j])
            if not math.isfinite(d) or d <= 0.0:
                continue
            if d > self.link_range:
                pull = self.link_keep * (d - self.link_range) / self.link_range
                direction = (p[j] - p[i]) / d
                steer[i] += pull * direction
                steer[j] -= pull * direction

        out = p + (steer + self.drift) * self.horizon
        return _clip_to_bounds(out, ctx)


# --------------------------------------------------------------------------
@register("topology", "gravity", "cgkar", "attract_repel")
class GravityTopology(TopologyDynamics):
    """The CGKAR attract/repel law, physically realized.

    Each device is drawn toward the affinity-weighted centroid of its partners,
    exactly as in ``cgkar_gravity.attract_step``::

        target[i] = p[i] + lr * (sum_j W[i,j] p[j] / sum_j W[i,j] - p[i])

    with two additions the abstract algorithm does not need.  Bodies are pushed
    apart so they cannot coincide, and :meth:`repel` is available so that a
    failed clique check can throw the offending devices out of the cluster and
    decay their affinity -- the same move as the original, but flown.
    """

    def __init__(self, learning_rate: float = 0.3, separation_margin: float = 2.5,
                 decay: float = 0.5, repel_strength: float = 0.5) -> None:
        self.learning_rate = float(learning_rate)
        self.separation_margin = float(separation_margin)
        self.decay = float(decay)
        self.repel_strength = float(repel_strength)
        self._repel: dict[int, np.ndarray] = {}

    def reset(self) -> None:
        self._repel.clear()

    def repel(self, indices, centre: np.ndarray, affinity: np.ndarray | None = None,
              members=()) -> None:
        """Push ``indices`` away from ``centre`` and decay their affinity.

        Called by the clique workload when an all-to-all verification round
        shows a candidate is not in fact a clique.
        """
        centre = np.asarray(centre, dtype=float)
        for idx in indices:
            self._repel[int(idx)] = centre.copy()
        if affinity is not None:
            for r in indices:
                for c in members:
                    affinity[r, c] *= self.decay
                    affinity[c, r] *= self.decay

    def targets(self, ctx: TopologyContext) -> np.ndarray:
        p = ctx.positions
        n = len(p)
        if n == 0:
            return p.copy()

        w = ctx.affinity
        if w is None:
            return p.copy()
        w = np.asarray(w, dtype=float)
        if ctx.alive is not None:
            dead = ~ctx.alive
            w = w.copy()
            w[dead, :] = 0.0
            w[:, dead] = 0.0

        w_sum = w.sum(axis=1, keepdims=True)
        attached = (w_sum > 0).ravel()
        centre = np.divide((w @ p), np.where(w_sum > 0, w_sum, 1.0))
        # A device with no affinity to anyone has no centroid to move toward, so
        # it holds station.  Letting it fall back to the origin would silently
        # drag every isolated vertex into the middle of the swarm.
        out = p.copy()
        out[attached] = p[attached] + self.learning_rate * (centre[attached] - p[attached])

        # Devices ejected by a failed clique check keep moving outward.
        for idx, from_point in self._repel.items():
            if idx >= n:
                continue
            direction = p[idx] - from_point
            mag = float(np.linalg.norm(direction))
            if mag < 1e-9:
                direction = np.array([1.0, 0.0, 0.0])
                mag = 1.0
            out[idx] = p[idx] + self.repel_strength * direction / mag

        radii = ctx.radii if ctx.radii is not None else np.full(n, 0.05)
        out = out + _separation_push(out, radii, self.separation_margin)
        return _clip_to_bounds(out, ctx)


# --------------------------------------------------------------------------
@register("topology", "annealed", "annealing")
class AnnealedTopology(TopologyDynamics):
    """Compiler goals plus decaying jitter.

    Straight gradient descent onto a placement can wedge in a local minimum
    where two devices each block the other's route.  A cooling noise term shakes
    those loose early and then gets out of the way.
    """

    def __init__(self, temperature: float = 2.0, cooling_time: float = 30.0,
                 separation_margin: float = 1.5) -> None:
        self.temperature = float(temperature)
        self.cooling_time = float(cooling_time)
        self.base = GradientTopology(separation_margin=separation_margin)

    def targets(self, ctx: TopologyContext) -> np.ndarray:
        out = self.base.targets(ctx)
        if ctx.rng is None or self.temperature <= 0.0:
            return out
        t = self.temperature * math.exp(-ctx.time / max(self.cooling_time, 1e-9))
        if t < 1e-6:
            return out
        noise = ctx.rng.normal(0.0, t, size=out.shape)
        return _clip_to_bounds(out + noise, ctx)


# --------------------------------------------------------------------------
@register("topology", "formation_seq", "waypoints", "script")
class FormationSequence(TopologyDynamics):
    """Play a scripted list of formations, one after another.

    ``waypoints`` is a list of ``(time, positions)`` pairs; between them the
    targets are linearly interpolated, which gives a repeatable manoeuvre for
    testing whether the swarm can hold its links while translating.
    """

    def __init__(self, waypoints=None, loop: bool = False) -> None:
        self.waypoints = [(float(t), np.asarray(p, dtype=float)) for t, p in (waypoints or [])]
        self.waypoints.sort(key=lambda item: item[0])
        self.loop = bool(loop)

    def targets(self, ctx: TopologyContext) -> np.ndarray:
        if not self.waypoints:
            return ctx.positions.copy()
        t = ctx.time
        span = self.waypoints[-1][0]
        if self.loop and span > 0:
            t = math.fmod(t, span)
        if t <= self.waypoints[0][0]:
            return self.waypoints[0][1].copy()
        for (t0, p0), (t1, p1) in zip(self.waypoints, self.waypoints[1:]):
            if t0 <= t <= t1:
                u = (t - t0) / max(t1 - t0, 1e-12)
                return _clip_to_bounds((1 - u) * p0 + u * p1, ctx)
        return self.waypoints[-1][1].copy()

"""Geometry: directions, quantized pointing, and line of sight.

Coordinates are a right-handed world frame in metres, ``+z`` up.  A pointing
direction is an (azimuth, elevation) pair in radians:

* **azimuth** is measured in the xy-plane from ``+x`` toward ``+y``, and lives on
  the full circle ``[-pi, pi)``;
* **elevation** is the angle out of the xy-plane toward ``+z``, in
  ``[-pi/2, +pi/2]``.

The devices can aim anywhere on the sphere but only in **quantized steps**, and
different device classes have different step sizes and different sweep limits.
That is what :class:`AngularGrid` models, and it is the reason a compiled aim
angle is never exactly the geometrically ideal one -- the leftover
:meth:`AngularGrid.snap` error feeds straight into the pointing-loss term of the
link budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------
# vectors
# --------------------------------------------------------------------------
def vec3(x: float, y: float, z: float) -> np.ndarray:
    """A single world-frame point/direction."""
    return np.array([x, y, z], dtype=float)


def norm(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """Euclidean length, broadcasting over the leading axes."""
    return np.sqrt(np.sum(np.asarray(v, dtype=float) ** 2, axis=axis))


def unit(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """Normalise to unit length; zero-length vectors are returned unchanged."""
    v = np.asarray(v, dtype=float)
    n = np.sqrt(np.sum(v**2, axis=axis, keepdims=True))
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0)


def wrap_pi(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap an angle (or array) into ``[-pi, pi)``."""
    return (np.asarray(angle, dtype=float) + math.pi) % TWO_PI - math.pi


def spherical_from_vec(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Direction vector(s) -> ``(azimuth, elevation)`` in radians."""
    v = np.asarray(v, dtype=float)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    az = np.arctan2(y, x)
    horiz = np.hypot(x, y)
    el = np.arctan2(z, horiz)
    return az, el


def vec_from_spherical(az: np.ndarray | float, el: np.ndarray | float) -> np.ndarray:
    """``(azimuth, elevation)`` -> unit direction vector(s)."""
    az = np.asarray(az, dtype=float)
    el = np.asarray(el, dtype=float)
    c = np.cos(el)
    return np.stack([c * np.cos(az), c * np.sin(az), np.sin(el)], axis=-1)


def angle_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Angle in radians between two (sets of) vectors, numerically safe."""
    a, b = unit(u), unit(v)
    dot = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
    return np.arccos(dot)


def angular_separation(az1, el1, az2, el2) -> np.ndarray:
    """Great-circle angle between two pointing directions [rad]."""
    return angle_between(vec_from_spherical(az1, el1), vec_from_spherical(az2, el2))


# --------------------------------------------------------------------------
# quantized pointing
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AngularGrid:
    """A transceiver's steering resolution and reach.

    Parameters
    ----------
    step_az, step_el
        Quantization step [rad].  A step of 0 means continuous steering.
    az_limits, el_limits
        Inclusive ``(lo, hi)`` sweep limits [rad], or ``None`` for unrestricted.
        Azimuth limits are interpreted on the wrapped circle, so
        ``(-pi/2, +pi/2)`` is a forward-facing 180 degree sector.
    origin_az, origin_el
        Where the grid's zero tick sits, so two device classes with the same
        step size need not share the same lattice.
    """

    step_az: float = 1.0 * math.pi / 180.0
    step_el: float = 1.0 * math.pi / 180.0
    az_limits: tuple[float, float] | None = None
    el_limits: tuple[float, float] | None = (-math.pi / 2, math.pi / 2)
    origin_az: float = 0.0
    origin_el: float = 0.0

    # -- reach ------------------------------------------------------------
    def in_range(self, az: float, el: float) -> bool:
        """Can this gimbal point at ``(az, el)`` at all?"""
        if self.el_limits is not None:
            lo, hi = self.el_limits
            if not (lo - 1e-12 <= el <= hi + 1e-12):
                return False
        if self.az_limits is not None:
            lo, hi = self.az_limits
            if (hi - lo) < TWO_PI - 1e-9:
                # Offset both the target and the window so the arc is contiguous.
                off = wrap_pi(az - lo)
                if not (-1e-12 <= off <= (hi - lo) + 1e-12):
                    return False
        return True

    def clamp(self, az: float, el: float) -> tuple[float, float]:
        """Nearest reachable direction, clamping into the sweep limits."""
        if self.el_limits is not None:
            el = min(max(el, self.el_limits[0]), self.el_limits[1])
        if self.az_limits is not None:
            lo, hi = self.az_limits
            if (hi - lo) < TWO_PI - 1e-9 and not self.in_range(az, el):
                # Snap to whichever endpoint is angularly closer.
                d_lo = abs(wrap_pi(az - lo))
                d_hi = abs(wrap_pi(az - hi))
                az = lo if d_lo <= d_hi else hi
        return float(wrap_pi(az)), float(el)

    # -- quantization ------------------------------------------------------
    def _q(self, value: float, step: float, origin: float) -> float:
        if step <= 0.0:
            return value
        return origin + round((value - origin) / step) * step

    def snap(self, az: float, el: float) -> tuple[float, float, float]:
        """Quantize a desired direction.

        Returns ``(az_q, el_q, residual)`` where ``residual`` is the true
        great-circle angle [rad] between the desired and achieved directions.
        This is the pointing error the link budget has to live with, and it is
        *not* simply half a step: near the poles an azimuth tick subtends less
        angle, so the residual shrinks.
        """
        az_c, el_c = self.clamp(az, el)
        el_q = self._q(el_c, self.step_el, self.origin_el)
        if self.el_limits is not None:
            el_q = min(max(el_q, self.el_limits[0]), self.el_limits[1])
        az_q = float(wrap_pi(self._q(az_c, self.step_az, self.origin_az)))
        if self.az_limits is not None and not self.in_range(az_q, el_q):
            az_q, el_q = self.clamp(az_q, el_q)
        residual = float(angular_separation(az, el, az_q, el_q))
        return az_q, el_q, residual

    def steps_between(self, az0: float, el0: float, az1: float, el1: float) -> int:
        """Number of quantization ticks the gimbal must traverse.

        Used for both actuation energy (J per step) and retarget latency; the
        two axes are assumed to slew concurrently, so the cost is the larger of
        the two travel distances.
        """
        n_az = abs(wrap_pi(az1 - az0)) / self.step_az if self.step_az > 0 else 0.0
        n_el = abs(el1 - el0) / self.step_el if self.step_el > 0 else 0.0
        return int(round(max(n_az, n_el)))

    def max_residual(self) -> float:
        """Worst-case pointing error [rad] this grid can leave behind."""
        return 0.5 * math.hypot(self.step_az, self.step_el)


# --------------------------------------------------------------------------
# line of sight
# --------------------------------------------------------------------------
def segment_sphere_blocked(
    p0: np.ndarray,
    p1: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    """Does the segment ``p0 -> p1`` clip any of the given spheres?

    Vectorised over the spheres.  Endpoints sitting inside their own body do not
    count -- callers exclude the two endpoint bodies themselves.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    centers = np.atleast_2d(np.asarray(centers, dtype=float))
    radii = np.atleast_1d(np.asarray(radii, dtype=float))
    if centers.size == 0:
        return np.zeros(0, dtype=bool)

    d = p1 - p0
    seg_len2 = float(np.dot(d, d))
    if seg_len2 <= 0.0:
        return np.zeros(len(centers), dtype=bool)

    # Parameter of the closest approach, clamped to the segment.
    t = np.clip(((centers - p0) @ d) / seg_len2, 0.0, 1.0)
    closest = p0 + t[:, None] * d
    dist2 = np.sum((centers - closest) ** 2, axis=1)
    return dist2 <= radii**2


def segment_segment_distance(
    p0: np.ndarray,
    p1: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    """Closest distance from one segment to each of many segments.

    The standard clamped-parameter solution (Ericson, *Real-Time Collision
    Detection*), vectorised over the second set.  Needed because a tether is a
    line in space, not a point, and a beam that passes near one is blocked.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    starts = np.atleast_2d(np.asarray(starts, dtype=float))
    ends = np.atleast_2d(np.asarray(ends, dtype=float))
    if starts.size == 0:
        return np.zeros(0)

    d1 = p1 - p0                      # (3,)
    d2 = ends - starts                # (m, 3)
    r = p0 - starts                   # (m, 3)

    a = float(np.dot(d1, d1))
    e = np.sum(d2 * d2, axis=1)
    f = np.sum(d2 * r, axis=1)
    c = r @ d1
    b = d2 @ d1

    eps = 1e-15
    s = np.zeros(len(starts))
    t = np.zeros(len(starts))

    degenerate_1 = bool(a <= eps)
    degenerate_2 = e <= eps        # (m,) array

    denom = a * e - b * b
    ok = (not degenerate_1) & (~degenerate_2)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(ok & (denom > eps),
                     np.clip((b * f - c * e) / np.where(denom > eps, denom, 1.0), 0.0, 1.0),
                     0.0)
        t = np.where(degenerate_2, 0.0, (b * s + f) / np.where(e > eps, e, 1.0))

        low = t < 0.0
        t = np.where(low, 0.0, t)
        if not degenerate_1:
            s = np.where(low, np.clip(-c / max(a, eps), 0.0, 1.0), s)

        high = t > 1.0
        t = np.where(high, 1.0, t)
        if not degenerate_1:
            s = np.where(high, np.clip((b - c) / max(a, eps), 0.0, 1.0), s)

    closest_1 = p0[None, :] + s[:, None] * d1[None, :]
    closest_2 = starts + t[:, None] * d2
    return np.linalg.norm(closest_1 - closest_2, axis=1)


def segment_capsule_blocked(
    p0: np.ndarray,
    p1: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    """Does the segment ``p0 -> p1`` clip any of the given capsules?"""
    if np.asarray(starts).size == 0:
        return np.zeros(0, dtype=bool)
    dist = segment_segment_distance(p0, p1, starts, ends)
    return dist <= np.atleast_1d(np.asarray(radii, dtype=float))


def has_line_of_sight(
    p0: np.ndarray,
    p1: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    exclude: tuple[int, ...] = (),
    capsules: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    exclude_capsules: tuple[int, ...] = (),
) -> bool:
    """True when nothing occludes the path between two points.

    ``capsules`` is ``(starts, ends, radii)`` and models the tethers of a
    remote-head deployment.  They matter: a dense field of vertical fibres is
    exactly the kind of obstacle that a placement pass optimising only for range
    will happily route a beam straight through.
    """
    centers = np.atleast_2d(np.asarray(centers, dtype=float))
    if centers.size:
        mask = np.ones(len(centers), dtype=bool)
        for idx in exclude:
            if 0 <= idx < len(centers):
                mask[idx] = False
        if mask.any():
            blocked = segment_sphere_blocked(p0, p1, centers[mask], np.asarray(radii)[mask])
            if bool(blocked.any()):
                return False

    if capsules is not None:
        starts, ends, cap_radii = capsules
        starts = np.atleast_2d(np.asarray(starts, dtype=float))
        if starts.size:
            ends = np.atleast_2d(np.asarray(ends, dtype=float))
            cap_radii = np.atleast_1d(np.asarray(cap_radii, dtype=float))
            keep = np.ones(len(starts), dtype=bool)
            for idx in exclude_capsules:
                if 0 <= idx < len(starts):
                    keep[idx] = False
            if keep.any():
                hit = segment_capsule_blocked(
                    p0, p1, starts[keep], ends[keep], cap_radii[keep]
                )
                if bool(hit.any()):
                    return False
    return True


# --------------------------------------------------------------------------
# formations
# --------------------------------------------------------------------------
def fibonacci_sphere(n: int, radius: float = 1.0) -> np.ndarray:
    """``n`` roughly equidistant points on a sphere -- a good default cluster."""
    if n <= 0:
        return np.zeros((0, 3))
    i = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = math.pi * (1.0 + 5.0**0.5) * i
    return radius * np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=-1
    )


def grid_lattice(n: int, spacing: float, shape: tuple[int, int, int] | None = None) -> np.ndarray:
    """``n`` points on a compact 3D lattice with the given spacing."""
    if n <= 0:
        return np.zeros((0, 3))
    if shape is None:
        side = int(math.ceil(n ** (1.0 / 3.0)))
        shape = (side, side, int(math.ceil(n / (side * side))))
    xs, ys, zs = (np.arange(s, dtype=float) for s in shape)
    pts = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    pts = pts[:n] * spacing
    return pts - pts.mean(axis=0)

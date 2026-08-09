"""Geometry, quantized pointing, and line of sight."""

from __future__ import annotations

import math

import numpy as np
import pytest

from iotc.geom import (
    AngularGrid,
    angular_separation,
    fibonacci_sphere,
    grid_lattice,
    has_line_of_sight,
    spherical_from_vec,
    unit,
    vec_from_spherical,
    wrap_pi,
)
from iotc.units import deg


def test_spherical_round_trip():
    rng = np.random.default_rng(0)
    v = unit(rng.normal(size=(200, 3)))
    az, el = spherical_from_vec(v)
    back = vec_from_spherical(az, el)
    assert np.allclose(back, v, atol=1e-12)


def test_wrap_pi_range():
    x = np.linspace(-20, 20, 1001)
    w = wrap_pi(x)
    assert np.all(w >= -math.pi - 1e-12)
    assert np.all(w < math.pi + 1e-12)
    # Wrapping must not change the direction it points to.
    assert np.allclose(np.cos(x), np.cos(w))
    assert np.allclose(np.sin(x), np.sin(w))


@pytest.mark.parametrize("step_deg", [0.02, 0.1, 0.5, 2.0])
def test_quantization_residual_is_bounded_by_half_a_cell(step_deg):
    """No snap may cost more than half the diagonal of one grid cell.

    This bound is what the link budget's pointing-loss floor is derived from,
    so if it ever fails the loss floor is wrong too.
    """
    grid = AngularGrid(step_az=step_deg * deg, step_el=step_deg * deg,
                       el_limits=(-90 * deg, 90 * deg))
    bound = grid.max_residual() + 1e-12
    rng = np.random.default_rng(1)
    for _ in range(500):
        az = rng.uniform(-math.pi, math.pi)
        el = math.asin(rng.uniform(-1.0, 1.0))
        _, _, residual = grid.snap(az, el)
        assert residual <= bound, (step_deg, az, el, residual, bound)


def test_finer_grid_never_points_worse():
    coarse = AngularGrid(step_az=0.5 * deg, step_el=0.5 * deg)
    fine = AngularGrid(step_az=0.02 * deg, step_el=0.02 * deg)
    rng = np.random.default_rng(2)
    worse = 0
    for _ in range(300):
        az = rng.uniform(-math.pi, math.pi)
        el = math.asin(rng.uniform(-0.99, 0.99))
        if fine.snap(az, el)[2] > coarse.snap(az, el)[2] + 1e-12:
            worse += 1
    # A finer lattice can lose on an individual sample only by luck; on average
    # it must be far better, and its worst case must be much smaller.
    assert worse < 30
    assert fine.max_residual() < coarse.max_residual() / 10


def test_sweep_limits_reject_and_clamp():
    grid = AngularGrid(step_az=0.1 * deg, step_el=0.1 * deg,
                       az_limits=(-45 * deg, 45 * deg),
                       el_limits=(-30 * deg, 30 * deg))
    assert grid.in_range(0.0, 0.0)
    assert not grid.in_range(120 * deg, 0.0)
    assert not grid.in_range(0.0, 60 * deg)

    az_q, el_q, residual = grid.snap(120 * deg, 60 * deg)
    assert abs(el_q) <= 30 * deg + 1e-9
    assert grid.in_range(az_q, el_q)
    # Being clamped is expensive, and the caller must be able to see that.
    assert residual > 30 * deg


def test_steps_between_counts_ticks():
    grid = AngularGrid(step_az=1.0 * deg, step_el=1.0 * deg)
    assert grid.steps_between(0.0, 0.0, 10 * deg, 0.0) == 10
    assert grid.steps_between(0.0, 0.0, 0.0, 7 * deg) == 7
    # Axes slew together, so the cost is the longer travel, not the sum.
    assert grid.steps_between(0.0, 0.0, 10 * deg, 7 * deg) == 10


def test_continuous_grid_has_no_residual():
    grid = AngularGrid(step_az=0.0, step_el=0.0)
    az_q, el_q, residual = grid.snap(0.7, 0.3)
    assert residual == pytest.approx(0.0, abs=1e-12)
    assert az_q == pytest.approx(0.7)
    assert el_q == pytest.approx(0.3)


def test_line_of_sight_blocked_by_a_body_in_between():
    centers = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    radii = np.array([0.03, 0.03, 0.5])
    assert not has_line_of_sight(centers[0], centers[1], centers, radii, exclude=(0, 1))
    # Move the obstacle aside and the path opens.
    centers[2] = [5.0, 3.0, 0.0]
    assert has_line_of_sight(centers[0], centers[1], centers, radii, exclude=(0, 1))


def test_line_of_sight_ignores_the_endpoints_themselves():
    centers = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    radii = np.array([0.4, 0.4])  # the bodies overlap the segment at both ends
    assert has_line_of_sight(centers[0], centers[1], centers, radii, exclude=(0, 1))


def test_angular_separation_matches_dot_product():
    assert angular_separation(0.0, 0.0, math.pi / 2, 0.0) == pytest.approx(math.pi / 2)
    assert angular_separation(0.0, 0.0, 0.0, math.pi / 2) == pytest.approx(math.pi / 2)
    assert angular_separation(1.0, 0.4, 1.0, 0.4) == pytest.approx(0.0, abs=1e-12)


def test_formations_are_well_formed():
    pts = fibonacci_sphere(64, radius=3.0)
    assert pts.shape == (64, 3)
    assert np.allclose(np.linalg.norm(pts, axis=1), 3.0)

    lattice = grid_lattice(30, 2.0)
    assert lattice.shape == (30, 3)
    assert np.allclose(lattice.mean(axis=0), 0.0, atol=1e-9)
    # No two devices at the same point.
    d = np.linalg.norm(lattice[:, None, :] - lattice[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    assert d.min() > 1.0

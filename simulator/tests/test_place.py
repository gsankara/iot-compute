"""B3 Place: the physics the projection has to respect."""

from __future__ import annotations

import math

import numpy as np
import pytest

from iotc.compiler import build_workload
from iotc.compiler.partition import partition
from iotc.compiler.place import (
    MotionBudget,
    derive_budget,
    enforce_separation,
    fit_to_reach,
    place,
    principal_axes,
    terminal_velocity,
)
from iotc.devices import MOTE_C, aerial, aquatic, tethered
from iotc.registry import available, create

WATER = create("medium", "water")
AIR = create("medium", "air")


def _devices():
    g = build_workload("stencil3d", nx=128, ny=128, nz=128, tx=8, ty=8, tz=8)
    return partition(g, MOTE_C, seed=0).coarsened()


# ------------------------------------------------------------------- budgets

def test_a_float_can_go_far_deeper_than_it_can_go_sideways():
    """Phase A's inverted finding, now a number the compiler acts on.

    Ballast sets a trim once and the body coasts to depth for free, so depth is
    bounded by time.  Thrusters burn charge every second, so lateral travel is
    bounded by the battery.  The gap is orders of magnitude, and a placer that
    treats the axes alike is spending the wrong one.
    """
    b = derive_budget(aquatic(MOTE_C), WATER)
    assert b.platform == "buoyant"
    assert b.z_reach > 50 * b.xy_reach
    assert b.z_cost_per_m == 0.0        # coasting to depth is free
    assert b.xy_cost_per_m > 0.0


def test_a_drone_pays_for_both_axes():
    b = derive_budget(aerial(MOTE_C), AIR)
    assert b.platform == "aerial"
    assert b.xy_cost_per_m > 0.0 and b.z_cost_per_m > 0.0
    # Vertical travel fights gravity, so the anisotropy does not run the
    # float's way -- the same placer must not assume it does.
    assert b.anisotropy < 10.0


def test_a_tethered_head_moves_for_free_and_is_bounded_by_its_wire():
    head = tethered(aquatic(MOTE_C))
    b = derive_budget(head, WATER)
    assert b.platform == "tethered"
    assert b.xy_cost_per_m == 0.0 and b.z_cost_per_m == 0.0
    assert b.z_reach <= head.tether.length + 1e-9


def test_a_tethered_head_sees_as_far_as_a_float():
    # The head carries the float's optics, so once it is converted for the
    # medium its reach is the float's.  A bare ``MOTE_C`` on 850 nm reports
    # ~1.75 m, which is the wavelength's limit masquerading as the platform's.
    head = derive_budget(tethered(aquatic(MOTE_C)), WATER)
    floater = derive_budget(aquatic(MOTE_C), WATER)
    assert head.max_link == pytest.approx(floater.max_link, rel=1e-9)
    assert derive_budget(tethered(MOTE_C), WATER).max_link < 0.1 * floater.max_link


def test_terminal_velocity_grows_with_force_and_falls_with_density():
    body = MOTE_C.body
    assert terminal_velocity(body, WATER, 1.0) > terminal_velocity(body, WATER, 0.25)
    assert terminal_velocity(body, AIR, 1.0) > terminal_velocity(body, WATER, 1.0)


def test_optical_reach_comes_from_the_link_budget():
    """Not a constant: the same part reaches much further in air than in water."""
    assert derive_budget(aerial(MOTE_C), AIR).max_link > derive_budget(
        aquatic(MOTE_C), WATER
    ).max_link


# ----------------------------------------------------------------- machinery

def test_principal_axes_finds_the_spread():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(200, 6)) * np.array([10.0, 5.0, 1.0, 0.01, 0.01, 0.01])
    axes = principal_axes(pts)
    assert axes.shape == (200, 3)
    spreads = axes.std(axis=0)
    assert spreads[0] > spreads[1] > spreads[2]


def test_principal_axes_pads_a_low_dimensional_embedding():
    assert principal_axes(np.zeros((10, 2))).shape == (10, 3)
    assert principal_axes(np.zeros((0, 4))).shape == (0, 3)


def test_separation_pushes_overlapping_bodies_apart():
    p = np.zeros((5, 3))          # every device on top of every other
    out = enforce_separation(p, 1.0)
    d = np.linalg.norm(out[:, None, :] - out[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    assert np.all(np.isfinite(out))
    assert d.min() > 0.0


def test_separation_produces_no_nan_from_self_distances():
    """The self-distance is infinite and its direction is zero; inf*0 is NaN."""
    out = enforce_separation(np.zeros((4, 3)), 0.5)
    assert np.all(np.isfinite(out))


def test_fit_to_reach_shrinks_only_when_it_must():
    g = build_workload("stencil3d", tx=2, ty=2, tz=2)
    p = np.random.default_rng(0).normal(size=(g.n_tasks, 3))
    assert np.allclose(fit_to_reach(p, g, 1e9), p)          # already fits
    shrunk = fit_to_reach(p, g, 0.1)
    e = g.edges
    assert np.linalg.norm(shrunk[e[:, 0]] - shrunk[e[:, 1]], axis=1).max() <= 0.1


def test_fit_to_reach_preserves_shape():
    """Scaling is uniform, so the anisotropy the projection chose survives."""
    g = build_workload("stencil3d", tx=2, ty=2, tz=2)
    p = np.random.default_rng(1).normal(size=(g.n_tasks, 3)) * [1.0, 1.0, 20.0]
    out = fit_to_reach(p, g, 0.5)
    before = (p.max(axis=0) - p.min(axis=0))
    after = (out.max(axis=0) - out.min(axis=0))
    assert np.allclose(after / after[0], before / before[0], rtol=1e-9)


# ------------------------------------------------------- the pass's own metric

def test_energy_placement_beats_the_velocity_reading_on_a_float():
    """The correction Phase A produced, measured end to end.

    Scaling by speed compresses depth because ballast is slow, which spends the
    rationed lateral axis to save the free vertical one.  Scaling by cost does
    the opposite and the formation is cheaper to fly.
    """
    devices = _devices()
    energy = place(devices, aquatic(MOTE_C), WATER, model="anisotropic", seed=0)
    naive = place(devices, aquatic(MOTE_C), WATER, model="velocity", seed=0)
    assert energy.motion_energy < naive.motion_energy / 2
    # And it does it by using the depth it was given.
    assert energy.extent[2] > 10 * naive.extent[2]


@pytest.mark.parametrize("model", sorted(available("placer")))
def test_every_placement_keeps_its_links_closable(model):
    """A formation the beams cannot span is not a placement."""
    devices = _devices()
    r = place(devices, aquatic(MOTE_C), WATER, model=model, seed=0)
    assert r.unreachable == 0, f"{model} left {r.unreachable} links beyond reach"


@pytest.mark.parametrize("model", sorted(available("placer")))
def test_every_placement_keeps_bodies_apart(model):
    devices = _devices()
    r = place(devices, aquatic(MOTE_C), WATER, model=model, seed=0)
    assert r.min_separation >= r.budget.separation * 0.99


@pytest.mark.parametrize("model", sorted(available("placer")))
def test_placement_is_deterministic(model):
    devices = _devices()
    a = place(devices, aquatic(MOTE_C), WATER, model=model, seed=0)
    b = place(devices, aquatic(MOTE_C), WATER, model=model, seed=0)
    assert np.allclose(a.positions, b.positions)


def test_positions_are_finite_and_shaped():
    devices = _devices()
    r = place(devices, aquatic(MOTE_C), WATER, seed=0)
    assert r.positions.shape == (devices.n_tasks, 3)
    assert np.all(np.isfinite(r.positions))


def test_weighted_link_length_favours_the_heavy_edges():
    """A chatty pair placed far away should hurt more than a quiet one."""
    from iotc.compiler.ir import TaskGraph

    g = TaskGraph.from_arrays([1.0, 1.0, 1.0], [1.0, 1.0, 1.0],
                              [0, 0], [1, 2], [1000.0, 1.0])
    budget = MotionBudget(xy_reach=10, z_reach=10, max_link=1e9, separation=0.0)
    near = np.array([[0.0, 0, 0], [1.0, 0, 0], [100.0, 0, 0]])
    far = np.array([[0.0, 0, 0], [100.0, 0, 0], [1.0, 0, 0]])
    from iotc.compiler.place import PlaceResult

    assert (PlaceResult(near, g, budget).weighted_link_length
            < PlaceResult(far, g, budget).weighted_link_length)


def test_the_placers_are_registered():
    assert {"anisotropic", "velocity", "isotropic"} <= set(available("placer"))


def test_a_two_device_graph_still_gets_three_axes():
    """SVD returns min(n, kappa) components; physical space always has three.

    A workload that coarsens onto two devices produced a two-column projection
    and crashed the placer on the z axis.
    """
    from iotc.compiler.ir import TaskGraph

    tiny = TaskGraph.from_arrays([1.0, 1.0], [1.0, 1.0], [0], [1], [512.0])
    assert principal_axes(np.zeros((2, 8))).shape == (2, 3)
    r = place(tiny, aquatic(MOTE_C), WATER, seed=0)
    assert r.positions.shape == (2, 3)
    assert np.all(np.isfinite(r.positions))

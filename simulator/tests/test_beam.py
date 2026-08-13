"""B5 Beam assignment: what a device can physically point at, and when."""

from __future__ import annotations

import math

import numpy as np
import pytest

from iotc.compiler import build_workload
from iotc.compiler.beam import (
    aimability,
    assign_beams,
    bearings,
    pointing_loss_db,
    refine_positions,
)
from iotc.compiler.ir import TaskGraph
from iotc.compiler.partition import partition
from iotc.compiler.place import place
from iotc.devices import MOTE_C, RELAY, aerial, aquatic
from iotc.registry import available, create

WATER = create("medium", "water")
AIR = create("medium", "air")


def _scenario(cls, medium):
    g = build_workload("stencil3d", nx=128, ny=128, nz=128, tx=8, ty=8, tz=8)
    devices = partition(g, MOTE_C, seed=0).coarsened()
    return devices, place(devices, cls, medium, seed=0).positions


# ---------------------------------------------------------------- geometry

def test_bearings_point_from_first_endpoint_to_second():
    p = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    az, el = bearings(p, np.array([[0, 1]]))
    assert el[0] == pytest.approx(0.0, abs=1e-12)


def test_pointing_loss_is_zero_on_axis_and_grows_with_error():
    assert pointing_loss_db(np.array([0.0]), 5e-3)[0] == pytest.approx(0.0)
    loss = pointing_loss_db(np.array([1e-3, 2e-3, 3e-3]), 5e-3)
    assert loss[0] < loss[1] < loss[2]


def test_a_narrower_beam_is_less_forgiving():
    """The relay's tighter divergence buys reach and costs pointing tolerance."""
    residual = np.array([1e-3])
    assert pointing_loss_db(residual, 3e-3)[0] > pointing_loss_db(residual, 5e-3)[0]


# ------------------------------------------------------------ the b-matching

def test_no_device_is_asked_for_more_beams_than_it_owns():
    """The constraint that makes this a b-matching rather than a topology."""
    cls = aquatic(MOTE_C)
    devices, pos = _scenario(cls, WATER)
    assign = assign_beams(devices, refine_positions(devices, pos, cls), cls)
    assert assign.feasible
    for r in range(assign.rounds):
        assert assign.degree_in_round(r).max() <= cls.optical.n_transceivers


def test_every_aimable_link_gets_exactly_one_round():
    cls = aquatic(MOTE_C)
    devices, pos = _scenario(cls, WATER)
    pos = refine_positions(devices, pos, cls)
    assign = assign_beams(devices, pos, cls)
    ok, _ = aimability(devices, pos, cls)
    assert np.array_equal(assign.round_of >= 0, ok)


def test_more_transceivers_need_fewer_rounds():
    """A relay carries six heads; it should collapse the round count."""
    cls = aquatic(MOTE_C)
    devices, pos = _scenario(cls, WATER)
    pos = refine_positions(devices, pos, cls)
    relay = aquatic(RELAY)
    assert relay.optical.n_transceivers > cls.optical.n_transceivers
    assert assign_beams(devices, pos, relay).rounds <= assign_beams(devices, pos, cls).rounds


def test_heaviest_first_front_loads_the_traffic():
    """Rounds cost retargets, so the opening round should carry the bytes."""
    cls = aquatic(MOTE_C)
    devices, pos = _scenario(cls, WATER)
    pos = refine_positions(devices, pos, cls)
    heavy = assign_beams(devices, pos, cls, model="heaviest_first")
    light = assign_beams(devices, pos, cls, model="lightest_first")
    assert heavy.first_round_traffic > light.first_round_traffic


@pytest.mark.parametrize("model", sorted(available("assigner")))
def test_assignment_is_deterministic(model):
    cls = aquatic(MOTE_C)
    devices, pos = _scenario(cls, WATER)
    a = assign_beams(devices, pos, cls, model=model)
    b = assign_beams(devices, pos, cls, model=model)
    assert np.array_equal(a.round_of, b.round_of)


def test_links_outside_the_sweep_cone_are_rejected_not_degraded():
    """A bearing beyond the elevation limit is impossible, not merely lossy."""
    cls = aquatic(MOTE_C)
    g = TaskGraph.from_arrays([1.0, 1.0], [1.0, 1.0], [0], [1], [1024.0])
    straight_up = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
    assert cls.optical.gimbal.grid.el_limits[1] < math.pi / 2
    assert assign_beams(g, straight_up, cls).rejected_fraction == 1.0

    sideways = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    assert assign_beams(g, sideways, cls).rejected_fraction == 0.0


# ------------------------------------------------------------- the B3 feedback

def test_the_energy_optimal_placement_is_almost_entirely_unaimable():
    """The cross-pass finding this whole feedback loop exists for.

    B3 puts a float's formation in a tall thin column, because depth is its free
    axis.  Nearly every bearing in a column points straight up or down, outside
    the gimbal's elevation limits -- so the placement that minimises motion
    energy cannot hold its links at all.
    """
    cls = aquatic(MOTE_C)
    devices, pos = _scenario(cls, WATER)
    extent = np.ptp(pos, axis=0)
    assert extent[2] > 50 * max(extent[0], extent[1])     # it really is a column
    assert assign_beams(devices, pos, cls).rejected_fraction > 0.9


def test_refinement_recovers_the_formation():
    cls = aquatic(MOTE_C)
    devices, pos = _scenario(cls, WATER)
    before = assign_beams(devices, pos, cls)
    after = assign_beams(devices, refine_positions(devices, pos, cls), cls)
    assert after.rejected_fraction < 0.1
    assert after.rejected_fraction < before.rejected_fraction / 5
    assert after.first_round_traffic > before.first_round_traffic


def test_refinement_widens_the_column_rather_than_nudging_it():
    cls = aquatic(MOTE_C)
    devices, pos = _scenario(cls, WATER)
    refined = refine_positions(devices, pos, cls)
    assert np.ptp(refined, axis=0)[:2].max() > 10 * np.ptp(pos, axis=0)[:2].max()


def test_refinement_helps_an_already_aimable_formation_too():
    """On a drone the placement is not a column, so only the residual is left."""
    cls = aerial(MOTE_C)
    devices, pos = _scenario(cls, AIR)
    before = assign_beams(devices, pos, cls)
    after = assign_beams(devices, refine_positions(devices, pos, cls), cls)
    assert after.mean_loss_db <= before.mean_loss_db


def test_refinement_is_deterministic():
    cls = aquatic(MOTE_C)
    devices, pos = _scenario(cls, WATER)
    assert np.allclose(refine_positions(devices, pos, cls),
                       refine_positions(devices, pos, cls))


def test_an_edgeless_graph_refines_to_itself():
    g = TaskGraph.from_arrays([1.0], [1.0], [], [], [])
    p = np.zeros((1, 3))
    assert np.allclose(refine_positions(g, p, aquatic(MOTE_C)), p)


def test_the_assigners_are_registered():
    assert {"heaviest_first", "lightest_first", "arbitrary"} <= set(available("assigner"))

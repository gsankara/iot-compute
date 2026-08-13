"""B4 Motion: what a manoeuvre must not break on the way."""

from __future__ import annotations

import numpy as np
import pytest

from iotc.compiler import build_workload
from iotc.compiler.ir import TaskGraph
from iotc.compiler.motion import (
    critical_edges,
    plan_motion,
    wave_colouring,
)
from iotc.compiler.partition import partition
from iotc.compiler.place import derive_budget, place
from iotc.devices import MOTE_C, aquatic
from iotc.registry import available, create

WATER = create("medium", "water")


def _scenario():
    g = build_workload("stencil3d", nx=128, ny=128, nz=128, tx=8, ty=8, tz=8)
    devices = partition(g, MOTE_C, seed=0).coarsened()
    cls = aquatic(MOTE_C)
    budget = derive_budget(cls, WATER)
    target = place(devices, cls, WATER, seed=0).positions
    start = np.random.default_rng(0).normal(scale=2.0, size=target.shape)
    return devices, budget, start, target


# ------------------------------------------------------------------ selection

def test_critical_edges_take_the_heaviest_traffic_first():
    g = TaskGraph.from_arrays([1.0] * 4, [1.0] * 4,
                              [0, 1, 2], [1, 2, 3], [1000.0, 1.0, 1.0])
    mask = critical_edges(g, fraction=0.5)
    assert mask[0]                       # the 1000-byte edge alone clears half
    assert mask.sum() == 1


def test_protecting_more_traffic_protects_more_edges():
    g = build_workload("stencil3d", tx=3, ty=3, tz=3)
    assert critical_edges(g, 0.9).sum() >= critical_edges(g, 0.3).sum()


def test_critical_edges_handles_an_edgeless_graph():
    g = TaskGraph.from_arrays([1.0], [1.0], [], [], [])
    assert critical_edges(g).size == 0


# ------------------------------------------------------------------ colouring

def test_colouring_never_moves_both_ends_of_a_protected_link():
    """The invariant the whole pass exists to hold."""
    g = build_workload("stencil3d", tx=4, ty=4, tz=4)
    protect = critical_edges(g, 1.0)
    colour = wave_colouring(g, protect)
    e = g.edges
    for idx in np.flatnonzero(protect):
        assert colour[e[idx, 0]] != colour[e[idx, 1]]


def test_colouring_uses_few_waves_relative_to_devices():
    """Waves are a makespan multiplier, so their count is the cost.

    Greedy needs at most max_degree + 1, which for a stencil is a small
    constant however many devices there are -- unlike moving one at a time.
    """
    g = build_workload("stencil3d", tx=4, ty=4, tz=4)
    colour = wave_colouring(g, critical_edges(g, 1.0))
    assert len(set(colour.tolist())) <= g.degree().max() + 1
    assert len(set(colour.tolist())) < g.n_tasks / 4


def test_colouring_is_deterministic():
    g = build_workload("stencil3d", tx=3, ty=3, tz=3)
    protect = critical_edges(g)
    assert np.array_equal(wave_colouring(g, protect), wave_colouring(g, protect))


# ------------------------------------------------------------- the manoeuvre

@pytest.mark.parametrize("model", sorted(available("planner")))
def test_every_plan_arrives(model):
    devices, budget, start, target = _scenario()
    plan = plan_motion(start, target, devices, budget, model=model)
    assert np.allclose(plan.path[-1], target, atol=1e-9)
    assert np.allclose(plan.path[0], start)


@pytest.mark.parametrize("model", sorted(available("planner")))
def test_plans_are_deterministic(model):
    devices, budget, start, target = _scenario()
    a = plan_motion(start, target, devices, budget, model=model)
    b = plan_motion(start, target, devices, budget, model=model)
    assert np.allclose(a.path, b.path)


def test_direct_flight_drops_every_protected_link():
    """Both gimbals in motion and no tracker: the link is gone, not degraded.

    Phase A leaves the closed-loop tracker out deliberately and measured
    open-loop tolerance in centimetres, so this is a consequence of the physics
    rather than a modelling choice.
    """
    devices, budget, start, target = _scenario()
    plan = plan_motion(start, target, devices, budget, model="direct")
    assert plan.broken_link_fraction == pytest.approx(1.0)


def test_wave_flight_drops_none_of_them():
    devices, budget, start, target = _scenario()
    plan = plan_motion(start, target, devices, budget, model="waves")
    assert plan.broken_link_steps == 0


def test_link_safety_costs_makespan_and_nothing_else():
    """Waves reorder the motion; they do not lengthen the paths.

    Energy is set by distance travelled, and every device still flies straight
    to its target -- so the whole price of keeping the links up is time.
    """
    devices, budget, start, target = _scenario()
    direct = plan_motion(start, target, devices, budget, model="direct")
    waved = plan_motion(start, target, devices, budget, model="waves")
    assert waved.makespan > direct.makespan
    assert waved.energy == pytest.approx(direct.energy, rel=1e-9)
    assert np.allclose(np.sort(waved.distance_travelled),
                       np.sort(direct.distance_travelled))


def test_colouring_beats_moving_one_at_a_time():
    """Sequential is also safe -- it is just far slower, and that is the point."""
    devices, budget, start, target = _scenario()
    waved = plan_motion(start, target, devices, budget, model="waves")
    one_by_one = plan_motion(start, target, devices, budget, model="sequential")
    assert one_by_one.broken_link_steps == 0
    assert waved.waves < one_by_one.waves
    assert waved.makespan < one_by_one.makespan / 2


def test_the_journey_is_checked_not_just_the_destination():
    """Endpoints can be properly separated while the paths cross."""
    g = TaskGraph.from_arrays([1.0, 1.0], [1.0, 1.0], [0], [1], [1024.0])
    budget = derive_budget(aquatic(MOTE_C), WATER)
    start = np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    target = np.array([[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]])   # they swap places
    plan = plan_motion(start, target, g, budget, model="direct")
    assert plan.min_separation < budget.separation
    assert plan.collisions > 0


def test_makespan_comes_from_real_speeds():
    devices, budget, start, target = _scenario()
    plan = plan_motion(start, target, devices, budget, model="direct")
    assert plan.makespan > 0
    # Nothing can arrive faster than the slowest axis allows.
    furthest = plan.distance_travelled.max()
    fastest = max(budget.xy_speed, budget.z_speed)
    assert plan.makespan >= furthest / fastest * 0.5


def test_mismatched_shapes_are_rejected():
    devices, budget, start, target = _scenario()
    with pytest.raises(ValueError, match="disagree"):
        plan_motion(start[:-1], target, devices, budget)


def test_the_planners_are_registered():
    assert {"direct", "waves", "sequential"} <= set(available("planner"))

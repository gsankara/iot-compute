"""B11 -- the omnidirectional plane.

Written as invariants rather than golden numbers: a colouring must actually be
a colouring, reach must trade against reuse in the stated direction, and a plan
that delivers nothing must never outrank one that delivers.
"""

from __future__ import annotations

import numpy as np
import pytest

from iotc.compiler import build_workload
from iotc.compiler.broadcast import (CARRIER_LADDER, POWER_FRACTIONS,
                                     RadioBudget, _endpoints, colour_greedy,
                                     conflict_matrix, derive_radio_budget,
                                     domain_sizes, evaluate_setting,
                                     hearing_matrix, plan_broadcast)
from iotc.compiler.partition import partition
from iotc.compiler.place import place
from iotc.devices import MOTE_C, aerial, aquatic
from iotc.registry import available, create

WATER = create("medium", "water")
AIR = create("medium", "air")


def _lattice(side: int = 4, pitch: float = 1.0):
    """A cubic lattice of tasks whose edges are nearest-neighbour."""
    g = build_workload(
        "stencil3d", nx=8 * side, ny=8 * side, nz=8 * side, tx=side, ty=side, tz=side
    )
    idx = np.array([(i, j, k) for i in range(side) for j in range(side) for k in range(side)])
    return g, idx[: g.n_tasks].astype(float) * pitch


def _drone_case(side: int = 10):
    cls, medium = aerial(MOTE_C), AIR
    g = build_workload(
        "stencil3d", nx=16 * side, ny=16 * side, nz=16 * side, tx=side, ty=side, tz=side
    )
    devices = partition(g, MOTE_C, seed=0).coarsened()
    positions = place(devices, cls, medium, seed=0).positions
    return devices, positions, cls, medium


# ------------------------------------------------------------------ domains


def test_hearing_is_symmetric_and_excludes_self():
    p = np.random.default_rng(0).normal(size=(12, 3)) * 3.0
    heard = hearing_matrix(p, 4.0)
    assert np.array_equal(heard, heard.T)
    assert not heard.diagonal().any()


def test_domains_grow_monotonically_with_reach():
    p = np.random.default_rng(1).normal(size=(30, 3)) * 5.0
    previous = -1
    for reach in (0.5, 1.0, 2.0, 5.0, 20.0, 1e3):
        total = int(domain_sizes(hearing_matrix(p, reach)).sum())
        assert total >= previous
        previous = total
    # At unbounded reach everything hears everything else.
    assert int(domain_sizes(hearing_matrix(p, 1e9)).sum()) == 30 * 29


# ----------------------------------------------------------------- colouring


def test_colouring_is_a_colouring():
    """The property that makes the slot count mean anything."""
    g, pos = _lattice(4)
    heard = hearing_matrix(pos, 1.5)
    conflict = conflict_matrix(heard, g)
    colours = colour_greedy(conflict)
    u, w = np.nonzero(conflict)
    assert not (colours[u] == colours[w]).any(), "two conflicting nodes share a slot"


def test_colouring_is_deterministic():
    g, pos = _lattice(4)
    conflict = conflict_matrix(hearing_matrix(pos, 2.5), g)
    first = colour_greedy(conflict)
    for _ in range(3):
        assert np.array_equal(colour_greedy(conflict), first)


def test_colouring_of_an_empty_conflict_graph_needs_one_slot():
    g, pos = _lattice(3)
    # Reach below the lattice pitch: nobody hears anybody.
    conflict = conflict_matrix(hearing_matrix(pos, 0.1), g)
    assert not conflict.any()
    assert int(colour_greedy(conflict).max()) + 1 == 1


def test_half_duplex_makes_audible_transmitters_conflict():
    """Two nodes that hear each other cannot transmit together, even with
    disjoint audiences -- a node cannot receive while it is transmitting."""
    g = build_workload("allreduce", count=1 << 10, p=4, topology="tree")
    pos = np.array([[0.0, 0, 0], [1.0, 0, 0], [10.0, 0, 0], [11.0, 0, 0]])[: g.n_tasks]
    heard = hearing_matrix(pos, 1.5)
    conflict = conflict_matrix(heard, g)
    assert np.all(conflict[heard]), "an audible pair was allowed into one slot"


# -------------------------------------------------------- reach versus reuse


def test_more_reach_costs_reuse():
    """B11's central claim: capacity on a shared medium is bought by shrinking
    domains, so growing the reach must not increase the reuse factor."""
    g, pos = _lattice(4, pitch=1.0)
    slots = []
    for reach in (1.1, 2.2, 4.4, 100.0):
        plan = evaluate_setting(g, pos, RadioBudget(2.4e9, 1e6, 1e-3, reach, "t"))
        slots.append(plan.slots)
    assert slots == sorted(slots), f"slot count fell as reach grew: {slots}"
    # At reach beyond the whole lattice every node conflicts with every other,
    # so the schedule degenerates to one node per slot: pure round-robin TDMA.
    assert slots[-1] == g.n_tasks


def test_per_node_rate_is_the_line_rate_divided_by_slots():
    g, pos = _lattice(4)
    plan = evaluate_setting(g, pos, RadioBudget(2.4e9, 1e6, 1e-3, 2.0, "t"))
    assert plan.per_node_rate == pytest.approx(1e6 / plan.slots)
    assert plan.reuse == pytest.approx(g.n_tasks / plan.slots)


def test_private_payload_never_cheaper_than_shared():
    """``frame_private`` sums a node's payloads where ``frame_shared`` takes the
    max, so the private bound is the pessimistic one by construction."""
    g, pos = _lattice(4)
    plan = evaluate_setting(g, pos, RadioBudget(2.4e9, 1e6, 1e-3, 2.0, "t"))
    assert plan.frame_private >= plan.frame_shared > 0.0


# ---------------------------------------------------------------- coverage


def test_an_edge_beyond_reach_makes_the_plan_infeasible():
    g, pos = _lattice(4, pitch=1.0)
    plan = evaluate_setting(g, pos, RadioBudget(2.4e9, 1e6, 1e-3, 0.5, "t"))
    assert not plan.feasible
    assert plan.uncovered_fraction == 1.0
    assert "exceed" in plan.reason


def test_swept_never_prefers_a_plan_that_delivers_nothing():
    """Regression: ranking on per-node rate alone picks the setting whose reach
    is below the shortest edge.  Nobody hears anybody, so the conflict graph is
    empty, the colouring needs one slot, and the plan advertises the full line
    rate while carrying not one byte.  B10 hit the same shape."""
    devices, positions, cls, medium = _drone_case()
    swept = plan_broadcast(devices, positions, cls, medium, model="swept")
    assert swept.feasible, "swept chose a plan that cannot carry the graph"
    assert swept.uncovered_fraction == 0.0
    # And it must be no worse than the model that only aims for coverage.
    tight = plan_broadcast(devices, positions, cls, medium, model="min_reach")
    assert swept.per_node_rate >= tight.per_node_rate


def test_min_reach_covers_every_edge_when_the_ladder_can():
    devices, positions, cls, medium = _drone_case()
    plan = plan_broadcast(devices, positions, cls, medium, model="min_reach")
    assert plan.feasible and plan.uncovered == 0
    src, dst = _endpoints(devices)
    longest = np.linalg.norm(positions[src] - positions[dst], axis=1).max()
    assert plan.budget.reach >= longest


def test_min_reach_does_not_overshoot_the_nominal_power():
    """The point of the model is to spend less power, not more."""
    devices, positions, cls, medium = _drone_case()
    plan = plan_broadcast(devices, positions, cls, medium, model="min_reach")
    assert plan.budget.tx_power <= cls.radio.tx_power


# ------------------------------------------------------------------ physics


def test_dropping_the_carrier_buys_range_in_water():
    """Phase A's finding, re-measured through the broadcast budget: seawater's
    conductivity term is what forces a subsea deployment off the microwave
    band, and the ladder has to move rate and range together."""
    reaches = [
        derive_radio_budget(
            aquatic(MOTE_C), WATER, carrier=c, line_rate=r
        ).reach
        for c, r, _ in CARRIER_LADDER
    ]
    assert reaches == sorted(reaches), f"range did not grow as carrier fell: {reaches}"
    assert reaches[0] < 0.1, "2.4 GHz should be all but dead in water"
    assert reaches[-1] > 10.0


def test_air_reaches_far_further_than_water_on_the_same_radio():
    water = derive_radio_budget(aquatic(MOTE_C), WATER, carrier=2.4e9, line_rate=1e6)
    air = derive_radio_budget(aerial(MOTE_C), AIR, carrier=2.4e9, line_rate=1e6)
    assert air.reach > 100.0 * water.reach


def test_cutting_power_shortens_reach_monotonically():
    previous = np.inf
    for frac in POWER_FRACTIONS:
        b = derive_radio_budget(
            aerial(MOTE_C), AIR, carrier=2.4e9, line_rate=1e6, power_fraction=frac
        )
        assert b.reach <= previous + 1e-9
        previous = b.reach


# --------------------------------------------------------------- integration


def test_every_registered_model_produces_a_usable_plan():
    devices, positions, cls, medium = _drone_case()
    assert set(available("broadcaster")) == {"nominal", "min_reach", "swept"}
    for model in available("broadcaster"):
        plan = plan_broadcast(devices, positions, cls, medium, model=model)
        assert plan.devices == devices.n_tasks
        assert plan.slots >= 1
        assert 0.0 <= plan.uncovered_fraction <= 1.0
        assert plan.report()


def test_planning_is_reproducible():
    devices, positions, cls, medium = _drone_case()
    a = plan_broadcast(devices, positions, cls, medium, model="min_reach")
    b = plan_broadcast(devices, positions, cls, medium, model="min_reach")
    assert (a.slots, a.budget.tx_power, a.budget.carrier) == (
        b.slots, b.budget.tx_power, b.budget.carrier
    )
    assert np.array_equal(a.colours, b.colours)


def test_no_subsea_radio_carries_an_optically_placed_float_swarm():
    """A composition result, not a bug: B3 scales a placement until its longest
    link closes *optically* -- 41.6 m of blue-green water -- and the whole
    carrier ladder tops out well below that, so the radio cannot be the data
    plane for a float swarm however it is tuned."""
    cls, medium = aquatic(MOTE_C), WATER
    g = build_workload("stencil3d", nx=128, ny=128, nz=128, tx=8, ty=8, tz=8)
    devices = partition(g, MOTE_C, seed=0).coarsened()
    positions = place(devices, cls, medium, seed=0).positions
    src, dst = _endpoints(devices)
    longest = np.linalg.norm(positions[src] - positions[dst], axis=1).max()
    best = max(
        derive_radio_budget(cls, medium, carrier=c, line_rate=r).reach
        for c, r, _ in CARRIER_LADDER
    )
    assert longest > best, "the ladder now spans the placement -- update this claim"
    plan = plan_broadcast(devices, positions, cls, medium, model="min_reach")
    assert not plan.feasible and plan.uncovered > 0

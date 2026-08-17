"""Phase C: the invariants, the classification, and the fault campaign."""

from __future__ import annotations

import math

import numpy as np
import pytest

from iotc.compiler.codegen import DeviceProgram, Instruction, Op
from iotc.testbench import DEGRADED, HUNG, PASS, WRONG, run_once, tally
from iotc.testbench.campaign import _inject, campaign, report, verdict
from iotc.testbench.invariants import (
    INVARIANTS,
    aims_are_reachable,
    no_double_delivery,
    no_task_executed_twice,
    schedule_is_feasible,
)
from iotc.devices import MOTE_C, aquatic

SMALL = dict(nx=128, ny=128, nz=128, tx=8, ty=8, tz=8)


class _State:
    def __init__(self, sent=0, received=0):
        self.bytes_sent = sent
        self.bytes_received = received


# ---------------------------------------------------------------- invariants

def test_the_five_invariants_are_the_ones_the_roadmap_names():
    assert set(INVARIANTS) == {
        "no_double_delivery", "no_task_executed_twice", "schedule_is_feasible",
        "energy_never_negative", "aims_are_reachable",
    }


def test_unmatched_bytes_are_caught():
    """A byte sent must be a byte received: a link needs both gimbals."""
    v = no_double_delivery([], [_State(sent=100), _State(received=40)])
    assert v is not None and v.invariant == "no_double_delivery"
    assert no_double_delivery([], [_State(sent=100), _State(received=100)]) is None


def test_a_repeated_transfer_to_the_same_peer_is_caught():
    p = DeviceProgram(device=0, frame=[
        Instruction(Op.SEND, 3, 64), Instruction(Op.SEND, 3, 64),
    ])
    v = no_double_delivery([p], [_State(sent=128, received=128)])
    assert v is not None and "twice" not in v.detail  # reported as repeated pair
    assert "2 times" in v.detail


def test_computing_twice_in_a_frame_is_caught():
    twice = DeviceProgram(device=0, frame=[Instruction(Op.COMPUTE, 5),
                                           Instruction(Op.COMPUTE, 5)])
    once = DeviceProgram(device=0, frame=[Instruction(Op.COMPUTE, 5)])
    assert no_task_executed_twice([twice]) is not None
    assert no_task_executed_twice([once]) is None


def test_an_over_subscribed_round_is_caught():
    class _Assign:
        rounds = 1

        def degree_in_round(self, r):
            return np.array([5, 1, 1])

    v = schedule_is_feasible(_Assign(), aquatic(MOTE_C))
    assert v is not None and "5 beams" in v.detail


def test_an_unreachable_bearing_is_caught():
    """Straight up is on the grid and outside the sweep limits."""
    cls = aquatic(MOTE_C)
    step_el = cls.optical.gimbal.grid.step_el
    straight_up = int(round((math.pi / 2) / step_el))
    bad = DeviceProgram(device=0, frame=[Instruction(Op.AIM, 1, 0, straight_up)])
    ok = DeviceProgram(device=0, frame=[Instruction(Op.AIM, 1, 0, 0)])
    assert aims_are_reachable([bad], cls) is not None
    assert aims_are_reachable([ok], cls) is None


# ------------------------------------------------------------ classification

def test_a_clean_run_finishes_and_is_classified():
    r = run_once("stencil3d", SMALL, seed=0)
    assert r.verdict in (PASS, DEGRADED)
    assert r.violations == []
    assert r.devices > 0 and r.tasks > r.devices
    assert r.golden is not None


def test_degraded_reports_what_was_lost_rather_than_hiding_it():
    """A swarm that drops traffic and knows it is usable; one that does not is not."""
    r = run_once("stencil3d", SMALL, seed=0)
    if r.verdict is DEGRADED:
        assert r.golden.unreachable_edges > 0 or r.delivery_ratio < 1.0
        assert r.note


def test_an_impossible_workload_is_hung_not_crashed():
    """A campaign must survive a run that cannot be compiled."""
    r = run_once("stencil3d", dict(nx=8, ny=8, nz=8, tx=1, ty=1, tz=1), seed=0)
    assert r.verdict in (PASS, DEGRADED, HUNG)
    assert isinstance(r.note, str)


def test_classification_is_deterministic():
    a = run_once("stencil3d", SMALL, seed=3)
    b = run_once("stencil3d", SMALL, seed=3)
    assert a.verdict == b.verdict
    assert a.delivered_bytes == b.delivered_bytes


# ------------------------------------------------------------------- faults

def test_drift_is_a_random_walk_over_an_exposure_not_a_fixed_offset():
    """Quoting m/sqrt(s) is meaningless without saying over how long."""
    pos = np.zeros((40, 3))
    short = _inject("position_drift", pos, 1.0, 0, exposure_s=1.0)
    long = _inject("position_drift", pos, 1.0, 0, exposure_s=100.0)
    assert np.linalg.norm(long, axis=1).mean() > 5 * np.linalg.norm(short, axis=1).mean()


def test_an_unknown_fault_is_rejected():
    with pytest.raises(ValueError, match="unknown fault"):
        _inject("gremlins", np.zeros((4, 3)), 0.5, 0)


def test_a_crashed_node_leaves_optical_reach():
    """Its bearing stays legal -- what fails is that the light cannot arrive."""
    clean = run_once("stencil3d", SMALL, seed=0, faults="none")
    crashed = run_once("stencil3d", SMALL, seed=0, faults="node_crash")
    assert crashed.out_of_reach > clean.out_of_reach


def test_the_pointing_tracker_recovers_drift_but_not_distance():
    """B8's mechanism, demonstrated end to end.

    Scatter moves devices within reach but outside the beam footprint, so a
    tracker recovers it entirely.  A crash moves them beyond reach, and no
    amount of re-aiming brings the photons back.
    """
    off = run_once("stencil3d", SMALL, seed=0, faults="scatter", tracker=False)
    on = run_once("stencil3d", SMALL, seed=0, faults="scatter", tracker=True)
    assert off.lost_to_drift > 0 and on.lost_to_drift == 0
    assert on.delivery_ratio > off.delivery_ratio

    crashed_on = run_once("stencil3d", SMALL, seed=0, faults="node_crash", tracker=True)
    assert crashed_on.out_of_reach > 0          # the tracker cannot fix distance


def test_short_links_are_the_fragile_ones_under_drift():
    """Counter-intuitive, and it falls straight out of the geometry.

    The beam footprint is the divergence half-angle at range, so it *grows* with
    link length while the drift does not.  A 3 m link has centimetres of
    tolerance; a 39 m link has decimetres.
    """
    cls = aquatic(MOTE_C)
    lengths = np.array([3.0, 40.0])
    footprints = cls.optical.divergence * lengths
    assert footprints[1] > 10 * footprints[0]


# ---------------------------------------------------------------- campaign

def test_the_campaign_runs_faults_with_and_without_a_tracker():
    results = campaign(["stencil3d"], seeds=1, params={"stencil3d": SMALL})
    settings = {(r.faults, r.tracker) for r in results}
    assert ("none", False) in settings
    assert ("scatter", False) in settings and ("scatter", True) in settings


def test_the_campaign_classifies_every_run():
    results = campaign(["stencil3d"], seeds=1, params={"stencil3d": SMALL})
    counts = tally(results)
    assert sum(counts.values()) == len(results)
    assert all(r.verdict in (PASS, DEGRADED, WRONG, HUNG) for r in results)


def test_no_invariant_breaks_on_a_clean_compile():
    """The compiler should not emit a plan that violates its own rules."""
    results = campaign(["stencil3d"], seeds=2, faults=("none",),
                       params={"stencil3d": SMALL})
    assert all(not r.violations for r in results)
    assert tally(results)[WRONG] == 0


def test_the_verdict_names_the_tracker_effect():
    results = campaign(["stencil3d"], seeds=2, faults=("none", "scatter"),
                       params={"stencil3d": SMALL})
    text = verdict(results)
    assert "without a pointing tracker" in text
    assert "invariant violations" in text
    assert report(results).count("stencil3d") >= 2

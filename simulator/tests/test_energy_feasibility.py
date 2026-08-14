"""B7 Energy feasibility: does the swarm afford the schedule B6 wrote?"""

from __future__ import annotations

import math

import numpy as np
import pytest

from iotc.compiler import build_workload
from iotc.compiler.beam import assign_beams, refine_positions
from iotc.compiler.energy import assess, harvest_power
from iotc.compiler.partition import partition
from iotc.compiler.place import place
from iotc.compiler.schedule import build_schedule
from iotc.devices import MOTE_C, MOTE_M, aerial, aquatic
from iotc.registry import available, create

WATER = create("medium", "water")
AIR = create("medium", "air")


def _pipeline(cls=None, medium=None, scheduler="amortised"):
    cls = cls or aquatic(MOTE_C)
    medium = medium or WATER
    g = build_workload("stencil3d", nx=128, ny=128, nz=128, tx=8, ty=8, tz=8)
    devices = partition(g, MOTE_C, seed=0).coarsened()
    pos = refine_positions(devices, place(devices, cls, medium, seed=0).positions, cls)
    sched = build_schedule(devices, assign_beams(devices, pos, cls), pos, cls,
                           model=scheduler)
    return sched, devices, pos, cls, medium


# ---------------------------------------------------------------- accounting

def test_the_terms_account_for_the_whole_draw():
    b = assess(*_pipeline(), policy="ignore")
    assert sum(b.terms.values()) == pytest.approx(b.draw_j)
    assert b.dominant in b.terms
    assert 0.0 < b.dominant_share <= 1.0


def test_a_device_is_charged_for_its_own_task_not_the_whole_graph():
    """The coarsened graph holds one task per device.

    Summing its FLOPs charges every device with the entire swarm's work, which
    inflated compute to 93% of the draw and made it look dominant.  The budget
    describes one device, so it must scale with the worst single task.
    """
    sched, devices, pos, cls, medium = _pipeline()
    b = assess(sched, devices, pos, cls, medium, policy="ignore")
    flops = cls.compute.flops
    per_device = float(devices.flop.max()) / flops * max(sched.batch, 1)
    whole_graph = float(devices.flop.sum()) / flops * max(sched.batch, 1)
    assert devices.n_tasks > 1 and whole_graph > per_device
    assert b.compute_j == pytest.approx(cls.power.mcu_active * per_device)


def test_the_receiver_listens_through_the_aim_not_only_the_wire():
    """A receiver cannot know when the packet starts, so it pays for the slot."""
    sched, devices, pos, cls, medium = _pipeline()
    b = assess(sched, devices, pos, cls, medium, policy="ignore")
    assert b.rx_j == pytest.approx(cls.optical.rx_power * sched.period)
    assert sched.period > sched.wire_time     # it really is more than the wire


def test_harvest_depends_on_the_medium():
    """Sunlight through water is not sunlight in air."""
    p = np.zeros(3)
    assert harvest_power(aerial(MOTE_C), p, AIR) > harvest_power(aquatic(MOTE_C), p, WATER)


# -------------------------------------------------------------------- policy

def test_ignoring_energy_never_sleeps():
    b = assess(*_pipeline(), policy="ignore")
    assert b.sleep == 0.0
    assert b.duty_cycle == 1.0


def test_the_duty_cycle_policy_reaches_a_positive_balance():
    sched, devices, pos, cls, medium = _pipeline(aquatic(MOTE_M))
    naive = assess(sched, devices, pos, cls, medium, policy="ignore")
    fixed = assess(sched, devices, pos, cls, medium, policy="duty_cycle")
    assert not naive.sustainable          # precondition: it really does drain
    assert fixed.sustainable
    assert fixed.duty_cycle < 1.0
    assert fixed.sleep > 0


def test_sleeping_lowers_the_duty_cycle_without_lowering_the_draw():
    """Sleep buys time, not efficiency: the frame's work costs what it costs."""
    sched, devices, pos, cls, medium = _pipeline(aquatic(MOTE_M))
    naive = assess(sched, devices, pos, cls, medium, policy="ignore")
    fixed = assess(sched, devices, pos, cls, medium, policy="duty_cycle")
    # Only leakage grows with the padded period; the rest is unchanged.
    assert fixed.retarget_j == pytest.approx(naive.retarget_j)
    assert fixed.compute_j == pytest.approx(naive.compute_j)
    assert fixed.leak_j >= naive.leak_j


def test_a_device_with_no_harvester_cannot_be_rescued_by_sleeping():
    """Reported as infinite sleep rather than clamped to something plausible."""
    sched, devices, pos, cls, medium = _pipeline()
    dark = cls.variant("dark", harvest=cls.harvest.__class__(kind="none"))
    b = assess(sched, devices, pos, dark, medium, policy="duty_cycle")
    assert b.harvest_w == 0.0
    assert math.isinf(b.sleep)
    assert b.duty_cycle == 0.0


def test_endurance_is_finite_only_when_draining():
    sched, devices, pos, cls, medium = _pipeline(aquatic(MOTE_M))
    draining = assess(sched, devices, pos, cls, medium, policy="ignore")
    fixed = assess(sched, devices, pos, cls, medium, policy="duty_cycle")
    assert not draining.sustainable and math.isfinite(draining.endurance)
    assert draining.endurance > 0
    assert math.isinf(fixed.endurance)


def test_a_fixed_duty_cycle_rule_is_not_a_substitute_for_computing_one():
    """Half duty is either wasteful or insufficient; here it is insufficient."""
    sched, devices, pos, cls, medium = _pipeline(aquatic(MOTE_M))
    half = assess(sched, devices, pos, cls, medium, policy="half_duty")
    computed = assess(sched, devices, pos, cls, medium, policy="duty_cycle")
    assert half.duty_cycle == pytest.approx(0.5)
    assert not half.sustainable
    assert computed.sustainable and computed.duty_cycle < half.duty_cycle


# --------------------------------------------------------- the B6/B7 tension

def test_b6s_batching_buys_link_efficiency_and_costs_energy():
    """The iteration the roadmap asks for, made measurable.

    B6 batches to amortise the retarget, and every batched superstep is another
    superstep of compute inside the same frame.  So the schedule that is best
    for the link is not the one that is cheapest to power, and B7 has to be
    allowed to push back.
    """
    batched, devices, pos, cls, medium = _pipeline(aquatic(MOTE_M), scheduler="amortised")
    single = build_schedule(devices, assign_beams(devices, pos, cls), pos, cls,
                            model="single")
    assert batched.batch > single.batch
    assert batched.efficiency > single.efficiency          # B6 wins on the link

    hot = assess(batched, devices, pos, cls, medium, policy="duty_cycle")
    cool = assess(single, devices, pos, cls, medium, policy="duty_cycle")
    assert hot.compute_j > cool.compute_j                  # and pays for it
    assert hot.duty_cycle < cool.duty_cycle


# ------------------------------------------------------------------ plumbing

@pytest.mark.parametrize("policy", sorted(available("energy_policy")))
def test_assessment_is_deterministic(policy):
    args = _pipeline()
    a = assess(*args, policy=policy)
    b = assess(*args, policy=policy)
    assert a.draw_j == pytest.approx(b.draw_j)
    assert a.sleep == pytest.approx(b.sleep) or (math.isinf(a.sleep) and math.isinf(b.sleep))


def test_the_policies_are_registered():
    assert {"ignore", "duty_cycle", "half_duty"} <= set(available("energy_policy"))

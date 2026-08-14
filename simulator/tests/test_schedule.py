"""B6 Schedule: amortising the aim, and what stops it."""

from __future__ import annotations

import numpy as np
import pytest

from iotc.compiler import build_workload
from iotc.compiler.beam import assign_beams, refine_positions
from iotc.compiler.partition import partition
from iotc.compiler.place import place
from iotc.compiler.schedule import build_schedule, retarget_steps
from iotc.devices import MOTE_C, MOTE_M, aquatic
from iotc.registry import available, create

WATER = create("medium", "water")


def _pipeline(cls=None):
    cls = cls or aquatic(MOTE_C)
    g = build_workload("stencil3d", nx=128, ny=128, nz=128, tx=8, ty=8, tz=8)
    devices = partition(g, MOTE_C, seed=0).coarsened()
    pos = refine_positions(devices, place(devices, cls, WATER, seed=0).positions, cls)
    return devices, assign_beams(devices, pos, cls), pos, cls


# ------------------------------------------------------------------ structure

def test_a_slot_accounts_for_all_of_its_time():
    devices, assign, pos, cls = _pipeline()
    s = build_schedule(devices, assign, pos, cls)
    for slot in s.slots:
        assert slot.duration == pytest.approx(slot.retarget + slot.wire + slot.guard)
    assert s.period == pytest.approx(s.wire_time + s.retarget_time + s.guard_time)


def test_every_round_with_traffic_gets_a_slot():
    devices, assign, pos, cls = _pipeline()
    s = build_schedule(devices, assign, pos, cls)
    assert len(s.slots) == assign.rounds
    assert {slot.round for slot in s.slots} == set(range(assign.rounds))


def test_guard_bands_grow_with_clock_drift():
    """Dead time is sized by the worst oscillator, not assumed away."""
    devices, assign, pos, cls = _pipeline()
    steady = build_schedule(devices, assign, pos, cls, resync_interval=0.1)
    loose = build_schedule(devices, assign, pos, cls, resync_interval=10.0)
    assert loose.guard_time > steady.guard_time


def test_retarget_cost_comes_from_real_bearings():
    devices, assign, pos, cls = _pipeline()
    steps = retarget_steps(assign, devices, pos, cls)
    assert len(steps) == assign.rounds
    assert all(s >= 0 for s in steps)
    assert max(steps) > 0        # the rounds really do point somewhere different


@pytest.mark.parametrize("model", sorted(available("scheduler")))
def test_scheduling_is_deterministic(model):
    devices, assign, pos, cls = _pipeline()
    a = build_schedule(devices, assign, pos, cls, model=model)
    b = build_schedule(devices, assign, pos, cls, model=model)
    assert a.period == pytest.approx(b.period)
    assert a.batch == b.batch


# ------------------------------------------------------- the pass's own metric

def test_the_unbatched_schedule_reproduces_the_phase_a_finding():
    """Under 1% efficiency: over 99% of link time spent pointing."""
    devices, assign, pos, cls = _pipeline()
    s = build_schedule(devices, assign, pos, cls, model="single")
    assert s.batch == 1
    assert s.efficiency < 0.01


def test_batching_amortises_the_aim():
    devices, assign, pos, cls = _pipeline()
    naive = build_schedule(devices, assign, pos, cls, model="single")
    smart = build_schedule(devices, assign, pos, cls, model="amortised")
    assert smart.batch > naive.batch
    assert smart.efficiency > 10 * naive.efficiency
    assert smart.throughput > 10 * naive.throughput


def test_batching_costs_latency_and_buffer():
    """Efficiency, latency and RAM are one trade, not three knobs."""
    devices, assign, pos, cls = _pipeline()
    naive = build_schedule(devices, assign, pos, cls, model="single")
    smart = build_schedule(devices, assign, pos, cls, model="amortised")
    assert smart.latency >= naive.latency
    assert smart.buffer_bytes > naive.buffer_bytes


def test_the_batch_never_exceeds_the_buffer_it_is_given():
    devices, assign, pos, cls = _pipeline()
    for model in ("amortised", "max_batch"):
        s = build_schedule(devices, assign, pos, cls, model=model, buffer_fraction=0.25)
        assert s.buffer_bytes <= cls.compute.ram_bytes * 0.25 + 1e-6
        assert s.fits()


# ------------------------------------------------------------ the real finding

def test_ram_starves_the_amortisation_long_before_it_works():
    """B6's headline, and it is a hardware verdict rather than a schedule.

    Amortising a ~0.5 s retarget against microseconds of wire time needs a batch
    of order a million supersteps.  The device has kilobytes.  So the schedule
    improves matters by more than an order of magnitude and still cannot reach
    its target -- the fabric stays aim-dominated because the buffer is too small,
    not because the schedule is poor.
    """
    devices, assign, pos, cls = _pipeline()
    s = build_schedule(devices, assign, pos, cls, model="amortised")
    assert s.starved
    assert s.batch_wanted > 100 * s.batch
    assert s.buffer_shortfall > 100        # times the whole RAM, not a fraction
    assert s.efficiency < 0.9              # the target is not reached


def test_a_memory_rich_part_gets_closer_but_not_there():
    """More RAM buys batch, and the shortfall shrinks without closing."""
    small = build_schedule(*_pipeline(aquatic(MOTE_C))[:3], aquatic(MOTE_C),
                           model="amortised")
    big = build_schedule(*_pipeline(aquatic(MOTE_M))[:3], aquatic(MOTE_M),
                         model="amortised")
    assert MOTE_M.compute.ram_bytes > MOTE_C.compute.ram_bytes
    assert big.batch > small.batch
    assert big.efficiency > small.efficiency
    assert big.starved                     # still not enough


def test_the_schedulers_are_registered():
    assert {"single", "amortised", "max_batch"} <= set(available("scheduler"))

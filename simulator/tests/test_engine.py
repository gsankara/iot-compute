"""World, pointing, and the discrete-event engine.

Determinism is treated as a correctness property here, not a convenience: a
fault campaign that cannot replay a failure is not evidence of anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from iotc.devices import ANCHOR, MOTE_C, MOTE_M, MOTE_S, RELAY, aquatic, tethered
from iotc.geom import grid_lattice
from iotc.sim import Simulation, World
from iotc.units import deg

AQ_C = aquatic(MOTE_C)
AQ_S = aquatic(MOTE_S)
AQ_M = aquatic(MOTE_M)


def two_nodes(distance=6.0, cls=None, seed=0, **kw):
    cls = cls or AQ_C
    w = World(medium="water", topology="static", seed=seed, **kw)
    w.add_device(cls, np.array([0.0, 0.0, -5.0]))
    w.add_device(cls, np.array([distance, 0.0, -5.0]))
    return w, Simulation(w, dt=0.01)


# --------------------------------------------------------------------------
# pointing
# --------------------------------------------------------------------------
def test_a_link_needs_both_ends_aimed():
    """Optical links are mutual: one end cannot open one on its own."""
    w, sim = two_nodes()
    w.aim(0, 1)
    sim.run(1.0)
    assert not w.link_budget(0, 1).feasible
    assert "aimed" in w.link_budget(0, 1).reason

    w.aim(1, 0)
    sim.run(2.0)
    assert w.link_budget(0, 1, stochastic=False).feasible


def test_retargeting_costs_time_and_energy():
    w = World(medium="water", topology="static", seed=0)
    w.add_device(AQ_C, np.array([0.0, 0.0, -5.0]))
    w.add_device(AQ_C, np.array([4.0, 3.0, -2.0]))  # off the initial boresight
    before = w.devices[0].store.energy
    result = w.aim(0, 1)
    assert result["ok"]
    assert result["steps"] > 0
    assert result["time"] > 0.0
    assert w.devices[0].store.energy < before
    assert w.devices[0].power.totals["gimbal"] == pytest.approx(result["energy"])


def test_a_finer_gimbal_points_more_accurately_at_the_same_target():
    """This is the whole reason mote-s exists."""
    residuals = {}
    for label, cls in (("coarse", AQ_M), ("fine", AQ_S)):
        w = World(medium="water", topology="static", seed=1)
        w.add_device(cls, np.array([0.0, 0.0, -5.0]))
        w.add_device(cls, np.array([7.3, 2.9, -3.1]))  # deliberately off-lattice
        residuals[label] = w.aim(0, 1)["residual"]
    assert residuals["fine"] < residuals["coarse"]


def test_sweep_limits_can_make_a_bearing_unreachable():
    w = World(medium="water", topology="static", seed=2)
    # mote-m is limited to +-60 degrees of elevation.
    w.add_device(AQ_M, np.array([0.0, 0.0, -20.0]))
    w.add_device(AQ_M, np.array([0.5, 0.0, -1.0]))  # almost straight up
    result = w.aim(0, 1)
    assert not result["ok"]
    assert "sweep" in result["reason"]


def test_moving_apart_degrades_a_link_without_anything_failing():
    """Pointing error is recomputed from live positions, so links decay."""
    w, sim = two_nodes(distance=4.0)
    sim.establish(0, 1)
    sim.run(2.0)
    before = w.pointing_error(0, 1)
    w.devices[1].motion.position += np.array([0.0, 2.0, 0.0])
    after = w.pointing_error(0, 1)
    assert after > before


def test_transceivers_are_a_finite_resource():
    w = World(medium="water", topology="static", seed=3)
    hub = w.add_device(AQ_C, np.zeros(3))  # two heads only
    for k in range(4):
        w.add_device(AQ_C, np.array([5.0 + k, 0.0, 0.0]))
    ok = sum(1 for j in (1, 2, 3, 4) if w.aim(0, j)["ok"])
    assert ok == hub.cls.optical.n_transceivers


# --------------------------------------------------------------------------
# occlusion
# --------------------------------------------------------------------------
def test_a_body_in_the_path_blocks_the_link():
    w = World(medium="water", topology="static", seed=4)
    w.add_device(AQ_C, np.array([-5.0, 0.0, 0.0]))
    w.add_device(AQ_C, np.array([5.0, 0.0, 0.0]))
    blocker = w.add_device(ANCHOR, np.array([0.0, 0.0, 0.0]))  # moored, stays put
    sim = Simulation(w, dt=0.01)
    sim.establish(0, 1)
    sim.run(2.0)
    assert not w.link_budget(0, 1).feasible
    blocker.motion.position[:] = [0.0, 5.0, 0.0]
    assert w.link_budget(0, 1, stochastic=False).feasible


def test_a_tether_blocks_a_beam_and_hauling_it_up_clears_it():
    """The cost of the remote-head architecture, made concrete."""
    head = tethered(AQ_C)
    w = World(medium="water", topology="static", seed=5)
    w.add_device(head, np.array([-4.0, 0.0, -10.0]))
    w.add_device(head, np.array([4.0, 0.0, -10.0]))
    assert w.line_of_sight(0, 1)

    third = w.add_device(head, np.array([0.0, 0.0, -10.0]))
    assert not w.line_of_sight(0, 1)

    # Winch that head above the beam plane: the wire no longer crosses it.
    third.motion.position[:] = [0.0, 0.0, -3.0]
    assert w.line_of_sight(0, 1)


# --------------------------------------------------------------------------
# transmission
# --------------------------------------------------------------------------
def test_a_message_is_delivered_over_an_established_link():
    w, sim = two_nodes()
    sim.establish(0, 1)
    sim.run(2.0)
    result = sim.send(0, 1, 8192)
    assert result["ok"]
    sim.drain(1.0)
    inbox = sim.receive(1)
    assert len(inbox) == 1
    assert inbox[0].bits == 8192
    assert inbox[0].deliver_at > inbox[0].sent_at


def test_sending_without_aiming_fails_cleanly():
    """An unreliable channel that raises exceptions is not a model of one."""
    w, sim = two_nodes()
    result = sim.send(0, 1, 8192)
    assert not result["ok"]
    assert "aimed" in result["reason"]


def test_a_transceiver_is_busy_while_it_transmits():
    w, sim = two_nodes()
    sim.establish(0, 1)
    sim.run(2.0)
    assert sim.send(0, 1, 10**7)["ok"]
    again = sim.send(0, 1, 8192)
    assert not again["ok"]
    assert again["reason"] == "transceiver busy"


def test_rf_needs_no_aiming_at_all():
    """The radio is what bootstraps the optical plane."""
    w = World(medium="air", topology="static", seed=6)
    w.add_device(MOTE_C, np.zeros(3))
    w.add_device(MOTE_C, np.array([20.0, 0.0, 0.0]))
    sim = Simulation(w, dt=0.01)
    assert sim.send_rf(0, 1, 512)["ok"]


def test_rf_broadcast_reaches_many_receivers_with_one_send():
    """One transmission, many receivers: the cheap verification primitive."""
    w = World(medium="air", topology="static", seed=7)
    for p in grid_lattice(8, 3.0):
        w.add_device(MOTE_C, p)
    sim = Simulation(w, dt=0.01)
    results = sim.broadcast_rf(0, 512)
    assert len(results) == 7


def test_optical_carries_far_more_than_rf():
    w, sim = two_nodes()
    sim.establish(0, 1)
    sim.run(2.0)
    optical = sim.send(0, 1, 10**6)
    rf = sim.send_rf(0, 1, 10**6)
    assert optical["ok"]
    if rf["ok"]:
        assert optical["duration"] < rf["duration"] / 100


# --------------------------------------------------------------------------
# accounting and determinism
# --------------------------------------------------------------------------
def test_summary_reports_the_optical_share_and_retarget_overhead():
    w, sim = two_nodes()
    sim.establish(0, 1)
    sim.run(2.0)
    for _ in range(20):
        sim.send(0, 1, 8192)
        sim.run(w.time + 0.05)
    s = sim.summary()
    assert 0.0 <= s["optical_share"] <= 1.0
    assert 0.0 <= s["retarget_overhead"] <= 1.0
    assert s["delivered"] > 0


def run_scenario(seed):
    w = World(medium="water", topology="gradient", seed=seed)
    for k, p in enumerate(grid_lattice(9, 5.0)):
        w.add_device([AQ_C, AQ_S, AQ_M][k % 3], p)
    w.goals = grid_lattice(9, 4.0)
    sim = Simulation(w, dt=0.05)
    sim.establish(0, 1)
    sim.run(3.0)
    for _ in range(30):
        sim.send(0, 1, 4096)
        sim.run(w.time + 0.1)
    sim.drain(1.0)
    return sim


def test_two_runs_with_the_same_seed_are_identical():
    a, b = run_scenario(11), run_scenario(11)
    assert a.summary() == b.summary()
    assert np.allclose(a.world.positions, b.world.positions, atol=0.0, rtol=0.0)
    assert [e.kind for e in a.world.trace.events] == [e.kind for e in b.world.trace.events]


def test_a_different_seed_gives_a_different_trace():
    a, c = run_scenario(11), run_scenario(12)
    assert a.summary() != c.summary() or a.world.trace.counters != c.world.trace.counters


def test_adding_a_subsystem_does_not_disturb_another_stream():
    """Named substreams keep one source of randomness from shifting another."""
    from iotc.rng import RandomSource

    plain = RandomSource(3)
    a = plain.stream("packet/0/1").random(20)

    busy = RandomSource(3)
    busy.stream("turbulence").random(1000)  # consume a lot elsewhere
    b = busy.stream("packet/0/1").random(20)
    assert np.array_equal(a, b)

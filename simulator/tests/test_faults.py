"""Fault injection.

Each model is checked for the *specific* behaviour it claims, not just that
something broke.  A stuck gimbal is not a dead laser; a failed link is not a
failed node; a transient stall is not a crash.  Conflating them would hide
exactly the recovery cases the resilience pass has to get right.
"""

from __future__ import annotations

import numpy as np
import pytest

from iotc.devices import ANCHOR, MOTE_C, aquatic
from iotc.geom import grid_lattice
from iotc.registry import create
from iotc.sim import Simulation, World, build_campaign

AQ = aquatic(MOTE_C)


def swarm(n=6, spacing=5.0, seed=0, medium="water", faults=()):
    """A ring of devices at one depth.

    Laid out laterally on purpose: a vertical stack puts every bearing straight
    up, which mote-c's +-75 degree elevation limit correctly refuses, and the
    test would then be measuring the sweep limit rather than the fault.
    """
    w = World(medium=medium, topology="static", seed=seed)
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    radius = spacing / (2 * np.sin(np.pi / n)) if n > 2 else spacing
    for a in angles:
        w.add_device(AQ, np.array([radius * np.cos(a), radius * np.sin(a), -5.0]))
    w.faults = list(faults)
    return w, Simulation(w, dt=0.05)


# --------------------------------------------------------------------------
# node faults
# --------------------------------------------------------------------------
def test_node_crash_is_permanent_and_drops_its_links():
    w, sim = swarm()
    sim.establish(0, 1)
    sim.run(2.0)
    assert w.link_budget(0, 1, stochastic=False).feasible

    w.faults = [create("fault", "node_crash", rate=1e6, targets=[0])]
    sim.run(3.0)
    assert w.devices[0].crashed
    assert not w.devices[0].alive
    assert all(t.target is None for t in w.devices[0].transceivers)

    # Permanent: no amount of further simulation brings it back.
    w.faults = []
    sim.run(20.0)
    assert not w.devices[0].alive


def test_node_stall_is_transient_and_the_node_returns():
    """The nasty case for a time-triggered schedule: it wakes up behind."""
    w, sim = swarm()
    w.faults = [create("fault", "node_stall", rate=1e6, duration=1.0, targets=[2],
                       max_events=1)]
    sim.run(0.5)
    assert not w.devices[2].operational(w.time)
    assert w.devices[2].alive          # stalled, not dead
    sim.run(3.0)
    assert w.devices[2].operational(w.time)


def test_memory_bitflip_corrupts_state_silently():
    """Nothing detects this unless the compiled program checksums."""
    w, sim = swarm()
    w.devices[3].state["tile"] = np.ones(64)
    w.faults = [create("fault", "memory_bitflip", rate=1e6, targets=[3], max_events=1)]
    sim.run(0.5)
    assert not np.array_equal(w.devices[3].state["tile"], np.ones(64))
    assert w.devices[3].alive          # it has no idea anything is wrong


def test_harvest_dropout_starves_and_then_restores():
    w, sim = swarm()
    dev = w.devices[1]
    baseline = dev.harvester.power(dev.cls.harvest, dev.position, w.medium, 0.0)
    assert baseline > 0.0

    w.faults = [create("fault", "harvest_dropout", rate=1e6, duration=2.0,
                       depth=0.0, targets=[1], max_events=1)]
    sim.run(0.5)
    assert dev.harvester.power(dev.cls.harvest, dev.position, w.medium, w.time) == 0.0
    sim.run(4.0)
    assert dev.harvester.power(dev.cls.harvest, dev.position, w.medium, w.time) > 0.0


def test_position_drift_degrades_the_formation():
    """An unmodelled current is how a good placement quietly stops working."""
    w, sim = swarm()
    start = w.positions.copy()
    w.faults = [create("fault", "position_drift", sigma=0.5)]
    sim.run(20.0)
    assert np.linalg.norm(w.positions - start) > 0.5


# --------------------------------------------------------------------------
# link faults
# --------------------------------------------------------------------------
def test_stuck_gimbal_keeps_firing_at_where_the_peer_used_to_be():
    """Worse than a dead laser: the sender believes it is still transmitting."""
    w, sim = swarm()
    sim.establish(0, 1)
    sim.run(2.0)
    tx = w.devices[0].transceiver_for(1)
    frozen_az, frozen_el = tx.az, tx.el

    w.faults = [create("fault", "gimbal_stuck", rate=1e6, targets=[0],
                       transceiver=tx.index, max_events=1)]
    sim.run(2.5)
    assert tx.stuck

    # Move the peer; the head cannot follow.
    w.devices[1].motion.position += np.array([0.0, 4.0, 0.0])
    assert not w.aim(0, 1)["ok"]
    assert (tx.az, tx.el) == (frozen_az, frozen_el)
    assert w.devices[0].alive          # the node itself is fine


def test_laser_degradation_erodes_margin_without_dropping_the_link():
    w, sim = swarm()
    sim.establish(0, 1)
    sim.run(2.0)
    before = w.link_budget(0, 1, stochastic=False).snr_db

    w.faults = [create("fault", "laser_degradation", half_life=5.0, targets=[0])]
    sim.run(20.0)
    after = w.link_budget(0, 1, stochastic=False).snr_db
    assert after < before - 3.0
    assert w.devices[0].laser_derate < 0.5


def test_link_loss_kills_one_pair_and_leaves_the_nodes_healthy():
    """The right response is a reroute, not a replacement."""
    w, sim = swarm()
    sim.establish(0, 1)
    sim.run(2.0)
    assert w.link_budget(0, 1, stochastic=False).feasible

    w.faults = [create("fault", "link_loss", pairs=[(0, 1)])]
    sim.run(2.5)
    budget = w.link_budget(0, 1, stochastic=False)
    assert not budget.feasible
    assert "permanently failed" in budget.reason
    assert w.devices[0].alive and w.devices[1].alive

    # A different peer still works.
    sim.establish(0, 2)
    sim.run(4.0)
    assert w.link_budget(0, 2, stochastic=False).feasible


def test_channel_noise_raises_the_loss_floor():
    w, sim = swarm()
    sim.establish(0, 1)
    sim.run(2.0)
    w.faults = [create("fault", "channel_noise", loss=0.5)]
    sim.run(2.1)
    assert w.extra_packet_loss == pytest.approx(0.5)

    lost = 0
    for _ in range(60):
        result = sim.send(0, 1, 4096)
        if result.get("ok") and result["lost"]:
            lost += 1
        sim.run(w.time + 0.05)
    assert lost > 10


def test_bursty_noise_clusters_its_losses():
    """Correlated loss is far harder on a protocol than the same average."""
    w, sim = swarm()
    fault = create("fault", "channel_noise", loss=0.9, burst=0.5)
    w.faults = [fault]
    states = []
    for _ in range(400):
        sim.run(w.time + 0.05)
        states.append(w.extra_packet_loss > 0.0)
    runs = sum(1 for a, b in zip(states, states[1:]) if a != b)
    assert any(states) and not all(states)
    # Far fewer transitions than samples means the losses arrive in bursts.
    assert runs < len(states) / 4


# --------------------------------------------------------------------------
# campaigns
# --------------------------------------------------------------------------
def test_a_campaign_is_declarative():
    models = build_campaign([
        {"type": "node_crash", "rate": 1e-3},
        {"type": "channel_noise", "loss": 0.01},
    ])
    assert len(models) == 2
    assert models[0].rate == pytest.approx(1e-3)


def test_onset_delays_a_fault_until_the_swarm_is_settled():
    w, sim = swarm()
    w.faults = [create("fault", "node_crash", rate=1e6, onset=5.0, targets=[0])]
    sim.run(2.0)
    assert w.devices[0].alive
    sim.run(6.0)
    assert not w.devices[0].alive


def test_max_events_caps_the_damage():
    w, sim = swarm(n=8)
    w.faults = [create("fault", "node_crash", rate=1e6, max_events=2)]
    sim.run(5.0)
    assert sum(d.crashed for d in w.devices) == 2


def test_faults_are_reproducible_under_a_seed():
    def run(seed):
        w, sim = swarm(n=8, seed=seed)
        w.faults = [create("fault", "node_crash", rate=0.05),
                    create("fault", "node_stall", rate=0.2, duration=0.5)]
        sim.run(30.0)
        return [d.crashed for d in w.devices], sorted(
            (e.time, e.kind, e.device) for e in w.trace.events if e.kind.startswith("fault")
        )

    assert run(7) == run(7)
    assert run(7) != run(8)

"""Kinematics and topology dynamics.

The three platforms differ in ways the compiler has to know about, so these
tests pin down the differences rather than just checking that things move.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from iotc.devices import MOTE_C, aerial, aquatic, tethered
from iotc.registry import create
from iotc.sim import Simulation, World
from iotc.sim.mobility import MotionState, TopologyContext
from iotc.units import G_EARTH


def settle(world, sim, device, tol=0.05, limit=600.0, step=1.0):
    """Run until the device reaches its goal; return the time it took."""
    t = 0.0
    goal = world.goals[device.id]
    while t < limit and np.linalg.norm(device.position - goal) > tol:
        t += step
        sim.run(t)
    return t


# --------------------------------------------------------------------------
# buoyancy
# --------------------------------------------------------------------------
def test_neutral_ballast_makes_the_body_hover():
    water = create("medium", "water")
    kin = create("kinematics", "buoyant")
    body = MOTE_C.body
    neutral = kin.neutral_ballast(water, body)
    assert 0.0 < neutral < body.ballast_capacity

    state = MotionState(position=np.array([0.0, 0.0, -5.0]))
    state.ballast = state.ballast_target = neutral
    for _ in range(2000):
        kin.step(state, np.array([0.0, 0.0, -5.0]), 0.01, water, body, MOTE_C.power)
    assert state.position[2] == pytest.approx(-5.0, abs=1e-3)


def test_terminal_velocity_matches_the_force_balance():
    """Free descent must settle where drag cancels the ballast force."""
    water = create("medium", "water")
    kin = create("kinematics", "buoyant")
    body = MOTE_C.body
    neutral = kin.neutral_ballast(water, body)

    state = MotionState(position=np.array([0.0, 0.0, -5.0]))
    state.ballast = state.ballast_target = body.ballast_capacity  # full down
    for _ in range(20000):
        # Hold the ballast fixed: this is the open-loop terminal velocity.
        state.ballast_target = body.ballast_capacity
        kin.step(state, np.array([0.0, 0.0, -1e6]), 0.01, water, body, MOTE_C.power)

    k = 0.5 * water.density * body.drag_coefficient * body.frontal_area
    force = (body.ballast_capacity - neutral) * G_EARTH
    expected = -math.sqrt(force / k)
    assert state.velocity[2] == pytest.approx(expected, rel=0.02)


def test_buoyant_descent_converges_without_a_limit_cycle():
    """A naive PD oscillates forever against a 25 s actuator; this must not."""
    w = World(medium="water", topology="gradient", seed=0)
    dev = w.add_device(aquatic(MOTE_C), np.array([0.0, 0.0, -2.0]))
    w.goals = np.array([[0.0, 0.0, -7.0]])
    sim = Simulation(w, dt=0.02)

    depths = []
    for t in range(1, 241):
        sim.run(float(t))
        depths.append(dev.position[2])

    assert dev.position[2] == pytest.approx(-7.0, abs=0.05)
    # Once settled, the depth must stop moving rather than hunting.
    tail = np.array(depths[-60:])
    assert tail.max() - tail.min() < 0.02


def test_ballast_actuation_lag_is_reported():
    kin = create("kinematics", "buoyant")
    lag = kin.actuation_lag(MOTE_C.body)
    assert lag == pytest.approx(MOTE_C.body.ballast_capacity / MOTE_C.body.ballast_rate)
    assert lag > 20.0  # slow enough that the compiler must plan around it


# --------------------------------------------------------------------------
# platform comparison
# --------------------------------------------------------------------------
def test_z_is_slow_on_a_float_and_fast_on_a_drone_and_a_tether():
    """The whole flotation study rests on this asymmetry, so pin it down."""
    times = {}
    energies = {}
    for label, cls, medium, sign in (
        ("buoyant", aquatic(MOTE_C), "water", -1.0),
        ("drone", aerial(MOTE_C), "air", +1.0),
        ("tethered", tethered(aquatic(MOTE_C)), "water", -1.0),
    ):
        w = World(medium=medium, topology="gradient", seed=1)
        start = np.array([0.0, 0.0, sign * 2.0])
        dev = w.add_device(cls, start)
        w.goals = np.array([[0.0, 0.0, sign * 12.0]])
        sim = Simulation(w, dt=0.02)
        times[label] = settle(w, sim, dev, tol=0.06, limit=400.0)
        energies[label] = dev.power.totals["motion"]

    assert times["buoyant"] > 3 * times["tethered"]
    assert times["drone"] < times["buoyant"]
    # The topside rig does the work, so the node's own budget is untouched.
    assert energies["tethered"] == 0.0
    assert energies["buoyant"] > 0.0
    assert energies["drone"] > energies["buoyant"]


def test_only_the_aerial_platform_has_a_finite_endurance():
    """Hovering costs watts; harvesting yields milliwatts.

    A float and a tethered head hold station indefinitely.  A drone is running
    down a battery from the moment it takes off, and how long it lasts is a
    hard limit on any schedule compiled for it.
    """
    endurance = {}
    for label, cls, medium, z in (
        ("buoyant", aquatic(MOTE_C), "water", -5.0),
        ("tethered", tethered(aquatic(MOTE_C)), "water", -5.0),
        ("drone", aerial(MOTE_C), "air", 5.0),
    ):
        w = World(medium=medium, topology="gradient", seed=5)
        dev = w.add_device(cls, np.array([0.0, 0.0, z]))
        w.goals = np.array([[0.0, 0.0, z]])  # pure station keeping
        sim = Simulation(w, dt=0.1)
        sim.run(600.0)
        endurance[label] = dev.store.fraction

    assert endurance["buoyant"] > 0.5
    assert endurance["tethered"] > 0.9
    # Ten minutes of hover eats a real slice of the pack, and extrapolating the
    # burn rate puts total endurance in hours, not days.
    burned = 1.0 - endurance["drone"]
    assert burned > 0.05
    assert 600.0 / burned < 4 * 3600.0


def test_tether_confines_the_head_to_its_workspace():
    cls = tethered(aquatic(MOTE_C), length=8.0, anchor_z=0.0)
    w = World(medium="water", topology="gradient", seed=2)
    dev = w.add_device(cls, np.array([0.0, 0.0, -2.0]))
    w.goals = np.array([[0.0, 0.0, -40.0]])  # far more wire than exists
    Simulation(w, dt=0.02).run(120.0)
    assert dev.position[2] >= -8.0 - 1e-6


def test_tethered_head_is_far_smaller_than_a_float():
    """Density is the point: a smaller immersed body packs far tighter."""
    head = tethered(aquatic(MOTE_C))
    float_body = aquatic(MOTE_C)
    ratio = (float_body.body.radius / head.body.radius) ** 3
    assert ratio > 50


def test_waveguide_loss_is_charged_to_a_tethered_head():
    w = World(medium="water", seed=3)
    plain = w.add_device(aquatic(MOTE_C), np.zeros(3))
    head = w.add_device(tethered(aquatic(MOTE_C), waveguide_loss_db=3.0), np.zeros(3))
    assert head.tx_power == pytest.approx(plain.tx_power * 10 ** -0.3, rel=1e-9)
    assert head.tx_power < plain.tx_power * 0.51


def test_browned_out_drone_falls_but_a_float_just_stops():
    """The same energy fault is survivable in a liquid and fatal in the air."""
    air, water = create("medium", "air"), create("medium", "water")
    drone, float_cls = aerial(MOTE_C), aquatic(MOTE_C)

    d_state = MotionState(position=np.array([0.0, 0.0, 20.0]), frozen=True)
    create("kinematics", "drone").step(
        d_state, np.array([0.0, 0.0, 20.0]), 1.0, air, drone.body, drone.power
    )
    assert d_state.position[2] < 20.0
    assert d_state.velocity[2] < 0.0

    kin = create("kinematics", "buoyant")
    f_state = MotionState(position=np.array([0.0, 0.0, -5.0]), frozen=True)
    f_state.ballast = f_state.ballast_target = kin.neutral_ballast(water, float_cls.body)
    for _ in range(500):
        kin.step(f_state, np.array([0.0, 0.0, -5.0]), 0.02, water,
                 float_cls.body, float_cls.power)
    assert f_state.position[2] == pytest.approx(-5.0, abs=1e-2)


# --------------------------------------------------------------------------
# topology dynamics
# --------------------------------------------------------------------------
def context(positions, **kw):
    positions = np.asarray(positions, dtype=float)
    kw.setdefault("velocities", np.zeros_like(positions))
    kw.setdefault("radii", np.full(len(positions), 0.03))
    return TopologyContext(positions=positions, **kw)


def test_static_topology_holds_station():
    top = create("topology", "static")
    p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert np.allclose(top.targets(context(p)), p)


def test_gradient_topology_follows_the_compiler_goal():
    top = create("topology", "gradient")
    p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    goals = np.array([[5.0, 0.0, 0.0], [np.nan, np.nan, np.nan]])
    out = top.targets(context(p, goals=goals))
    assert out[0][0] == pytest.approx(5.0, abs=0.01)
    # A device with no goal holds where it is.
    assert out[1][0] == pytest.approx(1.0, abs=0.01)


def test_separation_keeps_bodies_from_coinciding():
    """An embedding may put two points on top of each other; a swarm may not."""
    top = create("topology", "gradient")
    p = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]])
    out = top.targets(context(p, goals=np.zeros((2, 3))))
    assert np.linalg.norm(out[0] - out[1]) > np.linalg.norm(p[0] - p[1])


def test_flock_preserves_a_stretched_link():
    """Plain boids would let a beam-carrying pair drift apart; this must not."""
    top = create("topology", "flock", link_range=5.0, neighbour_radius=1.0)
    p = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    free = top.targets(context(p))
    held = top.targets(context(p, links=[(0, 1)]))
    assert np.linalg.norm(held[0] - held[1]) < np.linalg.norm(free[0] - free[1])


def test_flock_translates_as_a_group():
    top = create("topology", "flock", drift=(1.0, 0.0, 0.0), horizon=1.0)
    p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
    out = top.targets(context(p))
    assert np.all(out[:, 0] > p[:, 0])


def test_gravity_topology_pulls_affine_devices_together():
    """The CGKAR attract step, now expressed as somewhere to fly to."""
    top = create("topology", "gravity", learning_rate=0.3, separation_margin=1.0)
    p = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    w = np.array([[0.0, 5.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    out = top.targets(context(p, affinity=w))
    assert np.linalg.norm(out[0] - out[1]) < np.linalg.norm(p[0] - p[1])
    # The unaffiliated device is not dragged in.
    assert np.allclose(out[2], p[2], atol=1e-6)


def test_gravity_repel_ejects_and_decays_affinity():
    top = create("topology", "gravity")
    p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    w = np.full((3, 3), 4.0)
    np.fill_diagonal(w, 0.0)
    top.repel([2], centre=np.array([0.5, 0.0, 0.0]), affinity=w, members=[0, 1])
    assert w[2, 0] < 4.0 and w[0, 2] < 4.0
    out = top.targets(context(p, affinity=w))
    assert np.linalg.norm(out[2] - np.array([0.5, 0.0, 0.0])) > np.linalg.norm(
        p[2] - np.array([0.5, 0.0, 0.0])
    )


def test_annealed_noise_decays():
    rng = np.random.default_rng(0)
    top = create("topology", "annealed", temperature=5.0, cooling_time=1.0)
    p = np.zeros((6, 3))
    goals = np.zeros((6, 3))
    early = top.targets(context(p, goals=goals, time=0.0, rng=rng))
    late = top.targets(context(p, goals=goals, time=20.0, rng=rng))
    assert np.std(early) > np.std(late)


def test_formation_sequence_interpolates_between_waypoints():
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[10.0, 0.0, 0.0]])
    top = create("topology", "formation_seq", waypoints=[(0.0, a), (10.0, b)])
    mid = top.targets(context(a, time=5.0))
    assert mid[0][0] == pytest.approx(5.0)
    assert top.targets(context(a, time=100.0))[0][0] == pytest.approx(10.0)

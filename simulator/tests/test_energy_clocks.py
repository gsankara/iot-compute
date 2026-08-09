"""Energy harvesting, storage, and the clock discipline that sizes guard bands."""

from __future__ import annotations

import math

import numpy as np
import pytest

from iotc.devices import MOTE_C, HarvestSpec, StoreSpec, aquatic, tethered
from iotc.registry import create
from iotc.sim import Simulation, World
from iotc.sim.clocks import Clock, ClockDomain
from iotc.sim.energy import PowerAccount
from iotc.units import ppm


# --------------------------------------------------------------------------
# harvesting
# --------------------------------------------------------------------------
def test_photovoltaic_power_falls_off_with_depth():
    """Going deep costs link budget *and* income, using the same coefficient."""
    water = create("medium", "water")
    pv = create("harvester", "photovoltaic", surface_irradiance=600.0, surface_z=0.0)
    spec = MOTE_C.harvest
    shallow = pv.power(spec, np.array([0.0, 0.0, -1.0]), water, 0.0)
    deep = pv.power(spec, np.array([0.0, 0.0, -30.0]), water, 0.0)
    assert shallow > deep
    k = water.extinction(pv.band)
    assert deep / shallow == pytest.approx(math.exp(-k * 29.0), rel=1e-9)


def test_air_barely_shadows_a_swarm_but_water_does():
    """Altitude is nearly free for harvesting; depth is not.

    Over the tens of metres a swarm actually spans, the atmosphere costs well
    under a percent while clear water costs tens of percent -- which is why the
    placement pass has to treat depth as an energy decision and altitude not.
    """
    air = create("medium", "air")
    water = create("medium", "water")
    spec = MOTE_C.harvest
    pv_air = create("harvester", "photovoltaic", surface_z=40.0)
    pv_water = create("harvester", "photovoltaic", surface_z=0.0)

    air_ratio = (pv_air.power(spec, np.array([0.0, 0.0, 40.0]), air, 0.0)
                 / pv_air.power(spec, np.array([0.0, 0.0, 0.0]), air, 0.0))
    water_ratio = (pv_water.power(spec, np.array([0.0, 0.0, -40.0]), water, 0.0)
                   / pv_water.power(spec, np.array([0.0, 0.0, 0.0]), water, 0.0))
    assert air_ratio > 0.995
    assert water_ratio < 0.05


def test_diurnal_cycle_swings_between_day_and_night():
    air = create("medium", "air")
    pv = create("harvester", "photovoltaic", diurnal_period=100.0, night_fraction=0.02)
    spec = MOTE_C.harvest
    day = pv.power(spec, np.zeros(3), air, 25.0)
    night = pv.power(spec, np.zeros(3), air, 75.0)
    assert day > 10 * night


def test_rectenna_falls_off_as_inverse_square_and_dies_in_seawater():
    air = create("medium", "air")
    sea = create("medium", "water", conductivity=4.0)
    rect = create("harvester", "rf_rectenna", chargers=((np.zeros(3), 10.0),))
    spec = MOTE_C.harvest
    near = rect.power(spec, np.array([1.0, 0.0, 0.0]), air, 0.0)
    far = rect.power(spec, np.array([2.0, 0.0, 0.0]), air, 0.0)
    assert near / far == pytest.approx(4.0, rel=1e-6)
    # Wireless charging is simply not an option under salt water.
    assert rect.power(spec, np.array([1.0, 0.0, 0.0]), sea, 0.0) < near * 1e-6


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------
def test_store_conserves_energy():
    store = create("store", "supercap", spec=StoreSpec(capacity=10.0, initial_fraction=0.5,
                                                       leakage_power=0.0))
    start = store.energy
    store.harvest(1.0, 2.0)   # +2 J
    store.draw(0.5)
    assert store.energy == pytest.approx(start + 2.0 - 0.5)
    assert store.total_harvested == pytest.approx(2.0)
    assert store.total_consumed == pytest.approx(0.5)


def test_store_never_exceeds_capacity_and_reports_waste():
    store = create("store", "supercap", spec=StoreSpec(capacity=1.0, initial_fraction=0.9,
                                                       leakage_power=0.0))
    stored = store.harvest(10.0, 1.0)
    assert store.energy == pytest.approx(1.0)
    assert stored == pytest.approx(0.1)
    assert store.total_wasted == pytest.approx(9.9)


def test_brownout_latches_off_and_needs_real_recharge_to_come_back():
    """Without hysteresis a dead node reports itself alive forever."""
    store = create("store", "supercap", spec=StoreSpec(capacity=1.0, initial_fraction=0.1,
                                                       brownout_fraction=0.05,
                                                       leakage_power=0.0))
    assert store.alive
    assert store.draw(0.01)
    assert not store.draw(0.5)
    assert store.energy > 0.0          # never goes negative
    assert store.brownouts == 1
    assert not store.alive

    # A trickle that does not reach the recovery level must not revive it.
    store.harvest(0.01, 1.0)
    assert not store.alive
    store.harvest(1.0, 1.0)
    assert store.alive


def test_battery_charge_rate_limit_can_lose_a_burst():
    """A supercap swallows a short burst that a battery cannot."""
    spec = StoreSpec(capacity=100.0, initial_fraction=0.0, leakage_power=0.0)
    cap = create("store", "supercap", spec=spec)
    bat = create("store", "battery", spec=spec, max_charge_power=0.05)
    cap.harvest(5.0, 1.0)
    bat.harvest(5.0, 1.0)
    assert cap.energy > bat.energy


def test_power_account_breaks_down_by_subsystem():
    acct = PowerAccount()
    acct.add("mcu", 1.0)
    acct.add("gimbal", 3.0)
    assert acct.total == pytest.approx(4.0)
    assert acct.breakdown()["gimbal"] == pytest.approx(0.75)


def test_a_node_that_runs_dry_stops_moving():
    """Energy is a hard constraint, not a soft one."""
    w = World(medium="water", topology="gradient", seed=0)
    thirsty = aquatic(MOTE_C).variant(
        "thirsty",
        harvest=HarvestSpec(kind="none"),
        store=StoreSpec(capacity=0.5, initial_fraction=0.3, leakage_power=0.0),
    )
    dev = w.add_device(thirsty, np.array([0.0, 0.0, -2.0]))
    w.goals = np.array([[0.0, 0.0, -25.0]])
    Simulation(w, dt=0.05).run(300.0)
    assert not dev.store.alive
    assert dev.position[2] > -25.0  # it never got there


def test_tethered_node_is_not_depth_shadowed():
    """The box is outside the medium, so its supply does not fade with depth."""
    water = create("medium", "water")
    head = tethered(aquatic(MOTE_C))
    harvester = create("harvester", head.harvest.kind)
    shallow = harvester.power(head.harvest, np.array([0.0, 0.0, -1.0]), water, 0.0)
    deep = harvester.power(head.harvest, np.array([0.0, 0.0, -40.0]), water, 0.0)
    assert shallow == pytest.approx(deep)
    assert deep > 0.0


# --------------------------------------------------------------------------
# clocks
# --------------------------------------------------------------------------
def test_drift_is_a_fixed_property_of_the_part():
    """A crystal's offset is manufacturing, not noise: it must not resample."""
    rng = np.random.default_rng(0)
    clock = Clock(drift_ppm=20.0, jitter=0.0, rng=rng)
    d = clock.drift
    for t in (1.0, 2.0, 10.0):
        assert clock.error(t) == pytest.approx(d * t)


def test_uncertainty_grows_until_a_resync_and_then_collapses():
    clock = Clock(drift_ppm=50.0, jitter=1e-7, fixed_drift=50e-6)
    early = clock.uncertainty(1.0)
    late = clock.uncertainty(100.0)
    assert late > early
    clock.resync(100.0, accuracy=0.0)
    assert clock.uncertainty(100.0) < early
    assert clock.error(100.0) == pytest.approx(0.0, abs=1e-12)


def test_resync_corrects_the_offset_but_not_the_drift():
    """Learning the time is not learning how fast your crystal runs."""
    clock = Clock(drift_ppm=100.0, jitter=0.0, fixed_drift=100e-6)
    clock.resync(10.0, accuracy=0.0)
    assert clock.error(10.0) == pytest.approx(0.0, abs=1e-12)
    assert clock.error(20.0) == pytest.approx(100e-6 * 10.0, rel=1e-9)


def test_guard_band_scales_with_the_resync_interval():
    """This is the number the scheduler has to pay, so it must be monotone."""
    tight = ClockDomain.required_guard(drift_ppm=20.0, interval=0.1)
    loose = ClockDomain.required_guard(drift_ppm=20.0, interval=10.0)
    assert loose > tight
    assert loose == pytest.approx(2 * 2 * (20 * ppm * 10.0 + 1e-5 + 3e-6))


def test_domain_skew_is_bounded_by_the_beacon_interval():
    domain = ClockDomain(resync_interval=0.5, beacon_accuracy=1e-6)
    rng = np.random.default_rng(1)
    for i in range(20):
        domain.add(i, Clock(drift_ppm=50.0, jitter=1e-8, rng=rng))
    t = 0.0
    worst = 0.0
    while t < 20.0:
        t += 0.05
        domain.step(t)
        worst = max(worst, domain.max_skew(t))
    # Two nodes can drift apart for at most one beacon interval before a resync.
    assert worst < 2 * 50 * ppm * 0.5 + 1e-5


def test_unreachable_nodes_keep_drifting():
    """A swarm that spreads out loses its beacon, and the model must say so."""
    domain = ClockDomain(resync_interval=0.5, beacon_accuracy=0.0)
    near = domain.add(0, Clock(drift_ppm=100.0, jitter=0.0, fixed_drift=100e-6))
    far = domain.add(1, Clock(drift_ppm=100.0, jitter=0.0, fixed_drift=100e-6))
    t = 0.0
    while t < 10.0:
        t += 0.5
        domain.step(t, reachable={0})
    assert abs(near.error(t)) < abs(far.error(t))

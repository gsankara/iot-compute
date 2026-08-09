"""Invariants the stock device library has to satisfy.

These are the checks that catch a physically impossible part before it turns
into a mysterious simulation result.  A hull whose ballast tank cannot reach
neutral trim, for instance, simply floats away forever, and the symptom shows up
three layers downstream as "the link keeps dropping".
"""

from __future__ import annotations

import math

import pytest

from iotc.devices import LIBRARY, MOTE_C, aerial, aquatic, get_class, tethered
from iotc.registry import create
from iotc.sim.links import LinkGeometry
from iotc.units import KiB, deg


BUOYANT = [c for c in LIBRARY.values() if c.kinematics == "buoyant"]


@pytest.mark.parametrize("cls", BUOYANT, ids=lambda c: c.name)
def test_every_buoyant_hull_can_trim_neutral_in_water(cls):
    """The tank must straddle neutral, or the part can never hold a depth."""
    water = create("medium", "water")
    kin = create("kinematics", "buoyant")
    neutral = kin.neutral_ballast(water, cls.body)
    assert 0.0 < neutral < cls.body.ballast_capacity, (
        f"{cls.name}: neutral trim is {neutral * 1e3:.1f} g but the tank holds "
        f"{cls.body.ballast_capacity * 1e3:.1f} g"
    )


@pytest.mark.parametrize("cls", BUOYANT, ids=lambda c: c.name)
def test_every_buoyant_hull_can_move_in_both_vertical_directions(cls):
    water = create("medium", "water")
    kin = create("kinematics", "buoyant")
    neutral = kin.neutral_ballast(water, cls.body)
    assert neutral > 1e-4                                    # can rise
    assert cls.body.ballast_capacity - neutral > 1e-4        # can sink


@pytest.mark.parametrize("cls", list(LIBRARY.values()), ids=lambda c: c.name)
def test_every_class_is_internally_consistent(cls):
    assert cls.compute.flops > 0
    assert cls.compute.ram_bytes >= 64 * KiB
    assert cls.optical.n_transceivers >= 1
    assert 0.0 < cls.optical.tx_efficiency <= 1.0
    assert cls.store.capacity > 0
    assert 0.0 <= cls.store.brownout_fraction < cls.store.initial_fraction
    assert cls.optical.gimbal.grid.max_residual() < cls.optical.rx_fov


def test_the_library_actually_spans_its_axes_of_heterogeneity():
    """If every class were similar the compiler would have nothing to decide."""
    by = {c.name: c for c in LIBRARY.values()}
    assert by["mote-c"].compute.flops > 5 * by["mote-m"].compute.flops
    assert by["mote-m"].compute.ram_bytes > 8 * by["mote-s"].compute.ram_bytes
    assert (by["mote-s"].optical.gimbal.grid.step_az
            < by["mote-m"].optical.gimbal.grid.step_az / 10)
    assert by["relay"].optical.n_transceivers > 2 * by["mote-c"].optical.n_transceivers


def test_finer_steering_only_helps_if_the_sweep_is_wide_enough():
    """Resolution and reach are separate capabilities, and both are needed."""
    fine = get_class("mote-s").optical.gimbal.grid
    coarse = get_class("mote-m").optical.gimbal.grid
    assert fine.el_limits[1] > coarse.el_limits[1]
    assert fine.step_az < coarse.step_az


def test_variants_keep_the_electronics_and_change_the_platform():
    base = MOTE_C
    for variant in (aerial(base), aquatic(base), tethered(aquatic(base))):
        assert variant.compute == base.compute
        assert variant.name != base.name
    assert aerial(base).kinematics == "drone"
    assert tethered(aquatic(base)).kinematics == "tethered"
    assert aquatic(base).optical.wavelength < base.optical.wavelength


def test_aquatic_variant_is_the_difference_between_working_and_not():
    water = create("medium", "water")
    channel = create("channel", "fso", turbulence="none")

    def reach(cls):
        o = cls.optical
        geo = LinkGeometry(
            distance=1.0, wavelength=o.wavelength, tx_power=o.tx_power,
            divergence=o.divergence, tx_aperture_radius=o.tx_aperture_radius,
            rx_aperture_radius=o.rx_aperture_radius, rx_fov=o.rx_fov,
        )
        return channel.max_range(geo, water, line_rate=o.line_rate)

    assert reach(aquatic(MOTE_C)) > 10 * reach(MOTE_C)


def test_tethered_variant_removes_the_ballast_actuator_entirely():
    head = tethered(aquatic(MOTE_C))
    assert head.body.ballast_capacity == 0.0
    assert head.tether is not None
    assert head.tether.max_speed_z > 0.0
    assert head.power.thruster_power == 0.0

"""The link budget: monotonicity, closed-form checks, coding, and the surrogate."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from iotc.registry import create
from iotc.sim.links import LinkGeometry, q_function
from iotc.units import db, deg, mW


def geo(distance=10.0, wavelength=850e-9, misalignment=0.0, tx_power=1.0 * mW):
    return LinkGeometry(
        distance=distance,
        wavelength=wavelength,
        tx_power=tx_power,
        divergence=5e-3,
        tx_aperture_radius=1.5e-3,
        rx_aperture_radius=2.5e-3,
        rx_fov=15e-3,
        tx_misalignment=misalignment,
    )


@pytest.fixture
def air():
    return create("medium", "air")


@pytest.fixture
def channel():
    return create("channel", "fso", turbulence="none")


# --------------------------------------------------------------------------
# monotonicity: the properties every downstream bisection relies on
# --------------------------------------------------------------------------
def test_received_power_falls_monotonically_with_range(channel, air):
    powers = [
        channel.evaluate(geo(distance=d), air, None, stochastic=False).rx_power
        for d in (0.5, 1, 2, 5, 10, 20, 50, 100)
    ]
    assert all(a > b for a, b in zip(powers, powers[1:]))


def test_received_power_falls_monotonically_with_misalignment(channel, air):
    powers = [
        channel.evaluate(geo(misalignment=m), air, None, stochastic=False).rx_power
        for m in (0.0, 0.5e-3, 1e-3, 2e-3, 4e-3, 8e-3)
    ]
    assert all(a > b for a, b in zip(powers, powers[1:]))


def test_far_field_power_follows_inverse_square(channel, air):
    """Well beyond the Rayleigh range the spot grows linearly, so power ~ 1/d^2."""
    p1 = channel.evaluate(geo(distance=40.0), air, None, stochastic=False).rx_power
    p2 = channel.evaluate(geo(distance=80.0), air, None, stochastic=False).rx_power
    assert p1 / p2 == pytest.approx(4.0, rel=0.02)


def test_link_dies_outside_the_receiver_field_of_view(channel, air):
    g = geo()
    g.rx_misalignment = g.rx_fov * 1.01
    out = channel.evaluate(g, air, None, stochastic=False)
    assert not out.feasible
    assert "field of view" in out.reason


def test_max_range_is_the_bisection_boundary(channel, air):
    r = channel.max_range(geo(), air, line_rate=1e9)
    assert r > 1.0
    inside = channel.evaluate(geo(distance=r * 0.98), air, None, stochastic=False)
    outside = channel.evaluate(geo(distance=r * 1.05), air, None, stochastic=False)
    assert inside.feasible and not outside.feasible


# --------------------------------------------------------------------------
# beams
# --------------------------------------------------------------------------
def test_gaussian_on_axis_capture_matches_closed_form():
    beam = create("beam", "gaussian")
    g = geo(distance=30.0)
    w = beam.spot_radius(g)
    expected = 1.0 - math.exp(-2.0 * g.rx_aperture_radius**2 / w**2)
    assert beam.captured_fraction(g, 0.0) == pytest.approx(expected, rel=1e-9)


def test_gaussian_spot_radius_is_the_quadrature_sum():
    beam = create("beam", "gaussian")
    g = geo(distance=100.0)
    expected = math.hypot(g.tx_aperture_radius, g.divergence * g.distance)
    assert beam.spot_radius(g) == pytest.approx(expected)


def test_flat_top_is_more_pointing_tolerant_than_gaussian():
    """A top hat has no peak to slide off, at the cost of on-axis power.

    That trade is the whole reason the beam stage is a plugin.
    """
    gauss, flat = create("beam", "gaussian"), create("beam", "flat_top")
    g = geo(distance=20.0)
    w = gauss.spot_radius(g)
    small, large = 0.0, 0.35 * w
    g_ratio = gauss.captured_fraction(g, large) / gauss.captured_fraction(g, small)
    f_ratio = flat.captured_fraction(g, large) / flat.captured_fraction(g, small)
    assert f_ratio > g_ratio


def test_flat_top_capture_never_exceeds_unity():
    flat = create("beam", "flat_top")
    for d in (0.1, 1.0, 10.0, 100.0):
        for off in (0.0, 0.01, 0.5, 5.0):
            frac = flat.captured_fraction(geo(distance=d), off)
            assert 0.0 <= frac <= 1.0


# --------------------------------------------------------------------------
# detectors and modulation
# --------------------------------------------------------------------------
def test_apd_gain_has_an_optimum():
    """Excess noise eventually overtakes the thermal floor it was hiding."""
    g = geo()
    g.bandwidth = 1e9
    power = 3e-9
    snrs = {m: create("receiver", "apd", gain=m).snr(power, g) for m in (1, 5, 20, 50, 200, 1000)}
    best = max(snrs, key=snrs.get)
    assert 1 < best < 1000


def test_receiver_snr_grows_with_power():
    g = geo()
    g.bandwidth = 1e9
    for name in ("pin", "apd", "spad"):
        det = create("receiver", name)
        vals = [det.snr(p, g) for p in (1e-12, 1e-11, 1e-10, 1e-9, 1e-8)]
        assert all(a < b for a, b in zip(vals, vals[1:])), name


def test_ook_ber_matches_the_q_function():
    ook = create("modulation", "ook")
    for snr_db in (0, 5, 10, 15, 20):
        snr = 10 ** (snr_db / 10)
        assert ook.ber(snr) == pytest.approx(q_function(math.sqrt(snr) / 2.0), rel=1e-12)


def test_ber_falls_with_snr_for_every_modulation():
    for name in ("ook", "pam4", "ppm"):
        mod = create("modulation", name)
        bers = [mod.ber(10 ** (d / 10)) for d in (0, 6, 12, 18, 24, 30)]
        assert all(a >= b for a, b in zip(bers, bers[1:])), name
        assert bers[-1] < bers[0]


def test_pam4_needs_more_snr_than_ook_but_carries_more_bits():
    ook, pam4 = create("modulation", "ook"), create("modulation", "pam4")
    snr = 10 ** (14 / 10)
    assert pam4.ber(snr) > ook.ber(snr)
    assert pam4.bits_per_symbol == 2 * ook.bits_per_symbol


def test_ppm_beats_ook_when_photons_are_scarce():
    """Concentrating the energy into one of L slots is the point of PPM."""
    ook, ppm = create("modulation", "ook"), create("modulation", "ppm", order=8)
    snr = 10 ** (6 / 10)
    assert ppm.ber(snr) < ook.ber(snr)


# --------------------------------------------------------------------------
# coding
# --------------------------------------------------------------------------
def test_uncoded_per_matches_the_binomial():
    code = create("coding", "none")
    ber, bits = 1e-5, 8192
    assert code.per(ber, bits) == pytest.approx(1.0 - (1.0 - ber) ** bits, rel=1e-9)


def test_coding_reduces_packet_error_and_costs_rate():
    raw = create("coding", "none")
    rs = create("coding", "rs")
    ber = 1e-4
    assert rs.per(ber, 8192) < raw.per(ber, 8192)
    assert rs.code_rate < 1.0


def test_fountain_survives_whole_lost_packets():
    """The failure mode here is erasure, not scattered bit errors.

    A rateless block absorbs losses up to roughly its overhead and then falls
    off a cliff, which is the honest behaviour: 25 percent of redundancy cannot
    repair a 22 percent loss rate, and the model must not pretend otherwise.
    """
    raw = create("coding", "none")
    fountain = create("coding", "fountain", block_packets=16, overhead=0.25)

    # Comfortably inside the code's budget: four orders of magnitude better.
    easy = 3e-6
    assert raw.per(easy, 8192) > 1e-2
    assert fountain.per(easy, 8192) < raw.per(easy, 8192) / 1000

    # Beyond it, redundancy buys nothing and the block is simply lost.
    hard = 3e-5
    assert raw.per(hard, 8192) > 0.2
    assert fountain.per(hard, 8192) > 0.2


def test_per_is_monotone_in_ber():
    for name in ("none", "rs", "fountain"):
        code = create("coding", name)
        vals = [code.per(b, 4096) for b in (1e-9, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2)]
        assert all(a <= b + 1e-12 for a, b in zip(vals, vals[1:])), name


# --------------------------------------------------------------------------
# turbulence
# --------------------------------------------------------------------------
def test_turbulence_fading_has_unit_mean(air):
    """Fading redistributes power, it does not create or destroy it."""
    g = geo(distance=50.0)
    for name in ("lognormal", "gamma_gamma"):
        model = create("turbulence", name)
        rng = np.random.default_rng(3)
        samples = [model.sample_gain(g, air, rng) for _ in range(20000)]
        assert np.mean(samples) == pytest.approx(1.0, rel=0.06), name
        assert np.std(samples) > 0.0


def test_stronger_turbulence_fades_harder(air):
    g = geo(distance=200.0)
    model = create("turbulence", "gamma_gamma")
    calm = create("medium", "air", cn2=1e-16)
    rough = create("medium", "air", cn2=1e-13)
    rng = np.random.default_rng(4)
    calm_s = np.std([model.sample_gain(g, calm, rng) for _ in range(4000)])
    rough_s = np.std([model.sample_gain(g, rough, rng) for _ in range(4000)])
    assert rough_s > calm_s


def test_kolmogorov_series_is_temporally_correlated(air):
    """Independent draws per packet would overstate how well retries work."""
    model = create("turbulence", "kolmogorov_ts", coherence_time=1.0)
    g = geo(distance=200.0)
    rough = create("medium", "air", cn2=1e-13)
    rng = np.random.default_rng(5)
    series = [
        model.sample_gain(g, rough, rng, key="a", now=i * 0.02) for i in range(4000)
    ]
    x = np.log(series)
    lag1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    assert lag1 > 0.9


# --------------------------------------------------------------------------
# surrogate
# --------------------------------------------------------------------------
def test_surrogate_tracks_the_exact_model(air):
    sur = create("channel", "surrogate", turbulence="none",
                 min_distance=0.1, max_distance=200.0)
    stats = sur.verify(geo(), air, n=300, seed=6)
    assert stats["samples"] > 200
    assert stats["max_abs_db"] < 0.5, stats
    assert abs(stats["bias_db"]) < 0.1, stats


def test_surrogate_falls_back_outside_its_envelope(air):
    sur = create("channel", "surrogate", turbulence="none",
                 min_distance=1.0, max_distance=20.0)
    exact = sur.base.evaluate(geo(distance=500.0), air, None, stochastic=False)
    approx = sur.evaluate(geo(distance=500.0), air, None, stochastic=False)
    assert approx.rx_power == pytest.approx(exact.rx_power, rel=1e-12)


def test_surrogate_caches_across_calls(air):
    sur = create("channel", "surrogate", turbulence="none")
    for d in (1.0, 2.0, 4.0, 8.0):
        sur.evaluate(geo(distance=d), air, None, stochastic=False)
    assert sur.misses == 1
    assert sur.hits >= 3


# --------------------------------------------------------------------------
# media
# --------------------------------------------------------------------------
def test_water_forces_the_blue_green_window(channel):
    """850 nm cannot cross water; 470 nm can.  Any liquid config must move."""
    water = create("medium", "water")
    near_ir = channel.max_range(geo(wavelength=850e-9), water, line_rate=1e9)
    blue_green = channel.max_range(geo(wavelength=470e-9), water, line_rate=1e9)
    assert near_ir < 3.0
    assert blue_green > 10 * near_ir


def test_fog_destroys_an_air_link(channel):
    clear = create("medium", "air", visibility_km=50.0)
    fog = create("medium", "air", visibility_km=0.2)
    assert channel.max_range(geo(), fog, line_rate=1e9) < channel.max_range(
        geo(), clear, line_rate=1e9
    )


def test_rf_is_useless_in_seawater_and_free_in_air():
    air = create("medium", "air")
    sea = create("medium", "water", conductivity=4.0)
    assert air.rf_attenuation_db(2.4e9, 100.0) == pytest.approx(0.0)
    assert sea.rf_attenuation_db(2.4e9, 1.0) > 100.0
    # Dropping the carrier by five decades buys back two and a half.
    assert sea.rf_attenuation_db(1e4, 1.0) < sea.rf_attenuation_db(2.4e9, 1.0) / 100


def test_vacuum_is_lossless():
    vac = create("medium", "vacuum")
    assert vac.extinction(850e-9) == 0.0
    assert vac.optical_transmittance(1e6, 850e-9) == pytest.approx(1.0)

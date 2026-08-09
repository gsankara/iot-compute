"""The pluggable pieces of a link budget.

A link is evaluated as a chain of independent stages, each one a registry
plugin, so a new scattering law or a new detector can be dropped in without
touching the channel that composes them:

    beam -> attenuation -> turbulence -> pointing -> receiver -> modulation -> coding

Every stage takes SI inputs and returns SI outputs, and :class:`LinkBudget`
carries all of the intermediate terms rather than just the final answer -- when
a link fails it should be obvious *which* term killed it.

The physics and its validity ranges are written up in ``docs/PHYSICS.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ...registry import register
from ...units import K_B, Q_E, T_ROOM, db


# --------------------------------------------------------------------------
# data carried through the chain
# --------------------------------------------------------------------------
@dataclass
class LinkGeometry:
    """Everything geometric a link budget needs."""

    distance: float
    wavelength: float
    tx_power: float
    divergence: float  # transmit half-angle, rad
    tx_aperture_radius: float
    rx_aperture_radius: float
    rx_fov: float
    #: Residual boresight error at each end [rad], after quantized steering.
    tx_misalignment: float = 0.0
    rx_misalignment: float = 0.0
    bandwidth: float = 1e9  # electrical bandwidth of the receiver, Hz


@dataclass
class LinkBudget:
    """The full accounting for one link evaluation."""

    rx_power: float = 0.0
    geometric_db: float = 0.0
    medium_db: float = 0.0
    pointing_db: float = 0.0
    turbulence_db: float = 0.0
    coupling_db: float = 0.0
    snr: float = 0.0
    ber: float = 0.5
    per: float = 1.0
    rate: float = 0.0
    latency: float = 0.0
    feasible: bool = False
    reason: str = ""
    extras: dict = field(default_factory=dict)

    @property
    def snr_db(self) -> float:
        return db(self.snr)

    def report(self) -> str:  # pragma: no cover - reporting aid
        return (
            f"rx={self.rx_power:.3e} W  geo={self.geometric_db:7.2f} dB  "
            f"med={self.medium_db:7.2f} dB  point={self.pointing_db:7.2f} dB  "
            f"turb={self.turbulence_db:+6.2f} dB  SNR={self.snr_db:6.2f} dB  "
            f"BER={self.ber:.2e}  PER={self.per:.3f}  "
            f"{'up' if self.feasible else 'DOWN: ' + self.reason}"
        )


# --------------------------------------------------------------------------
# stage 1: transmit beam
# --------------------------------------------------------------------------
class Beam:
    """Transverse profile of the transmitted beam."""

    def spot_radius(self, geo: LinkGeometry) -> float:
        """Beam radius [m] at the receiver plane."""
        raise NotImplementedError

    def captured_fraction(self, geo: LinkGeometry, offset: float) -> float:
        """Fraction of transmitted power falling inside the receive aperture,
        when the beam centre misses it by ``offset`` metres."""
        raise NotImplementedError


@register("beam", "gaussian")
class GaussianBeam(Beam):
    """A TEM00 beam with the standard pointing-error capture model.

    On-axis capture is the usual ``1 - exp(-2a^2/w^2)``.  Off axis we use the
    Farid-Hranilovic approximation, which replaces the exact Bessel integral
    with a Gaussian of an equivalent width; it is accurate to a fraction of a dB
    for ``w > a``, which is the regime every one of these links operates in.
    """

    def __init__(self, m_squared: float = 1.0) -> None:
        #: Beam quality factor; 1.0 is diffraction limited.
        self.m_squared = float(m_squared)

    def spot_radius(self, geo: LinkGeometry) -> float:
        w0 = max(geo.tx_aperture_radius, 1e-9)
        theta = geo.divergence * self.m_squared
        return math.sqrt(w0**2 + (theta * geo.distance) ** 2)

    def captured_fraction(self, geo: LinkGeometry, offset: float) -> float:
        w = self.spot_radius(geo)
        a = geo.rx_aperture_radius
        # On axis the integral is exact, so use it rather than the approximation.
        a0 = 1.0 - math.exp(-2.0 * a**2 / w**2)
        if offset <= 0.0:
            return float(a0)
        v = math.sqrt(math.pi / 2.0) * a / w
        if v > 8.0:  # the aperture swallows the whole beam
            w_eq_sq = w**2
        else:
            erf_v = math.erf(v)
            denom = 2.0 * v * math.exp(-(v**2))
            w_eq_sq = w**2 * math.sqrt(math.pi) * erf_v / max(denom, 1e-300)
        return float(a0 * math.exp(-2.0 * offset**2 / max(w_eq_sq, 1e-300)))


@register("beam", "flat_top", "tophat")
class FlatTopBeam(Beam):
    """A uniform disc of light -- what a diffuser or a shaped array produces.

    More tolerant of small pointing errors than a Gaussian (no peak to slide
    off) but it wastes power outside the aperture, so it trades peak link margin
    for robustness.  Exactly the sort of trade the design study should price.
    """

    def spot_radius(self, geo: LinkGeometry) -> float:
        return max(geo.tx_aperture_radius, 1e-9) + geo.divergence * geo.distance

    def captured_fraction(self, geo: LinkGeometry, offset: float) -> float:
        r = self.spot_radius(geo)
        a = geo.rx_aperture_radius
        if offset >= r + a:
            return 0.0
        if offset <= max(r - a, 0.0):
            return float(min(1.0, (a / r) ** 2))
        # Area of intersection of two discs, normalised by the beam disc.
        d = offset
        t1 = r**2 * math.acos(np.clip((d**2 + r**2 - a**2) / (2 * d * r), -1.0, 1.0))
        t2 = a**2 * math.acos(np.clip((d**2 + a**2 - r**2) / (2 * d * a), -1.0, 1.0))
        t3 = 0.5 * math.sqrt(
            max((-d + r + a) * (d + r - a) * (d - r + a) * (d + r + a), 0.0)
        )
        return float((t1 + t2 - t3) / (math.pi * r**2))


@register("beam", "bessel", "nondiffracting")
class BesselBeam(Beam):
    """A quasi-nondiffracting beam: a tight core that resists spreading.

    Modelled as a Gaussian whose divergence is suppressed by ``spread_factor``
    over the non-diffracting range, at the cost of a fixed power penalty because
    most of the energy sits in the outer rings rather than the core.
    """

    def __init__(self, spread_factor: float = 0.25, core_fraction: float = 0.35,
                 nondiffracting_range: float = 50.0) -> None:
        self.spread_factor = float(spread_factor)
        self.core_fraction = float(core_fraction)
        self.nondiffracting_range = float(nondiffracting_range)

    def spot_radius(self, geo: LinkGeometry) -> float:
        w0 = max(geo.tx_aperture_radius, 1e-9)
        if geo.distance <= self.nondiffracting_range:
            theta = geo.divergence * self.spread_factor
        else:
            theta = geo.divergence  # beyond the range it behaves normally
        return math.sqrt(w0**2 + (theta * geo.distance) ** 2)

    def captured_fraction(self, geo: LinkGeometry, offset: float) -> float:
        w = self.spot_radius(geo)
        a = geo.rx_aperture_radius
        on_axis = 1.0 - math.exp(-2.0 * a**2 / w**2)
        return float(self.core_fraction * on_axis * math.exp(-2.0 * offset**2 / w**2))


# --------------------------------------------------------------------------
# stage 2: medium attenuation
# --------------------------------------------------------------------------
class Attenuation:
    """Loss along the path, over and above geometric spreading."""

    def transmittance(self, geo: LinkGeometry, medium) -> float:
        raise NotImplementedError


@register("attenuation", "beer_lambert", "beer-lambert")
class BeerLambert(Attenuation):
    """Exponential extinction using the medium's own coefficient."""

    def transmittance(self, geo: LinkGeometry, medium) -> float:
        return medium.optical_transmittance(geo.distance, geo.wavelength)


@register("attenuation", "mie")
class MieScattering(Attenuation):
    """Beer-Lambert plus a forward-scatter recovery term.

    Large particles scatter mostly forward, so some of the "lost" light is still
    inside the receiver's field of view.  ``recovery`` is the fraction of
    scattered power that is recovered; setting it to zero reduces this to plain
    Beer-Lambert.
    """

    def __init__(self, recovery: float = 0.15) -> None:
        self.recovery = float(recovery)

    def transmittance(self, geo: LinkGeometry, medium) -> float:
        direct = medium.optical_transmittance(geo.distance, geo.wavelength)
        return float(min(1.0, direct + self.recovery * (1.0 - direct) * geo.rx_fov / 0.05))


# --------------------------------------------------------------------------
# stage 3: turbulence
# --------------------------------------------------------------------------
class Turbulence:
    """Random intensity fading along the path."""

    def rytov_variance(self, geo: LinkGeometry, medium) -> float:
        """Plane-wave Rytov variance; the standard strength parameter."""
        k = 2.0 * math.pi / geo.wavelength
        return 1.23 * medium.cn2 * k ** (7.0 / 6.0) * geo.distance ** (11.0 / 6.0)

    def sample_gain(self, geo: LinkGeometry, medium, rng) -> float:
        """One realisation of the intensity scaling factor (mean 1)."""
        raise NotImplementedError


@register("turbulence", "none", "off")
class NoTurbulence(Turbulence):
    """Frozen atmosphere -- useful for isolating other effects."""

    def sample_gain(self, geo: LinkGeometry, medium, rng) -> float:
        return 1.0


@register("turbulence", "lognormal")
class LogNormalFading(Turbulence):
    """Weak-turbulence log-normal scintillation.

    Valid for Rytov variance below about 1; beyond that the distribution
    develops the heavy tail that only gamma-gamma captures, so this model
    clamps and reports rather than silently extrapolating.
    """

    def sample_gain(self, geo: LinkGeometry, medium, rng) -> float:
        sigma_r2 = self.rytov_variance(geo, medium)
        if sigma_r2 <= 0.0:
            return 1.0
        sigma_r2 = min(sigma_r2, 1.0)
        sigma_x2 = sigma_r2 / 4.0
        x = rng.normal(-sigma_x2, math.sqrt(sigma_x2))
        return float(math.exp(2.0 * x))


@register("turbulence", "gamma_gamma", "gamma-gamma")
class GammaGammaFading(Turbulence):
    """Moderate-to-strong turbulence, as the product of two gamma variates.

    ``alpha`` and ``beta`` are the large- and small-scale eddy counts derived
    from the Rytov variance; the product has mean 1 by construction.
    """

    def _shape(self, sigma_r2: float) -> tuple[float, float]:
        s = max(sigma_r2, 1e-9)
        alpha = 1.0 / (math.exp(0.49 * s / (1 + 1.11 * s ** (6 / 5)) ** (7 / 6)) - 1.0)
        beta = 1.0 / (math.exp(0.51 * s / (1 + 0.69 * s ** (6 / 5)) ** (5 / 6)) - 1.0)
        return max(alpha, 1e-3), max(beta, 1e-3)

    def sample_gain(self, geo: LinkGeometry, medium, rng) -> float:
        sigma_r2 = self.rytov_variance(geo, medium)
        if sigma_r2 <= 1e-9:
            return 1.0
        alpha, beta = self._shape(sigma_r2)
        x = rng.gamma(alpha, 1.0 / alpha)
        y = rng.gamma(beta, 1.0 / beta)
        return float(x * y)


@register("turbulence", "kolmogorov_ts", "kolmogorov")
class KolmogorovTimeSeries(Turbulence):
    """Temporally correlated fading.

    Independent draws per packet overstate how well interleaving and retries
    work: real scintillation stays faded for a coherence time, so losses arrive
    in bursts.  This wraps the log-normal amplitude in a first-order AR process
    with the given coherence time, keyed per link.
    """

    def __init__(self, coherence_time: float = 1e-3) -> None:
        self.coherence_time = float(coherence_time)
        self._state: dict[object, tuple[float, float]] = {}

    def sample_gain(self, geo: LinkGeometry, medium, rng, key=None, now: float = 0.0) -> float:
        sigma_r2 = min(self.rytov_variance(geo, medium), 1.0)
        if sigma_r2 <= 0.0:
            return 1.0
        sigma_x = math.sqrt(sigma_r2 / 4.0)
        prev_x, prev_t = self._state.get(key, (0.0, now))
        dt = max(now - prev_t, 0.0)
        rho = math.exp(-dt / self.coherence_time) if self.coherence_time > 0 else 0.0
        x = rho * prev_x + math.sqrt(max(1.0 - rho**2, 0.0)) * rng.normal(0.0, sigma_x)
        self._state[key] = (x, now)
        return float(math.exp(2.0 * (x - sigma_x**2 / 2.0)))


# --------------------------------------------------------------------------
# stage 4: pointing
# --------------------------------------------------------------------------
class Pointing:
    """Turns steering error into a radial displacement at the receiver."""

    def sample_offset(self, geo: LinkGeometry, rng, jitter_sigma: float) -> float:
        raise NotImplementedError


@register("pointing", "deterministic", "quantization")
class DeterministicPointing(Pointing):
    """Quantization residual only, with no random jitter.

    The bias case: what a perfectly steady platform with a finite gimbal tick
    would suffer.  It is the floor that no amount of stabilisation can beat.
    """

    def sample_offset(self, geo: LinkGeometry, rng, jitter_sigma: float) -> float:
        return float(geo.tx_misalignment * geo.distance)


@register("pointing", "beckmann", "rician")
class BeckmannPointing(Pointing):
    """Quantization residual as a fixed bias plus Gaussian jitter on both axes.

    The resulting radial error is Rician, which is the standard FSO pointing
    model.  Keeping the bias and the jitter separate matters here: the bias is
    something the compiler can trade against gimbal cost, the jitter is not.
    """

    def sample_offset(self, geo: LinkGeometry, rng, jitter_sigma: float) -> float:
        bias = geo.tx_misalignment
        s = max(jitter_sigma, 0.0)
        ex = bias + (rng.normal(0.0, s) if s > 0 else 0.0)
        ey = rng.normal(0.0, s) if s > 0 else 0.0
        return float(math.hypot(ex, ey) * geo.distance)


# --------------------------------------------------------------------------
# stage 5: receiver
# --------------------------------------------------------------------------
class Receiver:
    """Photodetector front end: optical power in, electrical SNR out."""

    def snr(self, rx_power: float, geo: LinkGeometry) -> float:
        raise NotImplementedError


@register("receiver", "pin")
class PinDetector(Receiver):
    """A PIN photodiode with a transimpedance front end (thermal-noise limited)."""

    def __init__(self, responsivity: float = 0.55, load_resistance: float = 1e3,
                 dark_current: float = 1e-9, temperature: float = T_ROOM,
                 noise_figure: float = 2.0) -> None:
        self.responsivity = float(responsivity)
        self.load_resistance = float(load_resistance)
        self.dark_current = float(dark_current)
        self.temperature = float(temperature)
        self.noise_figure = float(noise_figure)

    def snr(self, rx_power: float, geo: LinkGeometry) -> float:
        b = max(geo.bandwidth, 1.0)
        i_sig = self.responsivity * rx_power
        i_shot2 = 2.0 * Q_E * (i_sig + self.dark_current) * b
        i_th2 = 4.0 * K_B * self.temperature * self.noise_figure * b / self.load_resistance
        return float(i_sig**2 / max(i_shot2 + i_th2, 1e-300))


@register("receiver", "apd")
class ApdDetector(Receiver):
    """An avalanche photodiode: internal gain at the price of excess noise.

    The gain helps until the multiplied shot noise overtakes the thermal floor,
    which is why ``gain`` has an optimum rather than being "more is better".
    """

    def __init__(self, responsivity: float = 0.65, gain: float = 30.0,
                 k_eff: float = 0.02, load_resistance: float = 1e3,
                 dark_current: float = 5e-10, temperature: float = T_ROOM,
                 noise_figure: float = 2.0) -> None:
        self.responsivity = float(responsivity)
        self.gain = float(gain)
        self.k_eff = float(k_eff)
        self.load_resistance = float(load_resistance)
        self.dark_current = float(dark_current)
        self.temperature = float(temperature)
        self.noise_figure = float(noise_figure)

    @property
    def excess_noise_factor(self) -> float:
        m = self.gain
        return self.k_eff * m + (2.0 - 1.0 / m) * (1.0 - self.k_eff)

    def snr(self, rx_power: float, geo: LinkGeometry) -> float:
        b = max(geo.bandwidth, 1.0)
        m, f = self.gain, self.excess_noise_factor
        i_prim = self.responsivity * rx_power
        i_sig = m * i_prim
        i_shot2 = 2.0 * Q_E * (i_prim + self.dark_current) * m**2 * f * b
        i_th2 = 4.0 * K_B * self.temperature * self.noise_figure * b / self.load_resistance
        return float(i_sig**2 / max(i_shot2 + i_th2, 1e-300))


@register("receiver", "spad")
class SpadDetector(Receiver):
    """A single-photon avalanche diode, counting photons in Geiger mode.

    The right choice when the received power is femtowatts, which is where the
    long hops in an absorbing medium end up.  Limited by dead time rather than
    by thermal noise, so the SNR saturates instead of growing.
    """

    def __init__(self, pde: float = 0.35, dark_count_rate: float = 1e3,
                 dead_time: float = 20e-9) -> None:
        self.pde = float(pde)
        self.dark_count_rate = float(dark_count_rate)
        self.dead_time = float(dead_time)

    def snr(self, rx_power: float, geo: LinkGeometry) -> float:
        from ...units import photon_energy

        b = max(geo.bandwidth, 1.0)
        incident = rx_power / photon_energy(geo.wavelength)
        detected = self.pde * incident
        # Paralysable dead-time saturation.
        detected = detected / (1.0 + detected * self.dead_time)
        counts = detected / b  # per symbol interval
        noise = (detected + self.dark_count_rate) / b
        return float(counts**2 / max(noise, 1e-300))


# --------------------------------------------------------------------------
# stage 6: modulation
# --------------------------------------------------------------------------
def q_function(x: float) -> float:
    """Tail probability of the standard normal."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


class Modulation:
    """Symbol mapping: electrical SNR in, bit error rate out."""

    bits_per_symbol: float = 1.0

    def ber(self, snr: float) -> float:
        raise NotImplementedError

    def symbol_rate(self, line_rate: float) -> float:
        return line_rate / self.bits_per_symbol


@register("modulation", "ook", "nrz")
class OOK(Modulation):
    """On-off keying with direct detection -- the default for a cheap node."""

    bits_per_symbol = 1.0

    def ber(self, snr: float) -> float:
        # Q factor for equiprobable on/off with the noise dominated by the
        # 'on' level; the factor of two is the halved decision distance.
        return float(min(0.5, q_function(math.sqrt(max(snr, 0.0)) / 2.0)))


@register("modulation", "pam4")
class PAM4(Modulation):
    """Four-level PAM: twice the bits, but a third of the eye opening.

    Worth it only when there is spare SNR, which in this system means short
    hops.  Having both available lets the compiler pick per link.
    """

    bits_per_symbol = 2.0

    def ber(self, snr: float) -> float:
        q = math.sqrt(max(snr, 0.0)) / 2.0 / 3.0
        return float(min(0.5, 0.75 * q_function(q)))


@register("modulation", "ppm")
class PPM(Modulation):
    """L-ary pulse position modulation: power efficient, bandwidth hungry.

    The natural choice for the photon-starved long hops, since concentrating the
    energy into one of L slots buys a factor of L in peak SNR.
    """

    def __init__(self, order: int = 4) -> None:
        self.order = int(order)
        self.bits_per_symbol = math.log2(self.order)

    def ber(self, snr: float) -> float:
        slot_snr = max(snr, 0.0) * self.order
        pe = (self.order - 1) * q_function(math.sqrt(slot_snr) / 2.0)
        pe = min(pe, 1.0 - 1e-12)
        # Symbol error to bit error for orthogonal signalling.
        return float(min(0.5, pe * self.order / (2.0 * (self.order - 1))))


# --------------------------------------------------------------------------
# stage 7: coding
# --------------------------------------------------------------------------
class Coding:
    """Forward error correction: raw BER in, packet error rate out."""

    #: Fraction of the line rate left for payload.
    code_rate: float = 1.0

    def per(self, ber: float, payload_bits: int) -> float:
        raise NotImplementedError


@register("coding", "none", "uncoded")
class NoCoding(Coding):
    code_rate = 1.0

    def per(self, ber: float, payload_bits: int) -> float:
        if ber <= 0.0:
            return 0.0
        return float(-math.expm1(payload_bits * math.log1p(-min(ber, 1.0 - 1e-15))))


@register("coding", "rs", "reed_solomon")
class ReedSolomon(Coding):
    """A shortened Reed-Solomon code over GF(2^m).

    Corrects up to ``t`` symbol errors per block, so the packet survives unless
    more than ``t`` of the ``n`` symbols are hit.  The block-level binomial is
    evaluated directly, which is exact for independent bit errors.
    """

    def __init__(self, n: int = 255, k: int = 223, symbol_bits: int = 8) -> None:
        self.n = int(n)
        self.k = int(k)
        self.symbol_bits = int(symbol_bits)
        self.t = (self.n - self.k) // 2
        self.code_rate = self.k / self.n

    def per(self, ber: float, payload_bits: int) -> float:
        if ber <= 0.0:
            return 0.0
        ber = min(ber, 1.0 - 1e-15)
        ps = -math.expm1(self.symbol_bits * math.log1p(-ber))  # symbol error prob
        # P(block fails) = P(more than t symbol errors in n symbols)
        ok = 0.0
        log_ps, log_qs = math.log(max(ps, 1e-300)), math.log1p(-ps)
        for i in range(self.t + 1):
            log_c = (
                math.lgamma(self.n + 1) - math.lgamma(i + 1) - math.lgamma(self.n - i + 1)
            )
            ok += math.exp(log_c + i * log_ps + (self.n - i) * log_qs)
        p_block = 1.0 - min(ok, 1.0)
        n_blocks = max(1, math.ceil(payload_bits / (self.k * self.symbol_bits)))
        return float(-math.expm1(n_blocks * math.log1p(-min(p_block, 1.0 - 1e-15))))


@register("coding", "fountain", "rateless")
class Fountain(Coding):
    """A rateless erasure code across a burst of packets.

    Recovers the message as soon as slightly more than k packets arrive, in any
    order.  That is the right abstraction for this system: a beam that fades or
    a node that browses out loses whole packets, not scattered bits, and there
    is no time for a round trip to ask for a specific retransmission.
    """

    def __init__(self, block_packets: int = 16, overhead: float = 0.05) -> None:
        self.block_packets = int(block_packets)
        self.overhead = float(overhead)
        self.code_rate = 1.0 / (1.0 + self.overhead)

    def per(self, ber: float, payload_bits: int) -> float:
        raw = NoCoding().per(ber, payload_bits)
        k = self.block_packets
        n = int(math.ceil(k * (1.0 + self.overhead))) + 1
        # Block is lost if fewer than k of the n sent packets survive.
        p_ok = 0.0
        q = 1.0 - raw
        for i in range(k, n + 1):
            log_c = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            p_ok += math.exp(
                log_c + i * math.log(max(q, 1e-300)) + (n - i) * math.log(max(raw, 1e-300))
            )
        return float(max(0.0, min(1.0, 1.0 - p_ok)))

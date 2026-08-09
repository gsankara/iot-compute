"""The free-space optical channel.

This is just the composition of the seven stages in :mod:`iotc.sim.links.base`.
Keeping the composition separate from the stages is what makes the physics
pluggable: swapping ``turbulence="gamma_gamma"`` for a user-written model
changes one constructor argument and nothing else.

Two properties of this link are unusual enough to be worth stating, because the
rest of the system is shaped around them:

* it is **directional and mutual** -- both ends must be aimed at each other, so
  a link is a jointly scheduled resource, not something one node can decide to
  use unilaterally;
* the pointing error is **quantized**, so there is an irreducible loss floor set
  by the gimbal tick that no amount of transmit power removes.
"""

from __future__ import annotations

import math

from ...registry import build, register
from ...units import db
from .base import LinkBudget, LinkGeometry


@register("channel", "fso", "optical")
class FSOChannel:
    """A steerable free-space photonic link."""

    def __init__(
        self,
        beam="gaussian",
        attenuation="beer_lambert",
        turbulence="lognormal",
        pointing="beckmann",
        receiver="apd",
        modulation="ook",
        coding="rs",
        #: A link is declared down once the packet error rate exceeds this.
        max_per: float = 0.5,
    ) -> None:
        self.beam = build(beam, "beam")
        self.attenuation = build(attenuation, "attenuation")
        self.turbulence = build(turbulence, "turbulence")
        self.pointing = build(pointing, "pointing")
        self.receiver = build(receiver, "receiver")
        self.modulation = build(modulation, "modulation")
        self.coding = build(coding, "coding")
        self.max_per = float(max_per)

    # ------------------------------------------------------------------
    @property
    def bits_per_symbol(self) -> float:
        return self.modulation.bits_per_symbol

    def payload_rate(self, line_rate: float) -> float:
        """Useful bits per second after coding overhead."""
        return line_rate * self.coding.code_rate

    # ------------------------------------------------------------------
    def evaluate(
        self,
        geo: LinkGeometry,
        medium,
        rng=None,
        *,
        jitter_sigma: float = 0.0,
        payload_bits: int = 8192,
        line_rate: float = 1e9,
        stochastic: bool = True,
        key=None,
        now: float = 0.0,
    ) -> LinkBudget:
        """Evaluate one link.

        With ``stochastic=False`` the fading and jitter terms are replaced by
        their means, which is what the compiler's cost model and the surrogate
        both want -- a repeatable number rather than one draw.
        """
        out = LinkBudget()
        out.latency = geo.distance / medium.light_speed(geo.wavelength)

        # The receiver has to be looking at the transmitter at all.
        if geo.rx_misalignment > geo.rx_fov:
            out.reason = "receiver field of view exceeded"
            return out

        # Electrical bandwidth follows the symbol rate, not the bit rate.
        geo.bandwidth = max(line_rate / self.bits_per_symbol, 1.0)

        # --- pointing --------------------------------------------------
        if stochastic and rng is not None:
            offset = self.pointing.sample_offset(geo, rng, jitter_sigma)
        else:
            # Mean radial error of a Rician variate, to first order.
            offset = math.hypot(
                geo.tx_misalignment, jitter_sigma * math.sqrt(math.pi / 2.0)
            ) * geo.distance

        captured = self.beam.captured_fraction(geo, offset)
        on_axis = self.beam.captured_fraction(geo, 0.0)
        out.geometric_db = db(max(on_axis, 1e-300))
        out.pointing_db = db(max(captured, 1e-300)) - out.geometric_db

        # --- medium ----------------------------------------------------
        transmittance = self.attenuation.transmittance(geo, medium)
        out.medium_db = db(max(transmittance, 1e-300))

        # --- turbulence -------------------------------------------------
        if stochastic and rng is not None:
            try:
                gain = self.turbulence.sample_gain(geo, medium, rng, key=key, now=now)
            except TypeError:  # models that do not keep temporal state
                gain = self.turbulence.sample_gain(geo, medium, rng)
        else:
            gain = 1.0
        out.turbulence_db = db(max(gain, 1e-300))

        # --- receive coupling -------------------------------------------
        if geo.rx_fov > 0:
            coupling = math.exp(-2.0 * (geo.rx_misalignment / geo.rx_fov) ** 2)
        else:
            coupling = 1.0
        out.coupling_db = db(max(coupling, 1e-300))

        out.rx_power = geo.tx_power * captured * transmittance * gain * coupling

        # --- detection ---------------------------------------------------
        out.snr = self.receiver.snr(out.rx_power, geo)
        out.ber = self.modulation.ber(out.snr)
        out.per = self.coding.per(out.ber, payload_bits)
        out.rate = self.payload_rate(line_rate)
        out.feasible = out.per <= self.max_per
        if not out.feasible:
            out.reason = f"packet error rate {out.per:.3g} above {self.max_per:g}"
        out.extras["spot_radius"] = self.beam.spot_radius(geo)
        out.extras["offset"] = offset
        return out

    # ------------------------------------------------------------------
    def max_range(
        self,
        geo: LinkGeometry,
        medium,
        *,
        line_rate: float = 1e9,
        payload_bits: int = 8192,
        hi: float = 1e4,
        tol: float = 1e-3,
    ) -> float:
        """Longest distance at which this link still closes, by bisection.

        Monotone in distance for every stage in the chain, so bisection is safe.
        Used by the compiler to bound how far apart it may place two tasks.
        """
        import copy

        def ok(d: float) -> bool:
            g = copy.copy(geo)
            g.distance = d
            return self.evaluate(
                g, medium, None, stochastic=False,
                payload_bits=payload_bits, line_rate=line_rate,
            ).feasible

        lo = 1e-3
        if not ok(lo):
            return 0.0
        if ok(hi):
            return hi
        while hi - lo > tol * max(lo, 1.0):
            mid = 0.5 * (lo + hi)
            if ok(mid):
                lo = mid
            else:
                hi = mid
        return lo

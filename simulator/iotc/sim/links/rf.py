"""The omnidirectional RF channel.

Slow, shared and lossy -- and indispensable.  Nothing in the optical plane can
bootstrap itself: before two nodes can aim at each other they must already know
roughly where the other one is and roughly what time it is.  That chicken-and-egg
is what the radio breaks, so it carries discovery, beacons, clock sync, and the
control traffic that reroutes around a dead beam.

In water it barely works at all.  The conductivity term in
:meth:`iotc.sim.medium.Medium.rf_attenuation_db` is brutal at gigahertz
frequencies, which forces a liquid deployment either down to a very low carrier
or onto acoustic signalling.  The simulator reports that honestly rather than
quietly assuming a control plane exists.
"""

from __future__ import annotations

import math

from ...registry import build, register
from ...units import C0, K_B, T_ROOM, db, dbm, from_dbm
from .base import LinkBudget, LinkGeometry, q_function


@register("channel", "rf_omni", "rf")
class RFChannel:
    """Isotropic radio with Friis spreading, shadowing and medium absorption."""

    def __init__(
        self,
        coding="none",
        noise_figure_db: float = 6.0,
        shadowing_sigma_db: float = 4.0,
        implementation_loss_db: float = 3.0,
        max_per: float = 0.5,
    ) -> None:
        self.coding = build(coding, "coding")
        self.noise_figure_db = float(noise_figure_db)
        self.shadowing_sigma_db = float(shadowing_sigma_db)
        self.implementation_loss_db = float(implementation_loss_db)
        self.max_per = float(max_per)

    # ------------------------------------------------------------------
    def evaluate(
        self,
        distance: float,
        radio,
        medium,
        rng=None,
        *,
        payload_bits: int = 512,
        stochastic: bool = True,
    ) -> LinkBudget:
        """Evaluate an RF hop between two nodes carrying ``radio`` specs."""
        out = LinkBudget()
        distance = max(distance, 1e-3)
        wavelength = C0 / radio.frequency
        out.latency = distance / medium.light_speed(wavelength)

        # Friis, floored at zero: inside a wavelength the formula goes negative
        # and would hand out free gain.  That matters here because a subsea link
        # runs at tens of kilohertz, where the wavelength is kilometres and every
        # useful range is deep in the near field.
        fspl_db = max(0.0, 20.0 * math.log10(4.0 * math.pi * distance / wavelength))
        medium_db = medium.rf_attenuation_db(radio.frequency, distance)
        shadow_db = 0.0
        if stochastic and rng is not None and self.shadowing_sigma_db > 0:
            shadow_db = float(rng.normal(0.0, self.shadowing_sigma_db))

        rx_dbm = (
            dbm(radio.tx_power)
            + 2.0 * radio.antenna_gain_db
            - fspl_db
            - medium_db
            - shadow_db
            - self.implementation_loss_db
        )
        out.rx_power = from_dbm(rx_dbm)
        out.geometric_db = -fspl_db
        out.medium_db = -medium_db
        out.pointing_db = 0.0  # the whole point of the radio
        out.turbulence_db = -shadow_db

        if rx_dbm < radio.sensitivity_dbm:
            out.reason = f"below sensitivity ({rx_dbm:.1f} < {radio.sensitivity_dbm:.1f} dBm)"
            out.per = 1.0
            return out

        noise_w = K_B * T_ROOM * radio.line_rate * (10.0 ** (self.noise_figure_db / 10.0))
        out.snr = out.rx_power / max(noise_w, 1e-300)
        # Coherent BPSK, the usual assumption for a low-rate control radio.
        out.ber = float(min(0.5, q_function(math.sqrt(2.0 * max(out.snr, 0.0)))))
        out.per = self.coding.per(out.ber, payload_bits)
        out.rate = radio.line_rate * self.coding.code_rate
        out.feasible = out.per <= self.max_per
        if not out.feasible:
            out.reason = f"packet error rate {out.per:.3g} above {self.max_per:g}"
        return out

    # ------------------------------------------------------------------
    def max_range(self, radio, medium, *, payload_bits: int = 512,
                  hi: float = 1e4, tol: float = 1e-3) -> float:
        """Longest usable RF hop, by bisection."""

        def ok(d: float) -> bool:
            return self.evaluate(
                d, radio, medium, None, payload_bits=payload_bits, stochastic=False
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

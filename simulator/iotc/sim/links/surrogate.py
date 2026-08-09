"""A memoizing surrogate over the high-fidelity channel.

Scale here does *not* come from a cheaper physical model.  It comes from
noticing that the expensive half of a link evaluation -- the beam overlap
integral and the extinction along the path -- depends only on the geometry
``(distance, offset angle)`` and not on the random draws.  So that half is
tabulated once per configuration and interpolated afterwards, while the noise,
BER and coding algebra stay exact and stochastic on every packet.

The result is the same model, evaluated faster, and
:meth:`ChannelSurrogate.verify` measures exactly how much accuracy the
interpolation costs so the claim can be checked rather than asserted.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from ...registry import build, register
from ...units import db
from .base import LinkBudget, LinkGeometry


@register("channel", "surrogate", "fso_surrogate")
class ChannelSurrogate:
    """Wraps another channel and caches its optical throughput term."""

    def __init__(
        self,
        base="fso",
        n_distance: int = 96,
        n_offset: int = 48,
        min_distance: float = 0.05,
        max_distance: float = 500.0,
        max_offset: float = 0.05,  # rad
        **base_kwargs,
    ) -> None:
        self.base = build(base, "channel", **base_kwargs)
        self.n_distance = int(n_distance)
        self.n_offset = int(n_offset)
        self.min_distance = float(min_distance)
        self.max_distance = float(max_distance)
        self.max_offset = float(max_offset)
        self._tables: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------
    @property
    def bits_per_symbol(self) -> float:
        return self.base.bits_per_symbol

    def payload_rate(self, line_rate: float) -> float:
        return self.base.payload_rate(line_rate)

    # ------------------------------------------------------------------
    def _key(self, geo: LinkGeometry, medium) -> tuple:
        return (
            id(medium),
            round(geo.wavelength, 15),
            round(geo.divergence, 12),
            round(geo.tx_aperture_radius, 12),
            round(geo.rx_aperture_radius, 12),
        )

    def _table(self, geo: LinkGeometry, medium):
        key = self._key(geo, medium)
        table = self._tables.get(key)
        if table is not None:
            self.hits += 1
            return table
        self.misses += 1

        dists = np.geomspace(self.min_distance, self.max_distance, self.n_distance)
        offsets = np.linspace(0.0, self.max_offset, self.n_offset)
        grid = np.empty((self.n_distance, self.n_offset), dtype=float)

        probe = copy.copy(geo)
        for i, d in enumerate(dists):
            probe.distance = float(d)
            trans = self.base.attenuation.transmittance(probe, medium)
            for j, ang in enumerate(offsets):
                captured = self.base.beam.captured_fraction(probe, float(ang) * float(d))
                grid[i, j] = math.log(max(captured * trans, 1e-300))

        table = (dists, offsets, grid)
        self._tables[key] = table
        return table

    @staticmethod
    def _bilinear(dists, offsets, grid, d: float, ang: float) -> float:
        """Interpolate in (log distance, offset), clamping outside the table."""
        ld = math.log(max(d, 1e-12))
        lds = np.log(dists)
        i = int(np.clip(np.searchsorted(lds, ld) - 1, 0, len(dists) - 2))
        j = int(np.clip(np.searchsorted(offsets, ang) - 1, 0, len(offsets) - 2))
        t = np.clip((ld - lds[i]) / (lds[i + 1] - lds[i]), 0.0, 1.0)
        u_den = offsets[j + 1] - offsets[j]
        u = np.clip((ang - offsets[j]) / u_den, 0.0, 1.0) if u_den > 0 else 0.0
        g = grid
        return float(
            (1 - t) * (1 - u) * g[i, j]
            + t * (1 - u) * g[i + 1, j]
            + (1 - t) * u * g[i, j + 1]
            + t * u * g[i + 1, j + 1]
        )

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
        base = self.base
        out = LinkBudget()
        out.latency = geo.distance / medium.light_speed(geo.wavelength)

        if geo.rx_misalignment > geo.rx_fov:
            out.reason = "receiver field of view exceeded"
            return out

        # Outside the tabulated envelope, fall back to the exact model rather
        # than extrapolating; correctness beats speed at the edges.
        if not (self.min_distance <= geo.distance <= self.max_distance):
            return base.evaluate(
                geo, medium, rng, jitter_sigma=jitter_sigma, payload_bits=payload_bits,
                line_rate=line_rate, stochastic=stochastic, key=key, now=now,
            )

        geo.bandwidth = max(line_rate / base.bits_per_symbol, 1.0)

        if stochastic and rng is not None:
            offset_m = base.pointing.sample_offset(geo, rng, jitter_sigma)
        else:
            offset_m = math.hypot(
                geo.tx_misalignment, jitter_sigma * math.sqrt(math.pi / 2.0)
            ) * geo.distance
        offset_ang = offset_m / max(geo.distance, 1e-12)

        dists, offsets, grid = self._table(geo, medium)
        if offset_ang > self.max_offset:
            eta = 0.0
        else:
            eta = math.exp(self._bilinear(dists, offsets, grid, geo.distance, offset_ang))

        if stochastic and rng is not None:
            try:
                gain = base.turbulence.sample_gain(geo, medium, rng, key=key, now=now)
            except TypeError:
                gain = base.turbulence.sample_gain(geo, medium, rng)
        else:
            gain = 1.0

        coupling = (
            math.exp(-2.0 * (geo.rx_misalignment / geo.rx_fov) ** 2)
            if geo.rx_fov > 0 else 1.0
        )

        out.rx_power = geo.tx_power * eta * gain * coupling
        on_axis = math.exp(self._bilinear(dists, offsets, grid, geo.distance, 0.0))
        out.geometric_db = db(max(on_axis, 1e-300))
        out.pointing_db = db(max(eta, 1e-300)) - out.geometric_db
        out.medium_db = 0.0  # folded into the tabulated term
        out.turbulence_db = db(max(gain, 1e-300))
        out.coupling_db = db(max(coupling, 1e-300))

        out.snr = base.receiver.snr(out.rx_power, geo)
        out.ber = base.modulation.ber(out.snr)
        out.per = base.coding.per(out.ber, payload_bits)
        out.rate = base.payload_rate(line_rate)
        out.feasible = out.per <= base.max_per
        if not out.feasible:
            out.reason = f"packet error rate {out.per:.3g} above {base.max_per:g}"
        out.extras["surrogate"] = True
        return out

    def max_range(self, geo, medium, **kwargs) -> float:
        return self.base.max_range(geo, medium, **kwargs)

    # ------------------------------------------------------------------
    def verify(self, geo: LinkGeometry, medium, n: int = 200, seed: int = 0) -> dict:
        """Compare the surrogate against the exact model on random geometries.

        Returns the worst and RMS error in received power, in dB.  A surrogate
        that has not been checked is just an unvalidated approximation, so this
        is wired into the CLI as ``--verify-surrogate``.
        """
        rng = np.random.default_rng(seed)
        errors = []
        for _ in range(n):
            probe = copy.copy(geo)
            probe.distance = float(
                np.exp(rng.uniform(math.log(self.min_distance), math.log(self.max_distance)))
            )
            probe.tx_misalignment = float(rng.uniform(0.0, self.max_offset * 0.9))
            exact = self.base.evaluate(probe, medium, None, stochastic=False)
            approx = self.evaluate(copy.copy(probe), medium, None, stochastic=False)
            if exact.rx_power <= 0.0 and approx.rx_power <= 0.0:
                continue
            errors.append(db(max(approx.rx_power, 1e-300)) - db(max(exact.rx_power, 1e-300)))
        arr = np.asarray(errors) if errors else np.zeros(1)
        return {
            "samples": len(errors),
            "max_abs_db": float(np.max(np.abs(arr))),
            "rms_db": float(np.sqrt(np.mean(arr**2))),
            "bias_db": float(np.mean(arr)),
            "table_entries": len(self._tables),
        }

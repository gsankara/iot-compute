"""Clocks, drift, and the guard bands they force.

The premise of this machine is that computation *is* interaction timing: two
devices exchange data because both were told to point at each other during slot
17, not because either asked.  That only works if they agree on when slot 17 is,
so the disagreement has to be modelled rather than assumed away.

A node's local clock runs at ``1 + drift`` times real time, with drift drawn
once per device from its class specification and held (a crystal's offset is a
manufacturing property, not noise).  Reading the clock adds jitter.  Periodic
beacons over the radio pull the offset back toward truth, with a residual error
set by how well the beacon's own flight time is known.

:meth:`Clock.uncertainty` is the quantity the scheduler actually needs: the
bound on how far this node's idea of "now" can be from anyone else's.  Guard
bands are sized from it, and a schedule whose slots are shorter than the guard
band it needs is simply infeasible.
"""

from __future__ import annotations

import math

from ..units import ppm as PPM


class Clock:
    """One device's local time base."""

    def __init__(
        self,
        drift_ppm: float = 20.0,
        jitter: float = 1e-6,
        rng=None,
        initial_offset: float = 0.0,
        fixed_drift: float | None = None,
    ) -> None:
        self.nominal_drift_ppm = float(drift_ppm)
        self.jitter = float(jitter)
        self._rng = rng
        if fixed_drift is not None:
            self.drift = float(fixed_drift)
        elif rng is not None:
            # A crystal's error is a fixed property of the part, so draw once.
            self.drift = float(rng.normal(0.0, drift_ppm * PPM))
        else:
            self.drift = drift_ppm * PPM
        self.offset = float(initial_offset)
        self.last_sync = 0.0
        self.sync_error = 0.0
        self.syncs = 0

    # ------------------------------------------------------------------
    def local_time(self, true_time: float) -> float:
        """This node's belief about the current time, before reading noise."""
        return true_time * (1.0 + self.drift) + self.offset

    def read(self, true_time: float) -> float:
        """Read the clock, including per-read jitter."""
        t = self.local_time(true_time)
        if self.jitter > 0.0 and self._rng is not None:
            t += float(self._rng.normal(0.0, self.jitter))
        return t

    def error(self, true_time: float) -> float:
        """Signed difference between this clock and the truth."""
        return self.local_time(true_time) - true_time

    # ------------------------------------------------------------------
    def resync(self, true_time: float, accuracy: float = 1e-5) -> None:
        """Discipline the clock from a beacon of the given accuracy.

        The offset is corrected but the drift is not: the node learns what time
        it is, not how fast its crystal runs, so the error starts growing again
        immediately.  That is what makes the resync *interval* a design choice.
        """
        residual = 0.0
        if accuracy > 0.0 and self._rng is not None:
            residual = float(self._rng.normal(0.0, accuracy))
        self.offset = true_time + residual - true_time * (1.0 + self.drift)
        self.last_sync = true_time
        self.sync_error = abs(residual)
        self.syncs += 1

    def uncertainty(self, true_time: float) -> float:
        """Bound on |local - true| given how long it has been since a resync."""
        elapsed = max(true_time - self.last_sync, 0.0)
        return abs(self.nominal_drift_ppm * PPM) * elapsed + self.sync_error + 3.0 * self.jitter


class ClockDomain:
    """The whole swarm's timing, plus the guard band it implies."""

    def __init__(self, resync_interval: float = 1.0, beacon_accuracy: float = 1e-5) -> None:
        self.clocks: dict[int, Clock] = {}
        self.resync_interval = float(resync_interval)
        self.beacon_accuracy = float(beacon_accuracy)
        self._next_resync = 0.0

    def add(self, device_id: int, clock: Clock) -> Clock:
        self.clocks[device_id] = clock
        return clock

    def step(self, true_time: float, reachable=None) -> int:
        """Run any due beacon resync.  Returns how many clocks were corrected.

        ``reachable`` optionally restricts the resync to the devices the beacon
        actually reaches; nodes out of radio range keep drifting, which is
        exactly what should happen when a swarm spreads out.
        """
        if self.resync_interval <= 0.0 or true_time < self._next_resync:
            return 0
        self._next_resync = true_time + self.resync_interval
        n = 0
        for dev_id, clock in self.clocks.items():
            if reachable is not None and dev_id not in reachable:
                continue
            clock.resync(true_time, self.beacon_accuracy)
            n += 1
        return n

    # ------------------------------------------------------------------
    def max_uncertainty(self, true_time: float) -> float:
        if not self.clocks:
            return 0.0
        return max(c.uncertainty(true_time) for c in self.clocks.values())

    def max_skew(self, true_time: float) -> float:
        """Worst pairwise disagreement right now -- the quantity that breaks slots."""
        if len(self.clocks) < 2:
            return 0.0
        errors = [c.error(true_time) for c in self.clocks.values()]
        return max(errors) - min(errors)

    def guard_band(self, true_time: float, safety: float = 2.0) -> float:
        """Slot guard band [s] that keeps two nodes from overlapping.

        Both ends can be wrong in opposite directions, hence the factor of two,
        and ``safety`` covers the tail beyond the nominal drift spec.
        """
        return safety * 2.0 * self.max_uncertainty(true_time)

    @staticmethod
    def required_guard(drift_ppm: float, interval: float, beacon_accuracy: float = 1e-5,
                       jitter: float = 1e-6, safety: float = 2.0) -> float:
        """Guard band for a *planned* schedule, before any device exists.

        The compiler calls this: given the worst clock in the fleet and the
        chosen resync interval, this is the dead time every slot must carry.
        """
        worst = abs(drift_ppm) * PPM * interval + beacon_accuracy + 3.0 * jitter
        return safety * 2.0 * worst

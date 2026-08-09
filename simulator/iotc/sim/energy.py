"""Energy: harvesting, storage, and where the joules go.

Energy is a hard constraint here, not an afterthought.  A node with an empty
supercap does not run slowly, it stops -- and it stops in the middle of a
time-triggered schedule that other nodes are counting on, which is one of the
more interesting failure modes this simulator exists to expose.

One coupling is worth calling out because it links the two halves of the design.
In a liquid medium the ambient light a photovoltaic cell sees falls off
exponentially with depth, using the *same* extinction coefficient that governs
the optical links.  So descending buys nothing for free: it costs both link
budget and income.  The placement pass has to know that.
"""

from __future__ import annotations

import math

import numpy as np

from ..registry import register
from ..units import W


# --------------------------------------------------------------------------
# harvesting
# --------------------------------------------------------------------------
class Harvester:
    """Energy in."""

    def power(self, spec, position: np.ndarray, medium, now: float, env=None) -> float:
        """Instantaneous harvested power [W]."""
        raise NotImplementedError


@register("harvester", "photovoltaic", "solar", "pv", "ambient_light")
class Photovoltaic(Harvester):
    """Ambient light onto a small cell.

    ``surface_irradiance`` is the level at ``surface_z``; below that the light is
    attenuated by the medium at ``band`` (550 nm by default, near the peak of
    both daylight and silicon response).  ``diurnal_period`` of zero means a
    constant source, which is what most experiments want; setting it produces a
    day/night cycle that forces the schedule to bank energy.
    """

    def __init__(
        self,
        surface_irradiance: float = 600.0,  # W/m^2, overcast daylight
        surface_z: float = 0.0,
        band: float = 550e-9,
        diurnal_period: float = 0.0,
        night_fraction: float = 0.02,
        shading: float = 0.0,
    ) -> None:
        self.surface_irradiance = float(surface_irradiance)
        self.surface_z = float(surface_z)
        self.band = float(band)
        self.diurnal_period = float(diurnal_period)
        self.night_fraction = float(night_fraction)
        self.shading = float(shading)

    def irradiance_at(self, position: np.ndarray, medium, now: float) -> float:
        level = self.surface_irradiance
        if self.diurnal_period > 0.0:
            phase = math.sin(2.0 * math.pi * now / self.diurnal_period)
            level *= max(self.night_fraction, phase)
        depth = max(self.surface_z - float(position[2]), 0.0)
        if depth > 0.0:
            level *= math.exp(-medium.extinction(self.band) * depth)
        return level * (1.0 - self.shading)

    def power(self, spec, position, medium, now: float, env=None) -> float:
        return self.irradiance_at(position, medium, now) * spec.cell_area * spec.efficiency


@register("harvester", "rf_rectenna", "wireless_charging", "rectenna")
class RFRectenna(Harvester):
    """Wireless power beamed from fixed chargers.

    Unlike light this can be aimed and scheduled, but it falls off as 1/r^2 and,
    in salt water, is absorbed so hard that it is useless at any range -- which
    is precisely the sort of "this deployment cannot be powered that way"
    conclusion the simulator should reach on its own.
    """

    def __init__(
        self,
        chargers=((np.zeros(3), 10.0),),
        frequency: float = 915e6,
        polarisation_loss_db: float = 3.0,
    ) -> None:
        self.chargers = [(np.asarray(p, dtype=float), float(e)) for p, e in chargers]
        self.frequency = float(frequency)
        self.polarisation_loss_db = float(polarisation_loss_db)

    def power(self, spec, position, medium, now: float, env=None) -> float:
        total = 0.0
        pol = 10.0 ** (-self.polarisation_loss_db / 10.0)
        for centre, eirp in self.chargers:
            d = float(np.linalg.norm(np.asarray(position, dtype=float) - centre))
            d = max(d, 0.05)
            density = eirp / (4.0 * math.pi * d**2)
            loss_db = medium.rf_attenuation_db(self.frequency, d)
            density *= 10.0 ** (-loss_db / 10.0)
            total += density * spec.rectenna_area * spec.rectifier_efficiency * pol
        return total


@register("harvester", "none", "off", "tethered")
class NoHarvest(Harvester):
    """No income.  Runs the swarm purely off its stored energy."""

    def power(self, spec, position, medium, now: float, env=None) -> float:
        return 0.0


@register("harvester", "constant", "mains")
class ConstantHarvest(Harvester):
    """A fixed supply -- an anchor on shore power, or an idealised baseline."""

    def __init__(self, power_w: float = 1.0 * W) -> None:
        self.power_w = float(power_w)

    def power(self, spec, position, medium, now: float, env=None) -> float:
        return self.power_w


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------
class EnergyStore:
    """Energy buffer with an undervoltage lockout.

    The lockout needs hysteresis, and getting that wrong hides a whole class of
    failure.  Without it, a node that runs out simply has its draws refused, the
    charge pins a hair above the threshold, and it reports itself alive forever
    while doing no work.  Real hardware latches off at ``brownout_fraction`` and
    refuses to restart until the buffer has climbed back to
    ``recovery_fraction``, so that is what this does -- and a node that cannot
    reach the recovery level stays dark, which is the honest outcome.
    """

    #: Charge level, as a fraction of capacity, needed to come back up.
    recovery_multiple = 3.0

    def __init__(self, spec) -> None:
        self.capacity = float(spec.capacity)
        self.energy = float(spec.capacity * spec.initial_fraction)
        self.leakage_power = float(spec.leakage_power)
        self.brownout_energy = float(spec.capacity * spec.brownout_fraction)
        self.recovery_energy = min(
            self.capacity, self.brownout_energy * self.recovery_multiple
        )
        self.total_harvested = 0.0
        self.total_consumed = 0.0
        self.total_wasted = 0.0  # harvest arriving at a full buffer
        self.brownouts = 0
        self._locked = self.energy <= self.brownout_energy

    @property
    def fraction(self) -> float:
        return self.energy / self.capacity if self.capacity > 0 else 0.0

    @property
    def alive(self) -> bool:
        """False while the undervoltage lockout is latched."""
        return not self._locked

    def harvest(self, power: float, dt: float) -> float:
        """Add ``power * dt`` joules, returning what actually fit."""
        amount = max(power, 0.0) * dt
        room = self.capacity - self.energy
        stored = min(amount, room)
        self.energy += stored
        self.total_harvested += stored
        self.total_wasted += amount - stored
        if self._locked and self.energy >= self.recovery_energy:
            self._locked = False
        return stored

    def draw(self, joules: float) -> bool:
        """Spend energy.  Returns False (and spends nothing) if there is not enough."""
        if self._locked:
            return False
        if joules <= 0.0:
            return True
        if self.energy - joules <= self.brownout_energy:
            self.brownouts += 1
            self._locked = True
            return False
        self.energy -= joules
        self.total_consumed += joules
        return True

    def leak(self, dt: float) -> None:
        """Leakage drains regardless of the lockout; a dead node still bleeds."""
        self.energy = max(0.0, self.energy - self.leakage_power * dt)
        if self.energy <= self.brownout_energy:
            self._locked = True

    def snapshot(self) -> dict:
        return {
            "energy": self.energy,
            "fraction": self.fraction,
            "harvested": self.total_harvested,
            "consumed": self.total_consumed,
            "wasted": self.total_wasted,
            "brownouts": self.brownouts,
            "alive": self.alive,
        }


@register("store", "supercap", "capacitor")
class Supercapacitor(EnergyStore):
    """The default: small, leaky, and fast to charge."""


@register("store", "battery", "cell")
class Battery(EnergyStore):
    """Larger and far less leaky, but limited in charge rate.

    The rate limit is the interesting part: a battery cannot swallow a burst of
    harvest the way a supercap can, so a node that only sees light briefly may
    bank less energy from a battery than from a capacitor a tenth its size.
    """

    def __init__(self, spec, max_charge_power: float = 0.05) -> None:
        super().__init__(spec)
        self.leakage_power = float(spec.leakage_power) * 0.05
        self.max_charge_power = float(max_charge_power)

    def harvest(self, power: float, dt: float) -> float:
        return super().harvest(min(power, self.max_charge_power), dt)


# --------------------------------------------------------------------------
# accounting
# --------------------------------------------------------------------------
class PowerAccount:
    """Per-subsystem energy tally, so a report can say *where* the joules went."""

    SUBSYSTEMS = ("mcu", "optical_tx", "optical_rx", "radio", "gimbal", "motion", "leak")

    def __init__(self) -> None:
        self.totals = {k: 0.0 for k in self.SUBSYSTEMS}

    def add(self, subsystem: str, joules: float) -> None:
        self.totals[subsystem] = self.totals.get(subsystem, 0.0) + joules

    @property
    def total(self) -> float:
        return sum(self.totals.values())

    def breakdown(self) -> dict:
        total = self.total or 1.0
        return {k: v / total for k, v in self.totals.items()}

    def merge(self, other: "PowerAccount") -> None:
        for k, v in other.totals.items():
            self.totals[k] = self.totals.get(k, 0.0) + v

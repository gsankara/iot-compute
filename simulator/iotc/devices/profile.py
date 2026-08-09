"""Device capability descriptions.

A device class is pure data: no behaviour, no simulator state.  The simulator
reads it, the compiler reads it, and the two therefore cannot disagree about
what a node can do.

Heterogeneity is the point of this file.  Real deployments will mix a
compute-rich part, a memory-rich part and a part with a much finer steering
mirror, and every downstream pass -- tiling, placement, beam assignment,
scheduling -- has to cope with that.  So the natural unit is a *class*, and a
swarm is a bag of classes rather than N copies of one node.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from ..geom import AngularGrid
from ..units import GHz, KiB, MHz, MiB, deg, mm, mW, uJ, uW


@dataclass(frozen=True)
class ComputeSpec:
    """The processor and its memories."""

    cores: int = 1
    clock_hz: float = 120 * MHz
    ops_per_cycle: float = 1.0
    #: Cost of one 32-bit float operation in cycles; MCUs without an FPU pay more.
    cycles_per_flop: float = 1.0
    ram_bytes: float = 512 * KiB
    flash_bytes: float = 2 * MiB

    @property
    def flops(self) -> float:
        """Sustained float throughput [FLOP/s]."""
        return self.cores * self.clock_hz * self.ops_per_cycle / self.cycles_per_flop

    def time_for(self, flop: float) -> float:
        """Wall time [s] to execute ``flop`` floating point operations."""
        return flop / self.flops if self.flops > 0 else math.inf


@dataclass(frozen=True)
class GimbalSpec:
    """The steering mechanism for one optical transceiver.

    ``grid`` carries the quantization and the sweep limits; the rest is what it
    costs to move.  Retargeting is never free: it takes time the scheduler must
    budget for and energy the harvester must supply.
    """

    grid: AngularGrid = field(default_factory=AngularGrid)
    slew_rate: float = 200 * deg  # rad/s
    settle_time: float = 1e-3  # s, after the last step
    energy_per_step: float = 5 * uJ
    #: Standing pointing jitter (1-sigma, rad) from platform vibration.
    jitter_sigma: float = 0.1 * deg

    def retarget_time(self, n_steps: int) -> float:
        """Time [s] to traverse ``n_steps`` ticks and settle."""
        if n_steps <= 0:
            return 0.0
        travel = n_steps * max(self.grid.step_az, self.grid.step_el) / self.slew_rate
        return travel + self.settle_time

    def retarget_energy(self, n_steps: int) -> float:
        """Energy [J] to traverse ``n_steps`` ticks."""
        return max(n_steps, 0) * self.energy_per_step


@dataclass(frozen=True)
class OpticalSpec:
    """The free-space photonic transceiver bank."""

    n_transceivers: int = 2
    wavelength: float = 850e-9
    tx_power: float = 1.0 * mW  # optical power out of the aperture
    #: Half-angle 1/e^2 divergence of the transmit beam.
    divergence: float = 5e-3  # rad
    tx_aperture_radius: float = 1.5 * mm
    rx_aperture_radius: float = 2.5 * mm
    #: Receiver acceptance half-angle.  Beyond it the incoming beam misses the
    #: detector entirely, however much power arrives at the aperture.
    rx_fov: float = 15e-3  # rad
    line_rate: float = 1e9  # bit/s on the wire when the link is up
    #: Wall-plug efficiency of the laser driver.
    tx_efficiency: float = 0.2
    rx_power: float = 15 * mW  # receiver front end, while listening
    gimbal: GimbalSpec = field(default_factory=GimbalSpec)

    @property
    def tx_electrical_power(self) -> float:
        """Electrical draw [W] while transmitting."""
        return self.tx_power / max(self.tx_efficiency, 1e-9)


@dataclass(frozen=True)
class RadioSpec:
    """The omnidirectional RF side channel.

    Low rate and shared, so it is no good for bulk data -- but it needs no
    pointing, which makes it the right carrier for discovery, beacons, clock
    sync and the control plane that recovers from a lost optical link.
    """

    frequency: float = 2.4 * GHz
    tx_power: float = 1.0 * mW
    line_rate: float = 1e6  # bit/s
    antenna_gain_db: float = 0.0
    sensitivity_dbm: float = -95.0
    tx_electrical_power: float = 30 * mW
    rx_power: float = 12 * mW


@dataclass(frozen=True)
class HarvestSpec:
    """Energy in."""

    kind: str = "photovoltaic"  # resolved through the ``harvester`` registry
    #: Photovoltaic cell area [m^2] and conversion efficiency.
    cell_area: float = 10e-4
    efficiency: float = 0.20
    #: Rectenna effective aperture [m^2] and RF-DC conversion efficiency.
    rectenna_area: float = 4e-4
    rectifier_efficiency: float = 0.35


@dataclass(frozen=True)
class StoreSpec:
    """Energy buffer."""

    capacity: float = 5.4  # J -- a 1 F supercap at 3.3 V
    initial_fraction: float = 0.6
    leakage_power: float = 20 * uW
    #: Below this fraction the node browns out and stops executing.
    brownout_fraction: float = 0.05


@dataclass(frozen=True)
class PowerSpec:
    """Everything that draws current, other than the radios and the gimbal."""

    mcu_active: float = 100 * mW
    mcu_idle: float = 10 * mW
    mcu_sleep: float = 20 * uW
    #: Propulsion while translating in the xy-plane.
    thruster_power: float = 500 * mW
    #: Ballast pump while taking on or expelling water.
    ballast_power: float = 200 * mW


@dataclass(frozen=True)
class BodySpec:
    """Mass properties and actuation limits.

    ``ballast_capacity`` is what makes the liquid-medium variant interesting: a
    device changes depth by taking on or expelling water, which is both slow to
    actuate and strictly rate-limited, so the z axis behaves nothing like x and
    y.  A drone-style body leaves it at zero and uses ``max_thrust_z`` instead.
    """

    mass: float = 0.113  # kg -- neutrally buoyant sphere of r = 3 cm in water
    radius: float = 30 * mm
    drag_coefficient: float = 0.47  # sphere
    max_thrust_xy: float = 0.15  # N
    max_thrust_z: float = 0.0  # N -- zero for a buoyancy-driven platform
    ballast_capacity: float = 5e-3  # kg of water
    ballast_rate: float = 2e-4  # kg/s -- 25 s to swing the full range

    @property
    def frontal_area(self) -> float:
        return math.pi * self.radius**2

    @property
    def volume(self) -> float:
        return 4.0 / 3.0 * math.pi * self.radius**3


@dataclass(frozen=True)
class TetherSpec:
    """A remote transceiver head on a wire or waveguide.

    The compute box, the battery and the harvester stay *outside* the medium and
    only the small optical head is immersed, which buys two things.  Density: the
    volume is filled with 8 mm heads instead of 60 mm floats, so far more links
    fit in the same space.  And control authority: xyz comes from a gantry and a
    winch topside rather than from thrusters and ballast, so positioning is fast,
    precise and costs the node nothing.

    It is not free.  Every head trails a fibre back to the surface, and those
    fibres are opaque line obstacles -- a dense field of them blocks exactly the
    horizontal beams the placement pass most wants to use.  ``radius`` is what
    makes that cost real in the occlusion test.
    """

    #: Where the wire leaves the topside rig, relative to the head's column.
    anchor_z: float = 0.0
    #: Maximum payout, so the head is confined to a cylinder below the gantry.
    length: float = 30.0
    #: Physical radius of the wire/fibre, for line-of-sight occlusion.
    radius: float = 1.5 * mm
    #: Gantry translation rate and winch rate.
    max_speed_xy: float = 0.8  # m/s
    max_speed_z: float = 0.5  # m/s
    #: Insertion loss of the waveguide run between box and head.
    waveguide_loss_db: float = 1.5
    #: Gantry travel limits in x and y, or None for unrestricted.
    xy_limits: tuple[float, float, float, float] | None = None

    @property
    def transmittance(self) -> float:
        """Fraction of transmit power that survives the run out to the head."""
        return 10.0 ** (-self.waveguide_loss_db / 10.0)


@dataclass(frozen=True)
class DeviceClass:
    """A complete device capability profile."""

    name: str
    compute: ComputeSpec = field(default_factory=ComputeSpec)
    optical: OpticalSpec = field(default_factory=OpticalSpec)
    radio: RadioSpec = field(default_factory=RadioSpec)
    harvest: HarvestSpec = field(default_factory=HarvestSpec)
    store: StoreSpec = field(default_factory=StoreSpec)
    power: PowerSpec = field(default_factory=PowerSpec)
    body: BodySpec = field(default_factory=BodySpec)
    #: Set when only a remote head is immersed and the box sits outside.
    tether: TetherSpec | None = None
    #: Which ``kinematics`` plugin moves this body.
    kinematics: str = "buoyant"
    #: Clock stability, used to size scheduling guard bands.
    clock_drift_ppm: float = 20.0
    clock_jitter: float = 1e-6  # s

    def variant(self, name: str, **overrides) -> "DeviceClass":
        """A copy under a new name with some fields replaced."""
        return replace(self, name=name, **overrides)

    def summary(self) -> str:
        o = self.optical
        g = o.gimbal.grid
        return (
            f"{self.name:9s} "
            f"{self.compute.flops / 1e6:7.1f} MFLOP/s  "
            f"{self.compute.ram_bytes / KiB:7.0f} KiB RAM  "
            f"{o.n_transceivers}x optical @ {o.line_rate / 1e9:.2f} Gb/s  "
            f"step {g.step_az / deg:.3f} deg  "
            f"{self.kinematics}"
        )

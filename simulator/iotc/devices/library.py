"""The stock device classes.

Five parts that are deliberately *not* interchangeable.  A swarm built only from
``mote-c`` cannot hold a large tile; a swarm built only from ``mote-m`` computes
slowly; neither can retarget a beam precisely enough for a long hop without a
``mote-s`` or a ``relay`` in the path.  Getting a workload to run therefore
requires the compiler to actually reason about which task goes where.
"""

from __future__ import annotations

import math

from ..geom import AngularGrid
from ..units import GHz, KiB, MHz, MiB, deg, mm, mW, uJ, uW
from .profile import (
    BodySpec,
    ComputeSpec,
    DeviceClass,
    GimbalSpec,
    HarvestSpec,
    OpticalSpec,
    PowerSpec,
    RadioSpec,
    StoreSpec,
    TetherSpec,
)

# --------------------------------------------------------------------------
# gimbals: the steering granularity is the axis of heterogeneity that matters
# most for the optical link budget, so give it its own vocabulary.
# --------------------------------------------------------------------------
COARSE_GIMBAL = GimbalSpec(
    grid=AngularGrid(step_az=0.5 * deg, step_el=0.5 * deg,
                     el_limits=(-60 * deg, 60 * deg)),
    slew_rate=300 * deg,
    settle_time=2e-3,
    energy_per_step=8 * uJ,
    jitter_sigma=0.15 * deg,
)

STANDARD_GIMBAL = GimbalSpec(
    grid=AngularGrid(step_az=0.1 * deg, step_el=0.1 * deg,
                     el_limits=(-75 * deg, 75 * deg)),
    slew_rate=200 * deg,
    settle_time=1e-3,
    energy_per_step=5 * uJ,
    jitter_sigma=0.08 * deg,
)

#: Fine MEMS mirror with an unrestricted sweep -- the whole sphere, 0.02 deg ticks.
FINE_GIMBAL = GimbalSpec(
    grid=AngularGrid(step_az=0.02 * deg, step_el=0.02 * deg,
                     az_limits=None, el_limits=(-90 * deg, 90 * deg)),
    slew_rate=120 * deg,
    settle_time=0.5e-3,
    energy_per_step=3 * uJ,
    jitter_sigma=0.02 * deg,
)


# --------------------------------------------------------------------------
# the classes
# --------------------------------------------------------------------------
#: Compute-rich: a fast core, but only a small working set and coarse steering.
MOTE_C = DeviceClass(
    name="mote-c",
    compute=ComputeSpec(cores=1, clock_hz=480 * MHz, ops_per_cycle=1.0,
                        cycles_per_flop=1.0, ram_bytes=1 * MiB, flash_bytes=2 * MiB),
    optical=OpticalSpec(n_transceivers=2, gimbal=STANDARD_GIMBAL),
    power=PowerSpec(mcu_active=180 * mW),
)

#: Memory-rich: external PSRAM holds a big tile, but the core is slow and the
#: extra memory leaks, so it wants a bigger buffer to ride out dark periods.
MOTE_M = DeviceClass(
    name="mote-m",
    compute=ComputeSpec(cores=1, clock_hz=120 * MHz, ops_per_cycle=1.0,
                        cycles_per_flop=2.0, ram_bytes=8 * MiB, flash_bytes=16 * MiB),
    optical=OpticalSpec(n_transceivers=2, gimbal=COARSE_GIMBAL),
    store=StoreSpec(capacity=10.0),
    power=PowerSpec(mcu_active=90 * mW),
)

#: Steering-rich: fine ticks and a full-sphere sweep, so it can close the long
#: hops nobody else can.  Pays for it in cores and in gimbal duty cycle.
MOTE_S = DeviceClass(
    name="mote-s",
    compute=ComputeSpec(cores=1, clock_hz=120 * MHz, ops_per_cycle=1.0,
                        cycles_per_flop=2.0, ram_bytes=512 * KiB, flash_bytes=2 * MiB),
    optical=OpticalSpec(n_transceivers=3, divergence=2e-3, tx_power=2 * mW,
                        rx_aperture_radius=4 * mm, line_rate=2.5e9,
                        gimbal=FINE_GIMBAL),
    power=PowerSpec(mcu_active=90 * mW),
)

#: Relay: many transceivers and a large buffer, little compute.  The part the
#: facility-location pass picks when it needs an aggregator.
RELAY = DeviceClass(
    name="relay",
    compute=ComputeSpec(cores=1, clock_hz=200 * MHz, cycles_per_flop=2.0,
                        ram_bytes=2 * MiB, flash_bytes=8 * MiB),
    optical=OpticalSpec(n_transceivers=6, tx_power=3 * mW, divergence=3e-3,
                        rx_aperture_radius=5 * mm, line_rate=2.5e9,
                        gimbal=FINE_GIMBAL),
    harvest=HarvestSpec(cell_area=30e-4),
    store=StoreSpec(capacity=25.0),
    # Neutral trim for this hull in seawater is about 18 g, so the tank has to
    # straddle that; a capacity below it would leave the relay permanently
    # positively buoyant and unable to hold any depth at all.
    body=BodySpec(mass=0.30, radius=42 * mm, max_thrust_xy=0.25,
                  ballast_capacity=40e-3, ballast_rate=1.5e-3),
)

#: Anchor: moored, mains-fed, wide sweep.  Provides the fixed reference frame
#: the mobile parts localise and time-sync against.
ANCHOR = DeviceClass(
    name="anchor",
    compute=ComputeSpec(cores=2, clock_hz=1 * GHz, ram_bytes=64 * MiB,
                        flash_bytes=256 * MiB),
    optical=OpticalSpec(n_transceivers=8, tx_power=10 * mW, divergence=1.5e-3,
                        rx_aperture_radius=10 * mm, line_rate=10e9,
                        gimbal=FINE_GIMBAL),
    harvest=HarvestSpec(cell_area=200e-4),
    store=StoreSpec(capacity=500.0, initial_fraction=1.0),
    body=BodySpec(mass=2.0, radius=80 * mm, max_thrust_xy=0.0,
                  ballast_capacity=0.0, ballast_rate=0.0),
    kinematics="static",
    clock_drift_ppm=0.1,  # disciplined oscillator
    clock_jitter=1e-8,
)


LIBRARY: dict[str, DeviceClass] = {
    d.name: d for d in (MOTE_C, MOTE_M, MOTE_S, RELAY, ANCHOR)
}


def get_class(name: str) -> DeviceClass:
    """Look up a stock device class by name."""
    try:
        return LIBRARY[name]
    except KeyError:
        raise KeyError(
            f"unknown device class {name!r}; available: {sorted(LIBRARY)}"
        ) from None


def register_class(cls: DeviceClass) -> DeviceClass:
    """Add a custom device class to the library."""
    LIBRARY[cls.name] = cls
    return cls


def aerial(cls: DeviceClass, name: str | None = None,
           hover_power: float = 6.0, pack_energy: float = 13.3e3,
           cell_area: float = 100e-4) -> DeviceClass:
    """Re-body a class as a free-flying drone rather than a buoyant float.

    Same electronics, different physics: real thrust on all three axes, no
    ballast, and a much lighter airframe -- which makes z as cheap as x and y
    and removes the actuation lag that dominates the liquid variant.

    It also needs a battery, and that is not a detail.  Hovering costs about
    ``hover_power`` watts while a 100 square centimetre cell in full daylight
    harvests barely one, so an aerial swarm is **never** energy neutral: it runs
    on stored charge and has a finite endurance, whereas a neutrally buoyant
    float or a tethered head can hold station indefinitely.  Giving the drone a
    coin-cell supercap instead would simply make it fall out of the sky, which
    is a true statement about the platform and a useless simulation.
    """
    b = cls.body
    return cls.variant(
        name or f"{cls.name}-air",
        kinematics="drone",
        body=BodySpec(
            mass=b.mass * 0.4,
            radius=b.radius,
            drag_coefficient=b.drag_coefficient,
            max_thrust_xy=max(b.max_thrust_xy, 0.1) * 6.0,
            max_thrust_z=max(b.max_thrust_xy, 0.1) * 4.0,
            ballast_capacity=0.0,
            ballast_rate=0.0,
        ),
        harvest=HarvestSpec(kind=cls.harvest.kind, cell_area=cell_area,
                            efficiency=cls.harvest.efficiency),
        store=StoreSpec(capacity=pack_energy, initial_fraction=1.0,
                        leakage_power=cls.store.leakage_power,
                        brownout_fraction=0.03),
        power=PowerSpec(
            mcu_active=cls.power.mcu_active,
            mcu_idle=cls.power.mcu_idle,
            mcu_sleep=cls.power.mcu_sleep,
            thruster_power=hover_power,  # flight is expensive
            ballast_power=0.0,
        ),
    )


def aquatic(cls: DeviceClass, name: str | None = None,
            wavelength: float = 470e-9, radio_frequency: float = 30e3,
            radio_rate: float = 10e3) -> DeviceClass:
    """Re-spec a class for a liquid medium.

    Two changes, both forced by the physics rather than chosen:

    **Blue-green optics.**  Pure water absorbs roughly 4.4 per metre at 850 nm
    and 0.016 per metre at 470 nm, so the stock near-infrared transceiver cannot
    cross even two metres of water while the blue-green one reaches tens.

    **A very low RF carrier.**  Seawater absorbs something like 1700 dB per
    metre at 2.4 GHz, which makes the usual control radio not merely poor but
    absent.  Dropping to tens of kilohertz brings the absorption down to a few
    dB per metre and buys back a short-range, low-rate link -- enough for
    beacons and control, nowhere near enough for data.  A liquid deployment that
    keeps the 2.4 GHz radio simply has no control plane, and the reach report
    will show it as centimetres.
    """
    o = cls.optical
    r = cls.radio
    return cls.variant(
        name or f"{cls.name}-aq",
        radio=RadioSpec(
            frequency=radio_frequency,
            tx_power=r.tx_power,
            line_rate=radio_rate,
            antenna_gain_db=r.antenna_gain_db,
            sensitivity_dbm=r.sensitivity_dbm,
            tx_electrical_power=r.tx_electrical_power,
            rx_power=r.rx_power,
        ),
        optical=OpticalSpec(
            n_transceivers=o.n_transceivers,
            wavelength=wavelength,
            tx_power=o.tx_power,
            divergence=o.divergence,
            tx_aperture_radius=o.tx_aperture_radius,
            rx_aperture_radius=o.rx_aperture_radius,
            rx_fov=o.rx_fov,
            line_rate=o.line_rate,
            tx_efficiency=o.tx_efficiency * 0.6,  # blue lasers are less efficient
            rx_power=o.rx_power,
            gimbal=o.gimbal,
        ),
    )


def tethered(
    cls: DeviceClass,
    name: str | None = None,
    *,
    head_radius: float = 8 * mm,
    length: float = 30.0,
    anchor_z: float = 0.0,
    tether_radius: float = 1.5 * mm,
    max_speed_xy: float = 0.8,
    max_speed_z: float = 0.5,
    waveguide_loss_db: float = 1.5,
    xy_limits=None,
    external_power: float = 2.0,
) -> DeviceClass:
    """Re-body a class as a remote head on a wire, with the box outside the medium.

    This inverts most of the free-swimming trade-offs, and the numbers below say
    why it is worth having as an option:

    * the immersed object shrinks from a ~60 mm float to an ~8 mm head, so the
      same volume holds roughly *two orders of magnitude* more of them, and
      "dynamic proximity" stops being limited by how tightly bodies can pack;
    * xyz is commanded from a gantry and a winch, so depth is as quick as
      translation -- the ballast lag that dominates the buoyant platform is
      simply gone;
    * motion energy leaves the harvested budget entirely, because the topside
      rig does the work, which is modelled here as a constant supply.

    What it costs is line of sight.  Each head trails a fibre to the surface and
    those fibres are opaque, so a dense field of them blocks precisely the
    horizontal beams the placement pass wants most.  The occlusion test in
    :class:`~iotc.sim.world.World` charges that cost.
    """
    o = cls.optical
    return cls.variant(
        name or f"{cls.name}-tether",
        kinematics="tethered",
        tether=TetherSpec(
            anchor_z=anchor_z,
            length=length,
            radius=tether_radius,
            max_speed_xy=max_speed_xy,
            max_speed_z=max_speed_z,
            waveguide_loss_db=waveguide_loss_db,
            xy_limits=xy_limits,
        ),
        body=BodySpec(
            mass=0.02,
            radius=head_radius,
            drag_coefficient=cls.body.drag_coefficient,
            max_thrust_xy=0.0,
            max_thrust_z=0.0,
            ballast_capacity=0.0,
            ballast_rate=0.0,
        ),
        # The box is outside the medium, so it is neither depth-shadowed nor
        # limited to what a coin cell can hold.
        harvest=HarvestSpec(kind="constant"),
        store=StoreSpec(capacity=200.0, initial_fraction=1.0,
                        leakage_power=cls.store.leakage_power),
        power=PowerSpec(
            mcu_active=cls.power.mcu_active,
            mcu_idle=cls.power.mcu_idle,
            mcu_sleep=cls.power.mcu_sleep,
            thruster_power=0.0,
            ballast_power=0.0,
        ),
    )


def describe_library() -> str:
    """One line per stock class, for ``iotc devices``."""
    return "\n".join(LIBRARY[n].summary() for n in sorted(LIBRARY))

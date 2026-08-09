"""Devices and the world they live in.

:class:`Device` is one node's runtime state; :class:`World` holds them all plus
the medium, the channels, the topology dynamics and the clock domain, and knows
how to answer the two questions everything else is built on:

* **can these two nodes talk right now** -- which needs both gimbals aimed at
  each other, an unobstructed path, and a link budget that closes;
* **what does it cost to change that** -- retarget time, retarget energy, and
  the travel time to move the bodies somewhere better.

Pointing is mutual and quantized, so a link is a jointly held resource: neither
end can establish it alone, and neither end can point exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..devices.profile import DeviceClass
from ..geom import (
    has_line_of_sight,
    spherical_from_vec,
    vec_from_spherical,
)
from ..registry import build, create
from ..rng import RandomSource
from .clocks import Clock, ClockDomain
from .energy import EnergyStore, PowerAccount
from .links.base import LinkBudget, LinkGeometry
from .mobility.kinematics import MotionState
from .mobility.topology import TopologyContext
from .trace import Trace


# --------------------------------------------------------------------------
@dataclass
class Transceiver:
    """One steerable optical head."""

    index: int
    az: float = 0.0
    el: float = 0.0
    #: Device this head is currently aimed at, if any.
    target: int | None = None
    #: Residual pointing error left by quantization at the last aim.
    residual: float = 0.0
    #: Simulated time at which the gimbal finishes slewing and settling.
    ready_at: float = 0.0
    #: Simulated time at which the current transmission completes.
    busy_until: float = 0.0
    #: A stuck gimbal keeps its angles and refuses every retarget.
    stuck: bool = False

    def available(self, now: float) -> bool:
        return now >= self.ready_at and now >= self.busy_until

    def direction(self) -> np.ndarray:
        return vec_from_spherical(self.az, self.el)


class Device:
    """One node: body, radios, gimbals, energy, clock."""

    def __init__(
        self,
        device_id: int,
        cls: DeviceClass,
        position,
        *,
        medium,
        rng: RandomSource,
        name: str | None = None,
        harvester=None,
        store=None,
        store_type: str = "supercap",
        kinematics=None,
    ) -> None:
        self.id = int(device_id)
        self.name = name or f"{cls.name}-{device_id}"
        self.cls = cls
        sub = rng.child(f"device/{self.id}")

        self.motion = MotionState(position=np.asarray(position, dtype=float).copy())
        self.kinematics = kinematics or build(cls.kinematics, "kinematics")
        if getattr(self.kinematics, "slow_z", False) and medium.supports_buoyancy:
            # Start neutrally trimmed rather than at an arbitrary ballast.
            neutral = type(self.kinematics).neutral_ballast(medium, cls.body)
            self.motion.ballast = float(np.clip(neutral, 0.0, cls.body.ballast_capacity))
            self.motion.ballast_target = self.motion.ballast

        self.harvester = harvester if harvester is not None else build(cls.harvest.kind, "harvester")
        if store is None:
            store = create("store", store_type, spec=cls.store)
        elif isinstance(store, str):
            store = create("store", store, spec=cls.store)
        self.store: EnergyStore = store
        self.power = PowerAccount()
        self.clock = Clock(cls.clock_drift_ppm, cls.clock_jitter, rng=sub.stream("clock"))

        self.transceivers = [Transceiver(i) for i in range(cls.optical.n_transceivers)]
        self.rng = sub

        # --- health ------------------------------------------------------
        self.crashed = False
        self.stalled_until = 0.0
        #: Multiplies the transmit power; degradation and faults reduce it.
        self.laser_derate = 1.0
        self.blacklist: set[int] = set()

        # --- workload state ------------------------------------------------
        self.ram_used = 0
        self.program = None
        self.state: dict = {}

    # ------------------------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        return self.motion.position

    @property
    def alive(self) -> bool:
        return not self.crashed and self.store.alive

    def operational(self, now: float) -> bool:
        return self.alive and now >= self.stalled_until

    @property
    def radius(self) -> float:
        return self.cls.body.radius

    @property
    def tx_power(self) -> float:
        """Optical power leaving the aperture, after everything upstream of it.

        On a tethered head that includes the insertion loss of the waveguide run
        from the box down to the head, which is a real and often decisive cost
        of moving the electronics outside the medium.
        """
        power = self.cls.optical.tx_power * self.laser_derate
        if self.cls.tether is not None:
            power *= self.cls.tether.transmittance
        return power

    @property
    def tethered(self) -> bool:
        return self.cls.tether is not None

    def tether_segment(self) -> tuple[np.ndarray, np.ndarray, float] | None:
        """The wire, as ``(topside end, head end, radius)``.

        The gantry sits directly above the head, so the wire is the vertical
        column between the surface and the immersed head.
        """
        t = self.cls.tether
        if t is None:
            return None
        head = self.position
        top = np.array([head[0], head[1], t.anchor_z])
        return top, head.copy(), t.radius

    def free_transceiver(self, now: float) -> Transceiver | None:
        for t in self.transceivers:
            if t.target is None and t.available(now) and not t.stuck:
                return t
        return None

    def transceiver_for(self, peer_id: int) -> Transceiver | None:
        for t in self.transceivers:
            if t.target == peer_id:
                return t
        return None

    def spend(self, subsystem: str, joules: float) -> bool:
        """Draw energy, recording which subsystem used it."""
        ok = self.store.draw(joules)
        if ok:
            self.power.add(subsystem, joules)
        return ok

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "class": self.cls.name,
            "position": self.position.tolist(),
            "energy_fraction": self.store.fraction,
            "alive": self.alive,
            "links": [t.target for t in self.transceivers],
        }


# --------------------------------------------------------------------------
class World:
    """The simulated environment."""

    def __init__(
        self,
        medium="water",
        *,
        channel="fso",
        rf="rf_omni",
        topology="static",
        seed: int = 0,
        bounds=None,
        resync_interval: float = 1.0,
        beacon_accuracy: float = 1e-5,
        trace: Trace | None = None,
        payload_bits: int = 8192,
    ) -> None:
        self.rng = RandomSource(seed)
        self.medium = build(medium, "medium")
        self.channel = build(channel, "channel")
        self.rf = build(rf, "channel")
        self.topology = build(topology, "topology")
        self.clocks = ClockDomain(resync_interval, beacon_accuracy)
        self.trace = trace if trace is not None else Trace()
        self.devices: list[Device] = []
        self.time = 0.0
        self.payload_bits = int(payload_bits)
        self.bounds = None if bounds is None else (
            np.asarray(bounds[0], dtype=float),
            np.asarray(bounds[1], dtype=float),
        )
        #: Compiler-supplied goal positions, or NaN where unset.
        self.goals: np.ndarray | None = None
        #: Affinity matrix used by the gravity topology.
        self.affinity: np.ndarray | None = None
        self.faults: list = []
        #: Extra packet-loss floor imposed by a fault model, on top of physics.
        self.extra_packet_loss = 0.0
        self._pos_cache: np.ndarray | None = None

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def add_device(self, cls: DeviceClass, position, name: str | None = None, **kw) -> Device:
        dev = Device(len(self.devices), cls, position, medium=self.medium,
                     rng=self.rng, name=name, **kw)
        self.devices.append(dev)
        self.clocks.add(dev.id, dev.clock)
        self._pos_cache = None
        return dev

    def add_swarm(self, cls: DeviceClass, positions, prefix: str | None = None) -> list[Device]:
        return [
            self.add_device(cls, p, None if prefix is None else f"{prefix}-{i}")
            for i, p in enumerate(np.asarray(positions, dtype=float))
        ]

    def __len__(self) -> int:
        return len(self.devices)

    def __getitem__(self, i: int) -> Device:
        return self.devices[i]

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------
    @property
    def positions(self) -> np.ndarray:
        if self._pos_cache is None or len(self._pos_cache) != len(self.devices):
            self._pos_cache = np.zeros((len(self.devices), 3))
        for i, d in enumerate(self.devices):
            self._pos_cache[i] = d.position
        return self._pos_cache

    @property
    def radii(self) -> np.ndarray:
        return np.array([d.radius for d in self.devices])

    @property
    def alive_mask(self) -> np.ndarray:
        return np.array([d.alive for d in self.devices], dtype=bool)

    def distance(self, i: int, j: int) -> float:
        return float(np.linalg.norm(self.devices[i].position - self.devices[j].position))

    def neighbours(self, i: int, max_range: float) -> list[int]:
        """Every other device within ``max_range`` metres, nearest first."""
        p = self.positions
        d = np.linalg.norm(p - p[i], axis=1)
        d[i] = np.inf
        idx = np.where(d <= max_range)[0]
        return list(idx[np.argsort(d[idx])])

    def tether_capsules(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
        """Every tether as a capsule, plus which device each one belongs to.

        Rebuilt each call because the heads move.  The device index list is what
        lets the line-of-sight test excuse a link's own two tethers, which
        necessarily touch its endpoints.
        """
        starts, ends, radii, owners = [], [], [], []
        for dev in self.devices:
            seg = dev.tether_segment()
            if seg is None:
                continue
            top, head, r = seg
            starts.append(top)
            ends.append(head)
            radii.append(r)
            owners.append(dev.id)
        if not starts:
            return (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), [])
        return np.array(starts), np.array(ends), np.array(radii), owners

    def line_of_sight(self, i: int, j: int) -> bool:
        starts, ends, radii, owners = self.tether_capsules()
        capsules = None
        skip: tuple[int, ...] = ()
        if len(starts):
            capsules = (starts, ends, radii)
            skip = tuple(k for k, owner in enumerate(owners) if owner in (i, j))
        return has_line_of_sight(
            self.devices[i].position, self.devices[j].position,
            self.positions, self.radii, exclude=(i, j),
            capsules=capsules, exclude_capsules=skip,
        )

    def bearing(self, i: int, j: int) -> tuple[float, float, float]:
        """``(azimuth, elevation, distance)`` from device ``i`` to device ``j``."""
        delta = self.devices[j].position - self.devices[i].position
        dist = float(np.linalg.norm(delta))
        az, el = spherical_from_vec(delta)
        return float(az), float(el), dist

    # ------------------------------------------------------------------
    # pointing
    # ------------------------------------------------------------------
    def aim(self, i: int, j: int, tx: Transceiver | None = None,
            charge: bool = True) -> dict:
        """Aim one of ``i``'s heads at ``j``.

        Returns what it cost and how well it succeeded.  A stuck gimbal, an
        out-of-sweep bearing or an empty supercap all fail here rather than
        silently producing a link that cannot exist.
        """
        dev = self.devices[i]
        if not dev.operational(self.time):
            return {"ok": False, "reason": "device not operational"}
        if tx is None:
            tx = dev.transceiver_for(j) or dev.free_transceiver(self.time)
        if tx is None:
            return {"ok": False, "reason": "no free transceiver"}
        if tx.stuck:
            return {"ok": False, "reason": "gimbal stuck"}

        az, el, dist = self.bearing(i, j)
        grid = dev.cls.optical.gimbal.grid
        if not grid.in_range(az, el):
            return {"ok": False, "reason": "bearing outside sweep limits",
                    "az": az, "el": el}

        steps = grid.steps_between(tx.az, tx.el, az, el)
        az_q, el_q, residual = grid.snap(az, el)
        gimbal = dev.cls.optical.gimbal
        dt = gimbal.retarget_time(steps)
        energy = gimbal.retarget_energy(steps)

        if charge and not dev.spend("gimbal", energy):
            return {"ok": False, "reason": "insufficient energy to retarget"}

        tx.az, tx.el = az_q, el_q
        tx.residual = residual
        tx.target = j
        tx.ready_at = self.time + dt
        self.trace.event(self.time, "aim", device=i, peer=j,
                         steps=steps, residual=residual, dt=dt, energy=energy)
        self.trace.add("retarget_time", dt)
        self.trace.add("retarget_energy", energy)
        return {"ok": True, "steps": steps, "time": dt, "energy": energy,
                "residual": residual, "distance": dist, "transceiver": tx.index}

    def pointing_error(self, i: int, j: int) -> float:
        """How far ``i``'s head for ``j`` is off the true bearing, right now.

        Recomputed from live positions rather than cached, because the bodies
        keep moving after the gimbal stops -- which is precisely how a link
        decays without anything having "failed".
        """
        tx = self.devices[i].transceiver_for(j)
        if tx is None:
            return math.inf
        az, el, _ = self.bearing(i, j)
        want = vec_from_spherical(az, el)
        have = tx.direction()
        return float(np.arccos(np.clip(float(np.dot(want, have)), -1.0, 1.0)))

    # ------------------------------------------------------------------
    # links
    # ------------------------------------------------------------------
    def geometry(self, i: int, j: int) -> LinkGeometry:
        a, b = self.devices[i], self.devices[j]
        o = a.cls.optical
        return LinkGeometry(
            distance=self.distance(i, j),
            wavelength=o.wavelength,
            tx_power=a.tx_power,
            divergence=o.divergence,
            tx_aperture_radius=o.tx_aperture_radius,
            rx_aperture_radius=b.cls.optical.rx_aperture_radius,
            rx_fov=b.cls.optical.rx_fov,
            tx_misalignment=self.pointing_error(i, j),
            rx_misalignment=self.pointing_error(j, i),
        )

    def link_budget(self, i: int, j: int, *, stochastic: bool = True,
                    payload_bits: int | None = None) -> LinkBudget:
        """Evaluate the optical link from ``i`` to ``j`` as things stand."""
        a, b = self.devices[i], self.devices[j]
        if not (a.operational(self.time) and b.operational(self.time)):
            out = LinkBudget(reason="endpoint not operational")
            return out
        if j in a.blacklist or i in b.blacklist:
            return LinkBudget(reason="link permanently failed")
        if not self.line_of_sight(i, j):
            return LinkBudget(reason="occluded")

        geo = self.geometry(i, j)
        if not math.isfinite(geo.tx_misalignment):
            return LinkBudget(reason="transmitter not aimed")
        if not math.isfinite(geo.rx_misalignment):
            return LinkBudget(reason="receiver not aimed")

        return self.channel.evaluate(
            geo, self.medium, self.rng.stream(f"link/{min(i, j)}/{max(i, j)}"),
            jitter_sigma=a.cls.optical.gimbal.jitter_sigma,
            payload_bits=payload_bits or self.payload_bits,
            line_rate=min(a.cls.optical.line_rate, b.cls.optical.line_rate),
            stochastic=stochastic,
            key=(i, j),
            now=self.time,
        )

    def rf_budget(self, i: int, j: int, *, stochastic: bool = True) -> LinkBudget:
        a, b = self.devices[i], self.devices[j]
        if not (a.operational(self.time) and b.operational(self.time)):
            return LinkBudget(reason="endpoint not operational")
        return self.rf.evaluate(
            self.distance(i, j), a.cls.radio, self.medium,
            self.rng.stream(f"rf/{min(i, j)}/{max(i, j)}"),
            stochastic=stochastic,
        )

    def established_links(self) -> list[tuple[int, int]]:
        """Pairs whose heads are aimed at each other and whose budget closes."""
        out = []
        for i, a in enumerate(self.devices):
            for t in a.transceivers:
                j = t.target
                if j is None or j <= i:
                    continue
                if self.devices[j].transceiver_for(i) is None:
                    continue
                if self.link_budget(i, j, stochastic=False).feasible:
                    out.append((i, j))
        return out

    # ------------------------------------------------------------------
    # stepping
    # ------------------------------------------------------------------
    def _topology_context(self) -> TopologyContext:
        return TopologyContext(
            positions=self.positions.copy(),
            velocities=np.array([d.motion.velocity for d in self.devices]),
            time=self.time,
            goals=self.goals,
            affinity=self.affinity,
            links=self.established_links() if self._needs_links() else [],
            alive=self.alive_mask,
            radii=self.radii,
            bounds=self.bounds,
            rng=self.rng.stream("topology"),
        )

    def _needs_links(self) -> bool:
        # Only the flocking law cares, and computing them is not cheap.
        return hasattr(self.topology, "link_keep")

    def step(self, dt: float) -> None:
        """Advance the physical world by ``dt`` seconds."""
        if not self.devices:
            self.time += dt
            return

        targets = self.topology.targets(self._topology_context())

        for i, dev in enumerate(self.devices):
            # --- energy in --------------------------------------------
            p_in = dev.harvester.power(dev.cls.harvest, dev.position, self.medium, self.time)
            dev.store.harvest(p_in, dt)
            before = dev.store.energy
            dev.store.leak(dt)
            dev.power.add("leak", before - dev.store.energy)

            if not dev.alive:
                dev.motion.frozen = True
                continue
            dev.motion.frozen = dev.crashed or self.time < dev.stalled_until

            # --- housekeeping -----------------------------------------
            idle = dev.cls.power.mcu_idle * dt
            dev.spend("mcu", idle)

            # --- motion -------------------------------------------------
            cost = dev.kinematics.step(
                dev.motion, targets[i], dt, self.medium, dev.cls.body, dev.cls.power,
                tether=dev.cls.tether,
            )
            if cost.energy > 0.0 and not dev.spend("motion", cost.energy):
                # Out of energy mid-manoeuvre: stop actuating, keep drifting.
                dev.motion.frozen = True

            if self.bounds is not None:
                lo, hi = self.bounds
                clipped = np.clip(dev.motion.position, lo, hi)
                if not np.allclose(clipped, dev.motion.position):
                    dev.motion.velocity[clipped != dev.motion.position] = 0.0
                    dev.motion.position[:] = clipped

        self._pos_cache = None
        self.time += dt
        self.clocks.step(self.time)

        # Keyed on position in the list, never on id(): a memory address is not
        # stable across runs and would silently destroy reproducibility.
        for index, fault in enumerate(self.faults):
            fault.step(self, self.time, dt, self.rng.stream(f"fault/{index}"))

        self._record(dt)

    def _record(self, dt: float) -> None:
        t = self.time
        alive = int(sum(d.alive for d in self.devices))
        self.trace.sample(t, "alive", alive)
        self.trace.sample(t, "mean_energy_fraction",
                          float(np.mean([d.store.fraction for d in self.devices])))
        self.trace.sample(t, "clock_skew", self.clocks.max_skew(t))
        if self.goals is not None:
            err = np.linalg.norm(self.positions - self.goals, axis=1)
            valid = np.isfinite(err)
            if valid.any():
                self.trace.sample(t, "formation_rms", float(np.sqrt(np.mean(err[valid] ** 2))))
                self.trace.sample(t, "formation_max", float(np.max(err[valid])))

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        energies = [d.store.fraction for d in self.devices]
        return {
            "time": self.time,
            "devices": len(self.devices),
            "alive": int(sum(d.alive for d in self.devices)),
            "links": len(self.established_links()),
            "mean_energy_fraction": float(np.mean(energies)) if energies else 0.0,
            "min_energy_fraction": float(np.min(energies)) if energies else 0.0,
            "brownouts": int(sum(d.store.brownouts for d in self.devices)),
            "clock_skew": self.clocks.max_skew(self.time),
            "guard_band": self.clocks.guard_band(self.time),
        }

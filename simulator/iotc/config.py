"""Declarative configuration.

A config is a plain mapping -- YAML on disk, or a dict in a script -- and every
plugin slot in it is resolved through :mod:`iotc.registry`, so a user-written
channel model or topology law is selectable from a config file without touching
this module.

    medium: {type: water, turbidity: 0.02}
    channel:
      type: fso
      turbulence: {type: gamma_gamma}
      receiver: {type: apd, gain: 40}
    topology: gradient
    swarm:
      - {class: mote-c, variant: aquatic, count: 24,
         layout: {type: lattice, spacing: 6.0, centre: [0, 0, -10]}}

The ``variant`` key is worth knowing about: it applies ``aquatic``, ``aerial`` or
``tethered`` from :mod:`iotc.devices.library`, which is how one config selects a
whole platform -- blue-green optics and ballast, or near-infrared and rotors, or
a remote head on a wire.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .devices import aerial, aquatic, get_class, tethered
from .devices.profile import DeviceClass
from .geom import fibonacci_sphere, grid_lattice
from .registry import build
from .rng import RandomSource
from .sim.engine import Simulation
from .sim.faults import build_campaign
from .sim.trace import Trace
from .sim.world import World

VARIANTS = {"aquatic": aquatic, "aerial": aerial, "tethered": tethered, "none": None}


class ConfigError(ValueError):
    """Raised when a configuration cannot be interpreted."""


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_file(path) -> dict:
    """Read a YAML or JSON configuration."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ConfigError(
                "PyYAML is needed for .yaml configs; install it or use JSON"
            ) from exc
        return yaml.safe_load(text) or {}
    return json.loads(text)


# --------------------------------------------------------------------------
# device classes
# --------------------------------------------------------------------------
def resolve_class(spec) -> DeviceClass:
    """Turn a swarm-group spec into a concrete :class:`DeviceClass`.

    Accepts a bare name, or a mapping with ``class``, an optional ``variant``
    and optional ``variant_args``.

    ``variant`` may be a list, and usually should be.  The variants are
    orthogonal -- ``aquatic`` re-specs the *optics and radio* for a liquid, while
    ``tethered`` re-specs the *platform* -- so ``[aquatic, tethered]`` is a
    remote head that can actually see, whereas ``tethered`` alone leaves a
    near-infrared transceiver in water, where it cannot cross two metres.  A
    sweep that overwrites one with the other silently loses the optics, so
    composing them explicitly is the safe form.

    With a list, ``variant_args`` is keyed by variant name::

        variant: [aquatic, tethered]
        variant_args:
          aquatic: {wavelength: 5.0e-7}
          tethered: {length: 25.0}
    """
    if isinstance(spec, DeviceClass):
        return spec
    if isinstance(spec, str):
        return get_class(spec)
    if not isinstance(spec, dict):
        raise ConfigError(f"cannot interpret device spec {spec!r}")

    base = spec.get("class") or spec.get("device")
    if base is None:
        raise ConfigError(f"device spec needs a 'class' key: {spec!r}")
    cls = get_class(base) if isinstance(base, str) else base

    variants = spec.get("variant", "none")
    if isinstance(variants, str):
        variants = [variants]
    variants = [v for v in (variants or []) if v and v != "none"]

    args = spec.get("variant_args") or {}
    per_variant = len(variants) > 1 or (
        set(args) and set(args) <= set(VARIANTS)
    )

    for name in variants:
        fn = VARIANTS.get(name)
        if fn is None:
            raise ConfigError(
                f"unknown variant {name!r}; available: {sorted(VARIANTS)}"
            )
        kwargs = args.get(name, {}) if per_variant else args
        cls = fn(cls, **kwargs)

    if spec.get("name"):
        cls = cls.variant(spec["name"])
    return cls


# --------------------------------------------------------------------------
# layouts
# --------------------------------------------------------------------------
def build_layout(spec, count: int, rng) -> np.ndarray:
    """Produce ``count`` starting positions from a layout spec."""
    if spec is None:
        spec = {"type": "lattice"}
    if isinstance(spec, (list, tuple, np.ndarray)):
        pts = np.asarray(spec, dtype=float)
        if len(pts) != count:
            raise ConfigError(f"layout lists {len(pts)} points but count is {count}")
        return pts
    if isinstance(spec, str):
        spec = {"type": spec}

    kind = spec.get("type", "lattice")
    centre = np.asarray(spec.get("centre", (0.0, 0.0, 0.0)), dtype=float)

    if kind in ("lattice", "grid"):
        pts = grid_lattice(count, float(spec.get("spacing", 5.0)))
    elif kind in ("sphere", "shell"):
        pts = fibonacci_sphere(count, float(spec.get("radius", 10.0)))
    elif kind in ("ring", "circle"):
        radius = float(spec.get("radius", 10.0))
        a = np.linspace(0.0, 2 * math.pi, count, endpoint=False)
        pts = np.stack([radius * np.cos(a), radius * np.sin(a), np.zeros(count)], axis=-1)
    elif kind in ("line", "chain"):
        spacing = float(spec.get("spacing", 5.0))
        axis = np.asarray(spec.get("axis", (1.0, 0.0, 0.0)), dtype=float)
        offsets = (np.arange(count, dtype=float) - (count - 1) / 2.0) * spacing
        pts = offsets[:, None] * axis[None, :]
    elif kind in ("random", "uniform"):
        extent = np.asarray(spec.get("extent", (20.0, 20.0, 10.0)), dtype=float)
        pts = rng.uniform(-0.5, 0.5, size=(count, 3)) * extent
    else:
        raise ConfigError(f"unknown layout type {kind!r}")

    return pts + centre


# --------------------------------------------------------------------------
# world
# --------------------------------------------------------------------------
def build_world(cfg: dict, trace: Trace | None = None) -> World:
    """Construct a :class:`~iotc.sim.world.World` from a config mapping."""
    world_cfg = cfg.get("world", {}) or {}
    bounds = world_cfg.get("bounds")
    world = World(
        medium=cfg.get("medium", "water"),
        channel=cfg.get("channel", "fso"),
        rf=cfg.get("rf", "rf_omni"),
        topology=cfg.get("topology", "static"),
        seed=int(cfg.get("seed", 0)),
        bounds=bounds,
        resync_interval=float(world_cfg.get("resync_interval", 1.0)),
        beacon_accuracy=float(world_cfg.get("beacon_accuracy", 1e-5)),
        payload_bits=int(world_cfg.get("payload_bits", 8192)),
        trace=trace,
    )

    layout_rng = RandomSource(int(cfg.get("seed", 0))).stream("layout")
    groups = cfg.get("swarm") or cfg.get("devices") or []
    if not groups:
        raise ConfigError("config has no 'swarm' section")

    for group in groups:
        cls = resolve_class(group)
        count = int(group.get("count", 1))
        positions = build_layout(group.get("layout"), count, layout_rng)
        prefix = group.get("prefix", cls.name)
        for k, p in enumerate(positions):
            world.add_device(
                cls, p, name=f"{prefix}-{k}",
                store_type=group.get("store", "supercap"),
                harvester=build(group["harvester"], "harvester")
                if group.get("harvester") else None,
            )

    if cfg.get("goals") is not None:
        world.goals = np.asarray(cfg["goals"], dtype=float)
    world.faults = build_campaign(cfg.get("faults"))
    return world


def build_simulation(cfg: dict, world: World | None = None) -> Simulation:
    if world is None:
        world = build_world(cfg)
    dt = float((cfg.get("world", {}) or {}).get("dt", 0.05))
    return Simulation(world, dt=dt)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def dump_summary(world: World, cfg: dict | None = None) -> str:
    """A human-readable description of what was built."""
    from collections import Counter

    lines = []
    if cfg and cfg.get("name"):
        lines.append(f"configuration : {cfg['name']}")
    lines.append(f"medium        : {world.medium.summary()}")
    lines.append(f"channel       : {getattr(world.channel, 'plugin_name', world.channel)}")
    lines.append(f"topology      : {getattr(world.topology, 'plugin_name', world.topology)}")
    lines.append(f"devices       : {len(world.devices)}")
    counts = Counter(d.cls.name for d in world.devices)
    for name, n in sorted(counts.items()):
        cls = next(d.cls for d in world.devices if d.cls.name == name)
        lines.append(f"  {n:5d} x {cls.summary()}")
    if world.faults:
        lines.append("faults        :")
        for f in world.faults:
            lines.append(f"  {f.plugin_name} rate={f.rate:g} onset={f.onset:g}")
    return "\n".join(lines)


def reach_report(world: World) -> str:
    """Maximum optical and RF range for each device class, in this medium.

    The single most useful sanity check before running anything: if the reach is
    a metre and the layout spacing is ten, nothing will ever link up and every
    later number will be a confusing zero.
    """
    from .sim.links.base import LinkGeometry

    seen = {}
    for dev in world.devices:
        if dev.cls.name in seen:
            continue
        o = dev.cls.optical
        geo = LinkGeometry(
            distance=1.0, wavelength=o.wavelength, tx_power=dev.tx_power,
            divergence=o.divergence, tx_aperture_radius=o.tx_aperture_radius,
            rx_aperture_radius=o.rx_aperture_radius, rx_fov=o.rx_fov,
        )
        optical = world.channel.max_range(geo, world.medium, line_rate=o.line_rate)
        radio = world.rf.max_range(dev.cls.radio, world.medium)
        seen[dev.cls.name] = (optical, radio, o.wavelength)

    lines = ["class          lambda    optical reach    rf reach"]
    for name, (optical, radio, lam) in sorted(seen.items()):
        lines.append(
            f"{name:14s} {lam * 1e9:6.0f} nm   {optical:9.2f} m   {radio:9.2f} m"
        )
    return "\n".join(lines)

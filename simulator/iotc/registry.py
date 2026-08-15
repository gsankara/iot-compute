"""Plugin registry.

Every replaceable model in the package is registered here under a ``kind`` (the
abstraction slot) and a ``name`` (the specific model).  Two slots in particular
are meant to be extended by users, and the whole package is arranged so that
neither requires editing simulator code:

* **transmission physics** -- ``beam``, ``attenuation``, ``turbulence``,
  ``pointing``, ``receiver``, ``modulation``, ``coding``, ``channel``
* **topology dynamics** -- ``topology``, ``kinematics``

Adding a model is a subclass plus a decorator::

    from iotc.registry import register
    from iotc.sim.links.base import Turbulence

    @register("turbulence", "my_scintillation")
    class MyScintillation(Turbulence):
        ...

After that it is reachable from YAML configs and from
``iotc.registry.create("turbulence", "my_scintillation", ...)``.

Structure follows ``fsphotonix.registry`` deliberately, so someone who knows one
package knows the other.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Iterable

#: Abstraction slots that ship with the package.  Custom kinds are allowed.
KINDS = (
    # --- transmission physics (pluggable axis 1) -------------------------
    "medium",  # what the devices are immersed in
    "beam",  # transmit beam transverse profile / divergence
    "attenuation",  # absorption and scattering along the path
    "turbulence",  # scintillation / fading
    "pointing",  # gimbal quantization + jitter -> misalignment loss
    "receiver",  # photodetector front end and its noise
    "modulation",  # symbol mapping -> BER(SNR)
    "coding",  # FEC -> PER(BER)
    "channel",  # a whole link technology (fso, rf, surrogate)
    # --- topology dynamics (pluggable axis 2) ----------------------------
    "kinematics",  # how a body actually moves
    "topology",  # what the swarm is trying to do with its geometry
    # --- platform --------------------------------------------------------
    "harvester",  # energy in
    "store",  # energy buffer
    "fault",  # fault injector
    # --- compiler --------------------------------------------------------
    "tiler",  # how an oversized task is split into pieces
    "partitioner",  # task graph -> per-device groups
    "placer",  # task graph -> 3D coordinates
    "planner",  # start formation -> target formation, in flight
    "assigner",  # which links are open simultaneously
    "scheduler",  # traffic matrix -> time slots
    "energy_policy",  # what to do when harvest cannot cover the schedule
    "resilience",  # which tasks get a standby copy
    "encoder",  # bytecode image format
    "clique_solver",  # the B10 flagship workload
    "workload",  # problem frontend
)

_REGISTRY: dict[str, dict[str, type]] = defaultdict(dict)
_ALIASES: dict[str, dict[str, str]] = defaultdict(dict)


class PluginError(KeyError):
    """Raised when a plugin name cannot be resolved."""


def register(kind: str, name: str, *aliases: str) -> Callable[[type], type]:
    """Class decorator registering ``cls`` as ``kind``/``name``."""

    def _wrap(cls: type) -> type:
        key = name.lower()
        if key in _REGISTRY[kind] and _REGISTRY[kind][key] is not cls:
            raise PluginError(f"{kind}/{name} already registered by {_REGISTRY[kind][key]!r}")
        _REGISTRY[kind][key] = cls
        for alias in aliases:
            _ALIASES[kind][alias.lower()] = key
        cls.plugin_kind = kind  # type: ignore[attr-defined]
        cls.plugin_name = name  # type: ignore[attr-defined]
        return cls

    return _wrap


def get(kind: str, name: str) -> type:
    """Look up a registered class."""
    table = _REGISTRY.get(kind)
    if not table:
        raise PluginError(f"no plugins registered for kind {kind!r} (known: {sorted(_REGISTRY)})")
    key = name.lower()
    key = _ALIASES[kind].get(key, key)
    try:
        return table[key]
    except KeyError:
        raise PluginError(f"unknown {kind} {name!r}; available: {sorted(table)}") from None


def create(kind: str, name: str, /, **kwargs: Any) -> Any:
    """Instantiate a registered plugin."""
    return get(kind, name)(**kwargs)


def available(kind: str | None = None) -> dict[str, list[str]] | list[str]:
    """List registered plugin names, optionally for a single kind."""
    if kind is not None:
        return sorted(_REGISTRY.get(kind, {}))
    return {k: sorted(v) for k, v in sorted(_REGISTRY.items())}


def build(spec: Any, kind: str, **defaults: Any) -> Any:
    """Build an object from a declarative spec.

    ``spec`` may already be an instance (returned unchanged), a string naming a
    plugin, or a mapping ``{"type": name, ...kwargs}``.  Used by the config
    loader and by constructors that accept "either an object or a shorthand".
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        return create(kind, spec, **defaults)
    if isinstance(spec, dict):
        params = dict(defaults)
        params.update({k: v for k, v in spec.items() if k != "type"})
        name = spec.get("type")
        if name is None:
            raise PluginError(f"spec for kind {kind!r} needs a 'type' key: {spec!r}")
        return create(kind, name, **params)
    return spec  # already an instance / duck-typed object


def describe(kinds: Iterable[str] | None = None) -> str:
    """Human-readable plugin inventory."""
    lines = []
    for kind in kinds or KINDS:
        names = sorted(_REGISTRY.get(kind, {}))
        lines.append(f"{kind:13s} : {', '.join(names) if names else '-'}")
    extra = [k for k in sorted(_REGISTRY) if k not in KINDS]
    for kind in extra:
        lines.append(f"{kind:13s} : {', '.join(sorted(_REGISTRY[kind]))}  (custom)")
    return "\n".join(lines)

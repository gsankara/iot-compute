"""iot-compute: a mobility-driven IoT computation substrate.

Thousands of small, energy-harvesting devices that compute by *interaction
timing and coordination* rather than by instruction streams.  They are linked by
Gbps free-space photonics on quantized-step gimbals plus a slow omnidirectional
radio, they move -- as drones in air or as ballast-driven floats in a liquid --
and the interconnect topology is therefore something the compiler *produces* and
the swarm then physically realizes.

Three layers, usable independently:

``iotc.sim``
    The simulator.  High-fidelity transmission physics by default, with every
    stage of the link budget and every law of motion a swappable plugin.

``iotc.compiler``
    Lowers a problem onto thousands of devices: tile, partition, place (FastMap),
    plan motion, assign beams, schedule slots, check energy, harden, emit code.

``iotc.bench``
    Correctness and performance evaluation, with and without node/link faults
    and channel unreliability.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import devices, geom, registry, rng, sim, units
from .geom import AngularGrid, vec3
from .registry import available, create, describe, register
from .rng import RandomSource

__all__ = [
    "__version__",
    "units",
    "geom",
    "registry",
    "rng",
    "devices",
    "sim",
    "AngularGrid",
    "vec3",
    "RandomSource",
    "register",
    "create",
    "available",
    "describe",
]

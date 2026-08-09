"""Problem frontends.

Importing this package registers every stock workload under the ``workload``
plugin kind, so ``iotc.registry.create("workload", "stencil3d", ...)`` and the
``iotc graph`` CLI command both work without the caller naming a module.
"""

from __future__ import annotations

from . import dense, graphs, nphard, sparse  # noqa: F401  (import registers)
from .base import Workload
from .dense import AllReduce, FFTButterfly, Stencil2D, Stencil3D, Summa
from .nphard import (
    CapacitatedKMedian,
    GraphColouring,
    MaxClique,
    MaxCut,
    MaxIndependentSet,
    common_neighbour_weight,
)
from .sparse import BFS, SpMV

__all__ = [
    "Workload",
    "Stencil2D",
    "Stencil3D",
    "Summa",
    "FFTButterfly",
    "AllReduce",
    "SpMV",
    "BFS",
    "MaxClique",
    "MaxIndependentSet",
    "GraphColouring",
    "CapacitatedKMedian",
    "MaxCut",
    "common_neighbour_weight",
    "graphs",
]

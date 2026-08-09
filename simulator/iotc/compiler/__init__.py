"""The compiler: a problem graph lowered onto thousands of moving devices.

Phase B of the project plan, built pass by pass.  Each pass lands as
implementation, unit tests, and its own evaluation script reporting its own
quality metric -- no pass is finished without its bench.

Implemented:

``B0`` -- :mod:`iotc.compiler.ir` and :mod:`iotc.compiler.frontends`.  The
    :class:`~iotc.compiler.ir.TaskGraph` every later pass consumes, plus the
    stock workload family that produces one.

Still to come: tile, partition, place (FastMap), motion, beam assignment,
schedule, energy feasibility, resilience and codegen.  See ``docs/ROADMAP.md``.
"""

from __future__ import annotations

from . import frontends, ir
from .frontends import Workload
from .ir import Task, TaskGraph, TaskGraphError

__all__ = [
    "ir",
    "frontends",
    "Task",
    "TaskGraph",
    "TaskGraphError",
    "Workload",
    "build_workload",
]


def build_workload(name: str, /, **params) -> TaskGraph:
    """Build a task graph from a registered workload name.

    The one-liner that the CLI, the bench harness and the config loader all go
    through, so a workload named in YAML and a workload named on the command
    line cannot mean different things::

        graph = build_workload("stencil3d", tx=4, ty=4, tz=4)
    """
    from ..registry import create

    return create("workload", name, **params).build()

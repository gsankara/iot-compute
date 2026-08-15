"""The compiler: a problem graph lowered onto thousands of moving devices.

Phase B of the project plan, built pass by pass.  Each pass lands as
implementation, unit tests, and its own evaluation script reporting its own
quality metric -- no pass is finished without its bench.

Implemented:

``B0`` -- :mod:`iotc.compiler.ir` and :mod:`iotc.compiler.frontends`.  The
    :class:`~iotc.compiler.ir.TaskGraph` every later pass consumes, plus the
    stock workload family that produces one.

``B1`` -- :mod:`iotc.compiler.tile`.  Splits tasks until their working set fits
    the target class's RAM, choosing the *shape* of the cut rather than only its
    size, because added traffic depends on the factorization and not on the
    piece count.

``B2`` -- :mod:`iotc.compiler.partition`, on :mod:`iotc.compiler.fastmap`.  Packs
    tasks back into per-device groups so that as much traffic as possible
    becomes a memory copy instead of a beam.  FastMap is the spine and B3 uses
    the same embedding.

``B3`` -- :mod:`iotc.compiler.place`.  Projects the embedding onto physical
    coordinates, scaling each axis by what the *energy* budget allows rather
    than by how fast the body moves -- the correction Phase A produced.

``B4`` -- :mod:`iotc.compiler.motion`.  Flies the swarm to the placement without
    dropping the links on the way, by moving in colour classes so no protected
    link ever has both endpoints in motion.

``B5`` -- :mod:`iotc.compiler.beam`.  Chooses which links are open at the same
    time, as a sequence of degree-constrained b-matchings, and feeds the
    pointing problem back into B3.

``B6`` -- :mod:`iotc.compiler.schedule`.  Times the beam rounds into slots and
    amortises the retarget by batching, bounded by RAM and by latency.

``B7`` -- :mod:`iotc.compiler.energy`.  Prices the frame against what the class
    harvests and pads it with sleep until the balance closes.

``B8`` -- :mod:`iotc.compiler.resilience`.  Replication chosen by centrality,
    LSH failover targets, and the closed-loop pointing tracker Phase A omits.

``B9`` -- :mod:`iotc.compiler.codegen`.  Emits per-device bytecode and a global
    manifest; :mod:`iotc.sim.node` is the VM that executes it.

``B10`` -- :mod:`iotc.compiler.clique`.  Maximum clique end to end: physical
    proximity as the adjacency test, against four conventional baselines.

Phase B is complete.  See ``docs/ROADMAP.md``.
"""

from __future__ import annotations

from . import (beam, clique, codegen, energy, fastmap, frontends, ir, motion,
               partition, place, resilience, schedule, tile)
from .fastmap import embed_graph, traffic_distance
from .frontends import Workload
from .ir import Task, TaskGraph, TaskGraphError
from .partition import PartitionResult, Partitioner, minimum_devices
from .beam import BeamAssigner, BeamAssignment, assign_beams, refine_positions
from .motion import MotionPlan, Planner, plan_motion
from .place import MotionBudget, PlaceResult, Placer, derive_budget
from .energy import EnergyBudget, EnergyPolicy, assess
from .resilience import (FailoverIndex, ReplicationPolicy, ResilienceResult,
                         betweenness, plan_resilience, reaim_interval)
from .clique import CliqueResult, CliqueSolver, solve_clique
from .codegen import DeviceProgram, Instruction, Manifest, Op, emit
from .schedule import Schedule, Scheduler, Slot, build_schedule
from .tile import TileModel, TileResult, balanced_factors

__all__ = [
    "ir",
    "frontends",
    "tile",
    "fastmap",
    "partition",
    "place",
    "motion",
    "beam",
    "schedule",
    "energy",
    "resilience",
    "codegen",
    "clique",
    "Task",
    "TaskGraph",
    "TaskGraphError",
    "Workload",
    "TileModel",
    "TileResult",
    "Partitioner",
    "PartitionResult",
    "Placer",
    "PlaceResult",
    "MotionBudget",
    "derive_budget",
    "Planner",
    "MotionPlan",
    "plan_motion",
    "BeamAssigner",
    "BeamAssignment",
    "assign_beams",
    "refine_positions",
    "Scheduler",
    "Schedule",
    "Slot",
    "build_schedule",
    "EnergyPolicy",
    "EnergyBudget",
    "assess",
    "ReplicationPolicy",
    "ResilienceResult",
    "FailoverIndex",
    "betweenness",
    "plan_resilience",
    "reaim_interval",
    "Op",
    "Instruction",
    "DeviceProgram",
    "Manifest",
    "emit",
    "CliqueSolver",
    "CliqueResult",
    "solve_clique",
    "balanced_factors",
    "embed_graph",
    "traffic_distance",
    "minimum_devices",
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

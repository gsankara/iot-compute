"""Mobility: how bodies move, and where they are trying to go.

Two independent plugin axes:

``kinematics``
    The physics of one body -- :class:`DroneKinematics`, :class:`BuoyantKinematics`,
    :class:`StaticKinematics`.

``topology``
    The collective intent -- :class:`GradientTopology`, :class:`FlockTopology`,
    :class:`GravityTopology`, :class:`AnnealedTopology`, :class:`FormationSequence`.
"""

from __future__ import annotations

from .kinematics import (
    BuoyantKinematics,
    DroneKinematics,
    Kinematics,
    MotionCost,
    MotionState,
    StaticKinematics,
    drag_force,
)
from .topology import (
    AnnealedTopology,
    FlockTopology,
    FormationSequence,
    GradientTopology,
    GravityTopology,
    StaticTopology,
    TopologyContext,
    TopologyDynamics,
)

__all__ = [
    "MotionState",
    "MotionCost",
    "Kinematics",
    "StaticKinematics",
    "DroneKinematics",
    "BuoyantKinematics",
    "drag_force",
    "TopologyContext",
    "TopologyDynamics",
    "StaticTopology",
    "GradientTopology",
    "FlockTopology",
    "GravityTopology",
    "AnnealedTopology",
    "FormationSequence",
]

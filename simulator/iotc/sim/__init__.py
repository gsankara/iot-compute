"""The simulator: medium, links, mobility, energy, clocks, engine, faults.

Importing this package pulls in every simulator plugin so that the registry is
fully populated before a config is parsed.
"""

from __future__ import annotations

from . import energy, faults, links, medium, mobility  # noqa: F401  (registration)
from .clocks import Clock, ClockDomain
from .energy import EnergyStore, PowerAccount
from .engine import EventQueue, Message, Simulation
from .faults import FaultModel, build_campaign
from .medium import Air, IndexFluid, Medium, Vacuum, Water
from .mobility import MotionState, TopologyContext
from .trace import Event, Trace
from .world import Device, Transceiver, World

__all__ = [
    "Medium",
    "Vacuum",
    "Air",
    "Water",
    "IndexFluid",
    "World",
    "Device",
    "Transceiver",
    "Simulation",
    "Message",
    "EventQueue",
    "Trace",
    "Event",
    "Clock",
    "ClockDomain",
    "EnergyStore",
    "PowerAccount",
    "MotionState",
    "TopologyContext",
    "FaultModel",
    "build_campaign",
    "links",
    "medium",
    "mobility",
    "energy",
    "faults",
]

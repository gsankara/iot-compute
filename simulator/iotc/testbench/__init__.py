"""Phase C -- large-scale emulation and evaluation.

Where Phase B asked "is each pass right", Phase C asks "does the whole thing
hold together at scale, and does it still hold when the world misbehaves".  It
compiles and executes complete runs, checks the five invariants
``docs/ROADMAP.md`` names on every one, and classifies each into pass, degraded,
wrong or hung.

Built as an unoptimized reference by decision: time optimization comes later and
is expected to be hardware (GPU or FPGA) rather than CPU tuning, so a slow
campaign here is not a defect.
"""

from __future__ import annotations

from . import campaign, invariants
from .campaign import (
    DEGRADED,
    HUNG,
    PASS,
    WRONG,
    GoldenReference,
    RunResult,
    run_once,
    scale_demonstration,
    tally,
    violation_tally,
)
from .invariants import INVARIANTS, Violation, check_all

__all__ = [
    "campaign",
    "invariants",
    "GoldenReference",
    "RunResult",
    "run_once",
    "scale_demonstration",
    "tally",
    "violation_tally",
    "Violation",
    "check_all",
    "INVARIANTS",
    "PASS",
    "DEGRADED",
    "WRONG",
    "HUNG",
]

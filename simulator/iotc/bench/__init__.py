"""Evaluation harnesses.

Phase A ships :mod:`iotc.bench.sim_bench`, which sweeps the simulator's design
space -- medium, platform, device mix, channel model -- and writes tidy CSV.
Phase B adds one bench per compiler pass on top of the same aggregation and
reporting code, starting with :mod:`iotc.bench.ir_bench` for B0.  The
large-scale fault campaigns land here too.
"""

from __future__ import annotations

from . import ir_bench
from .ir_bench import characterize
from .sim_bench import (
    aggregate,
    expand,
    get_path,
    plot_sweep,
    run_one,
    run_sweep,
    set_path,
    write_csv,
)

__all__ = [
    "ir_bench",
    "characterize",
    "run_sweep",
    "run_one",
    "expand",
    "aggregate",
    "write_csv",
    "plot_sweep",
    "set_path",
    "get_path",
]

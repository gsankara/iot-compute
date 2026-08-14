"""Evaluation harnesses.

Phase A ships :mod:`iotc.bench.sim_bench`, which sweeps the simulator's design
space -- medium, platform, device mix, channel model -- and writes tidy CSV.
Phase B adds one bench per compiler pass on top of the same aggregation and
reporting code, starting with :mod:`iotc.bench.ir_bench` for B0.  The
large-scale fault campaigns land here too.
"""

from __future__ import annotations

from . import (beam_bench, energy_bench, ir_bench, motion_bench, partition_bench,
               place_bench, resilience_bench, schedule_bench, tile_bench)
from .ir_bench import characterize
from .partition_bench import fitted_exponent, scaling
from .tile_bench import evaluate
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
    "tile_bench",
    "partition_bench",
    "place_bench",
    "motion_bench",
    "beam_bench",
    "schedule_bench",
    "energy_bench",
    "resilience_bench",
    "characterize",
    "evaluate",
    "scaling",
    "fitted_exponent",
    "run_sweep",
    "run_one",
    "expand",
    "aggregate",
    "write_csv",
    "plot_sweep",
    "set_path",
    "get_path",
]

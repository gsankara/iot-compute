"""Evaluation harnesses.

Phase A ships :mod:`iotc.bench.sim_bench`, which sweeps the simulator's design
space -- medium, platform, device mix, channel model -- and writes tidy CSV.
The compiler-pass benches and the large-scale fault campaigns land on top of the
same aggregation and reporting code.
"""

from __future__ import annotations

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
    "run_sweep",
    "run_one",
    "expand",
    "aggregate",
    "write_csv",
    "plot_sweep",
    "set_path",
    "get_path",
]

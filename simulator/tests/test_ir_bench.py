"""B0's evaluation harness: does the characterization say true things?"""

from __future__ import annotations

import json

import numpy as np
import pytest

from iotc.bench import ir_bench
from iotc.compiler import build_workload
from iotc.compiler.ir import TaskGraph
from iotc.devices import MOTE_C, MOTE_M
from iotc.registry import available


def test_characterize_reports_the_fields_later_passes_read():
    row = ir_bench.characterize(build_workload("stencil3d"), MOTE_C)
    for key in (
        "workload", "device_class", "tasks", "edges", "ws_over_ram",
        "fit_fraction", "tiles_needed", "arithmetic_intensity", "aim_fraction",
        "degree_over_transceivers",
    ):
        assert key in row


def test_fit_fraction_tracks_the_class_it_is_scored_against():
    """The same workload is feasible or not depending on the part you pick."""
    ram_c = MOTE_C.compute.ram_bytes
    g = TaskGraph.from_arrays(
        [1.0, 1.0], [ram_c * 0.5, ram_c * 1.5], [0], [1], [1024.0]
    )
    assert ir_bench.characterize(g, MOTE_C)["fit_fraction"] == 0.5
    # The memory-rich part swallows both tasks.
    assert MOTE_M.compute.ram_bytes > ram_c
    assert ir_bench.characterize(g, MOTE_M)["fit_fraction"] == 1.0


def test_tiles_needed_is_the_ceiling_of_the_overflow():
    ram = MOTE_C.compute.ram_bytes
    g = TaskGraph.from_arrays([1.0, 1.0], [ram * 2.5, 1.0], [0], [1], [1.0])
    row = ir_bench.characterize(g, MOTE_C)
    assert row["ws_over_ram"] == pytest.approx(2.5)
    assert row["tiles_needed"] == 3


@pytest.mark.parametrize("name", sorted(available("workload")))
def test_aim_fraction_is_a_fraction(name):
    row = ir_bench.characterize(build_workload(name), MOTE_C)
    assert 0.0 <= row["aim_fraction"] <= 1.0


def test_every_stock_workload_is_aim_dominated_on_the_small_part():
    """The headline B0 result, and the reason B6 exists.

    Not one of the stock workloads spends most of its link time sending.  No
    choice available to B0 or B1 changes that -- only amortising the retarget
    does, which is B6's job.
    """
    rows = [r for r in ir_bench.run() if r["device_class"] == "mote-c"]
    assert len(rows) == 12
    assert all(r["aim_fraction"] > 0.9 for r in rows)


def test_the_verdict_names_the_pass_that_has_to_act():
    rows = ir_bench.run(classes=("mote-c",))
    text = ir_bench.verdict(rows)
    assert "B1 must tile" in text
    assert "B6 must amortise aiming" in text
    # The FFT holds two 1 MiB buffers, which does not fit a mote-c.
    tiling_line = [ln for ln in text.splitlines() if ln.startswith("B1")][0]
    assert "fft" in tiling_line


def test_run_writes_tidy_output(tmp_path):
    rows = ir_bench.run(["stencil2d", "gemm"], classes=("mote-c",), out=tmp_path)
    assert (tmp_path / "characterization.csv").exists()
    saved = json.loads((tmp_path / "characterization.json").read_text())
    assert len(saved) == len(rows) == 2


def test_characterization_is_reproducible():
    a = ir_bench.run(["spmv"], classes=("relay",))
    b = ir_bench.run(["spmv"], classes=("relay",))
    assert a == b

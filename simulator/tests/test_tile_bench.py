"""B1's evaluation harness: does the tiling report say true things?"""

from __future__ import annotations

import json

import pytest

from iotc.bench import tile_bench
from iotc.compiler import build_workload
from iotc.devices import MOTE_C


def test_evaluate_reports_both_halves_of_the_metric():
    """A tiling is scored on whether it fits *and* what fitting cost."""
    g = build_workload("stencil3d", **tile_bench.STRESS_PARAMS["stencil3d"])
    row = tile_bench.evaluate(g, MOTE_C, "geometric")
    assert row["fits"] is True
    assert row["tasks_after"] > row["tasks_before"]
    assert row["traffic_ratio"] > 1.0
    assert row["cut_shape"] == "3x3x2"


def test_the_stress_instances_actually_stress_the_part():
    """A tiling bench run on problems that already fit measures nothing."""
    tiled = 0
    for name, params in tile_bench.STRESS_PARAMS.items():
        if not build_workload(name, **params).fits(MOTE_C).all():
            tiled += 1
    assert tiled >= 6


def test_geometric_beats_slab_across_the_family():
    rows = tile_bench.run(
        ["stencil2d", "stencil3d", "gemm"], classes=("mote-c",)
    )
    by = {(r["workload"], r["tiler"]): r for r in rows}
    for w in ("stencil2d", "stencil3d", "summa"):
        assert by[(w, "geometric")]["mib_after"] < by[(w, "slab")]["mib_after"]


def test_a_two_dimensional_stencil_is_never_cut_in_three():
    """Cutting an axis the task does not have would be inventing geometry."""
    rows = tile_bench.run(["stencil2d"], classes=("mote-c",), models=("geometric",))
    assert rows[0]["cut_shape"].count("x") == 1


def test_the_verdict_names_what_cannot_be_tiled():
    rows = tile_bench.run(["fft", "stencil3d"], classes=("mote-c",))
    text = tile_bench.verdict(rows)
    assert "cannot be tiled" in text
    assert "fft" in text


def test_run_writes_tidy_output(tmp_path):
    rows = tile_bench.run(
        ["stencil3d"], classes=("mote-c",), models=("geometric",), out=tmp_path
    )
    assert (tmp_path / "tiling.csv").exists()
    saved = json.loads((tmp_path / "tiling.json").read_text())
    assert len(saved) == len(rows) == 1


def test_evaluation_is_reproducible():
    a = tile_bench.run(["stencil3d"], classes=("mote-c",), models=("geometric",))
    b = tile_bench.run(["stencil3d"], classes=("mote-c",), models=("geometric",))
    assert a == b

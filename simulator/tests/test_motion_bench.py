"""B4's evaluation harness."""

from __future__ import annotations

import json

from iotc.bench import motion_bench as mb


def test_the_trade_is_reported_in_both_directions():
    rows = mb.run(["stencil3d"])
    text = mb.verdict(rows)
    assert "drops 100% of protected links" in text
    assert "the price is makespan" in text


def test_waves_are_safe_and_far_faster_than_one_at_a_time():
    rows = mb.run(["stencil3d"])
    by = {r["planner"]: r for r in rows}
    assert by["direct"]["broken_link_fraction"] == 1.0
    assert by["waves"]["broken_link_fraction"] == 0.0
    assert by["sequential"]["broken_link_fraction"] == 0.0
    assert by["waves"]["waves"] < by["sequential"]["waves"]
    assert by["waves"]["makespan_s"] < by["sequential"]["makespan_s"]


def test_energy_is_unchanged_by_ordering():
    """If the planners differed in energy the comparison would be unfair."""
    rows = mb.run(["stencil3d"])
    assert len({round(r["energy_j"], 6) for r in rows}) == 1
    assert "energy is identical across planners" in mb.verdict(rows)


def test_run_writes_tidy_output(tmp_path):
    rows = mb.run(["stencil3d"], planners=("waves",), out=tmp_path)
    assert (tmp_path / "motion.csv").exists()
    assert len(json.loads((tmp_path / "motion.json").read_text())) == len(rows) == 1


def test_evaluation_is_reproducible():
    assert mb.run(["spmv"], planners=("waves",)) == mb.run(["spmv"], planners=("waves",))

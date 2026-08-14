"""B8's evaluation harness."""

from __future__ import annotations

import json

from iotc.bench import resilience_bench as rb


def test_redundancy_beats_no_redundancy():
    rows = rb.run(["stencil3d"], policies=("none", "traffic"), fractions=(0.3,))
    by = {r["policy"]: r["surviving_traffic"] for r in rows}
    assert by["traffic"] > by["none"]


def test_survival_falls_as_more_devices_die():
    rows = rb.run(["stencil3d"], policies=("traffic",))
    ordered = sorted(rows, key=lambda r: r["kill_fraction"])
    values = [r["surviving_traffic"] for r in ordered]
    assert values == sorted(values, reverse=True)


def test_comparisons_carry_an_error_bar():
    """A win that does not clear the noise is not a win."""
    rows = rb.run(["stencil3d"], policies=("uniform", "traffic"), fractions=(0.3,))
    assert all(r["stderr"] > 0 for r in rows)
    text = rb.verdict(rows, [])
    assert "+/-" in text
    assert "significant" in text


def test_the_verdict_reports_whether_betweenness_earns_its_cost():
    """The expensive policy has to justify itself against the cheap proxy."""
    rows = rb.run(["stencil3d"], policies=("traffic", "centrality"), fractions=(0.3,))
    text = rb.verdict(rows, [])
    assert "sampled betweenness over the cheap traffic proxy" in text


def test_dense_device_graphs_are_flagged():
    """B2 leaves little structure for centrality to exploit; say so."""
    rows = rb.run(["spmv"], policies=("traffic",), fractions=(0.3,))
    assert rows[0]["degree_mean"] >= rows[0]["devices"] - 1.5
    assert "dense" in rb.verdict(rows, [])


def test_the_tracker_table_covers_still_and_drifting_media():
    rows = rb.tracker_rows()
    assert any(r["drift_m_per_sqrt_s"] == 0.0 and not r["needed"] for r in rows)
    assert any(r["drift_m_per_sqrt_s"] == 0.04 and r["needed"] for r in rows)


def test_phase_a_drift_needs_many_reaims_per_frame():
    rows = rb.tracker_rows()
    hard = [r for r in rows
            if r["drift_m_per_sqrt_s"] == 0.04 and r["tolerance_m"] == 0.02][0]
    assert hard["reaims_per_frame"] > 10
    assert hard["energy_per_frame_mj"] > 0


def test_run_writes_tidy_output(tmp_path):
    rows = rb.run(["stencil3d"], policies=("traffic",), fractions=(0.3,), out=tmp_path)
    assert (tmp_path / "resilience.csv").exists()
    assert len(json.loads((tmp_path / "resilience.json").read_text())) == len(rows)


def test_evaluation_is_reproducible():
    kw = dict(policies=("centrality",), fractions=(0.3,))
    assert rb.run(["stencil3d"], **kw) == rb.run(["stencil3d"], **kw)

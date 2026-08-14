"""B7's evaluation harness."""

from __future__ import annotations

import json

from iotc.bench import energy_bench as eb


def test_the_dominant_term_is_counted_not_assumed():
    """Everything before B7 was about aiming; the energy says otherwise."""
    rows = eb.run(["stencil3d"], policies=("ignore",))
    text = eb.verdict(rows)
    assert "dominant energy term" in text
    assert "aim-dominated in TIME but not in energy" in text
    assert all(r["retarget_mj"] < r["draw_mj"] for r in rows)


def test_the_bench_reports_which_configurations_drain():
    rows = eb.run(["stencil3d"], policies=("ignore",))
    assert any(not r["sustainable"] for r in rows)
    assert "drain their store" in eb.verdict(rows)


def test_the_duty_cycle_needed_is_named_per_configuration():
    rows = eb.run(["stencil3d"], platforms=("float-m",))
    text = eb.verdict(rows)
    assert "must drop to a" in text
    assert "duty cycle to be sustainable" in text


def test_a_fixed_half_duty_rule_is_shown_to_be_insufficient():
    rows = eb.run(["stencil3d"], platforms=("float-m",))
    half = [r for r in rows if r["policy"] == "half_duty"][0]
    computed = [r for r in rows if r["policy"] == "duty_cycle"][0]
    assert not half["sustainable"]
    assert computed["sustainable"]


def test_terms_are_reported_per_row():
    rows = eb.run(["spmv"], platforms=("float",), policies=("ignore",))
    for key in ("retarget_mj", "tx_mj", "rx_mj", "compute_mj", "idle_mj", "leak_mj"):
        assert key in rows[0]


def test_run_writes_tidy_output(tmp_path):
    rows = eb.run(["spmv"], platforms=("float",), policies=("ignore",), out=tmp_path)
    assert (tmp_path / "energy.csv").exists()
    assert len(json.loads((tmp_path / "energy.json").read_text())) == len(rows) == 1


def test_evaluation_is_reproducible():
    kw = dict(platforms=("float",), policies=("duty_cycle",))
    assert eb.run(["spmv"], **kw) == eb.run(["spmv"], **kw)

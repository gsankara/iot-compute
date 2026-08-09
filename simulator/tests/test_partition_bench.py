"""B2's evaluation harness, including the near-linearity acceptance criterion."""

from __future__ import annotations

import json

import pytest

from iotc.bench import partition_bench as pb


def test_evaluate_reports_quality_and_feasibility_together():
    rows = pb.run(["stencil3d"], methods=("fmbm",))
    row = rows[0]
    assert row["method"] == "fmbm"
    assert 0.0 <= row["cut_fraction"] <= 1.0
    assert row["k"] >= row["min_devices"]
    assert row["feasible"] is True


def test_the_quality_instances_leave_the_partitioner_something_to_decide():
    """If each task fills a device, k is forced to n and every edge is cut."""
    from iotc.compiler import build_workload
    from iotc.compiler.partition import minimum_devices
    from iotc.devices import MOTE_C

    for name, params in pb.QUALITY_PARAMS.items():
        g = build_workload(name, **params)
        assert minimum_devices(g, MOTE_C) < g.n_tasks, name


def test_fitted_exponent_recovers_a_known_slope():
    """Sanity-check the estimator before trusting what it says about the pass."""
    rows = [{"method": "m", "tasks": n, "seconds": 1e-6 * n**2} for n in (100, 200, 400, 800)]
    assert pb.fitted_exponent(rows, "m") == pytest.approx(2.0, abs=0.01)
    rows = [{"method": "m", "tasks": n, "seconds": 1e-6 * n} for n in (100, 200, 400, 800)]
    assert pb.fitted_exponent(rows, "m") == pytest.approx(1.0, abs=0.01)


def test_fitted_exponent_is_nan_without_enough_points():
    assert pb.fitted_exponent([], "m") != pb.fitted_exponent([], "m")  # NaN


def test_fmbm_is_near_linear_and_beats_its_comparators_on_scaling():
    """B2's acceptance criterion, measured rather than asserted.

    The ladder is short so the tolerance is loose, but the ordering is the
    claim: FastMap partitioning must grow slower than spectral bisection and
    Kernighan-Lin, both of which are carried precisely because they do not
    scale.
    """
    rows = pb.scaling(sides=(6, 10, 14))
    fmbm = pb.fitted_exponent(rows, "fmbm")
    assert fmbm < 1.4, f"fmbm is not near-linear: n^{fmbm:.2f}"
    assert fmbm < pb.fitted_exponent(rows, "spectral")
    assert fmbm < pb.fitted_exponent(rows, "kernighan_lin")


def test_the_scaling_ladder_warms_up_before_timing():
    """A cold first measurement on a short ladder can invert the fitted slope."""
    rows = pb.scaling(methods=("block",), sides=(6, 8))
    assert all(r["seconds"] > 0 for r in rows)
    assert len(rows) == 2


def test_verdict_states_the_complexity_and_the_comparison():
    quality = pb.run(["stencil3d"], methods=("fmbm", "block"))
    scale = pb.scaling(methods=("fmbm", "block"), sides=(6, 8))
    text = pb.verdict(quality, scale)
    assert "empirical complexity" in text
    assert "fmbm" in text


def test_run_writes_tidy_output(tmp_path):
    pb.run(["stencil3d"], methods=("fmbm",), out=tmp_path)
    assert (tmp_path / "partition.csv").exists()
    assert len(json.loads((tmp_path / "partition.json").read_text())) == 1

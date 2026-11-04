"""B11's bench: that it runs, that its rows are consistent, and that its
verdict does not credit a setting for failing to deliver."""

from __future__ import annotations

import numpy as np
import pytest

from iotc.bench import broadcast_bench as bb

SMALL = {
    "stencil3d": dict(nx=64, ny=64, nz=64, tx=4, ty=4, tz=4),
    "allreduce": dict(count=1 << 12, p=16, topology="tree"),
}


@pytest.fixture(scope="module")
def rows():
    return bb.run(params=SMALL)


def test_every_platform_and_model_produces_a_row(rows):
    assert {r["platform"] for r in rows} == set(bb.PLATFORMS)
    assert {r["model"] for r in rows} == set(bb.DEFAULT_MODELS)
    assert {r["workload"] for r in rows} == set(SMALL)
    assert len(rows) == len(SMALL) * len(bb.PLATFORMS) * len(bb.DEFAULT_MODELS)


def test_row_quantities_are_internally_consistent(rows):
    for r in rows:
        assert r["slots"] >= 1
        assert r["reuse"] == pytest.approx(r["devices"] / r["slots"])
        assert r["per_node_bps"] == pytest.approx(r["line_rate_bps"] / r["slots"])
        assert 0.0 <= r["uncovered_fraction"] <= 1.0
        assert r["feasible"] == (r["uncovered_fraction"] == 0.0)
        assert r["frame_private_s"] >= r["frame_shared_s"] > 0.0
        assert r["mean_domain"] <= r["max_domain"] <= r["devices"] - 1
        assert r["reach_over_extent"] == pytest.approx(
            r["reach_m"] / r["extent_m"]
        ) or r["extent_m"] == 0.0


def test_the_optical_comparison_is_like_for_like(rows):
    """Both planes must be priced on the same workload, partition and
    placement, so the optical frame may not vary with the broadcast model."""
    for key in {(r["workload"], r["platform"]) for r in rows}:
        same = [r for r in rows if (r["workload"], r["platform"]) == key]
        periods = {round(r["optical_frame_s"], 12) for r in same}
        assert len(periods) == 1, f"optical frame moved with the radio model: {key}"


def test_a_dead_optical_plane_is_not_scored_as_infinitely_fast(rows):
    """At small device counts a float's column is entirely unaimable even after
    B5's refinement: every link is rejected, so there are no rounds and the
    optical frame is zero.  Dividing by that would report broadcast as
    infinitely worse, when the truth is that the optical plane carries nothing.
    Such rows must come back uncomparable, not scored."""
    for r in rows:
        if not r["optical_feasible"]:
            assert r["optical_frame_s"] == 0.0
            assert r["optical_rejected_fraction"] == pytest.approx(1.0)
            assert not r["comparable"]
            assert np.isnan(r["speedup_shared"]) and np.isnan(r["speedup_private"])
        if r["comparable"]:
            assert r["optical_frame_s"] > 0.0 and r["feasible"]
            assert r["speedup_shared"] > 0.0


def test_min_reach_spends_no_more_power_than_nominal(rows):
    for key in {(r["workload"], r["platform"]) for r in rows}:
        byname = {
            r["model"]: r for r in rows if (r["workload"], r["platform"]) == key
        }
        assert byname["min_reach"]["tx_power_w"] <= byname["nominal"]["tx_power_w"]


def test_swept_is_never_worse_covered_than_min_reach(rows):
    """The sweep sees every setting ``min_reach`` can choose, so it must never
    come back with worse coverage."""
    for key in {(r["workload"], r["platform"]) for r in rows}:
        byname = {
            r["model"]: r for r in rows if (r["workload"], r["platform"]) == key
        }
        assert (
            byname["swept"]["uncovered_fraction"]
            <= byname["min_reach"]["uncovered_fraction"] + 1e-12
        )


def test_the_verdict_does_not_credit_a_setting_for_not_delivering():
    """Regression on the reporting, not the pass: an earlier verdict ranked
    models by mean reuse across *all* rows, so a radio too weak to reach its
    neighbours scored best -- it has no conflicts because it has no links."""
    common = dict(
        platform="p", edges=10, optical_efficiency=0.01, optical_feasible=True,
        speedup_shared=1.0, speedup_private=1.0,
    )
    rows = [
        dict(common, model="nominal", reuse=9.0, feasible=False, comparable=False,
             uncovered_fraction=0.9, reach_over_extent=0.01, reach_m=0.1),
        dict(common, model="min_reach", reuse=1.0, feasible=True, comparable=True,
             uncovered_fraction=0.0, reach_over_extent=2.0, reach_m=50.0),
    ]
    text = bb.verdict(rows)
    assert "nominal lifts" not in text
    assert "9.00x" not in text, "the verdict credited the undelivering setting"


def test_scaling_sweep_is_monotonic_in_device_count():
    sweep = bb.scaling(sides=(6, 10, 14))
    counts = [r["devices"] for r in sweep]
    assert counts == sorted(counts) and len(set(counts)) == len(counts)
    for r in sweep:
        assert r["slots"] >= 1
        assert r["reuse"] >= 1.0 or r["devices"] < r["slots"]
        assert r["domains_across"] > 0.0


def test_reports_render(rows):
    assert bb.report(rows)
    assert bb.report(rows, platform="drone")
    assert bb.ladder_report("float")
    assert bb.ladder_report("drone")
    assert bb.verdict(rows)


def test_ladder_moves_rate_and_range_together():
    """A low carrier must not be allowed to keep a high line rate: that would
    manufacture capacity the physics does not provide."""
    carriers = [c for c, _, _ in bb.CARRIER_LADDER]
    rates = [r for _, r, _ in bb.CARRIER_LADDER]
    assert carriers == sorted(carriers, reverse=True)
    assert rates == sorted(rates, reverse=True)

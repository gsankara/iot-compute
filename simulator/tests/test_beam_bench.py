"""B5's evaluation harness."""

from __future__ import annotations

import json

from iotc.bench import beam_bench as bb


def test_the_oversubscription_is_reported_as_the_reason_for_rounds():
    rows = bb.run(["stencil3d"], platforms=("float",), assigners=("heaviest_first",))
    assert all(r["oversubscription"] > 1.0 for r in rows)
    assert "cannot be a topology" in bb.verdict(rows)


def test_the_float_column_is_reported_as_unaimable_and_then_recovered():
    """The cross-pass result: B3's energy-optimal placement cannot be aimed."""
    rows = bb.run(["stencil3d"], platforms=("float",), assigners=("heaviest_first",))
    raw = [r for r in rows if not r["refined"]][0]
    fixed = [r for r in rows if r["refined"]][0]
    assert raw["rejected_fraction"] > 0.9
    assert fixed["rejected_fraction"] < 0.2
    assert "unaimable" in bb.verdict(rows)


def test_more_heads_means_fewer_rounds():
    rows = bb.run(["stencil3d"], platforms=("float", "relay"),
                  assigners=("heaviest_first",))
    by = {r["platform"]: r for r in rows if r["refined"]}
    assert by["relay"]["transceivers"] > by["float"]["transceivers"]
    assert by["relay"]["rounds"] <= by["float"]["rounds"]


def test_ordering_moves_traffic_without_moving_rounds():
    rows = [r for r in bb.run(["stencil3d"], platforms=("relay",)) if r["refined"]]
    by = {r["assigner"]: r for r in rows}
    assert by["heaviest_first"]["first_round_traffic"] > by["lightest_first"]["first_round_traffic"]
    assert by["heaviest_first"]["rounds"] == by["lightest_first"]["rounds"]


def test_every_assignment_respects_the_head_count():
    rows = bb.run(["stencil3d"], platforms=("float", "drone", "relay"))
    assert all(r["feasible"] for r in rows)
    assert "every assignment is realisable" in bb.verdict(rows)


def test_run_writes_tidy_output(tmp_path):
    rows = bb.run(["stencil3d"], platforms=("relay",), assigners=("heaviest_first",),
                  out=tmp_path)
    assert (tmp_path / "beam.csv").exists()
    assert len(json.loads((tmp_path / "beam.json").read_text())) == len(rows)


def test_evaluation_is_reproducible():
    kw = dict(platforms=("relay",), assigners=("heaviest_first",))
    assert bb.run(["stencil3d"], **kw) == bb.run(["stencil3d"], **kw)

"""B3's evaluation harness."""

from __future__ import annotations

import json

from iotc.bench import place_bench as pb


def test_budget_report_covers_every_platform():
    text = pb.budget_report()
    for platform in pb.PLATFORMS:
        assert platform in text


def test_the_platforms_do_not_favour_the_same_axis():
    """A placer tuned to the float must be shown against the other two.

    The float's cheap axis is depth, the tethered head's is lateral, and the
    drone is nearly isotropic.  Anisotropy differs in *direction*, not degree.
    """
    rows = pb.run(["stencil3d"], placers=("anisotropic",))
    aniso = {r["platform"]: r["anisotropy"] for r in rows}
    assert aniso["float"] > 10.0        # depth is nearly free
    assert aniso["tether"] < 1.0        # the wire limits depth, the gantry does not
    assert 0.1 < aniso["drone"] < 10.0  # both axes cost


def test_energy_placement_wins_on_the_float_and_reports_it():
    rows = pb.run(["stencil3d"], platforms=("float",))
    text = pb.verdict(rows)
    assert "cheapest placer is anisotropic" in text
    assert "cheaper than the velocity reading" in text


def test_every_placement_is_connected():
    rows = pb.run(["stencil3d", "spmv"])
    assert all(r["unreachable"] == 0 for r in rows)
    assert "every placement is connected" in pb.verdict(rows)


def test_free_motion_is_reported_as_unrankable_rather_than_a_win():
    """A tethered head moves at zero node energy; ranking placers on it is noise."""
    rows = pb.run(["stencil3d"], platforms=("tether",))
    assert "energy cannot rank it" in pb.verdict(rows)


def test_run_writes_tidy_output(tmp_path):
    rows = pb.run(["stencil3d"], platforms=("float",), placers=("anisotropic",),
                  out=tmp_path)
    assert (tmp_path / "placement.csv").exists()
    assert len(json.loads((tmp_path / "placement.json").read_text())) == len(rows) == 1


def test_evaluation_is_reproducible():
    a = pb.run(["stencil3d"], platforms=("float",))
    b = pb.run(["stencil3d"], platforms=("float",))
    assert a == b

"""B10's evaluation harness."""

from __future__ import annotations

import json

from iotc.bench import clique_bench as cb


def test_every_answer_is_independently_verified():
    rows = cb.run(sizes=(400,), solvers=("physical", "greedy_degeneracy"))
    assert all(r["verified"] for r in rows)
    assert "are genuine cliques" in cb.verdict(rows)


def test_the_planted_clique_is_the_ground_truth():
    rows = cb.run(sizes=(400,), solvers=("physical",))
    assert rows[0]["planted"] == cb.PLANTED
    assert rows[0]["recovered_fraction"] == 1.0


def test_a_failing_solver_is_excluded_from_the_cost_comparison():
    """Its low test count is the cost of not answering, not efficiency.

    Annealing collapses on the larger instances, and ranking it as cheap
    alongside solvers that actually found the clique would invert the reading.
    """
    rows = cb.run(sizes=(400, 1500), solvers=("physical", "annealing"))
    text = cb.verdict(rows)
    assert "among solvers that found the clique" in text
    assert "the cost of not answering" in text


def test_the_verification_claim_is_stated_as_arithmetic():
    rows = cb.run(sizes=(400,), solvers=("physical",))
    text = cb.verdict(rows)
    assert "1 broadcast round" in text
    assert "pairwise tests conventionally" in text


def test_the_wall_clock_loss_is_reported_not_hidden():
    rows = cb.run(sizes=(400,), solvers=("physical", "greedy_degeneracy"))
    assert "slowest in wall-clock" in cb.verdict(rows)


def test_the_physical_axes_are_priced():
    rows = cb.run(sizes=(400,), solvers=("physical",))
    r = rows[0]
    assert r["converge_energy_j"] > 0
    assert r["broadcast_photons"] > 0
    assert r["aim_seconds"] > 0


def test_run_writes_tidy_output(tmp_path):
    rows = cb.run(sizes=(400,), solvers=("physical",), out=tmp_path)
    assert (tmp_path / "clique.csv").exists()
    assert len(json.loads((tmp_path / "clique.json").read_text())) == len(rows) == 1


def test_evaluation_is_reproducible():
    """Everything the bench *computes* is reproducible; the clock is not.

    This bench reports wall-clock as a metric in its own right -- the physical
    solver's slowness is one of its findings -- so ``seconds`` is deliberately
    part of the row and just as deliberately excluded from this comparison.
    Asserting on it would be asserting that two runs take identically long.
    """
    kw = dict(sizes=(400,), solvers=("physical",))
    a, b = cb.run(**kw), cb.run(**kw)
    assert [{k: v for k, v in row.items() if k != "seconds"} for row in a] == \
           [{k: v for k, v in row.items() if k != "seconds"} for row in b]
    assert all(row["seconds"] > 0 for row in a)

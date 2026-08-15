"""B10 Maximum clique: the flagship, and the claim it is actually making."""

from __future__ import annotations

import numpy as np
import pytest

from iotc.compiler.clique import (
    adjacency_sets,
    pairwise_tests_for,
    solve_clique,
)
from iotc.compiler.frontends import graphs
from iotc.registry import available

SOLVERS = sorted(available("clique_solver"))


def _planted(n: int = 400, k: int = 14, seed: int = 1):
    u, v, members = graphs.planted_clique(n, 10.0, k, seed=seed)
    return n, u, v, members


# ------------------------------------------------------------------ soundness

@pytest.mark.parametrize("solver", SOLVERS)
def test_every_answer_is_a_genuine_clique(solver):
    """Checked independently, so a solver cannot grade its own homework."""
    n, u, v, _ = _planted()
    result = solve_clique(n, u, v, model=solver, seed=0)
    assert result.verified
    assert result.is_clique(adjacency_sets(n, u, v))


@pytest.mark.parametrize("solver", SOLVERS)
def test_solving_is_deterministic(solver):
    n, u, v, _ = _planted()
    a = solve_clique(n, u, v, model=solver, seed=0)
    b = solve_clique(n, u, v, model=solver, seed=0)
    assert a.members == b.members


def test_an_empty_graph_does_not_crash_the_solvers():
    empty = np.zeros(0, dtype=np.int64)
    for solver in SOLVERS:
        result = solve_clique(4, empty, empty, model=solver, seed=0)
        assert result.size <= 1


def test_a_complete_graph_is_entirely_one_clique():
    n = 8
    idx = np.triu_indices(n, k=1)
    result = solve_clique(n, idx[0], idx[1], model="bron_kerbosch", seed=0)
    assert result.size == n


# -------------------------------------------------------------------- quality

def test_the_physical_scheme_recovers_a_planted_clique():
    n, u, v, members = _planted()
    result = solve_clique(n, u, v, model="physical", seed=0)
    assert result.size >= members.size
    assert set(members.tolist()) <= set(result.members)


def test_the_exact_solver_agrees_with_the_physical_one_on_size():
    n, u, v, _ = _planted(n=300, k=12)
    exact = solve_clique(n, u, v, model="bron_kerbosch", seed=0)
    physical = solve_clique(n, u, v, model="physical", seed=0)
    assert physical.size == exact.size


# ------------------------------------------------------------- the real claim

def test_verification_is_quadratic_conventionally_and_one_round_physically():
    """The narrow claim the flagship makes, stated as arithmetic.

    Confirming a k-clique needs every pair checked -- k(k-1)/2 lookups -- unless
    the members are in mutual line of sight, in which case one broadcast round
    tells everyone at once, whatever k is.
    """
    assert pairwise_tests_for(16) == 120
    assert pairwise_tests_for(64) == 2016
    # Quadratic: quadrupling k multiplies the conventional cost by ~16.
    assert pairwise_tests_for(64) / pairwise_tests_for(16) == pytest.approx(16.8, rel=0.05)


def test_the_physical_solver_uses_broadcast_rounds_and_the_others_do_not():
    n, u, v, _ = _planted()
    physical = solve_clique(n, u, v, model="physical", seed=0)
    greedy = solve_clique(n, u, v, model="greedy_degeneracy", seed=0)
    assert physical.broadcast_rounds > 0
    assert greedy.broadcast_rounds == 0


def test_broadcast_rounds_do_not_grow_with_the_problem():
    """The physical cost is set by how many balls are probed, not by n."""
    small = solve_clique(*_planted(n=400)[:3], model="physical", seed=0)
    large = solve_clique(*_planted(n=2000)[:3], model="physical", seed=0)
    assert large.broadcast_rounds == small.broadcast_rounds


def test_the_embedding_keeps_adjacency_tests_flatter_than_the_baselines():
    """What the common-neighbour weight buys: growth, not raw speed."""
    def tests(model, n):
        return solve_clique(*_planted(n=n)[:3], model=model, seed=0).adjacency_tests

    phys = tests("physical", 2000) / max(tests("physical", 400), 1)
    exact = tests("bron_kerbosch", 2000) / max(tests("bron_kerbosch", 400), 1)
    assert phys < exact


def test_the_physical_solver_is_slower_in_wall_clock_and_that_is_reported():
    """An honest limitation: the embedding is overhead a CPU need not pay.

    The pass's argument is about rounds of aiming on real hardware, not about
    Python microseconds, and the bench says so rather than burying it.
    """
    n, u, v, _ = _planted(n=2000)
    physical = solve_clique(n, u, v, model="physical", seed=0)
    greedy = solve_clique(n, u, v, model="greedy_degeneracy", seed=0)
    assert physical.seconds > greedy.seconds


# ------------------------------------------------------------------ machinery

def test_adjacency_sets_are_symmetric():
    n, u, v, _ = _planted(n=100, k=6)
    adj = adjacency_sets(n, u, v)
    for a, b in zip(u, v):
        assert int(b) in adj[int(a)] and int(a) in adj[int(b)]


def test_the_solvers_are_registered():
    assert {"physical", "bron_kerbosch", "greedy_degeneracy", "colour_bound",
            "annealing"} <= set(SOLVERS)

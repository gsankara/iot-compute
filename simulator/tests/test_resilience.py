"""B8 Resilience: replication, failover lookup, and the pointing tracker."""

from __future__ import annotations

import math

import numpy as np
import pytest

from iotc.compiler import build_workload
from iotc.compiler.ir import TaskGraph
from iotc.compiler.partition import partition
from iotc.compiler.resilience import (
    FailoverIndex,
    TrackerCost,
    betweenness,
    plan_resilience,
    reaim_interval,
    traffic_weight,
)
from iotc.devices import MOTE_C
from iotc.registry import available


def _devices():
    g = build_workload("stencil3d", nx=128, ny=128, nz=128, tx=8, ty=8, tz=8)
    return partition(g, MOTE_C, seed=0).coarsened()


def _barbell():
    """Two triangles joined through a single bridge vertex (index 3).

    Betweenness has an unambiguous answer here, which a mesh does not give.
    """
    src = [0, 1, 0, 2, 3, 4, 5, 4]
    dst = [1, 2, 2, 3, 4, 5, 6, 6]
    return TaskGraph.from_arrays(
        [1.0] * 7, [1.0] * 7, src, dst, [1024.0] * len(src)
    )


# ---------------------------------------------------------------- centrality

def test_betweenness_finds_the_bridge():
    score = betweenness(_barbell(), samples=7, seed=0)
    assert int(np.argmax(score)) in (3, 4)     # the two ends of the bridge
    assert score[3] > score[0]                 # and beats a triangle corner
    assert np.all(score >= 0)


def test_betweenness_is_zero_without_edges():
    g = TaskGraph.from_arrays([1.0, 1.0], [1.0, 1.0], [], [], [])
    assert np.all(betweenness(g) == 0.0)


def test_betweenness_is_deterministic():
    g = _devices()
    assert np.allclose(betweenness(g, 32, seed=1), betweenness(g, 32, seed=1))


def test_traffic_weight_matches_incident_payload():
    g = _barbell()
    assert np.allclose(traffic_weight(g), g.traffic_per_task())


# --------------------------------------------------------------- replication

@pytest.mark.parametrize("model", sorted(available("resilience")))
def test_every_policy_respects_its_budget(model):
    g = _devices()
    r = plan_resilience(g, budget=4, model=model, seed=0)
    assert r.budget <= 4
    if r.budget:
        assert r.replicated.min() >= 0 and r.replicated.max() < g.n_tasks
        assert len(set(r.replicated.tolist())) == r.budget


@pytest.mark.parametrize("model", sorted(available("resilience")))
def test_planning_is_deterministic(model):
    g = _devices()
    a = plan_resilience(g, 4, model=model, seed=0)
    b = plan_resilience(g, 4, model=model, seed=0)
    assert np.array_equal(a.replicated, b.replicated)


def test_no_replication_is_the_floor():
    g = _devices()
    n = g.n_tasks
    dead = np.zeros(n, dtype=bool)
    dead[:max(1, n // 5)] = True
    bare = plan_resilience(g, 0, model="none", seed=0)
    helped = plan_resilience(g, max(1, n // 5), model="traffic", seed=0)
    assert helped.surviving_traffic(dead) >= bare.surviving_traffic(dead)


def test_replicating_by_importance_beats_replicating_at_random():
    """The claim worth testing.  Averaged over trials, because a single draw of
    which devices die says nothing."""
    g = _devices()
    n = g.n_tasks
    budget = max(1, n // 5)
    scores = {}
    for model in ("uniform", "traffic"):
        r = plan_resilience(g, budget, model=model, seed=0)
        vals = []
        for trial in range(40):
            dead = np.zeros(n, dtype=bool)
            picks = np.random.default_rng(trial).choice(n, size=max(1, n // 3), replace=False)
            dead[picks] = True
            vals.append(r.surviving_traffic(dead))
        scores[model] = float(np.mean(vals))
    assert scores["traffic"] > scores["uniform"]


def test_a_replica_keeps_its_traffic_alive():
    g = _barbell()
    r = plan_resilience(g, 0, model="none", seed=0)
    dead = np.zeros(g.n_tasks, dtype=bool)
    dead[3] = True
    bare = r.surviving_traffic(dead)

    protected = plan_resilience(g, 1, model="traffic", seed=0)
    protected.replicated = np.array([3])
    assert protected.surviving_traffic(dead) > bare


def test_connectivity_measure_never_exceeds_direct_survival():
    """Traffic inside the largest component is a subset of surviving traffic."""
    g = _devices()
    n = g.n_tasks
    r = plan_resilience(g, 4, model="traffic", seed=0)
    for trial in range(10):
        dead = np.zeros(n, dtype=bool)
        dead[np.random.default_rng(trial).choice(n, size=n // 3, replace=False)] = True
        assert r.connected_traffic(dead) <= r.surviving_traffic(dead) + 1e-12


# ----------------------------------------------------------------- failover

def test_failover_returns_a_live_device_that_is_not_the_dead_one():
    g = _devices()
    coords = plan_resilience(g, 1, seed=0).coords
    index = FailoverIndex(coords, seed=0)
    alive = np.ones(g.n_tasks, dtype=bool)
    alive[3] = False
    pick = index.nearest_healthy(3, alive)
    assert pick != 3 and pick >= 0 and alive[pick]


def test_failover_reports_none_when_the_swarm_is_gone():
    g = _devices()
    index = FailoverIndex(plan_resilience(g, 1, seed=0).coords, seed=0)
    assert index.nearest_healthy(0, np.zeros(g.n_tasks, dtype=bool)) == -1


def test_failover_usually_finds_the_true_nearest():
    """LSH trades exactness for speed; it should not trade away too much."""
    g = _devices()
    coords = plan_resilience(g, 1, seed=0).coords
    index = FailoverIndex(coords, bits=6, seed=0)
    alive = np.ones(g.n_tasks, dtype=bool)
    hits = 0
    for device in range(g.n_tasks):
        alive[device] = False
        pick = index.nearest_healthy(device, alive)
        d = np.linalg.norm(coords - coords[device], axis=1)
        d[device] = np.inf
        hits += int(pick == int(np.argmin(d)))
        alive[device] = True
    assert hits >= g.n_tasks // 2


def test_failover_buckets_the_swarm_rather_than_scanning_it():
    g = _devices()
    index = FailoverIndex(plan_resilience(g, 1, seed=0).coords, bits=6, seed=0)
    assert len(index.buckets) > 1
    assert max(index.bucket_sizes()) < g.n_tasks


# ---------------------------------------------------------- pointing tracker

def test_the_reaim_interval_is_quadratic_in_tolerance():
    """Drift is a random walk, so displacement grows as sqrt(t).

    Halving the tolerance therefore quarters the interval -- a tighter beam
    costs slots, not just pointing loss.
    """
    wide = reaim_interval(0.04, 0.04)
    tight = reaim_interval(0.02, 0.04)
    assert wide == pytest.approx(4 * tight)


def test_a_still_medium_never_needs_re_aiming():
    assert math.isinf(reaim_interval(0.02, 0.0))


def test_phase_a_drift_forces_sub_second_tracking():
    """0.04 m/sqrt(s) against centimetres of tolerance, the case Phase A ran."""
    assert reaim_interval(0.02, 0.04) == pytest.approx(0.25)


def test_the_tracker_is_needed_when_the_frame_outlasts_the_tolerance():
    slow = TrackerCost(interval=10.0, frame=1.0, energy_per_reaim=1e-5, tracked_links=4)
    fast = TrackerCost(interval=0.25, frame=4.0, energy_per_reaim=1e-5, tracked_links=4)
    assert not slow.needed
    assert fast.needed
    assert fast.reaims_per_frame == pytest.approx(16.0)
    assert fast.energy_per_frame == pytest.approx(16.0 * 1e-5 * 4)


def test_a_tracker_with_no_drift_costs_nothing():
    free = TrackerCost(interval=math.inf, frame=4.0, energy_per_reaim=1e-5, tracked_links=8)
    assert free.reaims_per_frame == 0.0
    assert free.energy_per_frame == 0.0


def test_the_policies_are_registered():
    assert {"none", "uniform", "traffic", "centrality"} <= set(available("resilience"))

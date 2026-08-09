"""B2 Partition: what a grouping must guarantee, and what it should achieve."""

from __future__ import annotations

import numpy as np
import pytest

from iotc.compiler import build_workload
from iotc.compiler.partition import (
    kd_order,
    minimum_devices,
    partition,
    refine_boundary,
)
from iotc.devices import MOTE_C
from iotc.registry import available

#: A fine decomposition -- many tasks per device, so grouping has real freedom.
FINE = dict(nx=128, ny=128, nz=128, tx=8, ty=8, tz=8)


def _fine():
    return build_workload("stencil3d", **FINE)


# ------------------------------------------------------------------ guarantees

@pytest.mark.parametrize("method", sorted(available("partitioner")))
def test_every_partitioner_returns_a_total_valid_labelling(method):
    g = _fine()
    r = partition(g, MOTE_C, model=method, seed=0)
    assert r.labels.shape == (g.n_tasks,)
    assert r.labels.min() >= 0
    assert r.labels.max() < r.k


@pytest.mark.parametrize("method", sorted(available("partitioner")))
def test_partitioning_is_deterministic(method):
    g = _fine()
    a = partition(g, MOTE_C, model=method, seed=0)
    b = partition(g, MOTE_C, model=method, seed=0)
    assert np.array_equal(a.labels, b.labels)


def test_cut_and_internal_traffic_account_for_every_byte():
    g = _fine()
    r = partition(g, MOTE_C, seed=0)
    e = g.edges
    cross = r.labels[e[:, 0]] != r.labels[e[:, 1]]
    assert r.cut_bytes + g.payload[~cross].sum() == pytest.approx(g.payload.sum())
    assert 0.0 <= r.cut_fraction <= 1.0


def test_working_set_is_conserved_across_groups():
    g = _fine()
    r = partition(g, MOTE_C, seed=0)
    assert r.group_working_set.sum() == pytest.approx(g.working_set.sum())


def test_minimum_devices_is_a_real_lower_bound():
    g = _fine()
    need = minimum_devices(g, MOTE_C)
    ram = MOTE_C.compute.ram_bytes
    assert need * ram >= g.working_set.sum()
    assert (need - 1) * ram < g.working_set.sum()


def test_coarsening_keeps_only_the_cut():
    """What B3 places is one task per device, carrying just the optical traffic."""
    g = _fine()
    r = partition(g, MOTE_C, seed=0)
    coarse = r.coarsened()
    coarse.validate()
    assert coarse.n_tasks == r.k
    assert coarse.total_flop == pytest.approx(g.total_flop)
    assert coarse.payload.sum() == pytest.approx(r.cut_bytes)


def test_zero_ram_is_rejected():
    g = _fine()
    broken = MOTE_C.variant("no-ram", compute=MOTE_C.compute.__class__(ram_bytes=0))
    with pytest.raises(ValueError, match="no RAM"):
        partition(g, broken)


# ------------------------------------------------------------------- machinery

def test_kd_order_keeps_neighbours_adjacent():
    """The ordering has to beat the raw numbering, or chunking it is pointless."""
    rng = np.random.default_rng(0)
    pts = rng.random((512, 3))
    order = kd_order(pts)
    assert sorted(order.tolist()) == list(range(512))
    ordered = np.linalg.norm(pts[order[1:]] - pts[order[:-1]], axis=1).mean()
    shuffled = np.linalg.norm(pts[1:] - pts[:-1], axis=1).mean()
    assert ordered < shuffled / 2


def test_kd_order_handles_degenerate_input():
    assert kd_order(np.zeros((0, 3))).size == 0
    assert kd_order(np.zeros((5, 3))).size == 5      # all points identical
    assert kd_order(np.zeros((5, 0))).size == 5      # no dimensions at all


def test_refinement_never_makes_the_cut_worse():
    g = _fine()
    base = partition(g, MOTE_C, model="block", seed=0)
    capacity = float(g.working_set.sum()) / base.k * 1.10
    refined = refine_boundary(g, base.labels, capacity, sweeps=4)

    e = g.edges
    before = g.payload[base.labels[e[:, 0]] != base.labels[e[:, 1]]].sum()
    after = g.payload[refined[e[:, 0]] != refined[e[:, 1]]].sum()
    assert after <= before


def test_refinement_respects_capacity():
    g = _fine()
    base = partition(g, MOTE_C, model="block", seed=0)
    capacity = float(g.working_set.sum()) / base.k * 1.10
    refined = refine_boundary(g, base.labels, capacity, sweeps=4)
    load = np.zeros(base.k)
    np.add.at(load, refined, g.working_set)
    assert load.max() <= capacity + 1e-6


# -------------------------------------------------------- the pass's own metric

def test_fmbm_beats_the_structure_blind_baseline_on_a_lattice():
    """`block` is not weak here -- the frontend numbers tiles in lattice order,
    so contiguous ids are already coherent blocks.  Beating it means beating a
    partition that was handed the geometry."""
    g = _fine()
    fmbm = partition(g, MOTE_C, model="fmbm", seed=0)
    block = partition(g, MOTE_C, model="block", seed=0)
    assert fmbm.cut_fraction < block.cut_fraction


def test_fmbm_produces_feasible_groups():
    """Spectral gets better cuts on some inputs by ignoring capacity entirely."""
    g = _fine()
    assert partition(g, MOTE_C, model="fmbm", seed=0).feasible


def test_partitioning_reduces_traffic_against_one_task_per_device():
    """The point of grouping: bytes inside a device stop being beams."""
    g = _fine()
    r = partition(g, MOTE_C, seed=0)
    assert r.k < g.n_tasks
    assert r.cut_fraction < 1.0


def test_a_bigger_device_needs_fewer_groups():
    from iotc.devices import MOTE_M

    g = _fine()
    assert minimum_devices(g, MOTE_M) < minimum_devices(g, MOTE_C)


def test_the_partitioners_are_registered():
    assert {"fmbm", "block", "spectral", "kernighan_lin"} <= set(available("partitioner"))

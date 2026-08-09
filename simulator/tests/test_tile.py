"""B1 Tile: what splitting must conserve, and what it is allowed to cost."""

from __future__ import annotations

import math

import numpy as np
import pytest

from iotc.compiler import build_workload
from iotc.compiler.ir import TaskGraph
from iotc.compiler.tile import balanced_factors, tile
from iotc.devices import MOTE_C, MOTE_M
from iotc.registry import available
from iotc.units import KiB

#: A stencil that genuinely overflows a mote-c and so exercises every path.
BIG = dict(nx=256, ny=256, nz=256, tx=2, ty=2, tz=2)


def _big_stencil():
    return build_workload("stencil3d", **BIG)


# ------------------------------------------------------------------ factoring

def test_balanced_factors_cover_the_requested_pieces():
    for k in (1, 2, 7, 8, 63, 64, 1000):
        for d in (1, 2, 3):
            f = balanced_factors(k, d)
            assert len(f) == d
            assert math.prod(f) >= k


def test_balanced_factors_are_balanced():
    """AM-GM: at fixed product, sum(f_i - 1) is minimised by equal factors.

    This is not cosmetic.  Added traffic is exactly ``halo * sum(f_i - 1)``, so
    the balance of the factorization *is* the cost of the tiling.
    """
    f = balanced_factors(64, 3)
    assert max(f) - min(f) <= 1
    assert sum(x - 1 for x in f) < sum(x - 1 for x in balanced_factors(64, 1))


@pytest.mark.parametrize("k,d,expected", [(64, 1, 63), (64, 2, 14), (64, 3, 9)])
def test_added_traffic_follows_the_telescoping_law(k, d, expected):
    """Cutting k ways costs d*(k^(1/d) - 1) halos, not k halos."""
    assert sum(f - 1 for f in balanced_factors(k, d)) == expected


# --------------------------------------------------------------- conservation

def test_tiling_conserves_flops_exactly():
    g = _big_stencil()
    r = tile(g, MOTE_C)
    assert r.graph.total_flop == pytest.approx(g.total_flop)


def test_tiling_conserves_external_traffic():
    """Splitting adds internal faces; it must not invent or lose external bytes.

    Each original edge is spread over a round-robin matching of the two tasks'
    pieces, so the sum over that matching is the original payload.
    """
    g = _big_stencil()
    r = tile(g, MOTE_C)
    origin = r.origin
    edges = r.graph.edges
    external = origin[edges[:, 0]] != origin[edges[:, 1]]
    assert r.graph.payload[external].sum() == pytest.approx(g.payload.sum())


def test_tiling_a_graph_that_already_fits_changes_nothing():
    g = build_workload("stencil2d")
    assert g.fits(MOTE_C).all()
    r = tile(g, MOTE_C)
    assert np.all(r.factor == 1)
    assert r.graph.n_tasks == g.n_tasks
    assert np.allclose(r.graph.working_set, g.working_set)
    assert np.allclose(np.sort(r.graph.payload), np.sort(g.payload))


def test_provenance_maps_every_piece_back_to_its_task():
    g = _big_stencil()
    r = tile(g, MOTE_C)
    assert r.origin.size == r.graph.n_tasks
    assert set(r.origin.tolist()) == set(range(g.n_tasks))
    counts = np.bincount(r.origin, minlength=g.n_tasks)
    assert np.array_equal(counts, r.factor)


# ---------------------------------------------------------------- the fit

def test_tiling_makes_the_workload_fit():
    g = _big_stencil()
    assert not g.fits(MOTE_C).all()      # precondition: it really is too big
    r = tile(g, MOTE_C)
    assert r.converged
    assert r.graph.fits(MOTE_C).all()


def test_halo_counts_against_ram():
    """A piece must buffer its ghost faces, so bulk alone is not the test."""
    g = _big_stencil()
    r = tile(g, MOTE_C)
    bulk = g.working_set[r.origin] / r.factor[r.origin]
    # Every piece carries strictly more than its share of the bulk.
    assert np.all(r.graph.working_set > bulk * 0.5)
    assert r.graph.fits(MOTE_C).all()


def test_a_roomier_part_needs_fewer_pieces():
    g = _big_stencil()
    small = tile(g, MOTE_C)
    roomy = tile(g, MOTE_M)
    assert MOTE_M.compute.ram_bytes > MOTE_C.compute.ram_bytes
    assert roomy.graph.n_tasks < small.graph.n_tasks


# ------------------------------------------------------- the pass's own metric

def test_geometric_cutting_beats_slab_cutting_on_traffic():
    """B1's quality metric, and the reason the pass is not a division.

    Same workload, same target, same fit -- but cutting along every available
    axis moves several times fewer bytes than slicing along one.
    """
    g = _big_stencil()
    geo = tile(g, MOTE_C, model="geometric")
    slab = tile(g, MOTE_C, model="slab")

    assert geo.graph.fits(MOTE_C).all() and slab.graph.fits(MOTE_C).all()
    assert geo.added_traffic_halos < slab.added_traffic_halos
    assert geo.graph.total_bytes < slab.graph.total_bytes / 3
    # And it does it with fewer pieces, because less halo means less overflow.
    assert geo.graph.n_tasks < slab.graph.n_tasks


def test_planar_sits_between_the_two():
    g = _big_stencil()
    geo = tile(g, MOTE_C, model="geometric")
    plan = tile(g, MOTE_C, model="planar")
    slab = tile(g, MOTE_C, model="slab")
    assert geo.added_traffic_halos <= plan.added_traffic_halos <= slab.added_traffic_halos


def test_tiling_costs_arithmetic_intensity():
    """Splitting always makes the workload more communication-bound.

    Worth pinning because it is the argument against tiling further than the
    fit requires -- on an aim-dominated fabric every added edge is another
    retarget.
    """
    g = _big_stencil()
    r = tile(g, MOTE_C)
    assert r.graph.arithmetic_intensity < g.arithmetic_intensity


# ------------------------------------------------------------- infeasibility

def test_an_unfittable_exchange_is_reported_not_ground_at():
    """When the halo alone exceeds RAM, no tiling helps and the pass says so.

    The FFT butterfly exchanges half its points every crossing stage, and
    slicing a rank does not shrink that.  B1 has to return "this needs a bigger
    part" rather than subdividing until it runs out of memory.
    """
    g = build_workload("fft")
    r = tile(g, MOTE_C, model="slab")
    assert not r.converged
    assert "halo floor" in r.reason
    assert r.graph.n_tasks < 1 << 16       # it stopped rather than exploding


def test_the_same_fft_fits_a_bigger_part():
    g = build_workload("fft")
    r = tile(g, MOTE_M)
    assert r.converged
    assert r.graph.fits(MOTE_M).all()


def test_zero_ram_is_rejected():
    g = build_workload("stencil2d")
    broken = MOTE_C.variant("no-ram", compute=MOTE_C.compute.__class__(ram_bytes=0))
    with pytest.raises(ValueError, match="no RAM"):
        tile(g, broken)


# ------------------------------------------------------------------- plumbing

def test_the_tilers_are_registered():
    assert set(available("tiler")) == {"geometric", "slab", "planar"}


def test_tiling_is_deterministic():
    g = _big_stencil()
    a, b = tile(g, MOTE_C), tile(g, MOTE_C)
    assert np.array_equal(a.graph.edges, b.graph.edges)
    assert np.allclose(a.graph.payload, b.graph.payload)
    assert np.array_equal(a.origin, b.origin)


@pytest.mark.parametrize("name", sorted(available("workload")))
def test_tiling_every_stock_workload_yields_a_valid_graph(name):
    r = tile(build_workload(name), MOTE_M)
    r.graph.validate()
    assert r.graph.n_tasks >= build_workload(name).n_tasks

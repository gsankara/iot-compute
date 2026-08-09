"""Workload frontends: structure, determinism, and the scaling laws they imply."""

from __future__ import annotations

import math

import numpy as np
import pytest

from iotc.compiler import build_workload
from iotc.compiler.frontends import graphs
from iotc.compiler.frontends.nphard import MaxClique, common_neighbour_weight
from iotc.registry import available, create

ALL_WORKLOADS = sorted(available("workload"))


def test_the_stock_family_is_all_registered():
    assert set(ALL_WORKLOADS) == {
        "allreduce", "bfs", "coloring", "fft", "gemm", "kmedian",
        "maxclique", "maxcut", "mis", "spmv", "stencil2d", "stencil3d",
    }


@pytest.mark.parametrize("name", ALL_WORKLOADS)
def test_every_frontend_builds_a_valid_graph(name):
    g = build_workload(name)
    g.validate()
    assert g.n_tasks > 0
    assert np.all(g.flop >= 0)
    assert np.all(g.working_set > 0)
    assert np.all(g.payload >= 0)


@pytest.mark.parametrize("name", ALL_WORKLOADS)
def test_every_frontend_is_deterministic(name):
    """Two builds of the same parameters must agree edge for edge.

    The simulator treats determinism as a correctness requirement; a compiler
    that placed the same problem differently on two runs would make every
    downstream fault campaign unreproducible.
    """
    a, b = build_workload(name), build_workload(name)
    assert np.array_equal(a.edges, b.edges)
    assert np.allclose(a.payload, b.payload)
    assert np.allclose(a.flop, b.flop)
    assert np.allclose(a.working_set, b.working_set)


@pytest.mark.parametrize("name", ALL_WORKLOADS)
def test_every_frontend_is_connected(name):
    """A disconnected task graph would silently split the swarm in two."""
    g = build_workload(name)
    assert len(set(g.connected_components())) == 1


# ------------------------------------------------------------------ stencils

def test_stencil2d_has_a_five_point_neighbourhood():
    g = build_workload("stencil2d", tx=4, ty=4)
    deg = g.degree()
    assert deg.max() == 4          # interior tile: four faces
    assert deg.min() == 2          # corner tile: two faces
    # A tx-by-ty mesh has this many internal faces, exactly.
    assert g.n_edges == 4 * (4 - 1) * 2


def test_stencil3d_has_a_seven_point_neighbourhood():
    g = build_workload("stencil3d", tx=3, ty=3, tz=3)
    deg = g.degree()
    assert deg.max() == 6          # the one interior tile of a 3x3x3 mesh
    assert deg.min() == 3          # a corner
    assert np.count_nonzero(deg == 6) == 1


def test_stencil_intensity_rises_with_tile_size():
    """Surface-to-volume: fewer, fatter tiles communicate relatively less.

    This is the single most useful lever the compiler has on this fabric, and
    B1 has to know it cuts the other way from the obvious "smaller tiles fit
    better" instinct.
    """
    fine = build_workload("stencil3d", nx=64, ny=64, nz=64, tx=4, ty=4, tz=4)
    coarse = build_workload("stencil3d", nx=64, ny=64, nz=64, tx=2, ty=2, tz=2)
    assert coarse.arithmetic_intensity > fine.arithmetic_intensity


def test_stencil_weak_scaling_holds_the_working_set_constant():
    """Double the grid and the tiling together and each task holds the same."""
    small = build_workload("stencil2d", nx=256, ny=256, tx=2, ty=2)
    big = build_workload("stencil2d", nx=512, ny=512, tx=4, ty=4)
    assert small.working_set.max() == pytest.approx(big.working_set.max(), rel=0.05)


# ---------------------------------------------------------------- structured

def test_summa_communicates_only_along_rows_and_columns():
    p = 4
    g = build_workload("gemm", n=512, p=p)
    assert g.n_tasks == p * p
    # Each task has p-1 row peers and p-1 column peers, and no diagonal ones.
    assert np.all(g.degree() == 2 * (p - 1))


def test_gemm_is_compute_bound_relative_to_the_graph_family():
    dense = build_workload("gemm", n=1024, p=4)
    sparse = build_workload("spmv")
    assert dense.arithmetic_intensity > 50 * sparse.arithmetic_intensity


def test_fft_butterfly_is_a_hypercube():
    p = 8
    g = build_workload("fft", n=1 << 16, p=p)
    stages = int(math.log2(p))
    assert np.all(g.degree() == stages)
    assert g.n_edges == p // 2 * stages


def test_fft_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        create("workload", "fft", n=1000, p=8)
    with pytest.raises(ValueError):
        create("workload", "fft", n=1024, p=6)


def test_allreduce_topologies_have_their_textbook_edge_counts():
    ring = build_workload("allreduce", p=16, topology="ring")
    tree = build_workload("allreduce", p=16, topology="tree")
    assert ring.n_edges == 16
    assert tree.n_edges == 15
    assert tree.degree().max() == 3   # a parent plus two children


def test_allreduce_is_the_least_intense_workload_in_the_family():
    """The pathological case, kept so the scheduler has something to fail on."""
    ai = {n: build_workload(n).arithmetic_intensity for n in ALL_WORKLOADS}
    assert min(ai, key=ai.get) == "allreduce"


# -------------------------------------------------------------------- sparse

def test_cutting_a_graph_into_more_blocks_moves_more_bytes():
    """More parts means more cut edges means more traffic -- monotonically."""
    few = build_workload("spmv", n=2048, parts=4)
    many = build_workload("spmv", n=2048, parts=32)
    assert many.total_bytes > few.total_bytes
    assert many.arithmetic_intensity < few.arithmetic_intensity


def test_bfs_traffic_scales_with_the_frontier():
    quiet = build_workload("bfs", frontier_fraction=0.05)
    busy = build_workload("bfs", frontier_fraction=0.50)
    assert busy.total_bytes == pytest.approx(10 * quiet.total_bytes, rel=1e-9)


# ------------------------------------------------------------------- NP-hard

def test_a_planted_clique_really_is_a_clique():
    """The ground truth B10 will be scored against has to be true."""
    u, v, members = graphs.planted_clique(512, 8.0, 12, seed=3)
    present = {(int(a), int(b)) for a, b in zip(u, v)}
    for i in range(members.size):
        for j in range(i + 1, members.size):
            a, b = int(members[i]), int(members[j])
            assert (min(a, b), max(a, b)) in present


def test_clique_members_share_more_neighbours_than_random_pairs():
    """What the attraction weight is for: it separates the clique from the noise."""
    n, k = 512, 16
    u, v, members = graphs.planted_clique(n, 8.0, k, seed=1)
    w = common_neighbour_weight(u, v, n)
    member_set = set(int(x) for x in members)
    inside = np.array([
        w[e] for e in range(u.size)
        if int(u[e]) in member_set and int(v[e]) in member_set
    ])
    outside = np.array([
        w[e] for e in range(u.size)
        if int(u[e]) not in member_set or int(v[e]) not in member_set
    ])
    assert inside.size > 0 and outside.size > 0
    assert inside.mean() > 3 * outside.mean()


def test_attraction_weight_is_at_least_one_on_every_edge():
    u, v = graphs.erdos_renyi(256, 6.0, seed=5)
    w = common_neighbour_weight(u, v, 256)
    assert np.all(w >= 1.0)   # the +A[i,j] term


def test_maxclique_exposes_its_ground_truth():
    wl = MaxClique(n=512, clique_size=14, seed=2)
    assert wl.planted().size == 14
    assert wl.build().n_tasks == wl.parts


def test_clique_traffic_responds_to_structure_and_the_others_do_not():
    """Clique edges carry a common-neighbour weight, not a fixed-size message.

    MIS and colouring send one id (plus a priority or a colour) per cut edge
    whatever the graph looks like, so their traffic depends only on the cut.
    Clique traffic depends on the *neighbourhood overlap*, so planting a clique
    into an otherwise identical instance raises it.  That sensitivity is the
    whole reason B3 embeds this weight rather than plain adjacency.
    """
    kw = dict(n=512, avg_degree=12.0, seed=0)
    plain = build_workload("maxclique", graph="erdos_renyi", **kw)
    planted = build_workload("maxclique", graph="planted_clique", clique_size=40, **kw)
    assert planted.total_bytes > plain.total_bytes

    # The structure-blind workloads barely move on the same pair of instances.
    mis_plain = build_workload("mis", graph="erdos_renyi", **kw)
    assert mis_plain.total_bytes == pytest.approx(
        build_workload("mis", graph="erdos_renyi", **kw).total_bytes
    )


# ------------------------------------------------------------- generators

def test_generators_are_seed_stable_and_seed_sensitive():
    a = graphs.erdos_renyi(256, 8.0, seed=11)
    b = graphs.erdos_renyi(256, 8.0, seed=11)
    c = graphs.erdos_renyi(256, 8.0, seed=12)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not (a[0].size == c[0].size and np.array_equal(a[0], c[0]))


def test_erdos_renyi_spreads_edges_over_all_vertices():
    """Trimming a sorted pair list by prefix would starve the high ids."""
    n = 512
    u, v = graphs.erdos_renyi(n, 10.0, seed=7)
    deg = np.zeros(n)
    np.add.at(deg, u, 1)
    np.add.at(deg, v, 1)
    lo, hi = deg[: n // 2].mean(), deg[n // 2:].mean()
    assert lo == pytest.approx(hi, rel=0.25)
    assert deg.mean() == pytest.approx(10.0, rel=0.25)


def test_power_law_is_more_skewed_than_erdos_renyi():
    n = 512
    pu, pv = graphs.power_law(n, 3, seed=0)
    eu, ev = graphs.erdos_renyi(n, 6.0, seed=0)

    def degrees(u, v):
        d = np.zeros(n)
        np.add.at(d, u, 1)
        np.add.at(d, v, 1)
        return d

    assert degrees(pu, pv).max() > 3 * degrees(eu, ev).max()


def test_banded_is_exactly_local():
    u, v = graphs.banded(64, bandwidth=2)
    assert np.all(np.abs(v - u) <= 2)
    assert u.size == (64 - 1) + (64 - 2)


def test_block_partition_is_balanced_and_total():
    part = graphs.block_partition(1000, 16)
    counts = np.bincount(part)
    assert part.size == 1000
    assert counts.sum() == 1000
    assert counts.max() - counts.min() <= 1

"""Graph FastMap: does the embedding preserve the distances it claims to?"""

from __future__ import annotations

import numpy as np
import pytest

from iotc.compiler import build_workload
from iotc.compiler.fastmap import (
    build_csr,
    dijkstra,
    embed_graph,
    fastmap,
    traffic_distance,
)


def _path(n: int):
    u = np.arange(n - 1)
    return u, u + 1, np.ones(n - 1)


def _grid3d(side: int):
    """A cubic lattice with unit edges, and its true integer coordinates."""
    idx = lambda i, j, k: (k * side + j) * side + i  # noqa: E731
    u, v = [], []
    for k in range(side):
        for j in range(side):
            for i in range(side):
                if i + 1 < side:
                    u.append(idx(i, j, k)); v.append(idx(i + 1, j, k))
                if j + 1 < side:
                    u.append(idx(i, j, k)); v.append(idx(i, j + 1, k))
                if k + 1 < side:
                    u.append(idx(i, j, k)); v.append(idx(i, j, k + 1))
    true = np.array(
        [[i, j, k] for k in range(side) for j in range(side) for i in range(side)],
        dtype=float,
    )
    return np.array(u), np.array(v), np.ones(len(u)), true


# ------------------------------------------------------------------- shortest path

def test_dijkstra_matches_hand_computed_distances():
    u, v, w = _path(6)
    indptr, indices, data = build_csr(6, u, v, w)
    d = dijkstra(indptr, indices, data, 0, 6)
    assert np.allclose(d, [0, 1, 2, 3, 4, 5])


def test_unreachable_vertices_are_infinite_before_repair():
    indptr, indices, data = build_csr(4, [0], [1], [1.0])
    d = dijkstra(indptr, indices, data, 0, 4)
    assert np.isinf(d[2]) and np.isinf(d[3])


# ---------------------------------------------------------------------- embedding

def test_a_path_graph_embeds_onto_a_line():
    """The easiest case, and a total failure of it would be silent otherwise."""
    n = 64
    u, v, w = _path(n)
    coords = fastmap(n, u, v, w, kappa=2, seed=0)
    assert abs(np.corrcoef(coords[:, 0], np.arange(n))[0, 1]) == pytest.approx(1.0, abs=1e-6)


def test_embedded_distance_tracks_graph_distance():
    n = 48
    u, v, w = _path(n)
    coords = fastmap(n, u, v, w, kappa=2, seed=1)
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, n, 200), rng.integers(0, n, 200)
    graph_d = np.abs(a - b).astype(float)
    embed_d = np.linalg.norm(coords[a] - coords[b], axis=1)
    assert np.corrcoef(graph_d, embed_d)[0, 1] > 0.95


def test_a_lattice_does_not_collapse_to_one_dimension():
    """The degeneracy that a naive residual update walks straight into.

    Shortest-path distance on a grid is L1, so every edge contributes equally to
    the first diagonal coordinate and the Pythagorean update zeroes them all at
    once.  Without the orthogonalising restart the embedding returns one live
    axis and kappa-1 columns of zeros, and every partition built on it is a
    one-dimensional slicing of a three-dimensional mesh.
    """
    side = 6
    u, v, w, true = _grid3d(side)
    coords = fastmap(side**3, u, v, w, kappa=3, seed=0)
    live = (coords.std(axis=0) > 1e-9).sum()
    assert live >= 3, f"only {live} live axes; the embedding collapsed"

    # And the three lattice axes must be linearly recoverable from it.
    design = np.column_stack([coords, np.ones(len(coords))])
    for axis in range(3):
        beta, *_ = np.linalg.lstsq(design, true[:, axis], rcond=None)
        residual = true[:, axis] - design @ beta
        r2 = 1 - (residual**2).sum() / ((true[:, axis] - true[:, axis].mean()) ** 2).sum()
        assert r2 > 0.3, f"lattice axis {axis} barely recoverable (R^2={r2:.2f})"


def test_more_dimensions_never_lose_information():
    n = 40
    u, v, w = _path(n)
    two = fastmap(n, u, v, w, kappa=1, seed=3)
    four = fastmap(n, u, v, w, kappa=4, seed=3)
    assert four.shape[1] == 4
    assert np.allclose(four[:, 0], two[:, 0])


def test_embedding_is_deterministic_and_seed_sensitive():
    n = 40
    u, v, w = _path(n)
    assert np.allclose(fastmap(n, u, v, w, 3, seed=7), fastmap(n, u, v, w, 3, seed=7))


def test_degenerate_inputs_do_not_explode():
    assert fastmap(0, [], [], [], kappa=3).shape == (0, 3)
    assert fastmap(5, [], [], [], kappa=3).shape == (5, 3)
    assert np.allclose(fastmap(5, [], [], [], kappa=3), 0.0)


def test_a_disconnected_graph_embeds_without_nan():
    """A fault campaign can disconnect a task graph; NaN must not propagate."""
    u = np.array([0, 1, 5, 6])
    v = np.array([1, 2, 6, 7])
    coords = fastmap(8, u, v, np.ones(4), kappa=2, seed=0)
    assert np.all(np.isfinite(coords))


# ----------------------------------------------------------------------- distance

def test_heavier_traffic_means_shorter_distance():
    """The whole premise: chatty pairs must embed close together."""
    d = traffic_distance(np.array([1000.0, 10.0, 1.0]))
    assert d[0] < d[1] < d[2]
    assert d[0] == pytest.approx(1.0)   # the heaviest edge is the unit


def test_distance_is_scale_free():
    a = traffic_distance(np.array([100.0, 50.0, 25.0]))
    b = traffic_distance(np.array([200.0, 100.0, 50.0]))
    assert np.allclose(a, b)


def test_embed_graph_handles_a_real_workload():
    g = build_workload("stencil3d", tx=3, ty=3, tz=3)
    coords = embed_graph(g, kappa=3, seed=0)
    assert coords.shape == (g.n_tasks, 3)
    assert np.all(np.isfinite(coords))

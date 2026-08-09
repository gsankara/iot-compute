"""The compiler IR: invariants a TaskGraph must hold no matter who built it."""

from __future__ import annotations

import numpy as np
import pytest

from iotc.compiler.ir import TaskGraph, TaskGraphError
from iotc.devices import MOTE_C
from iotc.units import KiB, MiB


def _triangle() -> TaskGraph:
    g = TaskGraph("tri")
    for _ in range(3):
        g.add_task(flop=1e6, working_set=8 * KiB)
    g.add_edge(0, 1, 1024)
    g.add_edge(1, 2, 2048)
    g.add_edge(2, 0, 4096)
    return g.finalize()


def test_a_built_graph_validates():
    g = _triangle()
    assert g.n_tasks == 3
    assert g.n_edges == 3
    assert np.all(g.degree() == 2)


def test_self_loops_are_rejected():
    """A link needs two gimbals aimed at each other, so i--i is not expressible."""
    g = TaskGraph()
    g.add_task(1.0, 1.0)
    with pytest.raises(TaskGraphError, match="self-loop"):
        g.add_edge(0, 0, 128)


def test_edge_direction_does_not_matter():
    a, b = TaskGraph(), TaskGraph()
    for g in (a, b):
        g.add_task(1.0, 1.0)
        g.add_task(1.0, 1.0)
    a.add_edge(0, 1, 512)
    b.add_edge(1, 0, 512)
    assert np.array_equal(a.finalize().edges, b.finalize().edges)


def test_duplicate_edges_coalesce_by_summing_payload():
    """A frontend may emit the same pair twice for two different reasons."""
    g = TaskGraph()
    g.add_task(1.0, 1.0)
    g.add_task(1.0, 1.0)
    g.add_edge(0, 1, 100)
    g.add_edge(1, 0, 250)
    g.finalize()
    assert g.n_edges == 1
    assert g.payload[0] == pytest.approx(350.0)


def test_degree_and_traffic_agree_with_the_edge_list():
    g = _triangle()
    assert g.degree().sum() == 2 * g.n_edges
    # Every edge is counted once at each endpoint.
    assert g.traffic_per_task().sum() == pytest.approx(2 * g.payload.sum())


def test_negative_costs_are_rejected():
    g = TaskGraph()
    with pytest.raises(TaskGraphError):
        g.add_task(flop=-1.0, working_set=0.0)
    g.add_task(1.0, 1.0)
    g.add_task(1.0, 1.0)
    with pytest.raises(TaskGraphError):
        g.add_edge(0, 1, -5.0)


def test_out_of_range_edge_is_caught_by_validate():
    g = TaskGraph.__new__(TaskGraph)
    with pytest.raises(TaskGraphError):
        TaskGraph.from_arrays([1.0, 1.0], [1.0, 1.0], [0], [7], [1.0])


def test_arithmetic_intensity_counts_iterations_on_both_sides():
    """Scaling iterations scales FLOPs and bytes together, so AI is invariant."""
    one = TaskGraph.from_arrays([10.0], [1.0], [], [], [], iterations=1)
    assert one.arithmetic_intensity == np.inf  # no communication at all

    a = TaskGraph.from_arrays([10.0, 10.0], [1.0, 1.0], [0], [1], [5.0], iterations=1)
    b = TaskGraph.from_arrays([10.0, 10.0], [1.0, 1.0], [0], [1], [5.0], iterations=7)
    assert a.arithmetic_intensity == pytest.approx(b.arithmetic_intensity)
    assert b.total_flop == pytest.approx(7 * a.total_flop)
    assert b.total_bytes == pytest.approx(7 * a.total_bytes)


def test_fits_reads_the_device_class_rather_than_a_byte_count():
    """B1's stopping condition must come from the same data the simulator uses."""
    ram = MOTE_C.compute.ram_bytes
    g = TaskGraph.from_arrays([1.0, 1.0], [ram * 0.5, ram * 2.0], [0], [1], [1.0])
    assert list(g.fits(MOTE_C)) == [True, False]


def test_aim_fraction_reproduces_the_aim_dominated_finding():
    """Small payloads on this fabric are dominated by pointing, not by the wire.

    Phase A measured ~0.5 s to aim against microseconds to send an 8 kbit
    packet.  The IR has to show that before any placement effort is spent.
    """
    small = TaskGraph.from_arrays([1.0, 1.0], [1.0, 1.0], [0], [1], [1024.0])
    assert small.aim_fraction(MOTE_C) > 0.99

    # Make the payload enormous and the wire finally dominates.
    huge = TaskGraph.from_arrays([1.0, 1.0], [1.0, 1.0], [0], [1], [4e9])
    assert huge.aim_fraction(MOTE_C) < 0.5


def test_subgraph_keeps_only_internal_edges():
    g = _triangle()
    sub = g.subgraph(np.array([True, True, False]))
    assert sub.n_tasks == 2
    assert sub.n_edges == 1  # 0--1 survives; the two edges to task 2 do not
    assert sub.payload[0] == pytest.approx(1024.0)


def test_connected_components_finds_both_halves():
    g = TaskGraph.from_arrays(
        [1.0] * 4, [1.0] * 4, [0, 2], [1, 3], [1.0, 1.0]
    )
    labels = g.connected_components()
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    assert len(set(labels)) == 2


def test_json_round_trip_is_exact(tmp_path):
    g = _triangle()
    path = tmp_path / "g.json"
    g.to_json(path)
    back = TaskGraph.from_json(path)
    assert back.n_tasks == g.n_tasks
    assert np.array_equal(back.edges, g.edges)
    assert np.allclose(back.payload, g.payload)
    assert np.allclose(back.flop, g.flop)
    assert back.iterations == g.iterations


def test_copy_is_independent():
    g = _triangle()
    c = g.copy()
    c.flop[0] = 999.0  # a view write must not reach the original
    assert g.flop[0] != 999.0


def test_a_finalized_graph_refuses_further_edits():
    g = _triangle()
    with pytest.raises(TaskGraphError, match="finalized"):
        g.add_task(1.0, 1.0)


def test_iterations_must_be_at_least_one():
    with pytest.raises(TaskGraphError):
        TaskGraph("x", iterations=0)

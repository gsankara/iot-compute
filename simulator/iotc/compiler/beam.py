"""B5 -- Beam assignment: which links are open at the same time.

A device has a small fixed number of optical heads -- two on a mote, six on a
relay -- and a link occupies one at each end.  B0 measured the mismatch this
creates: eleven of the twelve stock workloads want **more simultaneous
neighbours than the part has transceivers**, by up to 7.5x.  So the demand graph
cannot be realised as a topology.  It has to be realised as a *sequence* of
topologies.

That makes the pass a **degree-constrained b-matching**, repeated: choose a
subgraph in which no device exceeds its transceiver count, then choose another
from what is left, until every link has had a turn.  Each subgraph is one
*round*, and the round count is what B6 must schedule around.

Two physical constraints sit on top of the combinatorics.

**A gimbal cannot point everywhere.**  Each class declares sweep limits, and a
link whose bearing falls outside them cannot be aimed at all -- not a weak link,
an impossible one.  Those are rejected here rather than discovered at runtime.

**A gimbal cannot point exactly.**  Angles snap to the quantization grid, and
the residual is a real pointing loss against a Gaussian beam.

The roadmap asks for that residual to be fed back into placement for one
refinement iteration, and building this pass showed the feedback matters far
more than that framing suggests.  B3 optimises motion energy, and on a float the
cheap axis is depth -- so it returns a tall, laterally thin column.  Nearly every
bearing in such a formation points almost straight up or down, *outside the
gimbal's elevation limits*: **96.5% of the links in the energy-optimal placement
cannot be aimed at all**.  :func:`refine_positions` is therefore not a nudge but
a structural correction, and it recovers the formation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..geom import spherical_from_vec
from ..registry import register
from .ir import TaskGraph


def bearings(positions: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Azimuth and elevation from each edge's first endpoint to its second."""
    if edges.size == 0:
        return np.zeros(0), np.zeros(0)
    delta = positions[edges[:, 1]] - positions[edges[:, 0]]
    return spherical_from_vec(delta)


def pointing_loss_db(residual: np.ndarray, divergence: float) -> np.ndarray:
    """Loss from a residual boresight error against a Gaussian beam.

    Far-field Gaussian: the on-axis gain falls as ``exp(-2 (theta/w)^2)`` where
    ``w`` is the 1/e^2 half-angle.  This is the same law the link budget's
    pointing stage uses, evaluated here directly so B5 can compare thousands of
    candidate assignments without running a full budget for each.
    """
    w = max(float(divergence), 1e-12)
    return -10.0 * np.log10(np.exp(-2.0 * (np.asarray(residual) / w) ** 2))


@dataclass
class BeamAssignment:
    """Which links are open in which round, and what pointing them costs."""

    #: Round index per edge; -1 marks an edge that could not be aimed at all.
    round_of: np.ndarray
    graph: TaskGraph
    positions: np.ndarray
    device_class: object
    residual: np.ndarray = field(default_factory=lambda: np.zeros(0))
    method: str = ""

    @property
    def rounds(self) -> int:
        live = self.round_of[self.round_of >= 0]
        return int(live.max()) + 1 if live.size else 0

    @property
    def rejected(self) -> np.ndarray:
        """Edges outside the gimbal's sweep limits."""
        return self.round_of < 0

    @property
    def rejected_fraction(self) -> float:
        return float(self.rejected.mean()) if self.round_of.size else 0.0

    @property
    def transceivers(self) -> int:
        return int(self.device_class.optical.n_transceivers)

    def degree_in_round(self, r: int) -> np.ndarray:
        """How many heads each device needs in round ``r``."""
        deg = np.zeros(self.graph.n_tasks, dtype=np.int64)
        e = self.graph.edges[self.round_of == r]
        if e.size:
            np.add.at(deg, e[:, 0], 1)
            np.add.at(deg, e[:, 1], 1)
        return deg

    @property
    def feasible(self) -> bool:
        """No device is asked for more beams at once than it owns."""
        return all(
            self.degree_in_round(r).max(initial=0) <= self.transceivers
            for r in range(self.rounds)
        )

    @property
    def first_round_traffic(self) -> float:
        """Share of payload carried by the opening round.

        The pass's quality metric.  Every round after the first is another
        retarget for every device in it, and Phase A priced retargeting at ~0.5 s
        against microseconds of transmission -- so front-loading the heavy edges
        is worth more than balancing the rounds.
        """
        total = float(self.graph.payload.sum())
        if total <= 0:
            return 0.0
        return float(self.graph.payload[self.round_of == 0].sum()) / total

    @property
    def mean_loss_db(self) -> float:
        live = self.round_of >= 0
        if not live.any():
            return 0.0
        loss = pointing_loss_db(self.residual[live], self.device_class.optical.divergence)
        return float(loss.mean())

    @property
    def worst_loss_db(self) -> float:
        live = self.round_of >= 0
        if not live.any():
            return 0.0
        return float(
            pointing_loss_db(self.residual[live], self.device_class.optical.divergence).max()
        )

    def summary(self) -> str:
        return (
            f"{self.method:12s} rounds={self.rounds:3d}  "
            f"first-round traffic {self.first_round_traffic:5.1%}  "
            f"rejected {self.rejected_fraction:5.1%}  "
            f"point loss {self.mean_loss_db:5.2f} dB (worst {self.worst_loss_db:5.2f})"
        )


class BeamAssigner:
    """The ``assigner`` plugin slot: edge priority for the b-matching."""

    def order(self, graph: TaskGraph, aimable: np.ndarray) -> np.ndarray:
        raise NotImplementedError  # pragma: no cover


@register("assigner", "heaviest_first", "greedy")
class HeaviestFirstAssigner(BeamAssigner):
    """Take the heaviest edges into the earliest rounds.  The pass.

    Rounds cost retargets, and payload is heavily skewed, so putting the bulk of
    the traffic in round 0 lets B6 amortise one aim over most of the bytes.
    """

    def order(self, graph: TaskGraph, aimable: np.ndarray) -> np.ndarray:
        idx = np.flatnonzero(aimable)
        return idx[np.argsort(-graph.payload[idx], kind="stable")]


@register("assigner", "lightest_first")
class LightestFirstAssigner(BeamAssigner):
    """The same machinery run backwards, as a control.

    It proves the ordering is doing the work rather than the b-matching: same
    round count, far less traffic in the opening round.
    """

    def order(self, graph: TaskGraph, aimable: np.ndarray) -> np.ndarray:
        idx = np.flatnonzero(aimable)
        return idx[np.argsort(graph.payload[idx], kind="stable")]


@register("assigner", "arbitrary")
class ArbitraryAssigner(BeamAssigner):
    """Edge-list order -- whatever the frontend happened to emit."""

    def order(self, graph: TaskGraph, aimable: np.ndarray) -> np.ndarray:
        return np.flatnonzero(aimable)


def assign_beams(
    graph: TaskGraph,
    positions: np.ndarray,
    device_class,
    model: BeamAssigner | str = "heaviest_first",
) -> BeamAssignment:
    """Cover every aimable link with as few transceiver-feasible rounds as possible."""
    if isinstance(model, str):
        from ..registry import create

        name, model = model, create("assigner", model)
    else:
        name = type(model).plugin_name

    edges = graph.edges
    grid = device_class.optical.gimbal.grid
    b = int(device_class.optical.n_transceivers)

    az, el = bearings(positions, edges)
    residual = np.zeros(edges.shape[0])
    aimable = np.ones(edges.shape[0], dtype=bool)
    for i in range(edges.shape[0]):
        # Both ends must be able to point at each other; the reverse bearing is
        # the forward one mirrored, so checking both is not the same test.
        if not grid.in_range(float(az[i]), float(el[i])) or not grid.in_range(
            float(az[i]) - math.pi, float(-el[i])
        ):
            aimable[i] = False
            continue
        residual[i] = grid.snap(float(az[i]), float(el[i]))[2]

    round_of = np.full(edges.shape[0], -1, dtype=np.int64)
    remaining = list(model.order(graph, aimable))

    r = 0
    while remaining:
        used = np.zeros(graph.n_tasks, dtype=np.int64)
        leftover = []
        for idx in remaining:
            u, v = int(edges[idx, 0]), int(edges[idx, 1])
            if used[u] < b and used[v] < b:
                round_of[idx] = r
                used[u] += 1
                used[v] += 1
            else:
                leftover.append(idx)
        if len(leftover) == len(remaining):
            break  # nothing placed: b is zero, or the graph is degenerate
        remaining = leftover
        r += 1

    return BeamAssignment(
        round_of=round_of, graph=graph, positions=positions,
        device_class=device_class, residual=residual, method=name,
    )


def aimability(
    graph: TaskGraph, positions: np.ndarray, device_class
) -> tuple[np.ndarray, np.ndarray]:
    """Per-edge (in-sweep, snap residual) for a formation."""
    grid = device_class.optical.gimbal.grid
    e = graph.edges
    az, el = bearings(positions, e)
    ok = np.zeros(e.shape[0], dtype=bool)
    residual = np.zeros(e.shape[0])
    for i in range(e.shape[0]):
        a, b = float(az[i]), float(el[i])
        ok[i] = grid.in_range(a, b) and grid.in_range(a - math.pi, -b)
        if ok[i]:
            residual[i] = grid.snap(a, b)[2]
    return ok, residual


def refine_positions(
    graph: TaskGraph,
    positions: np.ndarray,
    device_class,
    step: float | None = None,
    rounds: int = 6,
    reject_penalty_db: float = 30.0,
) -> np.ndarray:
    """Feed B5's pointing problem back into the placement, one iteration.

    The roadmap asks for the snap residual to be returned to B3, and building
    this pass showed the feedback is far more important than that framing
    suggests.  B3 optimises motion energy, and on a float the cheap axis is
    depth -- so it returns a tall, laterally thin column.  Almost every bearing
    in such a formation points nearly straight up or down, *outside the gimbal's
    elevation limits*.  The energy-optimal placement is unaimable.

    So the objective minimised here is the pointing cost that actually matters:
    quantization loss for links that can be aimed, plus a flat penalty for links
    that cannot be.  Coordinate descent on device positions; the penalty term
    dominates, which flattens the column until the bearings come inside the
    sweep cone, and the residual term then trims what is left.

    Minimising the raw residual instead -- the literal reading -- makes the loss
    *worse*, because loss is quadratic in the residual and a sum of residuals is
    not.
    """
    e = graph.edges
    if e.size == 0:
        return positions

    grid = device_class.optical.gimbal.grid
    divergence = device_class.optical.divergence
    if step is None:
        # Start at a fraction of the formation, not a fixed length: the fault is
        # the shape of the whole swarm, not a millimetre of jitter.
        extent = float(np.ptp(positions, axis=0).max())
        step = max(extent * 0.05, 1e-3)

    # Moving one device changes only the bearings of *its own* edges, so the
    # cost of a trial move is a local quantity.  Scoring it globally makes the
    # descent O(n * m) -- at a few hundred devices that is minutes per pass and
    # it puts Phase C's device counts out of reach entirely.  Incident edge
    # lists make each trial O(degree).
    incident: list[list[int]] = [[] for _ in range(positions.shape[0])]
    for idx in range(e.shape[0]):
        incident[int(e[idx, 0])].append(idx)
        incident[int(e[idx, 1])].append(idx)

    def edge_cost(p: np.ndarray, idx: int) -> float:
        a, b = int(e[idx, 0]), int(e[idx, 1])
        delta = p[b] - p[a]
        az, el = spherical_from_vec(delta.reshape(1, 3))
        az, el = float(az[0]), float(el[0])
        if not (grid.in_range(az, el) and grid.in_range(az - math.pi, -el)):
            return reject_penalty_db
        return float(pointing_loss_db(np.array([grid.snap(az, el)[2]]), divergence)[0])

    best = positions.astype(float).copy()
    edge_scores = np.array([edge_cost(best, idx) for idx in range(e.shape[0])])

    for _ in range(rounds):
        improved = False
        for i in range(best.shape[0]):
            mine = incident[i]
            if not mine:
                continue
            local_before = sum(edge_scores[idx] for idx in mine)
            for axis in range(3):
                for sign in (+1.0, -1.0):
                    original = best[i, axis]
                    best[i, axis] = original + sign * step
                    trial = {idx: edge_cost(best, idx) for idx in mine}
                    local_after = sum(trial.values())
                    if local_after < local_before - 1e-12:
                        for idx, value in trial.items():
                            edge_scores[idx] = value
                        local_before = local_after
                        improved = True
                    else:
                        best[i, axis] = original
        if not improved:
            step *= 0.5
            if step < 1e-4:
                break
    return best

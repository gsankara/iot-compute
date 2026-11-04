"""B11 -- Broadcast: the omnidirectional plane, where contention replaces aiming.

Every pass from B4 to B9 is shaped by one fact: a Gbps optical link needs two
gimbals pointed at each other, and pointing takes ~0.5 s against microseconds of
transmission.  B6 proved no *scheduler* escapes that -- reaching 90% link
efficiency would need 1,361 MiB of buffer against 4 MiB of RAM.  The part is too
small, not the schedule.

The radio does not point.  One transmission leaves the antenna in every
direction at once, so there is no retarget to amortise, no b-matching onto a
head count, and no wave colouring to keep a link's far end still.  B4, B5 and
B6's central problem all dissolve.

What arrives instead is **contention**, and it is governed by the same
surface-to-volume law B1 found for traffic and the harvest budget finds for
power.  A transmission is heard by everything within reach, so two transmitters
whose audiences overlap cannot speak at once.  Capacity is therefore areal --
one conversation per domain, and the number of domains is the deployment area
divided by the domain area -- while node count is volumetric.  At a million
nodes on a 60 mm pitch, a 2.4 GHz radio in air reaches 395 m and puts *136
million* nodes inside a single domain: the whole machine is one collision
domain, and each node's share of 1 Mb/s is one bit per second.

So this pass's job is the opposite of what a radio is normally for.  **Reach is
the thing to throw away.**  Per-node share is ``rate * s^2 / (pi * r^2)`` for
pitch ``s`` and domain radius ``r``, so power must be cut until domains are just
large enough to cover the task graph's longest edge and no larger.  That is what
:class:`MinReachBroadcaster` does; :class:`NominalBroadcaster` keeps the stock
power beside it to measure what not bothering would cost.

Two modelling notes, both of which keep this honest:

**The IR does not say whether a payload is shared.**  ``TaskGraph`` edges carry
bytes, not a multicast flag, and broadcast only beats point-to-point when the
*same* bytes serve several neighbours.  All-reduce and SpMV's vector exchange
are pure broadcast; a stencil's halo faces are all different bytes.  Rather than
guess, every plan reports **both bounds** -- ``frame_shared`` (one transmission
per superstep, the multicast case) and ``frame_private`` (distinct bytes per
neighbour, so the payloads sum) -- and the gap between them is how much of
broadcast's advantage a given workload can actually claim.

**Rate follows carrier.**  Dropping the carrier to get through water buys range
and costs bandwidth, so the ladder in :data:`CARRIER_LADDER` moves both together
instead of pretending a 30 kHz link can carry a megabit.

The conflict graph is built as a dense boolean matrix: O(n^2) memory, which is
deliberate while the project is still in its unoptimized pass (see
``docs/ROADMAP.md``).  It is the thing to replace first if this pass is ever
pointed at Phase C's full device counts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..registry import register
from ..sim.links.rf import RFChannel
from .ir import TaskGraph

#: Carrier, line rate and a human name, ordered from most bandwidth to most
#: range.  Rate moves with carrier because bandwidth does: a 30 kHz subsea link
#: physically cannot carry what a 2.4 GHz one can, and letting the pass pick a
#: low carrier *and* keep the high rate would manufacture capacity out of
#: nothing.  Phase A measured why the ladder has to exist at all -- seawater
#: absorbs ~1700 dB/m at 2.4 GHz.
CARRIER_LADDER: tuple[tuple[float, float, str], ...] = (
    (2.4e9, 1e6, "2.4 GHz"),
    (1e6, 1e5, "1 MHz"),
    (100e3, 1e4, "100 kHz"),
    (30e3, 1e3, "30 kHz"),
    (10e3, 1e3, "10 kHz"),
)

#: Power settings swept when a model is allowed to choose, as a fraction of the
#: class's nominal radio power.  Spanning six decades downward is the point:
#: the useful settings are the small ones.
POWER_FRACTIONS: tuple[float, ...] = (
    1.0, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6,
)


# --------------------------------------------------------------------- budget


@dataclass(frozen=True)
class RadioBudget:
    """What one radio setting reaches in one medium."""

    carrier: float
    line_rate: float
    tx_power: float
    reach: float
    label: str

    def report(self) -> str:
        return (
            f"{self.label} @ {self.line_rate / 1e3:.1f} kb/s, "
            f"{self.tx_power * 1e3:.4g} mW -> {self.reach:.3f} m"
        )


def _radio_at(device_class, carrier: float, line_rate: float, tx_power: float):
    """The class's own radio spec with carrier, rate and power overridden."""
    return type(device_class.radio)(
        frequency=float(carrier),
        tx_power=float(tx_power),
        line_rate=float(line_rate),
        antenna_gain_db=device_class.radio.antenna_gain_db,
        sensitivity_dbm=device_class.radio.sensitivity_dbm,
        tx_electrical_power=device_class.radio.tx_electrical_power,
        rx_power=device_class.radio.rx_power,
    )


def derive_radio_budget(
    device_class,
    medium,
    *,
    carrier: float | None = None,
    line_rate: float | None = None,
    power_fraction: float = 1.0,
    channel: RFChannel | None = None,
) -> RadioBudget:
    """Reach of one radio setting, measured through the RF channel model."""
    ch = channel or RFChannel()
    carrier = device_class.radio.frequency if carrier is None else carrier
    line_rate = device_class.radio.line_rate if line_rate is None else line_rate
    label = next(
        (name for c, _, name in CARRIER_LADDER if math.isclose(c, carrier, rel_tol=1e-9)),
        f"{carrier / 1e3:.3g} kHz",
    )
    tx = device_class.radio.tx_power * float(power_fraction)
    radio = _radio_at(device_class, carrier, line_rate, tx)
    return RadioBudget(carrier, line_rate, tx, ch.max_range(radio, medium), label)


# -------------------------------------------------------------------- domains


def _endpoints(graph: TaskGraph) -> tuple[np.ndarray, np.ndarray]:
    """Task-edge endpoints as two int arrays.

    ``TaskGraph.edges`` is the public accessor and returns an ``(m, 2)`` block;
    unpacking it here keeps the column order in one place.
    """
    e = np.asarray(graph.edges, dtype=np.int64).reshape(-1, 2)
    return e[:, 0], e[:, 1]


def hearing_matrix(positions: np.ndarray, reach: float) -> np.ndarray:
    """Boolean ``[i, j]``: j is inside i's broadcast domain (i excluded).

    Symmetric, because every node runs the same radio at the same power -- the
    asymmetric case (a relay shouting further than it listens) would need the
    pair of reaches and is not modelled here.
    """
    p = np.asarray(positions, dtype=float)
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    heard = d <= reach
    np.fill_diagonal(heard, False)
    return heard


def domain_sizes(heard: np.ndarray) -> np.ndarray:
    """How many nodes each transmission reaches."""
    return heard.sum(axis=1).astype(np.int64)


def conflict_matrix(heard: np.ndarray, graph: TaskGraph) -> np.ndarray:
    """Boolean ``[u, w]``: u and w cannot transmit in the same slot.

    A transmission from ``u`` is for its task-graph neighbours.  It collides at
    a receiver ``v`` if ``v`` also hears some other transmitter ``w`` in that
    slot, so ``u`` and ``w`` conflict whenever ``w`` is audible to any node
    ``u`` needs to talk to.  Note this is *not* simply the two-hop graph of the
    task graph: audibility and task adjacency are different relations, and the
    whole point of cutting power is to pull them apart.

    Half duplex is folded in last: a node cannot receive while it transmits, so
    two transmitters that hear each other conflict even when their audiences are
    otherwise disjoint.
    """
    n = graph.n_tasks
    audience = np.zeros((n, n), dtype=bool)
    src, dst = _endpoints(graph)
    # Task-graph edges are undirected: a payload crosses in both directions, so
    # each endpoint is in the other's audience.
    if len(src):
        audience[src, dst] = True
        audience[dst, src] = True
    # u conflicts with w if any member of u's audience hears w.
    conflict = (audience.astype(np.uint8) @ heard.astype(np.uint8)) > 0
    conflict |= conflict.T
    conflict |= heard
    np.fill_diagonal(conflict, False)
    return conflict


def colour_greedy(conflict: np.ndarray) -> np.ndarray:
    """Welsh-Powell colouring: descending degree, lowest free colour.

    Deterministic -- ties in degree break by node index -- because a slot count
    that moved between runs would not be a measurement.  Greedy is an upper
    bound on the chromatic number, and that is the honest direction for a
    capacity claim: it never promises more reuse than it can realise.
    """
    n = conflict.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    degree = conflict.sum(axis=1)
    order = np.lexsort((np.arange(n), -degree))
    colours = np.full(n, -1, dtype=np.int64)
    for u in order:
        taken = set(colours[conflict[u]].tolist())
        c = 0
        while c in taken:
            c += 1
        colours[u] = c
    return colours


# --------------------------------------------------------------------- result


@dataclass
class BroadcastPlan:
    """What one radio setting buys, in slots and in bits per second."""

    budget: RadioBudget
    colours: np.ndarray = field(repr=False)
    slots: int = 0
    devices: int = 0
    mean_domain: float = 0.0
    max_domain: int = 0
    uncovered: int = 0
    edges: int = 0
    frame_shared: float = 0.0
    frame_private: float = 0.0
    feasible: bool = True
    reason: str = ""

    @property
    def uncovered_fraction(self) -> float:
        return self.uncovered / self.edges if self.edges else 0.0

    @property
    def per_node_rate(self) -> float:
        """Each node transmits in one slot of ``slots``, so this is its share."""
        return self.budget.line_rate / self.slots if self.slots else 0.0

    @property
    def reuse(self) -> float:
        """Transmitters speaking at once, on average."""
        return self.devices / self.slots if self.slots else 0.0

    def report(self) -> str:
        return (
            f"{self.budget.report()}  slots={self.slots}  "
            f"reuse={self.reuse:.2f}x  per-node={self.per_node_rate:.4g} bit/s  "
            f"uncovered={self.uncovered_fraction:.1%}"
            + ("" if self.feasible else f"  INFEASIBLE: {self.reason}")
        )


def evaluate_setting(
    graph: TaskGraph,
    positions: np.ndarray,
    budget: RadioBudget,
) -> BroadcastPlan:
    """Price one radio setting: who hears whom, how many slots, what rate."""
    p = np.asarray(positions, dtype=float)
    n = graph.n_tasks
    heard = hearing_matrix(p, budget.reach)
    sizes = domain_sizes(heard)

    src, dst = _endpoints(graph)
    lengths = np.linalg.norm(p[src] - p[dst], axis=1) if len(src) else np.zeros(0)
    uncovered = int((lengths > budget.reach).sum())

    conflict = conflict_matrix(heard, graph)
    colours = colour_greedy(conflict)
    slots = int(colours.max()) + 1 if n else 0

    # Bytes one node must push per superstep, under the two sharing models.
    out = np.zeros(n)
    peak = np.zeros(n)
    if len(src):
        np.add.at(out, src, graph.payload)
        np.add.at(out, dst, graph.payload)
        np.maximum.at(peak, src, graph.payload)
        np.maximum.at(peak, dst, graph.payload)
    rate = budget.line_rate
    plan = BroadcastPlan(
        budget=budget,
        colours=colours,
        slots=slots,
        devices=n,
        mean_domain=float(sizes.mean()) if n else 0.0,
        max_domain=int(sizes.max()) if n else 0,
        uncovered=uncovered,
        edges=int(len(src)),
        frame_shared=float(slots * peak.max() * 8.0 / rate) if rate and n else 0.0,
        frame_private=float(slots * out.max() * 8.0 / rate) if rate and n else 0.0,
    )
    if uncovered:
        plan.feasible = False
        plan.reason = (
            f"{uncovered} of {plan.edges} task edges exceed the {budget.reach:.3f} m reach"
        )
    return plan


# ----------------------------------------------------------------- the models


class Broadcaster:
    """Chooses a carrier and a transmit power.  That is the whole decision."""

    def choose(self, graph, positions, device_class, medium, channel) -> BroadcastPlan:
        raise NotImplementedError


@register("broadcaster", "nominal", "full_power")
class NominalBroadcaster(Broadcaster):
    """The stock radio, untouched -- carried to measure what it costs.

    This is what a deployment gets by not thinking about it, and on any array
    bigger than one domain it is catastrophic: a single collision domain, every
    node contending, per-node share falling as 1/n.
    """

    def choose(self, graph, positions, device_class, medium, channel):
        b = derive_radio_budget(device_class, medium, channel=channel)
        return evaluate_setting(graph, positions, b)


@register("broadcaster", "min_reach", "tight")
class MinReachBroadcaster(Broadcaster):
    """The least reach that still covers every task edge.

    Spatial reuse is the only capacity lever a shared medium has, and reuse is
    bought by shrinking domains.  So take the highest-bandwidth rung of the
    ladder that can span the graph's longest edge at all, and on that rung use
    the smallest power which still spans it.
    """

    def choose(self, graph, positions, device_class, medium, channel):
        p = np.asarray(positions, dtype=float)
        src, dst = _endpoints(graph)
        need = float(np.linalg.norm(p[src] - p[dst], axis=1).max()) if len(src) else 0.0
        for carrier, rate, _ in CARRIER_LADDER:
            best = None
            for frac in POWER_FRACTIONS:
                b = derive_radio_budget(
                    device_class, medium, carrier=carrier, line_rate=rate,
                    power_fraction=frac, channel=channel,
                )
                if b.reach >= need:
                    best = b  # later fractions are smaller: keep shrinking
                else:
                    break
            if best is not None:
                return evaluate_setting(graph, positions, best)
        # Nothing on the ladder spans the graph.  Report the longest reach
        # available rather than silently returning a plan that cannot run.
        b = max(
            (derive_radio_budget(device_class, medium, carrier=c, line_rate=r,
                                 power_fraction=1.0, channel=channel)
             for c, r, _ in CARRIER_LADDER),
            key=lambda x: x.reach,
        )
        return evaluate_setting(graph, positions, b)


@register("broadcaster", "swept", "best_rate")
class SweptBroadcaster(Broadcaster):
    """Sweep the whole grid and keep the best per-node rate.

    The comparator that says whether ``min_reach``'s heuristic -- prefer
    bandwidth, then shrink power -- is the right trade, or whether a slower
    carrier with far better reuse wins.
    """

    def choose(self, graph, positions, device_class, medium, channel):
        plans = []
        for carrier, rate, _ in CARRIER_LADDER:
            for frac in POWER_FRACTIONS:
                b = derive_radio_budget(
                    device_class, medium, carrier=carrier, line_rate=rate,
                    power_fraction=frac, channel=channel,
                )
                plans.append(evaluate_setting(graph, positions, b))
        # Feasibility FIRST.  Ranking on per-node rate alone picks the setting
        # whose reach is below the shortest edge: nobody hears anybody, so the
        # conflict graph is empty, the colouring needs one slot, and the plan
        # reports the full line rate while delivering not one byte.  B10 hit
        # exactly this shape -- a solver that never answers looks cheapest --
        # so coverage is the primary key and rate only breaks ties among plans
        # that actually carry the graph.
        return max(
            plans,
            key=lambda p: (
                -p.uncovered_fraction,
                p.per_node_rate,
                -p.slots,
                p.budget.reach,
            ),
        )


def plan_broadcast(
    graph: TaskGraph,
    positions: np.ndarray,
    device_class,
    medium,
    model: str | Broadcaster = "min_reach",
    *,
    channel: RFChannel | None = None,
) -> BroadcastPlan:
    """Choose a radio setting and price the resulting broadcast schedule."""
    from ..registry import create

    chooser = create("broadcaster", model) if isinstance(model, str) else model
    return chooser.choose(graph, positions, device_class, medium, channel or RFChannel())

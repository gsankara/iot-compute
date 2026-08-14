"""B6 -- Schedule: time-slotted circuit switching over the beam rounds.

This is the pass every earlier one has been pointing at.  B0 measured that all
twelve stock workloads are **aim-dominated**: with 8 kbit packets at 1 Gb/s,
transmission takes microseconds and slewing a gimbal takes ~0.5 s, so over 99%
of link time is spent pointing.  B1 showed tiling makes it worse.  B5 turned the
demand graph into a sequence of rounds, each of which costs every device in it a
retarget.  Nothing so far can fix that.  This pass can, and it has exactly one
lever.

**Amortise the aim.**  Stay pointed and send more before moving.  A round that
carries ``batch`` supersteps' worth of payload pays its retarget once instead of
``batch`` times, and efficiency climbs towards the ratio of wire time to aim
time.  That is the whole content of B6.

The lever is not free, and what bounds it is the reason this is a compiler pass
and not a constant:

* **RAM bounds the batch.**  Buffering ``batch`` supersteps of traffic costs
  bytes on a device that has already been filled to its limit by B1 and B2.  The
  same ``DeviceClass.compute.ram_bytes`` that stopped the tiling stops the
  batching.
* **Latency grows with the batch.**  A link is served once per frame, so the
  frame period *is* the worst-case wait for every link in it.  Efficiency and
  responsiveness pull in opposite directions and the schedulers here differ
  precisely in where they stop.

Guard bands come from :meth:`ClockDomain.required_guard`, whose docstring says
the compiler is its caller: clocks drift, so every slot carries dead time sized
by the worst oscillator in the fleet and the resync interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..registry import register
from ..sim.clocks import ClockDomain
from .beam import BeamAssignment, bearings
from .ir import TaskGraph


@dataclass(frozen=True)
class Slot:
    """One round's turn on the wire."""

    round: int
    #: Seconds slewing into this round's aim, before anything can be sent.
    retarget: float
    #: Seconds actually transmitting.
    wire: float
    #: Dead time carried for clock uncertainty.
    guard: float
    #: Payload delivered in this slot, over all its parallel links.
    payload: float
    #: How many supersteps' traffic this slot carries.
    batch: int = 1

    @property
    def duration(self) -> float:
        return self.retarget + self.wire + self.guard

    @property
    def efficiency(self) -> float:
        d = self.duration
        return self.wire / d if d > 0 else 0.0


@dataclass
class Schedule:
    """One frame of slots, repeated for the life of the run."""

    slots: list[Slot]
    graph: TaskGraph
    device_class: object
    assignment: BeamAssignment
    method: str = ""
    buffer_bytes: float = 0.0
    unscheduled: int = 0
    #: Batch the amortisation *wanted* before RAM clamped it, and the buffer
    #: that batch would have needed.  The gap between this and what was granted
    #: is the pass's most important output.
    batch_wanted: int = 1
    buffer_wanted: float = 0.0

    @property
    def starved(self) -> bool:
        """Did RAM stop the batch short of amortising the aim?"""
        return self.batch_wanted > self.batch

    @property
    def buffer_shortfall(self) -> float:
        """How many times more buffer the target efficiency would need."""
        have = float(self.device_class.compute.ram_bytes)
        return self.buffer_wanted / have if have > 0 else math.inf

    @property
    def period(self) -> float:
        """Seconds for one frame -- and the worst-case wait for any link."""
        return sum(s.duration for s in self.slots)

    @property
    def wire_time(self) -> float:
        return sum(s.wire for s in self.slots)

    @property
    def retarget_time(self) -> float:
        return sum(s.retarget for s in self.slots)

    @property
    def guard_time(self) -> float:
        return sum(s.guard for s in self.slots)

    @property
    def efficiency(self) -> float:
        """Share of wall time spent transmitting rather than pointing or waiting.

        The pass's quality metric, and the number Phase A put at under 1%.
        """
        p = self.period
        return self.wire_time / p if p > 0 else 0.0

    @property
    def batch(self) -> int:
        return max((s.batch for s in self.slots), default=1)

    @property
    def payload_per_superstep(self) -> float:
        total = sum(s.payload for s in self.slots)
        return total / max(self.batch, 1)

    @property
    def throughput(self) -> float:
        """Delivered bytes per second, averaged over the frame."""
        p = self.period
        return sum(s.payload for s in self.slots) / p if p > 0 else 0.0

    @property
    def latency(self) -> float:
        """Worst-case seconds a superstep waits before its link is served."""
        return self.period

    def fits(self, device_class=None) -> bool:
        """Does the buffering the batch requires fit the device's RAM?"""
        cls = device_class or self.device_class
        return self.buffer_bytes <= float(cls.compute.ram_bytes)

    def summary(self) -> str:
        return (
            f"{self.method:12s} batch={self.batch:5d}  period {self.period:8.2f} s  "
            f"efficiency {self.efficiency:7.3%}  "
            f"throughput {self.throughput / 1e3:9.2f} kB/s  "
            f"buffer {self.buffer_bytes / 1024:8.1f} KiB  "
            f"{'fits' if self.fits() else 'OVER'}"
        )


def retarget_steps(
    assignment: BeamAssignment, graph: TaskGraph, positions: np.ndarray, device_class
) -> list[int]:
    """Gimbal ticks each device must traverse entering each round.

    Uses the real bearings rather than a nominal slew: a round whose links point
    near the previous round's is cheap to enter, and a scheduler that ignored
    that would misprice its own frame.
    """
    grid = device_class.optical.gimbal.grid
    az, el = bearings(positions, graph.edges)
    rounds = assignment.rounds
    # Per device, the bearing it holds in each round (first link it serves).
    held: dict[int, dict[int, tuple[float, float]]] = {}
    for idx in range(graph.n_edges):
        r = int(assignment.round_of[idx])
        if r < 0:
            continue
        u, v = int(graph.edges[idx, 0]), int(graph.edges[idx, 1])
        held.setdefault(u, {}).setdefault(r, (float(az[idx]), float(el[idx])))
        held.setdefault(v, {}).setdefault(r, (float(az[idx]) - math.pi, float(-el[idx])))

    steps = []
    for r in range(rounds):
        worst = 0
        for _, per_round in held.items():
            if r not in per_round:
                continue
            prev = per_round.get((r - 1) % rounds if rounds else 0)
            if prev is None:
                continue
            a0, e0 = prev
            a1, e1 = per_round[r]
            worst = max(worst, grid.steps_between(a0, e0, a1, e1))
        steps.append(worst)
    return steps


class Scheduler:
    """The ``scheduler`` plugin slot: how many supersteps a slot carries."""

    def batch_for(self, per_slot_bytes: float, ram: float, wire: float, overhead: float) -> int:
        raise NotImplementedError  # pragma: no cover


@register("scheduler", "single", "unbatched")
class SingleScheduler(Scheduler):
    """Retarget for every superstep.  The naive schedule, and the comparator.

    This is what Phase A measured at under 1% efficiency, and it is what any
    schedule looks like if the aim cost is not deliberately amortised.
    """

    def batch_for(self, per_slot_bytes: float, ram: float, wire: float, overhead: float) -> int:
        return 1


@register("scheduler", "amortised", "amortized")
class AmortisedScheduler(Scheduler):
    """Batch until the aim is amortised or the buffer runs out.  The pass.

    Targets an efficiency rather than a batch size: batch enough supersteps that
    wire time reaches ``target`` of the slot, then stop.  Stopping matters --
    every extra superstep past that point buys a fraction of a percent of
    efficiency and costs a full frame of latency.
    """

    def __init__(self, target: float = 0.9, cap: int = 1 << 16) -> None:
        self.target = float(target)
        self.cap = int(cap)

    def wanted(self, wire: float, overhead: float) -> int:
        """Batch that would hit the target, before RAM has a say."""
        if wire <= 0:
            return 1
        # k*wire / (k*wire + overhead) >= target  ->  k >= target*overhead/((1-target)*wire)
        t = min(max(self.target, 0.0), 0.999999)
        return int(max(1, math.ceil(t * overhead / max((1.0 - t) * wire, 1e-30))))

    def batch_for(self, per_slot_bytes: float, ram: float, wire: float, overhead: float) -> int:
        by_ram = ram / per_slot_bytes if per_slot_bytes > 0 else math.inf
        return int(max(1, min(self.wanted(wire, overhead), by_ram, self.cap)))


@register("scheduler", "max_batch", "greedy")
class MaxBatchScheduler(Scheduler):
    """Fill the buffer.  Best efficiency, worst latency -- the other extreme."""

    def __init__(self, cap: int = 1 << 16) -> None:
        self.cap = int(cap)

    def batch_for(self, per_slot_bytes: float, ram: float, wire: float, overhead: float) -> int:
        by_ram = ram / per_slot_bytes if per_slot_bytes > 0 else math.inf
        return int(max(1, min(by_ram, self.cap)))


def build_schedule(
    graph: TaskGraph,
    assignment: BeamAssignment,
    positions: np.ndarray,
    device_class,
    model: Scheduler | str = "amortised",
    resync_interval: float = 1.0,
    buffer_fraction: float = 0.25,
) -> Schedule:
    """Turn beam rounds into timed slots, amortising the aim where it can.

    ``buffer_fraction`` is the share of RAM the schedule may spend on batching.
    The rest belongs to the working set B1 and B2 already sized, so batching
    cannot simply consume the device.
    """
    if isinstance(model, str):
        from ..registry import create

        name, model = model, create("scheduler", model)
    else:
        name = type(model).plugin_name

    o = device_class.optical
    gimbal = o.gimbal
    ram = float(device_class.compute.ram_bytes) * buffer_fraction
    guard = ClockDomain.required_guard(
        device_class.clock_drift_ppm, resync_interval, jitter=device_class.clock_jitter
    )
    steps = retarget_steps(assignment, graph, positions, device_class)

    slots: list[Slot] = []
    worst_buffer = 0.0
    batch_wanted = 1
    buffer_wanted = 0.0
    for r in range(assignment.rounds):
        in_round = assignment.round_of == r
        payload = float(graph.payload[in_round].sum())
        if payload <= 0:
            continue
        # Links in a round run on separate heads, so the round takes as long as
        # its slowest link, not the sum.
        longest = float(graph.payload[in_round].max())
        wire = longest * 8.0 / max(o.line_rate, 1e-30)
        retarget = gimbal.retarget_time(steps[r] if r < len(steps) else 0)
        overhead = retarget + guard

        batch = model.batch_for(longest, ram, wire, overhead)
        worst_buffer = max(worst_buffer, longest * batch)
        if hasattr(model, "wanted"):
            want = model.wanted(wire, overhead)
            if want > batch_wanted:
                batch_wanted = want
            buffer_wanted = max(buffer_wanted, longest * want)
        slots.append(Slot(
            round=r, retarget=retarget, wire=wire * batch, guard=guard,
            payload=payload * batch, batch=batch,
        ))

    return Schedule(
        slots=slots, graph=graph, device_class=device_class, assignment=assignment,
        method=name, buffer_bytes=worst_buffer,
        unscheduled=int((assignment.round_of < 0).sum()),
        batch_wanted=batch_wanted, buffer_wanted=buffer_wanted,
    )

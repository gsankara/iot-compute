"""Phase C -- the invariants every large-scale run is checked against.

``docs/ROADMAP.md`` lists five, and they are checked on *every* run rather than
sampled, because the point of a campaign is to catch the run that misbehaves,
not to characterise the average one:

* **no double delivery** -- a payload crosses each link once per superstep;
* **no task executed twice** -- one COMPUTE per device per frame;
* **schedule feasibility** -- no device is asked for more simultaneous beams
  than it has transceivers;
* **energy never negative** -- the frame is inside what the harvester supplies;
* **aim angles on the quantization grid and inside sweep limits** -- every
  bearing the program commands is one the gimbal can actually reach.

Each returns a :class:`Violation` or ``None``, so a run's verdict is the list of
what broke rather than a boolean.  A campaign that reports "37 runs failed" and
cannot say how is not evidence of anything, which is the same standard Phase A
set for its fault campaigns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..compiler.codegen import Op


@dataclass(frozen=True)
class Violation:
    """One broken invariant, with enough detail to reproduce it."""

    invariant: str
    detail: str
    #: How far off it was, where that is meaningful.
    magnitude: float = 0.0

    def __str__(self) -> str:
        return f"{self.invariant}: {self.detail}"


def no_double_delivery(programs, states) -> Violation | None:
    """Every SEND is matched by exactly one RECV, and none is duplicated.

    A link needs both gimbals, so a byte sent must be a byte received.  A
    mismatch means either an edge was emitted one-way -- not physically
    representable -- or a payload was delivered twice.
    """
    sent = sum(s.bytes_sent for s in states)
    received = sum(s.bytes_received for s in states)
    if sent != received:
        return Violation(
            "no_double_delivery",
            f"{sent} bytes sent against {received} received",
            abs(sent - received),
        )

    # Per edge: each (device, peer) pair is transferred once per frame.
    for program in programs:
        seen: dict[tuple[int, int], int] = {}
        for ins in program.frame:
            if ins.op in (Op.SEND, Op.RECV):
                key = (program.device, int(ins.a))
                seen[key] = seen.get(key, 0) + 1
        repeated = {k: v for k, v in seen.items() if v > 1}
        if repeated:
            pair, count = next(iter(repeated.items()))
            return Violation(
                "no_double_delivery",
                f"device {pair[0]} transfers to peer {pair[1]} {count} times in one frame",
                float(count),
            )
    return None


def no_task_executed_twice(programs) -> Violation | None:
    """One COMPUTE per device per frame.

    The frame *is* the superstep, so a device that computes twice has run the
    task twice -- which for an iterative solver is not a slow answer but a wrong
    one.
    """
    for program in programs:
        computes = sum(1 for ins in program.frame if ins.op is Op.COMPUTE)
        if computes > 1:
            return Violation(
                "no_task_executed_twice",
                f"device {program.device} computes {computes} times in one frame",
                float(computes),
            )
    return None


def schedule_is_feasible(assignment, device_class) -> Violation | None:
    """No round asks a device for more beams than it owns."""
    b = int(device_class.optical.n_transceivers)
    for r in range(assignment.rounds):
        degree = assignment.degree_in_round(r)
        if degree.size and int(degree.max()) > b:
            worst = int(np.argmax(degree))
            return Violation(
                "schedule_is_feasible",
                f"round {r} needs {int(degree.max())} beams on device {worst}, "
                f"which has {b}",
                float(int(degree.max()) - b),
            )
    return None


def energy_never_negative(budget) -> Violation | None:
    """The frame is inside what the device can harvest and store."""
    if not math.isfinite(budget.period):
        return Violation(
            "energy_never_negative",
            "no duty cycle balances the draw -- the harvester supplies nothing",
            math.inf,
        )
    if budget.balance_j < 0:
        return Violation(
            "energy_never_negative",
            f"draw exceeds harvest by {-budget.balance_j * 1e3:.3f} mJ per frame; "
            f"store empties in {budget.endurance:.1f} s",
            -budget.balance_j,
        )
    return None


def aims_are_reachable(programs, device_class) -> Violation | None:
    """Every commanded bearing is on the grid and inside the sweep limits.

    The program carries *ticks*, so being on the grid is by construction; what
    is not automatic is staying inside the limits, and a program that commands
    an unreachable angle is one the hardware silently will not execute.
    """
    grid = device_class.optical.gimbal.grid
    step_az = grid.step_az or 1e-6
    step_el = grid.step_el or 1e-6
    for program in programs:
        for ins in program.frame:
            if ins.op is not Op.AIM:
                continue
            az, el = ins.b * step_az, ins.c * step_el
            if not grid.in_range(az, el):
                return Violation(
                    "aims_are_reachable",
                    f"device {program.device} commands az={math.degrees(az):.1f}deg "
                    f"el={math.degrees(el):.1f}deg, outside the sweep limits",
                    abs(el),
                )
    return None


def check_all(programs, states, assignment, device_class, budget) -> list[Violation]:
    """Run every invariant and return everything that broke."""
    found = [
        no_double_delivery(programs, states),
        no_task_executed_twice(programs),
        schedule_is_feasible(assignment, device_class),
        energy_never_negative(budget),
        aims_are_reachable(programs, device_class),
    ]
    return [v for v in found if v is not None]


#: Names in the order :func:`check_all` runs them, for tidy reporting.
INVARIANTS = (
    "no_double_delivery",
    "no_task_executed_twice",
    "schedule_is_feasible",
    "energy_never_negative",
    "aims_are_reachable",
)

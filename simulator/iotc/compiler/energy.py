"""B7 -- Energy feasibility: can the swarm afford the schedule B6 produced?

B6 produced a frame: slots that aim, send and guard.  Every one of those costs
joules, and these devices run on what they harvest.  This pass asks the question
nothing before it has asked -- *does the energy balance?* -- and when it does
not, it pads the frame with sleep until it does.

The accounting is per frame, per device class, and every term comes from the
``DeviceClass`` rather than a constant:

* **retargeting** -- ``GimbalSpec.energy_per_step`` times the ticks B6 already
  counted.  On an aim-dominated fabric this is the term to watch.
* **transmitting** -- ``OpticalSpec.tx_electrical_power``, which includes the
  laser driver's wall-plug efficiency, over the wire time.
* **listening** -- ``rx_power`` for the whole slot rather than the wire time
  alone: a receiver cannot know when the packet will start, so it listens
  through the aim and the guard band too.
* **computing** -- ``PowerSpec.mcu_active`` over the time the class needs for
  the FLOPs the task graph asks of it.
* **idling and leaking** -- ``mcu_idle`` and ``StoreSpec.leakage_power``.  Small
  per second, not small per frame.

Against that sits harvest: the class's ``Harvester`` evaluated at the device's
position in its medium, integrated over the frame.

**Sleep is the only lever, and it buys nothing but time.**  If draw exceeds
harvest the frame is padded until the average balances, which lowers the duty
cycle -- the swarm works a smaller share of the day.  That is the honest answer
rather than a faster schedule, because B6 already made the schedule as tight as
it goes.  A class that cannot balance at *any* duty cycle is not badly
scheduled; it is the wrong device for the job, and the pass says so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..registry import create, register
from .ir import TaskGraph
from .schedule import Schedule, retarget_steps


@dataclass
class EnergyBudget:
    """Joules in and out over one frame, for one device class."""

    frame: float
    harvest_w: float
    retarget_j: float
    tx_j: float
    rx_j: float
    compute_j: float
    idle_j: float
    leak_j: float
    sleep: float = 0.0
    #: Joules the store can supply before brownout, for the endurance estimate.
    usable_store_j: float = 0.0
    policy: str = ""

    @property
    def draw_j(self) -> float:
        """Joules consumed per frame.  Independent of sleep except for leakage."""
        return (self.retarget_j + self.tx_j + self.rx_j
                + self.compute_j + self.idle_j + self.leak_j)

    @property
    def period(self) -> float:
        return self.frame + self.sleep

    @property
    def harvest_j(self) -> float:
        return self.harvest_w * self.period

    @property
    def balance_j(self) -> float:
        """Positive means the device gains charge over the frame."""
        return self.harvest_j - self.draw_j

    @property
    def sustainable(self) -> bool:
        return self.balance_j >= -1e-12

    @property
    def duty_cycle(self) -> float:
        """Share of wall time the swarm is working rather than waiting to charge."""
        p = self.period
        return self.frame / p if p > 0 else 0.0

    @property
    def endurance(self) -> float:
        """Seconds until brownout when the balance is negative.

        Infinite when the device is sustainable.  This is what turns "draining"
        from a warning into a number a deployment can be planned around.
        """
        if self.sustainable:
            return math.inf
        deficit_per_second = -self.balance_j / max(self.period, 1e-30)
        return self.usable_store_j / deficit_per_second if deficit_per_second > 0 else math.inf

    @property
    def terms(self) -> dict[str, float]:
        return {
            "retarget": self.retarget_j, "tx": self.tx_j, "rx": self.rx_j,
            "compute": self.compute_j, "idle": self.idle_j, "leak": self.leak_j,
        }

    @property
    def dominant(self) -> str:
        return max(self.terms, key=self.terms.get)

    @property
    def dominant_share(self) -> float:
        d = self.draw_j
        return self.terms[self.dominant] / d if d > 0 else 0.0

    def summary(self) -> str:
        return (
            f"{self.policy:12s} draw {self.draw_j * 1e3:9.3f} mJ  "
            f"harvest {self.harvest_j * 1e3:9.3f} mJ  duty {self.duty_cycle:8.3%}  "
            f"{self.dominant}={self.dominant_share:5.1%}  "
            f"{'sustainable' if self.sustainable else 'DRAINING'}"
        )


class EnergyPolicy:
    """The ``energy_policy`` plugin slot: how to react to a negative balance."""

    def sleep_for(self, frame: float, draw: float, harvest_w: float) -> float:
        raise NotImplementedError  # pragma: no cover


@register("energy_policy", "ignore", "none")
class IgnoreEnergy(EnergyPolicy):
    """Run the schedule as B6 wrote it and report the balance.  The comparator.

    Not a straw man: it is what every pass before B7 implicitly assumed, so the
    gap between this and ``duty_cycle`` is exactly what ignoring energy costs.
    """

    def sleep_for(self, frame: float, draw: float, harvest_w: float) -> float:
        return 0.0


@register("energy_policy", "duty_cycle", "sleep")
class DutyCycle(EnergyPolicy):
    """Pad the frame with sleep until harvest covers the draw.  The pass.

    Solves ``harvest_w * (frame + sleep) = draw`` for the shortest sleep that
    balances.  If the harvester supplies nothing the answer is infinite, and
    that is reported rather than clamped -- a device with no power source does
    not have a long duty cycle, it has none.
    """

    def __init__(self, margin: float = 1.05) -> None:
        #: Balance to slightly better than break-even; a schedule that balances
        #: exactly has no headroom for a cloudy hour.
        self.margin = float(margin)

    def sleep_for(self, frame: float, draw: float, harvest_w: float) -> float:
        need = draw * self.margin
        if harvest_w <= 0:
            return math.inf if need > 0 else 0.0
        required_period = need / harvest_w
        return max(0.0, required_period - frame)


@register("energy_policy", "half_duty")
class HalfDuty(EnergyPolicy):
    """Cap the duty cycle at 50% whatever the balance says.

    A fixed policy, carried to show that a rule of thumb is either wasteful or
    insufficient depending on the class -- which is the argument for computing
    the sleep rather than picking one.
    """

    def sleep_for(self, frame: float, draw: float, harvest_w: float) -> float:
        return frame


def harvest_power(device_class, position: np.ndarray, medium, now: float = 0.0) -> float:
    """Watts the class's harvester supplies at this position, in this medium."""
    spec = device_class.harvest
    harvester = create("harvester", spec.kind)
    return float(harvester.power(spec, np.asarray(position, dtype=float), medium, now))


def assess(
    schedule: Schedule,
    graph: TaskGraph,
    positions: np.ndarray,
    device_class,
    medium,
    policy: EnergyPolicy | str = "duty_cycle",
) -> EnergyBudget:
    """Price one frame of ``schedule`` against what the class can harvest."""
    if isinstance(policy, str):
        name, policy = policy, create("energy_policy", policy)
    else:
        name = type(policy).plugin_name

    o = device_class.optical
    power = device_class.power
    store = device_class.store
    frame = schedule.period

    # Retargeting: the ticks B6 priced in time, priced again in joules.
    steps = retarget_steps(schedule.assignment, graph, positions, device_class)
    retarget_j = sum(
        o.gimbal.retarget_energy(steps[s.round] if s.round < len(steps) else 0)
        for s in schedule.slots
    )

    tx_j = o.tx_electrical_power * schedule.wire_time
    # The receiver cannot know when a packet starts, so it listens through the
    # aim and the guard band as well as the wire time.
    rx_j = o.rx_power * frame

    flops = max(device_class.compute.flops, 1e-30)
    # The budget describes one device, so it is charged for *its* task, not the
    # whole graph.  The coarsened graph holds one task per device, so summing
    # would overstate compute by the device count -- and compute is large enough
    # here to decide which term dominates.  The worst-loaded device sets the
    # budget, because a schedule that only the median device can afford is not
    # a feasible schedule.
    compute_time = float(graph.flop.max(initial=0.0)) / flops * max(schedule.batch, 1)
    compute_j = power.mcu_active * compute_time
    idle_j = power.mcu_idle * max(frame - compute_time, 0.0)

    harvest_w = harvest_power(device_class, np.asarray(positions).mean(axis=0), medium)
    draw_without_leak = retarget_j + tx_j + rx_j + compute_j + idle_j
    sleep = policy.sleep_for(frame, draw_without_leak, harvest_w)
    # Leakage is charged over the *padded* period: sleeping does not stop it,
    # which is why an arbitrarily long duty cycle cannot rescue every class.
    leak_j = store.leakage_power * (frame + (sleep if math.isfinite(sleep) else 0.0))

    usable = max(store.capacity * (store.initial_fraction - store.brownout_fraction), 0.0)
    return EnergyBudget(
        frame=frame, harvest_w=harvest_w, retarget_j=retarget_j, tx_j=tx_j,
        rx_j=rx_j, compute_j=compute_j, idle_j=idle_j, leak_j=leak_j,
        sleep=sleep, usable_store_j=usable, policy=name,
    )

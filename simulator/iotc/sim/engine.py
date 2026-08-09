"""The simulation engine.

Hybrid by necessity.  Motion, energy, harvesting and clock drift are continuous
and are integrated on a fixed outer tick; packets are discrete and are handled
by an event queue.  Running one clock for both would either integrate the
physics at nanosecond resolution or quantise a 1 Gb/s transfer to a 10 ms grid,
and neither is acceptable.

Determinism is a hard requirement rather than a nice property, because a fault
campaign that cannot replay a failure is not much use.  Every random draw comes
from a named substream (see :mod:`iotc.rng`) and the event queue breaks ties on
a monotonic sequence number, so two runs with the same seed produce byte
identical traces.
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from .world import Device, World


# --------------------------------------------------------------------------
@dataclass
class Message:
    """One transmission in flight."""

    id: int
    src: int
    dst: int
    bits: int
    kind: str = "data"
    payload: object = None
    sent_at: float = 0.0
    deliver_at: float = 0.0
    #: False if the physics or a fault destroyed it in flight.
    ok: bool = True
    via: str = "optical"
    detail: dict = field(default_factory=dict)


class EventQueue:
    """A deterministic priority queue keyed on simulated time."""

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, object]] = []
        self._seq = itertools.count()

    def push(self, time: float, item) -> int:
        seq = next(self._seq)
        heapq.heappush(self._heap, (time, seq, item))
        return seq

    def pop(self):
        return heapq.heappop(self._heap)

    def peek_time(self) -> float:
        return self._heap[0][0] if self._heap else math.inf

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)


# --------------------------------------------------------------------------
class Simulation:
    """Drives a :class:`~iotc.sim.world.World` forward."""

    def __init__(self, world: World, dt: float = 0.01) -> None:
        self.world = world
        self.dt = float(dt)
        self.events = EventQueue()
        self.inbox: dict[int, list[Message]] = {}
        self._msg_ids = itertools.count()
        self.delivered = 0
        self.lost = 0
        self.bits_delivered = 0

    # ------------------------------------------------------------------
    # scheduling
    # ------------------------------------------------------------------
    def schedule(self, time: float, callback) -> None:
        """Run ``callback(sim, time)`` at the given simulated time."""
        self.events.push(max(time, self.world.time), ("call", callback))

    def after(self, delay: float, callback) -> None:
        self.schedule(self.world.time + max(delay, 0.0), callback)

    # ------------------------------------------------------------------
    # link establishment
    # ------------------------------------------------------------------
    def establish(self, i: int, j: int) -> dict:
        """Aim both ends at each other.

        Optical links are mutual, so this either sets up both heads or reports
        which end failed; there is no useful half-open state.
        """
        a = self.world.aim(i, j)
        if not a["ok"]:
            return {"ok": False, "end": i, **a}
        b = self.world.aim(j, i)
        if not b["ok"]:
            return {"ok": False, "end": j, **b}
        ready = max(a["time"], b["time"])
        return {
            "ok": True,
            "ready_at": self.world.time + ready,
            "setup_time": ready,
            "energy": a["energy"] + b["energy"],
            "residual": max(a["residual"], b["residual"]),
        }

    def teardown(self, i: int, j: int) -> None:
        for dev_id, peer in ((i, j), (j, i)):
            tx = self.world.devices[dev_id].transceiver_for(peer)
            if tx is not None:
                tx.target = None

    # ------------------------------------------------------------------
    # transmission
    # ------------------------------------------------------------------
    def send(self, src: int, dst: int, bits: int, payload=None,
             kind: str = "data") -> dict:
        """Send over an already-established optical link.

        Returns a dict describing what happened.  Failure is normal and is
        reported, not raised: an unreliable channel that throws exceptions is
        not a model of an unreliable channel.
        """
        world = self.world
        now = world.time
        a, b = world.devices[src], world.devices[dst]

        if not (a.operational(now) and b.operational(now)):
            return self._fail(src, dst, "endpoint not operational")

        tx = a.transceiver_for(dst)
        if tx is None:
            return self._fail(src, dst, "transmitter not aimed at peer")
        if b.transceiver_for(src) is None:
            return self._fail(src, dst, "receiver not aimed at peer")
        if now < tx.ready_at:
            return self._fail(src, dst, "gimbal still slewing",
                              retry_at=tx.ready_at)
        if now < tx.busy_until:
            return self._fail(src, dst, "transceiver busy", retry_at=tx.busy_until)

        budget = world.link_budget(src, dst, payload_bits=bits)
        if not budget.feasible:
            return self._fail(src, dst, budget.reason or "link budget does not close",
                              snr_db=budget.snr_db, per=budget.per)

        duration = bits / max(budget.rate, 1.0)
        energy_tx = a.cls.optical.tx_electrical_power * duration
        energy_rx = b.cls.optical.rx_power * duration
        if not a.spend("optical_tx", energy_tx):
            return self._fail(src, dst, "transmitter out of energy")
        if not b.spend("optical_rx", energy_rx):
            return self._fail(src, dst, "receiver out of energy")

        per = min(1.0, budget.per + world.extra_packet_loss)
        rng = world.rng.stream(f"packet/{src}/{dst}")
        lost = bool(rng.random() < per)

        msg = Message(
            id=next(self._msg_ids), src=src, dst=dst, bits=bits, kind=kind,
            payload=payload, sent_at=now,
            deliver_at=now + duration + budget.latency, ok=not lost,
            via="optical",
            detail={"snr_db": budget.snr_db, "per": per, "rate": budget.rate},
        )
        tx.busy_until = msg.deliver_at
        self.events.push(msg.deliver_at, ("deliver", msg))

        world.trace.add("optical_bits", bits)
        world.trace.add("optical_time", duration)
        world.trace.count("send.optical")
        return {"ok": True, "message": msg, "duration": duration,
                "energy": energy_tx + energy_rx, "per": per, "lost": lost,
                "snr_db": budget.snr_db}

    def send_rf(self, src: int, dst: int, bits: int, payload=None,
                kind: str = "control") -> dict:
        """Send over the omnidirectional radio: no aiming, but slow and shared."""
        world = self.world
        now = world.time
        a, b = world.devices[src], world.devices[dst]
        if not (a.operational(now) and b.operational(now)):
            return self._fail(src, dst, "endpoint not operational", via="rf")

        budget = world.rf_budget(src, dst)
        if not budget.feasible:
            return self._fail(src, dst, budget.reason or "rf link does not close", via="rf")

        duration = bits / max(budget.rate, 1.0)
        if not a.spend("radio", a.cls.radio.tx_electrical_power * duration):
            return self._fail(src, dst, "transmitter out of energy", via="rf")
        b.spend("radio", b.cls.radio.rx_power * duration)

        per = min(1.0, budget.per + world.extra_packet_loss)
        rng = world.rng.stream(f"rf_packet/{src}/{dst}")
        lost = bool(rng.random() < per)
        msg = Message(
            id=next(self._msg_ids), src=src, dst=dst, bits=bits, kind=kind,
            payload=payload, sent_at=now,
            deliver_at=now + duration + budget.latency, ok=not lost, via="rf",
            detail={"snr_db": budget.snr_db, "per": per},
        )
        self.events.push(msg.deliver_at, ("deliver", msg))
        world.trace.add("rf_bits", bits)
        world.trace.count("send.rf")
        return {"ok": True, "message": msg, "duration": duration, "lost": lost}

    def broadcast_rf(self, src: int, bits: int, payload=None) -> list[dict]:
        """One radio transmission heard by everyone in range.

        This is the primitive that makes the clique verification round cheap:
        one send, many receivers, no pointing.
        """
        out = []
        for j in range(len(self.world.devices)):
            if j == src:
                continue
            res = self.send_rf(src, j, bits, payload, kind="broadcast")
            if res.get("ok"):
                out.append(res)
        self.world.trace.count("broadcast.rf")
        return out

    def _fail(self, src: int, dst: int, reason: str, via: str = "optical", **extra) -> dict:
        self.world.trace.count(f"send_failed.{via}")
        self.world.trace.event(self.world.time, "send_failed", device=src, peer=dst,
                               reason=reason, **extra)
        return {"ok": False, "reason": reason, **extra}

    # ------------------------------------------------------------------
    # running
    # ------------------------------------------------------------------
    def _drain_events(self, until: float) -> None:
        while self.events and self.events.peek_time() <= until:
            time, _, item = self.events.pop()
            self.world.time = max(self.world.time, time)
            tag, obj = item
            if tag == "deliver":
                self._deliver(obj)
            elif tag == "call":
                obj(self, time)

    def _deliver(self, msg: Message) -> None:
        world = self.world
        dst = world.devices[msg.dst]
        if not dst.operational(world.time):
            msg.ok = False
        if msg.ok:
            self.inbox.setdefault(msg.dst, []).append(msg)
            self.delivered += 1
            self.bits_delivered += msg.bits
            world.trace.count("delivered")
        else:
            self.lost += 1
            world.trace.count("lost")
            world.trace.event(world.time, "packet_lost", device=msg.src, peer=msg.dst,
                              bits=msg.bits, via=msg.via)

    def step(self) -> None:
        """One outer tick: physics, then any events that fall inside it."""
        target = self.world.time + self.dt
        self.world.step(self.dt)
        self._drain_events(self.world.time)
        self.world.time = max(self.world.time, target)

    def run(self, until: float, on_step=None) -> "Simulation":
        """Advance to ``until`` seconds of simulated time."""
        while self.world.time < until - 1e-12:
            self.step()
            if on_step is not None:
                on_step(self)
        return self

    def run_steps(self, n: int, on_step=None) -> "Simulation":
        for _ in range(int(n)):
            self.step()
            if on_step is not None:
                on_step(self)
        return self

    def drain(self, max_time: float | None = None) -> "Simulation":
        """Run until the event queue empties (or ``max_time`` is reached)."""
        limit = math.inf if max_time is None else self.world.time + max_time
        while self.events and self.world.time < limit:
            self.step()
        return self

    # ------------------------------------------------------------------
    def receive(self, device: int) -> list[Message]:
        """Take everything waiting for ``device``."""
        msgs = self.inbox.pop(device, [])
        return msgs

    def summary(self) -> dict:
        out = self.world.summary()
        total = self.delivered + self.lost
        out.update({
            "delivered": self.delivered,
            "lost": self.lost,
            "delivery_rate": self.delivered / total if total else 1.0,
            "bits_delivered": self.bits_delivered,
            "pending_events": len(self.events),
        })
        tr = self.world.trace
        opt = tr.sums.get("optical_bits", 0.0)
        rf = tr.sums.get("rf_bits", 0.0)
        out["optical_share"] = opt / (opt + rf) if (opt + rf) > 0 else 0.0
        busy = tr.sums.get("optical_time", 0.0)
        retarget = tr.sums.get("retarget_time", 0.0)
        out["retarget_overhead"] = retarget / (busy + retarget) if (busy + retarget) > 0 else 0.0
        return out

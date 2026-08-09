"""Fault injection.

Everything here is declarative and seeded, so a campaign is a table of
``(fault type, rate, seed)`` and a failing run replays exactly.

The models are split the way the hardware fails, not the way it is convenient to
simulate.  A crashed node and a stalled node look identical for one slot and
completely different after ten.  A stuck gimbal keeps its *last* angle, so the
node still has a working laser pointed at the wrong place -- which is worse than
a dead laser, because its neighbours keep waiting for it.  A permanently failed
link is not a dead node: the rest of that node's beams still work, so the right
recovery is a reroute, not a replacement.
"""

from __future__ import annotations

import math

import numpy as np

from ..registry import register


class FaultModel:
    """Base class for a fault injector.

    ``rate`` is a per-device (or per-link) hazard rate in events per second, so
    the probability of firing in a step of ``dt`` is ``1 - exp(-rate * dt)``.
    ``onset`` delays the whole model, which is how a campaign asks "what happens
    if this starts once the swarm is already in formation".
    """

    def __init__(self, rate: float = 0.0, onset: float = 0.0, targets=None,
                 max_events: int = 0) -> None:
        self.rate = float(rate)
        self.onset = float(onset)
        self.targets = None if targets is None else set(int(t) for t in targets)
        self.max_events = int(max_events)
        self.fired = 0

    # ------------------------------------------------------------------
    def _armed(self, now: float) -> bool:
        if now < self.onset:
            return False
        return not (self.max_events and self.fired >= self.max_events)

    def _hazard(self, dt: float) -> float:
        return -math.expm1(-self.rate * dt)

    def _candidates(self, world):
        if self.targets is None:
            return range(len(world.devices))
        return [i for i in self.targets if i < len(world.devices)]

    def step(self, world, now: float, dt: float, rng) -> None:
        raise NotImplementedError

    def _fire(self, world, now: float, kind: str, device=None, peer=None, **detail) -> None:
        self.fired += 1
        world.trace.event(now, f"fault.{kind}", device=device, peer=peer, **detail)


# --------------------------------------------------------------------------
# node faults
# --------------------------------------------------------------------------
@register("fault", "node_crash", "crash", "fail_stop")
class NodeCrash(FaultModel):
    """Permanent fail-stop.  The node never comes back."""

    def step(self, world, now, dt, rng):
        if not self._armed(now):
            return
        p = self._hazard(dt)
        for i in self._candidates(world):
            if not self._armed(now):
                break
            dev = world.devices[i]
            if dev.crashed:
                continue
            if rng.random() < p:
                dev.crashed = True
                for t in dev.transceivers:
                    t.target = None
                self._fire(world, now, "node_crash", device=i)


@register("fault", "node_stall", "stall", "transient_hang")
class NodeStall(FaultModel):
    """Transient hang: the node stops executing for a while, then resumes.

    The nasty case for a time-triggered schedule, because the node wakes up
    believing it is still in the slot it went to sleep in.
    """

    def __init__(self, rate: float = 0.0, duration: float = 0.5, onset: float = 0.0,
                 targets=None, max_events: int = 0, jitter: float = 0.0) -> None:
        super().__init__(rate, onset, targets, max_events)
        self.duration = float(duration)
        self.jitter = float(jitter)

    def step(self, world, now, dt, rng):
        if not self._armed(now):
            return
        p = self._hazard(dt)
        for i in self._candidates(world):
            if not self._armed(now):
                break
            dev = world.devices[i]
            if dev.crashed or now < dev.stalled_until:
                continue
            if rng.random() < p:
                d = self.duration
                if self.jitter > 0:
                    d = max(0.0, d + float(rng.normal(0.0, self.jitter)))
                dev.stalled_until = now + d
                self._fire(world, now, "node_stall", device=i, duration=d)


@register("fault", "memory_bitflip", "bitflip", "seu")
class MemoryBitFlip(FaultModel):
    """A single-event upset in the node's working set.

    Corrupts one entry of the device's state dictionary.  Whether this is
    detected depends entirely on whether the compiled program bothered to
    checksum -- which is the point of having it.
    """

    def step(self, world, now, dt, rng):
        if not self._armed(now):
            return
        p = self._hazard(dt)
        for i in self._candidates(world):
            if not self._armed(now):
                break
            dev = world.devices[i]
            if not dev.alive or not dev.state:
                continue
            if rng.random() < p:
                key = list(dev.state)[int(rng.integers(len(dev.state)))]
                value = dev.state[key]
                if isinstance(value, np.ndarray) and value.size:
                    flat = value.reshape(-1)
                    idx = int(rng.integers(flat.size))
                    flat[idx] = _flip_bit(flat[idx], rng)
                elif isinstance(value, (int, float)):
                    dev.state[key] = _flip_bit(value, rng)
                else:
                    continue
                dev.state.setdefault("_corrupted", set()).add(key)
                self._fire(world, now, "memory_bitflip", device=i, key=str(key))


def _flip_bit(value, rng):
    """Flip one bit of the IEEE-754 representation."""
    import struct

    try:
        packed = struct.pack("<d", float(value))
    except (TypeError, ValueError, OverflowError):
        return value
    bits = int.from_bytes(packed, "little")
    bits ^= 1 << int(rng.integers(64))
    out = struct.unpack("<d", bits.to_bytes(8, "little"))[0]
    return out if math.isfinite(out) else float(value)


@register("fault", "harvest_dropout", "shading", "power_loss")
class HarvestDropout(FaultModel):
    """The energy source goes away -- a cloud, a shadow, a charger switched off.

    Scales the harvested power of the affected devices for ``duration``.  With
    ``depth=0`` this is total loss of income, which forces the node to live off
    its buffer and shows how long the schedule can coast.
    """

    def __init__(self, rate: float = 0.0, duration: float = 10.0, depth: float = 0.0,
                 onset: float = 0.0, targets=None, max_events: int = 0) -> None:
        super().__init__(rate, onset, targets, max_events)
        self.duration = float(duration)
        self.depth = float(depth)
        self._active: dict[int, tuple[float, object]] = {}

    def step(self, world, now, dt, rng):
        for i, (until, original) in list(self._active.items()):
            if now >= until:
                world.devices[i].harvester = original
                del self._active[i]
        if not self._armed(now):
            return
        p = self._hazard(dt)
        for i in self._candidates(world):
            if not self._armed(now):
                break
            if i in self._active:
                continue
            if rng.random() < p:
                dev = world.devices[i]
                self._active[i] = (now + self.duration, dev.harvester)
                dev.harvester = _ScaledHarvester(dev.harvester, self.depth)
                self._fire(world, now, "harvest_dropout", device=i,
                           duration=self.duration, depth=self.depth)


class _ScaledHarvester:
    """Wraps a harvester and multiplies its output."""

    def __init__(self, inner, scale: float) -> None:
        self.inner = inner
        self.scale = float(scale)

    def power(self, spec, position, medium, now, env=None) -> float:
        return self.scale * self.inner.power(spec, position, medium, now, env)


@register("fault", "position_drift", "drift", "current")
class PositionDrift(FaultModel):
    """An unmodelled current or wind pushing bodies off station.

    Continuous rather than event-driven: every device gets a random-walk shove
    each step, so the formation slowly degrades and the links with it.  The
    honest version of "the world is not as tidy as the placement pass assumed".
    """

    def __init__(self, sigma: float = 0.0, onset: float = 0.0, targets=None,
                 bias=(0.0, 0.0, 0.0)) -> None:
        super().__init__(0.0, onset, targets, 0)
        self.sigma = float(sigma)
        self.bias = np.asarray(bias, dtype=float)

    def step(self, world, now, dt, rng):
        if now < self.onset or (self.sigma <= 0.0 and not self.bias.any()):
            return
        scale = self.sigma * math.sqrt(dt)
        for i in self._candidates(world):
            dev = world.devices[i]
            shove = self.bias * dt
            if scale > 0:
                shove = shove + rng.normal(0.0, scale, size=3)
            dev.motion.position += shove


# --------------------------------------------------------------------------
# link faults
# --------------------------------------------------------------------------
@register("fault", "gimbal_stuck", "stuck_mirror")
class GimbalStuck(FaultModel):
    """A steering head freezes at its current angle.

    Deliberately *not* the same as losing the transceiver: the laser still fires
    into wherever it was last aimed, so a neighbour that has since moved sees
    nothing while the sender believes it is transmitting.
    """

    def __init__(self, rate: float = 0.0, duration: float = math.inf, onset: float = 0.0,
                 targets=None, max_events: int = 0, transceiver: int | None = None) -> None:
        super().__init__(rate, onset, targets, max_events)
        self.duration = float(duration)
        #: Which head to freeze; None picks one at random.
        self.transceiver = transceiver
        self._release: dict[tuple[int, int], float] = {}

    def step(self, world, now, dt, rng):
        for (i, k), until in list(self._release.items()):
            if now >= until:
                world.devices[i].transceivers[k].stuck = False
                del self._release[(i, k)]
        if not self._armed(now):
            return
        p = self._hazard(dt)
        for i in self._candidates(world):
            if not self._armed(now):
                break
            dev = world.devices[i]
            if not dev.alive or not dev.transceivers:
                continue
            if rng.random() < p:
                if self.transceiver is not None:
                    k = int(self.transceiver)
                    if k >= len(dev.transceivers):
                        continue
                else:
                    k = int(rng.integers(len(dev.transceivers)))
                if dev.transceivers[k].stuck:
                    continue
                dev.transceivers[k].stuck = True
                if math.isfinite(self.duration):
                    self._release[(i, k)] = now + self.duration
                self._fire(world, now, "gimbal_stuck", device=i, transceiver=k)


@register("fault", "laser_degradation", "laser_derate", "tx_degradation")
class LaserDegradation(FaultModel):
    """Slow loss of transmit power -- ageing, contamination, thermal droop.

    A gradual fault rather than a discrete one: links do not drop, their margin
    just erodes until the longest hop stops closing.  Tests whether the plan has
    any margin at all.
    """

    def __init__(self, half_life: float = 600.0, floor: float = 0.1, onset: float = 0.0,
                 targets=None) -> None:
        super().__init__(0.0, onset, targets, 0)
        self.half_life = float(half_life)
        self.floor = float(floor)

    def step(self, world, now, dt, rng):
        if now < self.onset or self.half_life <= 0.0:
            return
        decay = 0.5 ** (dt / self.half_life)
        for i in self._candidates(world):
            dev = world.devices[i]
            dev.laser_derate = max(self.floor, dev.laser_derate * decay)


@register("fault", "link_loss", "link_failure", "permanent_link")
class LinkLoss(FaultModel):
    """A specific pair can never talk optically again.

    Models a shattered lens or a persistent obstruction between exactly those
    two nodes.  Both endpoints stay healthy, so the correct response is to
    reroute, and a plan that has no alternative path fails here.
    """

    def __init__(self, rate: float = 0.0, onset: float = 0.0, pairs=None,
                 max_events: int = 0) -> None:
        super().__init__(rate, onset, None, max_events)
        self.pairs = None if pairs is None else [tuple(int(x) for x in p) for p in pairs]

    def step(self, world, now, dt, rng):
        if not self._armed(now):
            return
        if self.pairs is not None:
            for i, j in self.pairs:
                if j not in world.devices[i].blacklist:
                    world.devices[i].blacklist.add(j)
                    world.devices[j].blacklist.add(i)
                    self._fire(world, now, "link_loss", device=i, peer=j)
            self.pairs = None
            return
        established = world.established_links()
        if not established:
            return
        p = self._hazard(dt)
        for i, j in established:
            if not self._armed(now):
                break
            if rng.random() < p:
                world.devices[i].blacklist.add(j)
                world.devices[j].blacklist.add(i)
                self._fire(world, now, "link_loss", device=i, peer=j)


@register("fault", "channel_noise", "packet_loss", "extra_loss")
class ChannelNoise(FaultModel):
    """An extra packet-loss floor on top of whatever the physics produces.

    The blunt instrument: it lets a campaign sweep "how unreliable can the
    channel get before the answer is wrong" without having to find a physical
    parameter that produces exactly that loss rate.  Bursty mode correlates the
    losses, which is far harder on a protocol than the same average spread out.
    """

    def __init__(self, loss: float = 0.0, burst: float = 0.0, onset: float = 0.0,
                 targets=None) -> None:
        super().__init__(0.0, onset, targets, 0)
        self.loss = float(loss)
        self.burst = float(burst)
        self._bursting = False

    def step(self, world, now, dt, rng):
        if now < self.onset:
            world.extra_packet_loss = 0.0
            return
        if self.burst > 0.0:
            # Two-state Gilbert-Elliott style toggle.
            switch = -math.expm1(-dt / self.burst)
            if rng.random() < switch:
                self._bursting = not self._bursting
            world.extra_packet_loss = self.loss if self._bursting else 0.0
        else:
            world.extra_packet_loss = self.loss


# --------------------------------------------------------------------------
def build_campaign(specs) -> list[FaultModel]:
    """Build a list of fault models from declarative specs.

    Each spec is ``{"type": name, ...kwargs}``, exactly as it appears in YAML.
    """
    from ..registry import build

    return [build(spec, "fault") for spec in (specs or [])]

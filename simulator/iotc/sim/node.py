"""The node VM: execute the bytecode B9 emits.

``docs/ROADMAP.md`` puts this file at the end of the compiler chain -- "``node.py``
becomes the VM that executes it".  It is the thing that makes the bytecode a
program rather than a data structure: if the VM's trace does not reproduce the
schedule the compiler intended, one of them is wrong, and the round trip is the
only way to find out which.

The interpreter is deliberately dull.  A flat dispatch over fourteen opcodes,
one integer accumulator's worth of state, no control stack -- because the part
running it has kilobytes of RAM and a compiler that needed a clever VM would be
a compiler that had pushed its problems downstream.

Repetition lives in :class:`~iotc.compiler.codegen.DeviceProgram`, not in an
opcode, so the loop is here in the runner rather than in the instruction set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from ..compiler.codegen import DeviceProgram, Instruction


def _ops():
    """Fetch the opcode enum on first use.

    ``docs/ARCHITECTURE.md`` requires each layer to depend only downward --
    ``compiler`` sits *above* ``sim``, so importing the instruction set at module
    scope would invert that and, concretely, deadlock the import graph.  The VM
    needs the enum only while executing, so it is fetched then.
    """
    global _OPS
    if _OPS is None:
        from ..compiler.codegen import Op

        _OPS = Op
    return _OPS


_OPS = None


@dataclass
class NodeState:
    """What one device knows while it runs."""

    device: int
    #: Elapsed program time, microseconds.
    clock_us: int = 0
    #: Currently aimed peer per head, in aim order.
    aimed: list[int] = field(default_factory=list)
    bytes_sent: int = 0
    bytes_received: int = 0
    compute_us: int = 0
    sleep_us: int = 0
    guard_us: int = 0
    harvest_us: int = 0
    aims: int = 0
    barriers: int = 0
    checks: int = 0
    replications: int = 0
    scans: int = 0
    moves: int = 0
    #: Net displacement in millimetres, accumulated from MOVE.
    displacement_mm: tuple[int, int, int] = (0, 0, 0)
    ballast_ug: int = 0

    def summary(self) -> str:
        return (
            f"dev {self.device:4d}  t={self.clock_us / 1e6:9.3f} s  "
            f"tx {self.bytes_sent / 1024:10.1f} KiB  rx {self.bytes_received / 1024:10.1f} KiB  "
            f"aims {self.aims:6d}  barriers {self.barriers:5d}"
        )


class VMError(RuntimeError):
    """Raised on an instruction the VM cannot execute."""


class NodeVM:
    """Executes a :class:`DeviceProgram` and records what it did.

    ``transceivers`` bounds how many peers may be aimed at once, so a program
    that violates B5's b-matching is caught here rather than silently working in
    simulation and failing on hardware.
    """

    def __init__(self, transceivers: int = 2, strict: bool = True) -> None:
        self.transceivers = int(transceivers)
        self.strict = bool(strict)

    def run(self, program: "DeviceProgram", repeat: int | None = None) -> NodeState:
        state = NodeState(device=program.device)
        for ins in program.prologue:
            self._step(ins, state)
        rounds = program.repeat if repeat is None else repeat
        for _ in range(rounds):
            # A frame starts with every head free: the previous frame's
            # BARRIER is the point at which the aims are released.
            for ins in program.frame:
                self._step(ins, state)
        return state

    # ------------------------------------------------------------------ core
    def _step(self, ins: "Instruction", s: NodeState) -> None:
        Op = _ops()
        op = ins.op
        if op is Op.AIM:
            if ins.a not in s.aimed:
                if self.strict and len(s.aimed) >= self.transceivers:
                    raise VMError(
                        f"device {s.device} aimed at {len(s.aimed) + 1} peers with "
                        f"{self.transceivers} transceivers -- the beam assignment is "
                        f"not realisable"
                    )
                s.aimed.append(ins.a)
            s.aims += 1
        elif op is Op.SEND:
            self._require_aim(ins.a, s)
            s.bytes_sent += max(ins.b, 0)
        elif op is Op.RECV:
            self._require_aim(ins.a, s)
            s.bytes_received += max(ins.b, 0)
        elif op is Op.BCAST:
            if self.strict and not s.aimed:
                raise VMError(f"device {s.device} broadcast with no head aimed")
            s.bytes_sent += max(ins.b, 0) * max(len(s.aimed), 1)
        elif op is Op.COMPUTE:
            s.compute_us += max(ins.a, 0)
            s.clock_us += max(ins.a, 0)
        elif op is Op.GUARD:
            s.guard_us += max(ins.a, 0)
            s.clock_us += max(ins.a, 0)
        elif op is Op.SLEEP:
            s.sleep_us += max(ins.a, 0)
            s.clock_us += max(ins.a, 0)
        elif op is Op.HARVEST:
            s.harvest_us += max(ins.a, 0)
            s.clock_us += max(ins.a, 0)
        elif op is Op.BARRIER:
            s.barriers += 1
            s.aimed.clear()          # heads are released at the superstep edge
        elif op is Op.MOVE:
            x, y, z = s.displacement_mm
            s.displacement_mm = (x + ins.a, y + ins.b, z + ins.c)
            s.moves += 1
        elif op is Op.BALLAST:
            s.ballast_ug += ins.a
        elif op is Op.SCAN:
            s.scans += 1
        elif op is Op.REPLICATE:
            s.replications += 1
        elif op is Op.CHECK:
            s.checks += 1
        else:  # pragma: no cover - the enum is closed
            raise VMError(f"unknown opcode {op!r}")

    def _require_aim(self, peer: int, s: NodeState) -> None:
        if self.strict and peer not in s.aimed:
            raise VMError(
                f"device {s.device} transferred to peer {peer} without aiming at it"
            )


def run_swarm(programs, transceivers: int = 2, repeat: int | None = None,
              strict: bool = True) -> list[NodeState]:
    """Execute every device's program independently."""
    vm = NodeVM(transceivers=transceivers, strict=strict)
    return [vm.run(p, repeat=repeat) for p in programs]


def delivered_bytes(states) -> tuple[int, int]:
    """Total sent and received across the swarm.

    These must match: every SEND on one device is a RECV on its peer, because
    a link needs both gimbals.  A mismatch means the emitter produced a one-way
    edge, which is not physically representable on this fabric.
    """
    return sum(s.bytes_sent for s in states), sum(s.bytes_received for s in states)

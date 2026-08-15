"""B9 -- Codegen: emit the per-device bytecode the swarm actually runs.

Everything before this produced *plans*.  B9 turns them into something a mote
can hold in flash and execute without asking anyone anything: a prologue that
flies the device to its place, then a frame that repeats for the life of the run.

**The program is a loop, not a stream.**  B6's schedule is one frame repeated,
so emitting it as a linear instruction sequence is the naive thing and it is
what ``flat`` does, for comparison.  The real encoding stores the frame once and
a repeat count.  That is the whole lever, and it is worth orders of magnitude
because a run is thousands of frames long.

**Operands are integers, because that is what the hardware has.**  Angles are
gimbal *ticks* rather than radians -- B5 already snapped them to the grid, so
the float was a fiction -- and durations are microseconds.  Integers varint-
encode, floats do not, and this has to fit ``ComputeSpec.flash_bytes``.

The opcode set is fixed by the roadmap.  Note there is deliberately **no loop
opcode**: repetition is a property of the program, not an instruction, which
keeps the interpreter a flat switch with no control stack to get wrong on a part
with 512 KiB of RAM.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from ..registry import register
from .beam import BeamAssignment, bearings
from .ir import TaskGraph
from .schedule import Schedule


class Op(IntEnum):
    """The instruction set named in ``docs/ROADMAP.md`` B9."""

    AIM = 1        # point a head at a peer:      (peer, az_ticks, el_ticks)
    SCAN = 2       # search for a failover peer:  (hint, radius_ticks, 0)
    SEND = 3       # transmit to an aimed peer:   (peer, bytes, 0)
    RECV = 4       # listen for an aimed peer:    (peer, bytes, 0)
    BCAST = 5      # send to every aimed head:    (0, bytes, 0)
    COMPUTE = 6    # run local work:              (kflop, 0, 0)
    MOVE = 7       # translate:                   (dx_mm, dy_mm, dz_mm)
    BALLAST = 8    # set buoyancy trim:           (milligrams, 0, 0)
    BARRIER = 9    # wait for the superstep edge: (0, 0, 0)
    SLEEP = 10     # idle to save charge:         (microseconds, 0, 0)
    HARVEST = 11   # hold still and charge:       (microseconds, 0, 0)
    GUARD = 12     # clock guard band:            (microseconds, 0, 0)
    REPLICATE = 13 # mirror a task to a standby:  (peer, task, 0)
    CHECK = 14     # checkpoint / integrity test: (tag, 0, 0)


@dataclass(frozen=True)
class Instruction:
    """One operation and up to three integer operands."""

    op: Op
    a: int = 0
    b: int = 0
    c: int = 0

    def __str__(self) -> str:
        return f"{self.op.name:9s} {self.a:8d} {self.b:8d} {self.c:8d}"


def _varint(value: int) -> bytes:
    """LEB128 with zig-zag, so small negatives cost one byte like small positives.

    Operands here are mostly small and often negative -- a millimetre offset, a
    tick delta -- and a naive two's-complement encoding would spend ten bytes on
    ``-1``.
    """
    zig = (value << 1) ^ (value >> 63) if value < 0 else value << 1
    out = bytearray()
    while True:
        byte = zig & 0x7F
        zig >>= 7
        if zig:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


@dataclass
class DeviceProgram:
    """What one device holds in flash."""

    device: int
    #: Executed once: fly to station, take up the formation.
    prologue: list[Instruction] = field(default_factory=list)
    #: Executed ``repeat`` times: the steady-state superstep.
    frame: list[Instruction] = field(default_factory=list)
    repeat: int = 1
    encoding: str = "looped"

    @property
    def instructions(self) -> int:
        """Instructions *stored*, which is not the number executed."""
        return len(self.prologue) + len(self.frame)

    @property
    def executed(self) -> int:
        return len(self.prologue) + len(self.frame) * self.repeat

    def opcode_mix(self) -> dict[str, int]:
        mix: dict[str, int] = {}
        for ins in list(self.prologue) + list(self.frame):
            mix[ins.op.name] = mix.get(ins.op.name, 0) + 1
        return mix

    def checksum(self, encoder: "Encoder | None" = None) -> str:
        blob = (encoder or PackedEncoder()).encode(self)
        return hashlib.sha256(blob).hexdigest()[:16]

    def listing(self, limit: int = 20) -> str:
        lines = [f"device {self.device}  repeat={self.repeat}"]
        for label, block in (("prologue", self.prologue), ("frame", self.frame)):
            lines.append(f"  .{label}")
            for ins in block[:limit]:
                lines.append(f"    {ins}")
            if len(block) > limit:
                lines.append(f"    ... {len(block) - limit} more")
        return "\n".join(lines)


@dataclass
class Manifest:
    """What the swarm needs to agree on before any program runs."""

    devices: int
    frame_period_us: int
    repeat: int
    tick_az: float
    tick_el: float
    device_class: str
    seed: int = 0
    checksums: dict[int, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "devices": self.devices,
            "frame_period_us": self.frame_period_us,
            "repeat": self.repeat,
            "tick_az": self.tick_az,
            "tick_el": self.tick_el,
            "device_class": self.device_class,
            "seed": self.seed,
            "checksums": {str(k): v for k, v in self.checksums.items()},
        }


# ----------------------------------------------------------------- encoders

class Encoder:
    """The ``encoder`` plugin slot: program to bytes."""

    def encode(self, program: DeviceProgram) -> bytes:
        raise NotImplementedError  # pragma: no cover

    def size(self, program: DeviceProgram) -> int:
        """Encoded length in bytes, without materialising the image.

        An unrolled image of a long run is hundreds of megabytes, and the only
        thing anyone asks of it is how big it is.  Every encoding here has a
        closed form -- varint lengths do not depend on how often an instruction
        repeats -- so the size is computed rather than measured.  Subclasses
        that cannot do that inherit the honest fallback.
        """
        return len(self.encode(program))


@register("encoder", "flat", "unrolled")
class FlatEncoder(Encoder):
    """Fixed 13-byte records, frame unrolled ``repeat`` times.

    The comparator: no varints, no loop.  It is what you get by writing out what
    the device does, step by step, which is the obvious first implementation.
    """

    #: opcode byte plus three 32-bit operands
    RECORD = 13

    def encode(self, program: DeviceProgram) -> bytes:
        out = bytearray()
        stream = list(program.prologue) + list(program.frame) * program.repeat
        for ins in stream:
            out.append(int(ins.op))
            for operand in (ins.a, ins.b, ins.c):
                out += int(operand).to_bytes(4, "little", signed=True)
        return bytes(out)

    def size(self, program: DeviceProgram) -> int:
        return self.RECORD * (len(program.prologue)
                              + len(program.frame) * program.repeat)


@register("encoder", "packed", "varint")
class PackedEncoder(Encoder):
    """Varint operands, frame still unrolled.

    Isolates the encoding from the looping, so the bench can say which of the
    two actually buys the space.
    """

    def encode(self, program: DeviceProgram) -> bytes:
        out = bytearray()
        stream = list(program.prologue) + list(program.frame) * program.repeat
        for ins in stream:
            out.append(int(ins.op))
            out += _varint(ins.a) + _varint(ins.b) + _varint(ins.c)
        return bytes(out)

    def size(self, program: DeviceProgram) -> int:
        def block(instructions) -> int:
            return sum(1 + len(_varint(i.a)) + len(_varint(i.b)) + len(_varint(i.c))
                       for i in instructions)

        return block(program.prologue) + block(program.frame) * program.repeat


@register("encoder", "looped", "rolled")
class LoopedEncoder(Encoder):
    """Varints, and the frame stored **once** with a repeat count.  The pass.

    The schedule repeats by construction, so storing it once is not a
    compression trick -- it is emitting what the program actually is.
    """

    def encode(self, program: DeviceProgram) -> bytes:
        out = bytearray()
        out += _varint(len(program.prologue))
        out += _varint(len(program.frame))
        out += _varint(program.repeat)
        for ins in list(program.prologue) + list(program.frame):
            out.append(int(ins.op))
            out += _varint(ins.a) + _varint(ins.b) + _varint(ins.c)
        return bytes(out)


# -------------------------------------------------------------------- emitter

def _ticks(angle: float, step: float) -> int:
    return int(round(angle / step)) if step > 0 else int(round(angle * 1e6))


def emit(
    graph: TaskGraph,
    schedule: Schedule,
    assignment: BeamAssignment,
    positions: np.ndarray,
    device_class,
    start: np.ndarray | None = None,
    replicated: np.ndarray | None = None,
    repeat: int = 1000,
    encoding: str = "looped",
) -> tuple[Manifest, list[DeviceProgram]]:
    """Lower the compiled plan into one program per device."""
    grid = device_class.optical.gimbal.grid
    step_az = grid.step_az or 1e-6
    step_el = grid.step_el or 1e-6
    n = graph.n_tasks
    e = graph.edges
    az, el = bearings(positions, e)
    replicated = np.zeros(0, dtype=np.int64) if replicated is None else np.asarray(replicated)
    flops = max(device_class.compute.flops, 1e-30)

    programs: list[DeviceProgram] = []
    for d in range(n):
        prologue: list[Instruction] = []
        if start is not None:
            delta = np.asarray(positions[d]) - np.asarray(start[d])
            prologue.append(Instruction(
                Op.MOVE, int(round(delta[0] * 1e3)), int(round(delta[1] * 1e3)),
                int(round(delta[2] * 1e3)),
            ))
            if device_class.body.ballast_capacity > 0:
                # Depth is trimmed, not thrusted, on a buoyancy platform.
                trim = delta[2] / max(abs(delta[2]), 1e-9) * device_class.body.ballast_capacity
                prologue.append(Instruction(Op.BALLAST, int(round(trim * 1e6))))
        if d in set(replicated.tolist()):
            prologue.append(Instruction(Op.CHECK, tag := 1))
            del tag

        frame: list[Instruction] = []
        for slot in schedule.slots:
            in_round = np.flatnonzero(assignment.round_of == slot.round)
            mine = [i for i in in_round if d in (int(e[i, 0]), int(e[i, 1]))]
            for i in mine:
                u, v = int(e[i, 0]), int(e[i, 1])
                peer = v if u == d else u
                # Both ends aim; the far end's bearing is the near end mirrored.
                a = float(az[i]) if u == d else float(az[i]) - math.pi
                b = float(el[i]) if u == d else -float(el[i])
                frame.append(Instruction(Op.AIM, peer, _ticks(a, step_az), _ticks(b, step_el)))
                frame.append(Instruction(Op.GUARD, int(round(slot.guard * 1e6))))
                payload = int(round(float(graph.payload[i]) * slot.batch))
                frame.append(Instruction(
                    Op.SEND if u == d else Op.RECV, peer, payload,
                ))
            if mine and int(replicated.size) and d in set(replicated.tolist()):
                frame.append(Instruction(Op.REPLICATE, int(mine[0]), d))
            if mine:
                # A slot boundary is where the heads are released and retargeted.
                # Without this the device accumulates aims across every round of
                # the frame and appears to need one transceiver per neighbour --
                # which is exactly the constraint B5 exists to respect.
                frame.append(Instruction(Op.BARRIER))

        compute_us = int(round(float(graph.flop[d]) / flops * schedule.batch * 1e6))
        frame.append(Instruction(Op.COMPUTE, max(compute_us, 0)))
        frame.append(Instruction(Op.BARRIER))
        if getattr(schedule, "sleep_us", 0):
            frame.append(Instruction(Op.SLEEP, int(schedule.sleep_us)))
        frame.append(Instruction(Op.CHECK, 0))

        programs.append(DeviceProgram(
            device=d, prologue=prologue, frame=frame, repeat=repeat, encoding=encoding,
        ))

    from ..registry import create

    encoder = create("encoder", encoding)
    manifest = Manifest(
        devices=n,
        frame_period_us=int(round(schedule.period * 1e6)),
        repeat=repeat,
        tick_az=step_az,
        tick_el=step_el,
        device_class=device_class.name,
        checksums={p.device: p.checksum(encoder) for p in programs},
    )
    return manifest, programs


def program_bytes(program: DeviceProgram, encoding: str | None = None) -> int:
    from ..registry import create

    return create("encoder", encoding or program.encoding).size(program)


def fits_flash(program: DeviceProgram, device_class, encoding: str | None = None) -> bool:
    """Does the emitted image fit the part's flash?

    The same shape of question B1 asked about RAM, and the reason the encoding
    is a compiler decision rather than a serialization detail.
    """
    return program_bytes(program, encoding) <= float(device_class.compute.flash_bytes)

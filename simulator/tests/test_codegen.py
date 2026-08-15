"""B9 Codegen: the bytecode, its encodings, and the VM round trip."""

from __future__ import annotations

import numpy as np
import pytest

from iotc.compiler import build_workload
from iotc.compiler.beam import assign_beams, refine_positions
from iotc.compiler.codegen import (
    DeviceProgram,
    Instruction,
    Op,
    _varint,
    emit,
    fits_flash,
    program_bytes,
)
from iotc.compiler.partition import partition
from iotc.compiler.place import place
from iotc.compiler.schedule import build_schedule
from iotc.devices import MOTE_C, aquatic
from iotc.registry import available, create
from iotc.sim.node import NodeVM, VMError, delivered_bytes, run_swarm

WATER = create("medium", "water")


def _compiled(repeat: int = 100):
    cls = aquatic(MOTE_C)
    g = build_workload("stencil3d", nx=128, ny=128, nz=128, tx=8, ty=8, tz=8)
    devices = partition(g, MOTE_C, seed=0).coarsened()
    pos = refine_positions(devices, place(devices, cls, WATER, seed=0).positions, cls)
    assign = assign_beams(devices, pos, cls)
    sched = build_schedule(devices, assign, pos, cls, model="amortised")
    manifest, programs = emit(devices, sched, assign, pos, cls,
                              start=np.zeros_like(pos), repeat=repeat)
    return devices, sched, assign, cls, manifest, programs


# ------------------------------------------------------------ instruction set

def test_the_opcode_set_is_the_one_the_roadmap_names():
    assert {op.name for op in Op} == {
        "AIM", "SCAN", "SEND", "RECV", "BCAST", "COMPUTE", "MOVE", "BALLAST",
        "BARRIER", "SLEEP", "HARVEST", "GUARD", "REPLICATE", "CHECK",
    }


def test_there_is_no_loop_opcode():
    """Repetition is a property of the program, so the VM needs no control stack."""
    assert not any("LOOP" in op.name or "JUMP" in op.name for op in Op)


def test_small_negatives_encode_as_cheaply_as_small_positives():
    """Zig-zag: a millimetre offset of -1 must not cost ten bytes."""
    assert len(_varint(-1)) == len(_varint(1)) == 1
    assert len(_varint(-63)) <= 2
    assert len(_varint(0)) == 1


def test_varint_is_shorter_than_fixed_width_for_small_values():
    assert len(_varint(5)) < 4


# -------------------------------------------------------------------- encoding

def test_looping_the_frame_is_worth_orders_of_magnitude():
    """The schedule repeats by construction, so storing it once is the point."""
    _, _, _, _, _, programs = _compiled(repeat=1000)
    flat = sum(program_bytes(p, "flat") for p in programs)
    packed = sum(program_bytes(p, "packed") for p in programs)
    looped = sum(program_bytes(p, "looped") for p in programs)
    assert packed < flat
    assert looped < packed / 100


def test_only_the_looped_encoding_is_independent_of_run_length():
    short = _compiled(repeat=10)[5]
    long = _compiled(repeat=10_000)[5]
    assert program_bytes(long[0], "flat") > 100 * program_bytes(short[0], "flat")
    assert program_bytes(long[0], "looped") == pytest.approx(
        program_bytes(short[0], "looped"), rel=0.2
    )


def test_every_program_fits_the_part_it_targets():
    _, _, _, cls, _, programs = _compiled(repeat=1000)
    assert all(fits_flash(p, cls, "looped") for p in programs)


def test_a_long_unrolled_run_overflows_flash():
    """Why the encoding is a compiler decision and not a serialization detail."""
    _, _, _, cls, _, programs = _compiled(repeat=200_000)
    assert not fits_flash(programs[0], cls, "flat")
    assert fits_flash(programs[0], cls, "looped")


def test_the_encoders_are_registered():
    assert {"flat", "packed", "looped"} <= set(available("encoder"))


def test_checksums_are_deterministic_and_distinguish_programs():
    _, _, _, _, manifest, programs = _compiled()
    assert programs[0].checksum() == programs[0].checksum()
    assert len(set(manifest.checksums.values())) > 1


# ------------------------------------------------------------------- emission

def test_the_prologue_flies_the_device_to_its_place():
    devices, _, _, cls, _, programs = _compiled()
    moves = [i for i in programs[0].prologue if i.op is Op.MOVE]
    assert len(moves) == 1
    assert any(abs(v) > 0 for v in (moves[0].a, moves[0].b, moves[0].c))


def test_a_buoyancy_platform_trims_ballast_rather_than_thrusting_down():
    _, _, _, _, _, programs = _compiled()
    assert any(i.op is Op.BALLAST for i in programs[0].prologue)


def test_the_frame_ends_with_a_barrier_and_a_check():
    _, _, _, _, _, programs = _compiled()
    ops = [i.op for i in programs[0].frame]
    assert Op.BARRIER in ops
    assert ops[-1] is Op.CHECK


def test_angles_are_emitted_as_gimbal_ticks_not_radians():
    """B5 already snapped them, so the float was a fiction and would not varint."""
    _, _, _, _, manifest, programs = _compiled()
    aims = [i for i in programs[0].frame if i.op is Op.AIM]
    assert aims
    assert all(isinstance(i.b, int) and isinstance(i.c, int) for i in aims)
    assert manifest.tick_az > 0


# ------------------------------------------------------------- the VM round trip

def test_the_vm_delivers_exactly_what_the_schedule_planned():
    """The round trip that makes the bytecode a program rather than a structure."""
    _, sched, _, cls, _, programs = _compiled()
    states = run_swarm(programs, transceivers=cls.optical.n_transceivers, repeat=1)
    tx, rx = delivered_bytes(states)
    assert tx == rx                                  # every SEND has its RECV
    assert tx == pytest.approx(sum(s.payload for s in sched.slots), rel=1e-9)


def test_the_vm_rejects_transfers_to_an_unaimed_peer():
    vm = NodeVM(transceivers=2)
    bad = DeviceProgram(device=0, frame=[Instruction(Op.SEND, 7, 100)])
    with pytest.raises(VMError, match="without aiming"):
        vm.run(bad)


def test_the_vm_enforces_the_transceiver_count():
    """The check that caught the emitter accumulating aims across rounds.

    Heads are released at a slot boundary, not at the end of the frame.  Without
    that BARRIER a device appears to need one transceiver per neighbour, which
    is exactly the constraint B5 exists to respect.
    """
    vm = NodeVM(transceivers=2)
    over = DeviceProgram(device=0, frame=[
        Instruction(Op.AIM, 1), Instruction(Op.AIM, 2), Instruction(Op.AIM, 3),
    ])
    with pytest.raises(VMError, match="not realisable"):
        vm.run(over)

    ok = DeviceProgram(device=0, frame=[
        Instruction(Op.AIM, 1), Instruction(Op.AIM, 2), Instruction(Op.BARRIER),
        Instruction(Op.AIM, 3),
    ])
    vm.run(ok)


def test_the_emitted_programs_respect_the_transceiver_count():
    _, _, _, cls, _, programs = _compiled()
    run_swarm(programs, transceivers=cls.optical.n_transceivers, repeat=2)


def test_the_vm_clock_advances_only_on_timed_instructions():
    vm = NodeVM(transceivers=2)
    p = DeviceProgram(device=0, frame=[
        Instruction(Op.AIM, 1), Instruction(Op.GUARD, 500),
        Instruction(Op.COMPUTE, 1500), Instruction(Op.SLEEP, 1000),
    ])
    s = vm.run(p)
    assert s.clock_us == 3000
    assert s.guard_us == 500 and s.compute_us == 1500 and s.sleep_us == 1000


def test_running_more_frames_scales_the_work_linearly():
    _, _, _, cls, _, programs = _compiled()
    one = run_swarm(programs, transceivers=cls.optical.n_transceivers, repeat=1)
    five = run_swarm(programs, transceivers=cls.optical.n_transceivers, repeat=5)
    assert delivered_bytes(five)[0] == 5 * delivered_bytes(one)[0]


def test_a_broadcast_needs_at_least_one_aimed_head():
    vm = NodeVM(transceivers=2)
    with pytest.raises(VMError, match="no head aimed"):
        vm.run(DeviceProgram(device=0, frame=[Instruction(Op.BCAST, 0, 64)]))


def test_execution_is_deterministic():
    _, _, _, cls, _, programs = _compiled()
    a = run_swarm(programs, transceivers=cls.optical.n_transceivers, repeat=2)
    b = run_swarm(programs, transceivers=cls.optical.n_transceivers, repeat=2)
    assert [s.bytes_sent for s in a] == [s.bytes_sent for s in b]
    assert [s.clock_us for s in a] == [s.clock_us for s in b]

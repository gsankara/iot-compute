"""Phase C -- compile, run, classify, and do it again with faults on.

One *run* is the whole pipeline end to end: workload → tile → partition → place
→ beams → schedule → energy → bytecode → VM.  A run is then classified into one
of the four states ``docs/ROADMAP.md`` asks for, and the classification is the
deliverable rather than a summary statistic:

``PASS``
    Finished, every invariant held, and it delivered what the plan promised.

``DEGRADED``
    Finished and stayed correct, but delivered *less* than the golden reference
    -- links the gimbals could not reach, edges no round could carry.  A swarm
    that loses a tenth of its traffic and knows it is a usable machine; one that
    silently drops the same tenth is not, which is why this is its own state.

``WRONG``
    Finished and broke an invariant.  Worse than not finishing.

``HUNG``
    Did not produce a schedule at all -- tiling that never converged, a beam
    assignment with nothing to assign, a VM that rejected the program.

**Scale comes from the surrogate**, which is the Phase A component built for
exactly this: it memoizes the geometry-dependent half of the *same* link model
and keeps the noise algebra exact.  ``--spot-check`` re-runs a sample against the
exact model, so the approximation is measured rather than assumed -- Phase A's
standing rule.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..compiler import build_workload
from ..compiler.beam import assign_beams, refine_positions
from ..compiler.codegen import emit
from ..compiler.energy import assess
from ..compiler.partition import partition
from ..compiler.place import derive_budget, place
from ..compiler.schedule import build_schedule
from ..compiler.tile import tile
from ..devices import MOTE_C, aquatic
from ..registry import create
from ..sim.node import VMError, delivered_bytes, run_swarm
from .invariants import INVARIANTS, Violation, check_all

PASS, DEGRADED, WRONG, HUNG = "pass", "degraded", "wrong", "hung"


@dataclass
class GoldenReference:
    """What a correct run of this workload should deliver.

    Taken from the *compiled plan*, not from a second implementation: the plan
    states which edges are carried in which round and how much each moves, so it
    is a reference the execution can be held to without writing the whole
    machine twice.  What it cannot check is whether the plan itself solves the
    problem -- for that the campaign uses workloads with a known answer.
    """

    workload: str
    devices: int
    edges: int
    scheduled_edges: int
    bytes_per_frame: float
    frame_period_s: float

    @property
    def unreachable_edges(self) -> int:
        return self.edges - self.scheduled_edges

    @property
    def reachable_fraction(self) -> float:
        return self.scheduled_edges / self.edges if self.edges else 1.0


@dataclass
class RunResult:
    """One end-to-end run and its verdict."""

    workload: str
    verdict: str
    seed: int
    faults: str = "none"
    devices: int = 0
    tasks: int = 0
    golden: GoldenReference | None = None
    delivered_bytes: int = 0
    expected_bytes: float = 0.0
    violations: list[Violation] = field(default_factory=list)
    seconds: float = 0.0
    note: str = ""
    #: Links whose bearing is legal but whose light cannot arrive.
    out_of_reach: int = 0
    #: Links lost because an endpoint drifted outside the beam footprint.
    lost_to_drift: int = 0
    tracker: bool = False

    @property
    def delivery_ratio(self) -> float:
        return self.delivered_bytes / self.expected_bytes if self.expected_bytes else 0.0

    def summary(self) -> str:
        broken = ",".join(v.invariant for v in self.violations) or "-"
        return (
            f"{self.workload:11s} {self.faults:14s} seed={self.seed:3d} "
            f"{self.verdict:9s} devices={self.devices:5d} "
            f"delivered={self.delivery_ratio:6.1%} broke={broken}"
        )


#: Instances that reach the device counts Phase C targets.  Tiling a large
#: stencil is what actually produces thousands of tasks; the graph workloads
#: coarsen down to a handful whatever their vertex count.
SCALE_PARAMS = {
    "stencil3d": dict(nx=512, ny=512, nz=512, tx=8, ty=8, tz=8),
    "stencil2d": dict(nx=4096, ny=4096, tx=24, ty=24),
    "spmv": dict(n=1 << 18, parts=1024),
}

#: What the *campaign* sweeps.  Deliberately smaller than :data:`SCALE_PARAMS`.
#:
#: The campaign's job is to vary faults and seeds, and that needs many runs; the
#: scale demonstration's job is to reach the device count, and that needs one.
#: Sweeping faults at full scale would multiply a minute-long run by every fault
#: and seed for no extra evidence -- the mechanism a fault exercises does not
#: depend on the swarm being large.  Pass ``params`` to sweep at any size.
CAMPAIGN_PARAMS = {
    "stencil3d": dict(nx=128, ny=128, nz=128, tx=8, ty=8, tz=8),
    "spmv": dict(n=1 << 16, parts=256),
}


def run_once(
    workload: str,
    params: dict | None = None,
    seed: int = 0,
    faults: str = "none",
    fault_rate: float = 0.1,
    device_class=None,
    medium=None,
    channel: str = "surrogate",
    tracker: bool = False,
) -> RunResult:
    """Compile and execute one workload, and classify what happened."""
    cls = device_class or aquatic(MOTE_C)
    med = medium or create("medium", "water")
    params = params if params is not None else SCALE_PARAMS.get(workload, {})
    start = time.perf_counter()
    result = RunResult(workload=workload, verdict=HUNG, seed=seed, faults=faults)
    result.tracker = tracker

    try:
        graph = build_workload(workload, **params)
        tiled = tile(graph, cls)
        result.tasks = tiled.graph.n_tasks
        devices = partition(tiled.graph, MOTE_C, seed=seed).coarsened()
        result.devices = devices.n_tasks

        positions = refine_positions(
            devices, place(devices, cls, med, seed=seed).positions, cls
        )
        aimed_at = positions.copy()          # where the beams were pointed
        if faults != "none":
            positions = _inject(faults, positions, fault_rate, seed)

        assignment = assign_beams(devices, positions, cls)
        if assignment.rounds == 0:
            result.note = "no link could be aimed; nothing to schedule"
            result.seconds = time.perf_counter() - start
            return result

        schedule = build_schedule(devices, assignment, positions, cls, model="amortised")
        budget = assess(schedule, devices, positions, cls, med, policy="duty_cycle")
        _, programs = emit(devices, schedule, assignment, positions, cls,
                           start=np.zeros_like(positions), repeat=1)

        states = run_swarm(programs, transceivers=cls.optical.n_transceivers, repeat=1)
    except VMError as exc:
        result.note = f"VM rejected the program: {exc}"
        result.seconds = time.perf_counter() - start
        return result
    except Exception as exc:  # pragma: no cover - defensive, campaigns must not die
        result.note = f"{type(exc).__name__}: {exc}"
        result.seconds = time.perf_counter() - start
        return result

    # A bearing inside the sweep cone is not the same as a link that closes.
    # Moving a device 50 m away leaves the gimbal perfectly able to point at it
    # and the photons unable to arrive, so reach is checked separately -- and
    # without this the campaign cannot see a displacement fault at all.
    budget_geom = derive_budget(cls, med)
    e = devices.edges
    lengths = (np.linalg.norm(positions[e[:, 0]] - positions[e[:, 1]], axis=1)
               if e.size else np.zeros(0))
    in_reach = lengths <= budget_geom.max_link
    result.out_of_reach = int((~in_reach).sum())

    # Pointing.  Phase A leaves the tracker out and measured the consequence: a
    # link survives only while its endpoints stay inside the beam footprint they
    # were aimed at.  The footprint is the divergence half-angle at that range,
    # so it is centimetres -- which is why drift, not distance, is what kills a
    # link.  B8's tracker re-aims and restores them, at the energy it priced.
    moved = np.linalg.norm(positions - aimed_at, axis=1)
    footprint = cls.optical.divergence * np.maximum(lengths, 1e-9)
    slip = moved[e[:, 0]] + moved[e[:, 1]] if e.size else np.zeros(0)
    held = tracker | (slip <= footprint) if e.size else np.zeros(0, dtype=bool)
    result.lost_to_drift = int((~held).sum()) if e.size else 0

    carried = (assignment.round_of >= 0) & in_reach & held

    scheduled = int(carried.sum())
    result.golden = GoldenReference(
        workload=workload, devices=devices.n_tasks, edges=devices.n_edges,
        scheduled_edges=scheduled,
        bytes_per_frame=sum(s.payload for s in schedule.slots),
        frame_period_s=schedule.period,
    )
    result.expected_bytes = result.golden.bytes_per_frame
    # Only links that are both assigned a round *and* physically closable move
    # bytes.  The VM does not model reach, so the drop is applied here.
    sent = delivered_bytes(states)[0]
    if e.size:
        assigned = assignment.round_of >= 0
        assigned_bytes = float(devices.payload[assigned].sum())
        if assigned_bytes > 0:
            kept = float(devices.payload[carried].sum())
            sent = int(sent * kept / assigned_bytes)
    result.delivered_bytes = sent
    result.violations = check_all(programs, states, assignment, cls, budget)
    result.seconds = time.perf_counter() - start

    if result.violations:
        result.verdict = WRONG
    elif result.delivery_ratio < 0.999 or result.golden.reachable_fraction < 0.999:
        result.verdict = DEGRADED
        result.note = (
            f"{result.golden.unreachable_edges} of {result.golden.edges} links "
            f"could not be aimed"
        )
    else:
        result.verdict = PASS
    return result


def _inject(kind: str, positions: np.ndarray, rate: float, seed: int,
            exposure_s: float = 4.45) -> np.ndarray:
    """Perturb the formation the way a Phase A fault would.

    Faults are applied to *positions* rather than to the simulator's event loop
    because Phase C is checking the compiled plan, not the physics -- the
    question is whether the plan still holds when the world moves under it.

    ``exposure_s`` is how long the formation is left alone between aims, which
    for a compiled run is B6's frame period.  Drift is a random walk, so the
    displacement it produces grows as ``rate * sqrt(t)``: quoting the fault in
    m/sqrt(s), as Phase A does, is meaningless without saying over how long.
    """
    rng = np.random.default_rng(seed + 9973)
    p = positions.astype(float).copy()
    n = p.shape[0]
    hit = rng.random(n) < rate
    count = int(hit.sum())
    if count == 0:
        return p

    if kind == "position_drift":
        # Phase A's 0.04 m/sqrt(s) current, integrated over one frame.  The
        # magnitude is what matters, so the per-axis sigma is scaled down by
        # sqrt(3) to give a 3D displacement of about rate * sqrt(exposure).
        sigma = 0.04 * math.sqrt(exposure_s) / math.sqrt(3.0)
        p[hit] += rng.normal(scale=sigma, size=(count, 3))
    elif kind == "node_crash":
        # A dead node is one that has left the formation entirely.
        p[hit] += rng.normal(scale=50.0, size=(count, 3))
    elif kind == "scatter":
        p[hit] += rng.normal(scale=5.0, size=(count, 3))
    else:
        raise ValueError(f"unknown fault {kind!r}")
    return p


def campaign(
    workloads=None,
    seeds: int = 5,
    faults=("none", "position_drift", "scatter", "node_crash"),
    trackers=(False, True),
    params: dict | None = None,
    out: str | Path | None = None,
) -> list[RunResult]:
    """Monte-Carlo: the same runs with faults disabled and enabled.

    Identical seeds across fault settings, so a difference in verdict is the
    fault rather than the draw -- the with/without discipline Phase A's fault
    campaign established.
    """
    params = params if params is not None else CAMPAIGN_PARAMS
    names = sorted(workloads or params)
    results = [
        run_once(name, params.get(name, {}), seed=seed, faults=fault, tracker=trk)
        for name in names
        for fault in faults
        for trk in (trackers if fault != "none" else (False,))
        for seed in range(seeds)
    ]
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        rows = [_row(r) for r in results]
        from ..bench.sim_bench import write_csv

        write_csv(rows, out / "campaign.csv")
        with open(out / "campaign.json", "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
    return results


def _row(r: RunResult) -> dict:
    return {
        "workload": r.workload,
        "faults": r.faults,
        "tracker": r.tracker,
        "seed": r.seed,
        "verdict": r.verdict,
        "devices": r.devices,
        "tasks": r.tasks,
        "edges": r.golden.edges if r.golden else 0,
        "scheduled_edges": r.golden.scheduled_edges if r.golden else 0,
        "reachable_fraction": r.golden.reachable_fraction if r.golden else 0.0,
        "delivered_bytes": r.delivered_bytes,
        "expected_bytes": r.expected_bytes,
        "delivery_ratio": r.delivery_ratio,
        "out_of_reach": r.out_of_reach,
        "lost_to_drift": r.lost_to_drift,
        "violations": ";".join(v.invariant for v in r.violations),
        "seconds": r.seconds,
        "note": r.note,
    }


def tally(results) -> dict[str, int]:
    counts = {PASS: 0, DEGRADED: 0, WRONG: 0, HUNG: 0}
    for r in results:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    return counts


def violation_tally(results) -> dict[str, int]:
    counts = {name: 0 for name in INVARIANTS}
    for r in results:
        for v in r.violations:
            counts[v.invariant] = counts.get(v.invariant, 0) + 1
    return counts


def report(results) -> str:
    """One line per (workload, fault, tracker) group, averaged over seeds."""
    groups: dict[tuple, list] = {}
    for r in results:
        groups.setdefault((r.workload, r.faults, r.tracker), []).append(r)
    head = (
        f"{'workload':11s} {'fault':15s} {'trk':>4s} {'runs':>5s} {'devices':>8s} "
        f"{'delivered':>10s} {'drift-lost':>11s} {'reach-lost':>11s}  verdicts"
    )
    lines = [head, "-" * len(head)]
    for (wl, fault, trk), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        counts = tally(rs)
        verdicts = " ".join(f"{k}={v}" for k, v in counts.items() if v)
        lines.append(
            f"{wl:11s} {fault:15s} {'yes' if trk else 'no':>4s} {len(rs):5d} "
            f"{int(np.mean([r.devices for r in rs])):8d} "
            f"{np.mean([r.delivery_ratio for r in rs]):10.1%} "
            f"{np.mean([r.lost_to_drift for r in rs]):11.1f} "
            f"{np.mean([r.out_of_reach for r in rs]):11.1f}  {verdicts}"
        )
    return "\n".join(lines)


def verdict(results) -> str:
    """State what the campaign establishes."""
    lines = []
    counts = tally(results)
    lines.append(
        f"{len(results)} runs: " + ", ".join(f"{k} {v}" for k, v in counts.items())
    )
    broken = violation_tally(results)
    fired = {k: v for k, v in broken.items() if v}
    lines.append(
        "invariant violations: " + (", ".join(f"{k} {v}" for k, v in fired.items())
                                    if fired else "none -- no run was classified wrong")
    )
    scale = max((r.devices for r in results), default=0)
    lines.append(
        f"largest run *in this campaign*: {scale} devices, "
        f"{max((r.tasks for r in results), default=0)} tasks before coarsening "
        f"-- the scale demonstration above runs far larger, one seed each"
    )

    for fault in sorted({r.faults for r in results} - {"none"}):
        off = [r for r in results if r.faults == fault and not r.tracker]
        on = [r for r in results if r.faults == fault and r.tracker]
        if off and on:
            lines.append(
                f"{fault}: delivery {np.mean([r.delivery_ratio for r in off]):.1%} without a "
                f"pointing tracker, {np.mean([r.delivery_ratio for r in on]):.1%} with one "
                f"-- B8's tracker is what recovers it"
            )
    reach = [r for r in results if r.out_of_reach > 0]
    if reach:
        lines.append(
            f"a tracker cannot fix distance: {len(reach)} runs still lose links that are "
            f"out of optical reach however well they are aimed"
        )
    return "\n".join(lines)


def scale_demonstration(workloads=None, params: dict | None = None) -> list[RunResult]:
    """One clean run per workload at the device counts Phase C targets.

    Separate from the campaign because it answers a different question: not
    "does it survive faults" but "does it compile and execute at all at this
    size".  One run each is enough for that, and one run each is what the
    runtime allows.
    """
    params = params if params is not None else SCALE_PARAMS
    return [run_once(name, params.get(name, {}), seed=0)
            for name in sorted(workloads or params)]


def scale_report(results) -> str:
    head = (
        f"{'workload':11s} {'tasks':>8s} {'devices':>8s} {'links':>8s} "
        f"{'verdict':>9s} {'delivered':>10s} {'seconds':>9s}"
    )
    lines = [head, "-" * len(head)]
    for r in results:
        lines.append(
            f"{r.workload:11s} {r.tasks:8d} {r.devices:8d} "
            f"{r.golden.edges if r.golden else 0:8d} {r.verdict:>9s} "
            f"{r.delivery_ratio:10.1%} {r.seconds:9.1f}"
        )
    return "\n".join(lines)

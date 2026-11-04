# Roadmap

Sequencing follows the agreed plan: **simulator first; then the compiler pass by
pass, each pass shipping with its own testbench and evaluation; then a
large-scale emulation and evaluation testbench.**

---

## Phase A — Simulator ✅ complete

Medium, seven-stage pluggable link budget, RF, memoizing surrogate, three
kinematics, six topology laws, energy harvesting and storage, clocks and guard
bands, hybrid engine, nine fault models, trace and metrics, config loader, CLI,
and a design-space sweep harness.

Acceptance, all met:

- physics unit tests pass (see `docs/PHYSICS.md` for what each one pins down);
- two seeded runs are bit-identical, and adding a stochastic subsystem does not
  disturb an unrelated one;
- a heterogeneous swarm holds formation and its links, and energy balances;
- a fault campaign produces a graded with/without table.

---

## Phase B — Compiler, pass by pass

Each pass lands as implementation → unit tests → a pass-level evaluation script
reporting its own quality metric. No pass is done without its bench.

The spine is **FastMap** (Li & Kumar, *Revisiting FastMap: New Applications*),
which embeds a graph into ℝ^κ in near-linear time preserving shortest-path
distance.  Placing thousands of tasks so that physical distance tracks
communication distance *is* that problem, and it has to be near-linear at this
scale.

| Dissertation chapter | Pass |
|---|---|
| Ch.1 graph FastMap (Alg. 2) | **Place** — embed the task graph, get 3D coordinates |
| Ch.1 FastMap + LSH | **Runtime lookup** — nearest healthy device to a point; failover beam targets |
| Ch.2 (capacitated) vertex k-median | **Aggregator selection** under device memory capacity |
| Ch.3 top-K centrality | **Resilience** — replicate the highest-centrality tasks |
| Ch.4 FMBM block modeling | **Partition** into per-device groups; also the clique-amplified distance |
| Ch.5 graph convex hull | **Formation envelope** — perimeter and halo devices |

**B0 IR** ✅ **complete** — `TaskGraph` of tasks (FLOPs, working-set bytes) and
edges (payload bytes), in `iotc/compiler/ir.py`.  Twelve frontends ship, all
registered under the `workload` plugin kind: stencil 2D/3D Jacobi, GEMM/SUMMA,
FFT butterfly, all-reduce, SpMV/BFS, and the NP-hard family (max clique, MIS,
colouring, capacitated k-median, max-cut).

Three modelling decisions the rest of Phase B rests on: the graph describes one
*superstep* rather than a dependence DAG, because a fabric that re-planned its
topology per edge would spend all its time pointing; edges are undirected and
canonical, because a link needs both gimbals aimed and a one-way edge is not
physically representable; and the numeric fields are arrays, so B3 can stay
near-linear.

Bench: `iotc graph` (`iotc/bench/ir_bench.py`) scores every workload against
real device classes and reports which later pass has to act.  Its results:

- **every one of the twelve workloads is aim-dominated on a mote-c** (aim
  fraction ≥ 0.92) — no choice available to B0 or B1 changes this, only B6 can;
- **eleven of twelve want more simultaneous neighbours than the part has
  transceivers**, up to 7.5×, so B5 must time-share beams rather than match;
- arithmetic intensity spans 0.27 FLOP/B (all-reduce) to 170.7 (SUMMA), a 630×
  range, so a single scheduling strategy will not serve the whole family;
- the FFT is the only stock workload that fails to fit a mote-c at default size,
  which makes it B1's first customer.

**B1 Tile** ✅ **complete** — split until the working set fits the target class's
RAM, in `iotc/compiler/tile.py`.  The stopping condition is `TaskGraph.fits`,
read straight from the `DeviceClass`.

The pass is not a division, it is a choice of *shape*.  Model a task as a
d-dimensional block cut into `f₁ × … × f_d` pieces; the added traffic telescopes
to `halo × Σ(fᵢ − 1)`, so cutting `k` ways along one axis costs `(k−1)` halos
while cutting a cube costs `d·(k^(1/d) − 1)`.  Three tilers are registered under
the `tiler` kind — `geometric` (the pass), `planar`, and `slab` (the naive
comparator) — and the halo is *measured* from the task's existing edges rather
than assumed.  Halo counts against RAM, so the fit is a fixed point and the pass
iterates to convergence.

Bench: `iotc tile` (`iotc/bench/tile_bench.py`), on instances sized to overflow
the part.  Its results:

- **geometric cutting moves 18.9× less traffic than slab on average**, best case
  133×, with 2.1× fewer pieces — same workload, same target, same fit;
- for SUMMA that gap is the difference between fitting (70×70, 139× traffic) and
  **not fitting at all** after 18,480 slices;
- **some workloads cannot be tiled onto a part at any granularity**: the FFT
  butterfly and the all-reduce hit a halo floor, where the boundary exchange
  alone exceeds RAM.  The pass reports that rather than subdividing forever —
  the answer is a bigger part, not a finer cut;
- **tiling does not rescue B6**: 58 of the 61 tilings that fit are still
  aim-dominated, and every one has more edges than before.  Making a problem fit
  makes the scheduling problem harder.

**B2 Partition** ✅ **complete** — `fmbm` by default, with Kernighan–Lin,
spectral and a contiguous-block baseline as comparators, in
`iotc/compiler/partition.py` on `iotc/compiler/fastmap.py`.

Where B1 split tasks to fit, B2 packs them back together: every task boundary
that survives becomes a beam, and bytes inside a group are a memory copy.  The
method is embed with graph FastMap → KD-order the embedding → cut into
capacity-balanced runs → bounded boundary refinement.  Ordering rather than
clustering is deliberate: capacitated k-means is `O(n·k)` and the device count
grows with the problem, so it is quadratic in disguise and would sink the
acceptance criterion.

Bench: `iotc partition` (`iotc/bench/partition_bench.py`), which reports cut
quality *and* a fitted log-log runtime slope.  Its results:

- **near-linear runtime, measured: `n^0.87`** over a 16× range of task counts,
  against `n^1.17` for Kernighan–Lin and `n^2.16` for spectral bisection — 6× in
  wall-clock at 8,000 tasks and widening.  The acceptance criterion is met;
- **spectral gets the best cuts and cannot be used** — it produces groups that
  do not fit the device on five of eight workloads, because Fiedler bisection
  has no notion of capacity;
- `fmbm` beats the contiguous baseline on **six of eight** workloads.  It loses
  on BFS, where a power-law degree distribution defeats a geometric ordering,
  and on SUMMA, whose row-and-column pattern has no low-dimensional embedding.

Two findings worth carrying forward.  Shortest-path distance on a lattice is L1,
so every edge contributes equally to the first FastMap coordinate and the
Pythagorean residual zeroes them all at once — a naive implementation returns a
*one-dimensional* embedding of a 3D mesh.  `fastmap.py` detects the collapse and
recovers with fresh pivots orthogonalised against the axes already found.  And
`scipy.sparse.linalg.eigsh` draws its Krylov start vector at random, which made
spectral partitioning non-deterministic until the start vector was pinned.

**B3 Place** ✅ **complete** — graph FastMap into ℝ^κ, then an **anisotropic
projection** to physical 3D, in `iotc/compiler/place.py`.  The correction Phase A
produced is the whole content of the pass: for a photovoltaic float, z is *slow
but free* and xy is *fast but rationed*, so each axis is scaled by what the
**energy budget** allows rather than by terminal velocity.

Three placers under the `placer` kind — `anisotropic` (the pass), `velocity`
(the pre-correction reading, kept as the comparator) and `isotropic`.  Motion
budgets are derived from the `DeviceClass` and the medium, and optical reach
comes from `FSOChannel.max_range`, which was written for exactly this.

Bench: `iotc place` (`iotc/bench/place_bench.py`), across all three platforms.
Its results:

- **a float reaches 231× further down than sideways**, and placing by cost
  rather than speed makes the formation **2.78× cheaper to fly**;
- **the axes do not favour the same direction on every platform** — float 231,
  tethered head 0.06 (the wire limits depth, the gantry does not), drone 0.82
  (vertical travel fights gravity).  On the drone the naive velocity scaling is
  not penalised, and that is reported rather than hidden;
- **energy is not the binding constraint; light is.**  The float's charge allows
  161 m of depth spread but 470 nm crosses only 41.6 m of water, so every
  placement is scaled until its longest link closes, keeping 5% back as
  tolerance against the drift Phase A measured.

Two bugs worth recording.  Coincident bodies never separate — the push direction
is undefined at zero distance — so identically-connected tasks would be emitted
stacked; a deterministic spiral offset breaks the tie before relaxation.  And
`numpy.linalg.svd` returns `min(n, κ)` components, so a workload coarsened onto
two devices produced a two-column projection and crashed on the z axis.

**B4 Motion** ✅ **complete** — collision-free trajectories that preserve
critical links in flight, in `iotc/compiler/motion.py`.

The pass exists because of a gap Phase A left deliberately: there is no
closed-loop pointing tracker, and open-loop pointing tolerates centimetres.  A
link whose *both* endpoints are in motion is therefore dropped, not degraded.
That turns trajectory planning into graph colouring — colour the critical links
so no two adjacent devices move in the same wave, and every protected link keeps
a stationary endpoint for the mover to re-aim against.

Three planners under the `planner` kind: `direct` (interpolate, the comparator),
`waves` (the pass) and `sequential` (one device at a time, the other extreme).
Critical links are the smallest set carrying the heaviest half of the traffic,
since payload is heavily skewed and protecting everything costs waves.

Bench: `iotc motion` (`iotc/bench/motion_bench.py`).  Its results:

- **flying straight drops 100% of protected links** for the whole manoeuvre;
  wave-ordered flight drops **none**;
- the price is **2.18× makespan** on average, worst case 2.36×;
- **energy is identical to the digit across all three planners** — waves reorder
  the motion, they do not lengthen the paths, so link safety costs only time;
- **wave count follows degree, not device count**: 3.3 waves on average where
  moving one at a time needs 22 and runs 4.5× slower.

Still outstanding from the original scope: the harvest duty cycle is not yet
enforced here.  Motion energy is measured and charged per axis, but checking it
against what the harvester can supply belongs to B7, which iterates with B6.

**B5 Beam assignment** ✅ **complete** — degree-constrained b-matching onto
transceivers under per-class sweep limits, in `iotc/compiler/beam.py`.

A mote has two heads and wants up to thirteen neighbours, so the demand graph
cannot be a topology — it is realised as a *sequence* of b-matchings, one per
round.  Three assigners under the `assigner` kind: `heaviest_first` (the pass),
`lightest_first` (the same matching reversed, isolating the ordering) and
`arbitrary`.

Bench: `iotc beam` (`iotc/bench/beam_bench.py`).  Its results:

- **B3's energy-optimal placement cannot be aimed.**  The float formation is
  81 m tall and 0.4 m wide, so nearly every bearing points straight up or down,
  outside the ±75° elevation limit: **96.5% of links rejected**.  Each pass was
  locally right and the composition was unusable;
- **the feedback iteration recovers it** — widening the column to 20 m drops
  rejection to 3.5% and takes first-round traffic from 5.5% to 48.1%;
- **but 96.5% is a property of columns, not of gimbals** — the tethered head's
  slab placement (74.1 × 74.1 × 4.6 m, anisotropy 0.06) is **0.0% unaimable with
  no feedback iteration at all**, since a slab's bearings are near-horizontal.
  B5 does not charge for occlusion, though, so this says the slab is *pointable*,
  not that the beams arrive — and horizontal beams are the ones the wires block;
- demand exceeds transceivers by **3.8×** on average; a relay's six heads cut
  rounds from 4.9 to 2.0 and let one round carry 90% of the traffic, which is
  the strongest argument yet for heterogeneity;
- ordering by payload puts **57.6%** of traffic in the opening round against
  33.9% reversed, at identical round count.

Two corrections to the original framing.  The roadmap asks for the *snap
residual* to be fed back into B3; the residual is the minor term, and the sweep
limits are what actually break the placement.  And minimising raw residual makes
the loss **worse** — loss is quadratic in the residual, so the objective must be
the loss itself.

**B6 Schedule** ✅ **complete** — time-slotted circuit switching over B5's beam
rounds, in `iotc/compiler/schedule.py`.  Guard bands come from
`ClockDomain.required_guard`, whose docstring already named the compiler as its
caller.

The pass has exactly one lever: **stay aimed and send more per aim**.  A slot
carrying `batch` supersteps pays its retarget once instead of `batch` times.
Three schedulers under the `scheduler` kind: `single` (retarget every superstep,
the comparator), `amortised` (batch to a target efficiency, the pass) and
`max_batch` (fill the buffer).

Bench: `iotc schedule` (`iotc/bench/schedule_bench.py`).  Its results:

- unbatched efficiency is **0.002%–0.09%** on the mote classes, reproducing
  Phase A's "over 99% overhead" end to end;
- amortising lifts it by **5×–241×** depending on workload and part;
- **and it still is not enough.**  Five of six schedules are RAM-starved: the
  batch the amortisation wants is ~2,800× larger than the buffer allows, and
  reaching 90% efficiency would need **1,361 MiB of buffer against 4 MiB of
  RAM**.  The fabric stays aim-dominated because the part is too small, not
  because the schedule is poor;
- holding the gimbal fixed and varying only RAM (mote-c → mote-m): stencil3d
  goes 0.325% → 3.874%, SpMV 0.308% → 3.944%.  Buffer is the axis that moves it.

This is the clearest hardware verdict the compiler has produced: **B6 cannot fix
aim-domination, and no scheduler can.**  Closing it needs a larger buffer, a
larger packet, a faster gimbal, or a slower line rate — all decisions outside the
compiler.  The relay is deliberately excluded from the RAM comparison, since it
differs in head count and step size as well.

**B7 Energy feasibility** ✅ **complete** — price the frame against harvest and
storage, pad with sleep until it balances, in `iotc/compiler/energy.py`.

Six draw terms, each read from the `DeviceClass`: retargeting from
`GimbalSpec.energy_per_step`, transmitting at the laser driver's wall-plug
efficiency, listening for the *whole slot* (a receiver cannot know when a packet
starts, so it stays powered through the aim and the guard band), computing,
idling and leaking.  Against that, the class's `Harvester` at its position in its
medium.  Three policies under the `energy_policy` kind: `ignore` (the
comparator — what every pass before B7 implicitly assumed), `duty_cycle` (the
pass) and `half_duty` (a fixed rule, carried to show a rule of thumb is not
enough).

Bench: `iotc energy` (`iotc/bench/energy_bench.py`).  Its results:

- **the fabric is aim-dominated in time but not in energy.**  Retargeting is
  **18.1%** of the draw; listening dominates in four of six configurations and
  computing in two.  Anyone optimising this platform for energy should start
  somewhere other than the gimbal;
- **B6 and B7 pull against each other**, which is why the roadmap pairs them.
  B6 batches to amortise the retarget, and each batched superstep is another
  superstep of compute in the same frame.  The memory-rich part B6 favours most
  (batch 2048) is the only one that cannot power the result: compute reaches 98%
  of the draw and the store empties in **6 seconds**;
- padding with sleep fixes it at an **11.5% duty cycle**;
- a fixed 50% duty rule leaves that configuration still draining while costing
  the five healthy ones half their throughput — the sleep has to be computed.

One accounting bug worth recording: the first version charged every device with
the *whole graph's* FLOPs.  The coarsened graph holds one task per device, so
that overstated compute by the device count and made it look like the dominant
term at 93%.  Corrected to the worst single task, compute falls to 34% on a
mote-c and the real answer — listening — becomes visible.

**B8 Resilience** ✅ **complete (partly)** — replication chosen by centrality,
backup beam targets via LSH, and the closed-loop pointing tracker, in
`iotc/compiler/resilience.py`.  **Erasure coding, standby spares and checkpoint
intervals are not implemented** and remain open.

Four policies under the `resilience` kind: `none`, `uniform` (the comparator),
`traffic` (a cheap degree-weighted proxy) and `centrality` (sampled Brandes
betweenness — near-linear at `O(samples·(n+m))`, since ranking converges long
before absolute values do).  Failover is random-hyperplane LSH over the FastMap
embedding B2 and B3 already built, so the index costs nothing extra.

Bench: `iotc resilience` (`iotc/bench/resilience_bench.py`), 60 random failure
draws per point with standard errors, so a win has to clear the noise:

- **redundancy pays**: replicating a fifth of the swarm lifts 30%-loss survival
  from 45.9% to 58.1% on the mesh;
- **but the expensive choice does not.**  Sampled betweenness never beats the
  cheap traffic proxy — −0.005 ± 0.011 on the mesh, −0.000 ± 0.020 on SpMV,
  inside the noise both times.  Use the proxy;
- the reason is a consequence of B2, not of centrality: **B2 leaves device
  graphs dense.**  SpMV coarsens to a *complete* 7-device graph, where there are
  no bottlenecks to find and even importance-vs-random fails significance
  (−0.031 ± 0.021).  Only the mesh rewards choosing at all;
- **the pointing tracker is not optional and not cheap.**  Drift is a random
  walk so the safe interval is quadratic in tolerance: at Phase A's 0.04 m/√s
  and 2 cm tolerance a link survives **0.25 s** against a 4.45 s frame —
  **17.8 re-aims per frame** at 72.9 mJ, purely to hold links open.  Relaxing to
  5 cm cuts it to 2.8.

A modelling note worth carrying: the first survival metric counted only direct
edge survival, which is not what betweenness protects.  A connectivity-aware
measure was added — but on these instances it never differs from the direct one,
because B2's dense device graphs do not fragment.  That is reported rather than
kept as a metric that quietly never fires.

**B9 Codegen** ✅ **complete** — per-device bytecode and a global manifest, in
`iotc/compiler/codegen.py`, executed by `iotc/sim/node.py` — the VM the roadmap
asks this file to become.

The fourteen opcodes are as specified (`AIM`, `SCAN`, `SEND`, `RECV`, `BCAST`,
`COMPUTE`, `MOVE`, `BALLAST`, `BARRIER`, `SLEEP`, `HARVEST`, `GUARD`,
`REPLICATE`, `CHECK`), with **no loop opcode**: repetition is a property of the
`DeviceProgram`, so the interpreter stays a flat dispatch with no control stack.
Operands are integers — angles in gimbal *ticks*, since B5 already snapped them
and the float was a fiction — so they varint-encode.  A program is a prologue
(fly to station) plus a frame that repeats.

Three encoders under the `encoder` kind: `flat` (fixed 13-byte records,
unrolled), `packed` (varints, unrolled) and `looped` (varints, frame stored
once).

Bench: `iotc codegen` (`iotc/bench/codegen_bench.py`).  Its results:

- **the program is a loop, not a stream.**  A device stores **17 instructions**
  and executes **1,500,002**.  At 100,000 frames: flat 516,700 KiB, packed
  208,643 KiB, **looped 2.3 KiB** — varints save 2.5×, looping a further
  **89,021×**;
- **unrolling overflows a mote's 2 MiB flash from 10,000 frames on**, while the
  looped image is flat in run length (1.01× across a 1000× range).  Same shape
  of question B1 asked of RAM, which is why the encoding is a compiler decision;
- **the VM round trip passes on all 24 images**: each delivers exactly the bytes
  B6 planned, with every SEND matched by a RECV.

Two things the VM caught that review had not.  On its first run it rejected the
emitter — *"device 0 aimed at 3 peers with 2 transceivers"* — because heads were
released only at the frame edge, when B5's rounds exist precisely so devices
retarget between them; a `BARRIER` per slot boundary fixed it.  And `node.py`
initially imported the instruction set at module scope, which inverts the
downward-only layering `docs/ARCHITECTURE.md` requires (`compiler` sits *above*
`sim`) and deadlocked the import graph; the enum is now fetched at execution
time.

**B10 Maximum clique, end to end** ✅ **complete** — the flagship, in
`iotc/compiler/clique.py`.  The pipeline is as specified: FastMap-embed under the
common-neighbour attraction weight `W[i,j] = |N(i) ∩ N(j)| + A[i,j]`, LSH buckets
nominate dense balls, the devices holding a candidate converge, and one broadcast
round verifies it.  Five solvers under the `clique_solver` kind — `physical` plus
Bron–Kerbosch, colour-bound branch and bound, simulated annealing and greedy
degeneracy order.

Bench: `iotc clique` (`iotc/bench/clique_bench.py`), on a planted clique so
recovery has a ground truth that is not itself exponential to establish.

**The claim, narrowly.**  Verifying a k-clique costs `k(k−1)/2` pairwise tests
conventionally — 120 for k=16, 2,016 for k=64 — and **one broadcast round** once
the members are in mutual line of sight, whatever k is.

Its results:

- among the four solvers that find the planted clique, the physical scheme's
  **adjacency tests grow slowest over a 10× range of n: 2.1×**, against 4.8×
  (greedy), 7.9× (colour bound) and 9.1× (Bron–Kerbosch);
- **broadcast rounds are constant at 36** regardless of problem size;
- **and it is the slowest in wall-clock** — 624 ms against greedy's 3.2 ms at
  n=4000.  The FastMap embedding and LSH probing are overhead a greedy pass
  never pays.  The argument is that hardware cost is rounds of aiming (~0.5 s
  each), not CPU microseconds; on a laptop the physical scheme loses, and the
  bench reports that rather than burying it;
- physically priced: 340.9 J to converge the members, 1.82e10 photons for the
  round;
- every answer is checked independently against the adjacency sets — 15 of 15
  are genuine cliques.

Simulated annealing collapses on the larger instances (size 2 of a planted 16 at
n=4000) and is excluded from the cost comparison: its low test count is the cost
of not answering, and ranking it as cheap would invert the reading.

---

## Phase C — Large-scale emulation and evaluation ✅ **first cut complete**

**Sequencing decision (2026-08-14): correctness first, speed later.**  Phase C is
built as an unoptimized reference; time optimization comes afterwards and is
expected to be **hardware** — GPU or FPGA — rather than CPU tuning.  So a slow
pipeline here is not a defect to be fixed in passing.  Asymptotic complexity is
still in scope where the roadmap makes it an acceptance criterion (B2's
near-linear partitioning is a requirement, not an optimisation); constant factors
are not.

Implemented in `iotc/testbench/`, driven by `iotc testbench`.  A *run* is the
whole pipeline end to end — workload → tile → partition → place → beams →
schedule → energy → bytecode → VM — classified into the four states this section
asks for: **pass**, **degraded**, **wrong**, **hung**.  Degraded is its own state
on purpose: a swarm that loses a tenth of its traffic *and knows it* is a usable
machine, and one that silently drops the same tenth is not.

The five invariants are checked on **every** run, not sampled: no double
delivery, no task executed twice, schedule feasibility, energy never negative,
and aim angles on the grid and inside sweep limits.  Each returns what broke
rather than a boolean, because a campaign that reports "37 runs failed" and
cannot say how is not evidence of anything.

Results:

- **1,270 devices** from a 2,048-task tiled stencil, compiled and executed with
  every invariant holding — the roadmap's 1k–10k band is reached at the low end;
- **no run was ever classified `wrong`**: across the campaign the compiler never
  emitted a plan that violated its own rules;
- **the pointing tracker B8 built is what survives faults.**  Under scatter,
  delivery is 84.0% without it and 99.6% with it; under drift, 92.5% against
  99.6%.  Under a node crash it recovers only 83.6% → 89.5%, because **a tracker
  cannot fix distance** — those links are out of optical reach however well they
  are aimed;
- **short links are the fragile ones under drift**, which inverts the obvious
  reading.  The beam footprint is the divergence half-angle *at range*, so it
  grows with link length while the drift does not: 1.6 cm of tolerance on a 3 m
  link against 20 cm on a 39 m one.  Measured, the links drift breaks average
  15.4 m and the survivors 30.5 m.

Two things this exposed in earlier passes.  B5's refinement scored a *local*
move with a *global* cost, making it `O(n·m)` — 329 s at 159 devices, which put
these device counts out of reach entirely; scoring only the moved device's
incident edges made it 4.5 s and left B5's published results byte-identical.
And drift quoted in m/√s is meaningless without an exposure, since the
displacement grows as `rate·√t`; the injector now takes the frame period.

Still open: erasure coding, standby spares and checkpoint intervals from B8;
the golden reference is taken from the compiled plan rather than from an
independent implementation, so it checks that the *execution* matches the
*plan*, not that the plan solves the problem — only the clique workload has a
ground truth strong enough for that.

---

## Known gaps carried forward

Listed in `docs/PHYSICS.md` §7.  The two that most affect Phase B are the absence
of a **closed-loop pointing tracker** (B8) and the **aim-dominated overhead** of
small packets (B6).

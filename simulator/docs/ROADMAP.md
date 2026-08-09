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

**B2 Partition** — `fmbm` by default, with Kernighan–Lin and spectral as
comparators. Must demonstrate near-linear runtime.

**B3 Place** — graph FastMap into ℝ^κ, then an **anisotropic projection** to
physical 3D.  Note the correction Phase A produced: for a photovoltaic float,
z is *slow but free* and xy is *fast but rationed*, so the projection must be
driven by the energy budget, not by terminal velocity.

**B4 Motion** — collision-free trajectories that preserve critical links in
flight, and that respect the duty cycle a harvested budget allows.

**B5 Beam assignment** — degree-constrained b-matching onto transceivers under
per-class sweep limits; angles snapped to the grid with the residual fed back
into B3 for one refinement iteration.

**B6 Schedule** — time-slotted circuit switching, since a head serves one
neighbour at a time and retargeting costs real gimbal time.  Phase A measured
that overhead at over 99% for small packets, so amortising it is the central
job here.  Guard bands come from `ClockDomain.required_guard`.

**B7 Energy feasibility** — check against harvest and storage, insert harvest and
sleep slots, iterate with B6.

**B8 Resilience** — replication chosen by centrality, erasure coding, standby
spares, backup beam targets via LSH, checkpoint intervals.  This is also where
the **closed-loop pointing tracker** lands, which Phase A deliberately omits and
whose absence the drift fault exposes.

**B9 Codegen** — compact bytecode (`AIM`, `SCAN`, `SEND`, `RECV`, `BCAST`,
`COMPUTE`, `MOVE`, `BALLAST`, `BARRIER`, `SLEEP`, `HARVEST`, `GUARD`,
`REPLICATE`, `CHECK`) plus a per-device `DeviceProgram` and a global `Manifest`.
`iotc/sim/node.py` becomes the VM that executes it.

**B10 Maximum clique, end to end** — the flagship demonstration that physical
proximity *is* the adjacency test:

1. FastMap-embed under the **common-neighbour attraction weight**
   `W[i,j] = |N(i) ∩ N(j)| + A[i,j]`, replacing spectral initialization —
   near-linear instead of eigendecomposition;
2. **LSH** buckets nominate dense balls as candidate cliques;
3. the devices holding those vertices **physically converge**, forming an
   all-to-all optical mesh, so O(k²) mutual-adjacency verification becomes **one
   physical broadcast round**;
4. non-members are repelled and their affinity decayed — a **force-directed
   attraction/repulsion** law, now flown.  `GravityTopology.repel()` already
   implements this.

Evaluated against **Bron–Kerbosch**, **colour-bound branch and bound**,
**simulated annealing** and a **greedy degeneracy-order** baseline, on solution
quality, wall-clock, *and* the new physical axes (rounds, motion energy,
photons).

---

## Phase C — Large-scale emulation and evaluation

- 1k–10k devices via the surrogate, with periodic exact spot-checks.
- Golden host reference per workload; every run classified pass / degraded /
  wrong / hung.
- Monte-Carlo fault campaigns, run identically with faults disabled and enabled.
- Invariants checked every run: no double delivery, no task executed twice,
  schedule feasibility, energy never negative, aim angles on the quantization
  grid and inside sweep limits.
- Tidy CSV plus figures in `docs/figures/`.

---

## Known gaps carried forward

Listed in `docs/PHYSICS.md` §7.  The two that most affect Phase B are the absence
of a **closed-loop pointing tracker** (B8) and the **aim-dominated overhead** of
small packets (B6).

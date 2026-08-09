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

**B0 IR** — `TaskGraph` of tasks (FLOPs, working-set bytes) and edges (payload
bytes).  Frontends: stencil 2D/3D Jacobi, GEMM/SUMMA, FFT butterfly, all-reduce,
SpMV/BFS, and the NP-hard family (max clique, MIS, colouring, capacitated
k-median, max-cut).

**B1 Tile** — split until the working set fits the target class's RAM.

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

**B10 The clique workload, end to end** — the flagship demonstration that
physical proximity *is* the adjacency test.  Based on the user's CGKAR work in
`C:\Users\Ganesh C S\Documents\cgkar\`:

1. FastMap-embed under the CGKAR attraction weight
   `W[i,j] = |N(i) ∩ N(j)| + A[i,j]`, replacing `cgkar_gravity.spectral_init` —
   near-linear instead of eigendecomposition;
2. LSH buckets nominate dense balls as candidate cliques;
3. the devices holding those vertices **physically converge**, forming an
   all-to-all optical mesh, so O(k²) mutual-adjacency verification becomes **one
   physical broadcast round**;
4. non-members are repelled and their affinity decayed — the `cgkar_gravity`
   law, now flown.  `GravityTopology.repel()` already implements this.

Evaluated against Bron–Kerbosch, colour-bound branch and bound, simulated
annealing and the existing RINGS/GRAVITY baselines, on solution quality,
wall-clock, *and* the new physical axes (rounds, motion energy, photons).

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

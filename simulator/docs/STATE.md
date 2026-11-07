# Position is state

What a node remembers, and why this machine's memory behaves nothing like a
CPU's.  Companion to `ARCHITECTURE.md`, which describes the engine; this note
describes the thing the engine is simulating a computer *out of*.

---

## The claim

Each node holds two kinds of state.

**Resident state** is conventional: the working set a node keeps in its own
memory.  B1 and B2 size it.

**Positional state** is the node's physical coordinates.  Through the distance
conversion `d = W_max / W`, where `W` is an attraction weight, the coordinates
encode the structure of the problem itself.

A CPU or GPU is stateless with respect to the problem.  Its wires are fixed, so
the problem's structure has to be *loaded* as operands before anything can be
computed.  Here the operands do not travel; the machine does, and that travel is
the computation.  Adjacency is not stored anywhere.  It is a geometric fact
about where the bodies are.

## The measured instance: B10

Verifying a 16-clique costs **120 pairwise adjacency tests** conventionally and
**one broadcast round** physically, whatever k is.

Under the attraction weight `W[i,j] = |N(i) ∩ N(j)| + A[i,j]`, converted by
`d = W_max / W`, a dense region contracts.  In a planted 7-clique on 26 vertices
the members contract **26.6x** and end **9.5x tighter** than non-members,
collapsing into a knot.  The 120 tests are loads.  The broadcast round fetches
nothing.

The scaling agrees.  Adjacency tests grow **2.1x over a 10x range of n**,
against 4.8x for greedy degeneracy, 7.9x for the colour bound and 9.1x for
Bron-Kerbosch.  Broadcast rounds are **constant at 36** regardless of problem
size.

## Positional state is a strange memory

Five properties, all measured.

### 1. Expensive to write, and anisotropically

Writing positional state means moving a body.  Per-axis write cost:

| platform | lateral | vertical | anisotropy |
|---|---|---|---|
| float | 1.065 J/m | **0.000 J/m** | 231.01 |
| drone | 0.180 J/m | 0.221 J/m | 0.82 |
| tethered head | 0.000 J/m | 0.000 J/m | 0.06 |

A float's ballast sets a trim once and the body coasts to depth, so depth is
bounded by *time* rather than energy.  One axis of its positional state is free
to write.

### 2. Free to read, but conditionally

Nothing is fetched; photons arrive if the geometry is right.  But the standard
gimbal's elevation limit of +/-75 deg means a link is readable only while
`|dz| <= 3.73 x horizontal separation`.  That is a **15 deg blind cone about the
vertical, at both ends of the link**.

### 3. Write cost and read feasibility are anti-correlated

The free-to-write axis lies inside the blind cone.  B3 spends it and returns a
0.4 x 0.4 x 81.0 m column, an aspect ratio of **202** against a legal ceiling of
**3.73**, exceeding it **54x**.  B5 then measures **96.5% of its links
unaimable**.  The framing predicts this: a pass that minimises write cost will
drive the formation into the read-blind direction.

This is not a law, though.  It is a part property, and that was measured as a
single-variable control: the `float-hemi` platform in `beam_bench`, identical to
the float except `el_limits`.

| gimbal | el limit | step | slew | raw unaimable |
|---|---|---|---|---|
| standard | +/-75 deg | 0.10 deg | 200 deg/s | 96.5% |
| standard but +/-90 | +/-90 deg | 0.10 deg | 200 deg/s | **0.0%** |
| fine | +/-90 deg | 0.02 deg | 120 deg/s | **0.0%** |

The step contributes nothing; the limit is the whole effect.  Across workloads,
standard to hemispherical: stencil3d 96.5% to 0.0%, stencil2d 97.0% to 0.0%,
spmv 100.0% to 0.0%.

Lifting the limit is not free.  Through B6 on stencil3d/float the frame goes
**4446.37 ms to 5762.37 ms** and link efficiency **0.32% to 0.25%**, because the
+/-75 deg limit was *hiding work*: the links it rejected were never in the
schedule, and once legal they must be carried.  A true fine gimbal is worse
again at **9904.37 ms**, since it slews **1.67x slower** and costs **3.0x the
retarget energy per degree**.  A 60 deg retarget is 300.50 ms / 2995.0 uJ on the
standard gimbal against 500.33 ms / 8997.0 uJ on the fine one.

### 4. It decays, and refresh can outrun access

Drift is a random walk, so displacement grows as `rate * sqrt(t)` and the
interval before a link must be re-aimed is `(tolerance / rate)^2`.  At the
measured 0.04 m/sqrt(s):

| tolerance | re-aim every |
|---|---|
| 1 cm | 0.062 s |
| 2 cm | 0.25 s |
| 5 cm | 1.56 s |
| 10 cm | 6.25 s |
| 20 cm | 25 s |

A retarget takes about 0.5 s.  So at 2 cm tolerance the state decays faster than
it can be rewritten: **0% usable duty**.  At 5 cm tolerance the link is usable
**68%** of the time.  This is a refresh-bound regime, structurally like DRAM
where refresh consumes the bus.  The interval is *quadratic* in tolerance, so
halving the tolerance quarters it.

### 5. It is not randomly addressable

Moving one node changes its relation to every neighbour at once; you cannot
update one adjacency without perturbing others.  This is why B4 exists at all.
With no closed-loop pointing tracker, a link with both endpoints in motion is
dropped rather than degraded, so motion must be planned as graph colouring, at
**2.18x makespan** for identical energy.

## The honest form of the claim

The machine replaces **loads with geometry**, and pays in write cost, refresh,
and the loss of random access.  It wins when the read-to-write ratio is high:
many computations against one placement.  B10 is exactly that shape, with 36
broadcast rounds constant in n against baselines whose test counts grow 4.8x to
9.1x.

The same data holds the counter-example, and it should be read as one:
**B10 is the slowest solver in wall clock**, at 624 ms against greedy
degeneracy's 3.2 ms at n=4000.  The embedding and the convergence are a one-off
write that a greedy pass never pays.  With a low read-to-write ratio, geometry
loses.

The argument for the architecture is therefore about what the hardware cost is,
not about this simulator's wall clock: it is rounds of aiming at roughly 0.5 s
each, not CPU microseconds.

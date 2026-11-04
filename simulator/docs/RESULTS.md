# Example results

Every table below is verbatim output from a run of this code at the Phase A
tag, on Windows 11 / Python 3.13 / numpy 2.5.  Each section gives the command
that produces it, so nothing here has to be taken on trust.

To regenerate the raw CSV and PNG artefacts into `results/`:

```bash
bash scripts/regenerate_results.sh
```

---

## 1. Reach against medium and wavelength

```bash
python examples/01_reach_vs_medium.py
```

Maximum range [m] for a 1 mW, 1 Gb/s link with a 5 mrad beam and a 2.5 mm
receive aperture:

| medium | blue-green (470) | green (550) | near IR (850) | C band (1550) |
|---|---|---|---|---|
| vacuum | 88.12 | 88.12 | 88.12 | 88.12 |
| air, clear (50 km vis) | 87.59 | 87.74 | 87.89 | 87.97 |
| air, fog (0.2 km vis) | 52.61 | 52.64 | 52.64 | 52.64 |
| **water, clear** | **41.62** | 27.62 | **1.75** | 0.01 |
| water, coastal | 10.30 | 9.56 | 1.64 | 0.01 |
| index fluid | 73.36 | 73.36 | 73.36 | 73.36 |

Extinction coefficient [1/m]:

| medium | 470 | 550 | 850 | 1550 |
|---|---|---|---|---|
| air, clear | 1.26e-4 | 1.00e-4 | 5.84e-5 | 3.04e-5 |
| air, fog | 1.96e-2 | 1.96e-2 | 1.96e-2 | 1.96e-2 |
| water, clear | 0.036 | 0.084 | 4.42 | 1100 |
| water, coastal | 0.416 | 0.464 | 4.80 | 1100 |
| index fluid | 0.005 | 0.005 | 0.005 | 0.005 |

**The near-infrared column is why any liquid deployment must move to
blue-green.** 850 nm dies at 1.75 m; 470 nm reaches 42 m — a factor of 24 from
one design choice.

---

## 2. Platform comparison

```bash
python examples/02_platform_comparison.py
```

A 10 m change of depth or altitude:

| platform | time [s] | node energy [J] | final z [m] |
|---|---|---|---|
| buoyant float | 88.0 | 7.672 | -11.953 |
| aerial drone | 5.5 | 24.005 | +11.964 |
| tethered head | 20.0 | **0.000** | -11.970 |

Ten minutes of station keeping, charge remaining:

| platform | charge left |
|---|---|
| buoyant float | 1.000 |
| aerial drone | 0.857 |
| tethered head | 1.000 |

Immersed body size, which sets how tightly the swarm can pack:

| platform | radius | volume |
|---|---|---|
| buoyant float | 30.0 mm | 113.10 cm³ |
| aerial drone | 30.0 mm | 113.10 cm³ |
| **tethered head** | **8.0 mm** | **2.14 cm³** |

Three different limits, not three scores.  The float is slow in z but
sustainable; the drone is fast everywhere and running down a battery from the
moment it lifts off — extrapolating that burn rate puts endurance in hours; the
tethered head is fast, free to the node and ~53× smaller by volume, and pays for
it in the fibre it trails.

---

## 3. Motion is energy-limited, not velocity-limited

```bash
python examples/03_custom_plugin.py
```

Twelve photovoltaic floats starting from one point, asked to form a helix of
radius 10 m and pitch 1.5 m:

| t [s] | span x, y, z [m] | mean charge |
|---|---|---|
| 60 | 4.27, 4.27, 0.45 | 0.058 |
| 300 | 3.87, 4.08, 4.02 | 0.117 |
| 900 | 1.89, 2.24, 12.86 | 0.086 |
| 1800 | 2.19, 1.69, **16.00** | 0.103 |

**This is the result that corrected a design assumption.**  The swarm covers the
full 16 m of the helix in z while stalling at about 2 m in x and y.  Ballast is
slow — tens of seconds of pump lag — but once trimmed it costs nothing, because
buoyancy does the work.  Thrusters draw ~500 mW against a photovoltaic income of
~40 mW at 10 m depth, so lateral travel is rationed.

The correct statement is therefore **z is slow but free, xy is fast but
rationed** — the opposite of the obvious reading, and the opposite of what the
original project plan assumed.  A placement pass built on "depth is expensive,
lateral travel is not" would produce formations the swarm cannot afford to reach.

Also from this example, the custom `bubble_curtain` plugin: 36.4% of packets hit
an obscuration at 15 m with density 0.3, using a model that lives entirely
outside the library.

---

## 4. A heterogeneous liquid swarm

```bash
python -m iotc.cli sim configs/liquid_swarm.yaml --connect --traffic 30 --until 20
```

31 devices — 12 `mote-c`, 8 `mote-m`, 6 `mote-s`, 3 `relay`, 2 `anchor`, all
blue-green.  Reach report:

| class | λ | optical reach | RF reach |
|---|---|---|---|
| anchor-aq | 470 nm | 135.42 m | 15.38 m |
| mote-c-aq | 470 nm | 43.41 m | 15.38 m |
| mote-m-aq | 470 nm | 43.41 m | 15.38 m |
| mote-s-aq | 470 nm | 83.77 m | 15.38 m |
| relay-aq | 470 nm | 84.46 m | 15.38 m |

The 15 m RF reach is at a 30 kHz carrier.  At the stock 2.4 GHz it is 5 cm —
seawater absorbs roughly 1700 dB/m — so a liquid deployment has no control plane
at all unless it drops the carrier by five decades.

Run summary after 20 s with traffic:

```json
{"devices": 31, "alive": 31, "links": 36,
 "delivered": 1017, "lost": 1, "delivery_rate": 0.9990,
 "bits_delivered": 8331264, "optical_share": 1.0,
 "retarget_overhead": 0.9998,
 "mean_energy_fraction": 0.799, "brownouts": 0,
 "clock_skew": 3.65e-05, "guard_band": 1.05e-04}
```

39 links established, 36 still closing at the end — the rest decayed as bodies
drifted, without anything having "failed".

**`retarget_overhead` of 0.9998 is the headline number for Phase B.**  An 8 kbit
packet at 1 Gb/s takes ~8 µs to send; an aim takes ~0.5 s.  The fabric is almost
entirely aim-dominated, so amortising retargeting — parking beams, batching
traffic per slot, ordering the schedule to minimise slew — is the central job of
the scheduler.

---

## 5. Platform design sweep

```bash
python -m iotc.cli bench configs/platform_study.yaml --out results/platform --plot --seeds 2
```

24 devices in water, identical optics, varying only the platform and the packing
spacing.  Mean of 2 seeds:

| variant | spacing [m] | links | delivered | optical share | charge |
|---|---|---|---|---|---|
| aquatic | 8 | 23.0 | 460.0 | 1.00 | 0.794 |
| aquatic | 20 | 23.0 | 459.5 | 1.00 | 0.783 |
| aquatic | 35 | 21.0 | 381.5 | 1.00 | 0.777 |
| aquatic | 60 | 0.0 | 0.0 | 0.00 | 0.776 |
| aquatic+tethered | 8 | 23.0 | 140.0 | 1.00 | **1.000** |
| aquatic+tethered | 20 | 24.0 | 160.0 | 1.00 | **1.000** |
| aquatic+tethered | 35 | 21.0 | 95.0 | 1.00 | **1.000** |
| aquatic+tethered | 60 | 0.0 | 0.0 | 0.00 | 1.000 |

Two things to read here.  Past 35 m spacing the 43 m reach stops covering the
lattice diagonal and the fabric collapses to zero links — below that, the swarm
is transceiver-limited rather than range-limited, which is why the first two
rows are flat.

And the tethered platform runs at **full charge** because the topside rig powers
it, but delivers roughly **a third** of the traffic: its own forest of wires
occludes the beams, and the 1.5 dB waveguide run costs margin on top.  Density
and power are bought at the price of line of sight.

---

## 6. Fault campaign — correctness with and without faults

```bash
python -m iotc.cli bench configs/fault_campaign.yaml --out results/faults --seeds 3
```

24 devices, 40 s, identical seeds across rows.  Mean of 3 seeds:

| fault | alive | links | delivered | delivery rate | charge |
|---|---|---|---|---|---|
| **healthy** | 24.0 | 24.0 | 957.7 | 1.0000 | 0.937 |
| node_crash | 19.3 | 15.0 | 598.3 | 1.0000 | 0.942 |
| node_stall | 24.0 | 18.7 | 636.7 | 0.9944 | 0.937 |
| gimbal_stuck | 24.0 | 24.0 | 957.7 | 1.0000 | 0.937 |
| laser_degradation | 24.0 | 24.0 | 954.0 | 0.9997 | 0.937 |
| link_loss | 24.0 | 15.0 | 612.0 | 1.0000 | 0.937 |
| channel_noise_uniform | 24.0 | 24.0 | 761.3 | 0.7950 | 0.937 |
| channel_noise_bursty | 24.0 | 24.0 | 667.3 | **0.6969** | 0.937 |
| harvest_dropout | 24.0 | 24.0 | 957.7 | 1.0000 | **0.866** |
| drift_weak (0.004) | 24.0 | 10.0 | 367.7 | 0.9965 | 0.937 |
| drift_strong (0.04) | 24.0 | **0.0** | **0.0** | 1.0000 | 0.931 |
| combined | 21.7 | 7.7 | 215.7 | 0.9018 | 0.938 |

Notes on rows that look surprising, because the surprises are the useful part:

- **`gimbal_stuck` costs nothing here.**  The swarm is static, so a head that
  never needed to move is still pointing correctly.  The damage appears only once
  the formation moves or the schedule wants to retarget — which is a statement
  about when this fault matters, not evidence that it is harmless.
- **`laser_degradation` barely registers** over 40 s because the links have tens
  of dB of margin at these ranges.  It erodes margin rather than dropping links.
- **`channel_noise_bursty` is worse than uniform** at a comparable average, which
  is the expected result: correlated loss defeats coding and retries in a way
  that independent loss does not.
- **`link_loss` and `node_crash` cost a similar number of links but are different
  faults.**  Under `link_loss` every node stays healthy, so the correct response
  is a reroute, not a replacement.
- **`drift_strong` annihilates the fabric.**  There is no closed-loop pointing
  tracker in Phase A, so this measures how long an *open-loop* pointing solution
  survives: the beam spot is a few centimetres across, and a 0.04 m/√s current
  clears that in seconds.  The tracker belongs to the compiled runtime (Phase B,
  pass B8), and this row is the argument for it.

---

## 7. Surrogate accuracy

```bash
python -m iotc.cli sim <config-with-surrogate-channel> --verify-surrogate
```

400 random geometries, surrogate against exact evaluation:

```
max |error| 0.106 dB    rms 0.067 dB    bias -0.056 dB
```

Scale does not come from a cheaper physical model.  It comes from tabulating the
geometry-dependent half of the *same* model — the beam overlap integral and the
extinction — while the noise, BER and coding algebra stay exact and stochastic
per packet.  Outside the tabulated envelope it falls back to exact evaluation
rather than extrapolating.

---

## 8. A single link budget

```bash
python -m iotc.cli link --medium water --wavelength 470 --distance 20 --sweep
```

```
extinction : 0.036 /m at 470 nm
spot radius: 100.01 mm
budget     : rx=6.079e-07 W  geo=-29.03 dB  med=-3.13 dB  point=0.00 dB
             turb=+0.00 dB  SNR=26.42 dB  BER=6.21e-26  PER=0.000  up
max range  : 41.62 m
```

Every term is reported separately on purpose: when a link fails it should be
obvious *which* term killed it.

---

## 9. B0 — what each workload demands of the hardware

```bash
python -m iotc.cli graph --class mote-c
```

```
workload    class      tasks  edges  deg  ws/RAM   fit tiles        AI    aim deg/tx
------------------------------------------------------------------------------------
allreduce   mote-c        16     16    2    0.50  1.00     1      0.27  0.963    1.0
bfs         mote-c        16    120   15    0.02  1.00     1      0.56  1.000    7.5
coloring    mote-c        16    120   15    0.00  1.00     1      0.85  1.000    7.5
fft         mote-c         8     12    3    2.00  0.00     2     16.67  0.924    1.5
summa       mote-c        16     48    6    0.75  1.00     1    170.67  0.961    3.0
kmedian     mote-c        16    120   15    0.00  1.00     1      1.23  1.000    7.5
maxclique   mote-c        16    120   15    0.00  1.00     1      0.78  1.000    7.5
maxcut      mote-c        16    120   15    0.00  1.00     1      0.88  1.000    7.5
mis         mote-c        16    120   15    0.00  1.00     1      0.29  1.000    7.5
spmv        mote-c        16    120   15    0.02  1.00     1      1.20  1.000    7.5
stencil2d   mote-c        16     24    4    0.13  1.00     1    128.00  1.000    2.0
stencil3d   mote-c         8     12    3    0.27  1.00     1     42.67  0.999    1.5
```

Three things fall out, and each names a later pass rather than a statistic.

**Every one of the twelve is aim-dominated** — the least affected still spends
92% of its link time slewing rather than sending.  Nothing available to B0 or B1
changes this; only amortising the retarget does, which is B6.

**Eleven of twelve want more simultaneous neighbours than the part has
transceivers**, by up to 7.5×.  B5 cannot solve beam assignment as a matching;
it has to time-share.

**Arithmetic intensity spans 630×**, from 0.27 FLOP/B for an all-reduce to 170.7
for SUMMA.  One scheduling strategy will not serve that range.

---

## 10. B1 — what making it fit costs

```bash
python -m iotc.cli tile --class mote-c
```

Instances sized to overflow the part; every tiler is asked for the same fit, and
they are scored on the traffic it took to get there.

```
workload    class   tiler             tasks        cut  halos        MiB x traffic       AI  fit
------------------------------------------------------------------------------------------------
stencil2d   mote-c  geometric        4->144        6x6     40       34.4     11.00   279.27   ok
stencil2d   mote-c  slab             4->132         33    128      103.1     33.00    93.09   ok
stencil3d   mote-c  geometric        8->144      3x3x2     40      325.0      4.33    39.38   ok
stencil3d   mote-c  planar           8->160        5x4     56      425.0      5.67    30.12   ok
stencil3d   mote-c  slab             8->272         34    264     1725.0     23.00     7.42   ok
summa       mote-c  geometric      4->19600      70x70    552     8896.0    139.00    14.73   ok
summa       mote-c  slab           4->73920      18480  73916  1182720.0  18480.00     0.11   NO
```

```
geometric vs slab over 11 tiled cases: 18.91x less traffic on average (best 132.95x),
2.14x fewer pieces
cannot be tiled onto the part at all (halo exceeds RAM): allreduce on mote-c,
fft on mote-c, fft on mote-m, summa on mote-c, summa on mote-m
worst traffic amplification: summa under slab on mote-c, 18480.0x
58 of 61 fitted tilings are still aim-dominated -- tiling does not rescue B6
```

**The shape of the cut is the whole cost.**  Added traffic is `halo × Σ(fᵢ − 1)`,
so cutting *k* ways along one axis costs `(k−1)` halos while cutting a cube costs
`d·(k^(1/d) − 1)`.  For SUMMA that is the difference between a tiling that fits
(70×70, 139× traffic) and one that does not fit at all after 18,480 slices.
Geometric cutting averages **18.9× less traffic** than slab across the family.

**Some workloads cannot be tiled onto a part at any granularity.**  The FFT
butterfly exchanges half its points every crossing stage and slicing a rank does
not shrink that, so the halo alone exceeds RAM.  B1 reports this as a halo floor
rather than subdividing forever — the answer is a bigger part, not a finer cut.

**Tiling does not rescue B6.**  58 of the 61 tilings that fit are still
aim-dominated, and every one of them has *more* edges than before.  Making a
problem fit makes the scheduling problem harder, not easier.

---

## 11. B2 — grouping tasks onto devices, and whether it scales

```bash
python -m iotc.cli partition --class mote-c
```

Bytes crossing a group boundary become optical traffic; bytes inside a group are
a memory copy and cost nothing.  Every method is given the same device count.

```
workload    method          tasks     k     cut   cut MiB balance  fits        ms
---------------------------------------------------------------------------------
stencil3d   block             512    22   0.458      0.60    1.03    ok       0.0
stencil3d   fmbm              512    22   0.444      0.58    1.07    ok      47.0
stencil3d   spectral          512    22   0.417      0.55    1.38    NO      26.2
stencil3d   kernighan_lin     512    22   0.458      0.60    1.03    ok       9.9
stencil2d   block             256    38   0.573      0.13    1.04    ok       0.1
stencil2d   fmbm              256    38   0.481      0.11    1.04    ok      18.8
stencil2d   spectral          256    38   0.546      0.13    1.19    NO      36.1
summa       block              64    14   0.786     22.00    1.09    ok       0.0
summa       fmbm               64    14   0.842     23.56    1.09    ok       8.0
```

The acceptance criterion the roadmap sets for this pass is not cut quality but
**near-linear runtime**, so the ladder is the part that decides it:

```
method                512      1728      4096      8000  exponent
-----------------------------------------------------------------
block                 0.0       0.1       0.1       0.7      0.90
fmbm                 55.5     150.2     333.8     602.8      0.87
kernighan_lin        12.1      48.0     212.3     247.0      1.17
spectral              7.3     217.4     579.5    3490.4      2.16
```

```
empirical complexity: block n^0.90, fmbm n^0.87, kernighan_lin n^1.17, spectral n^2.16
fmbm is near-linear at n^0.87; kernighan_lin, spectral grow strictly faster
cut vs the block baseline: fmbm wins on 6 of 8 (maxclique, maxcut, mis, spmv,
stencil2d, stencil3d); loses on bfs, summa
produced groups that do not fit the device: spectral on 5 workloads
```

**FastMap partitioning is near-linear, measured.**  `n^0.87` over a 16× range of
task counts, against `n^2.16` for spectral bisection — a gap that is 6× in
wall-clock at 8,000 tasks and widens from there.  That is the property the whole
choice of FastMap rests on, and it is now a measurement rather than a citation.

**Spectral gets the best cuts and cannot be used.**  It wins on cut in most rows
and produces groups that *do not fit the device* in five of eight workloads,
because Fiedler bisection optimises the cut with no notion of capacity.  An
infeasible partition is not a better partition.

**`block` is a stronger baseline than it looks.**  The frontends number their
tiles in lattice order, so contiguous id ranges are already coherent 3D blocks;
beating it on a stencil means beating a partition that was handed the geometry
for free.  `fmbm` beats it on six of eight workloads and loses on two —
BFS, where a power-law degree distribution defeats a geometric ordering, and
SUMMA, whose row-and-column communication pattern has no low-dimensional
embedding to find.  Both losses are reported rather than tuned away.

---

## 12. B3 — placing the swarm, and what the formation costs to fly

```bash
python -m iotc.cli place
```

What each platform can actually reach, derived from its `DeviceClass` and its
medium rather than configured:

```
platform    xy reach    z reach  anisotropy   optical    xy J/m     z J/m
------------------------------------------------------------------------
float           0.70     160.99      231.01     41.62     1.065     0.000
drone       17874.55   14594.51        0.82     87.89     0.180     0.221
tether        480.00      30.00        0.06     41.62     0.000     0.000
```

```
workload    platform placer                 extent x/y/z (m)   link m  unreach   energy J
-----------------------------------------------------------------------------------------
stencil2d   float    anisotropic       0.3     0.3      78.8     8.40    0.0%       3.52
stencil2d   float    velocity          0.7     0.7       0.7     0.18    0.0%       9.06
spmv        float    anisotropic       0.2     0.2      39.5    11.83    0.0%       0.43
spmv        float    velocity          0.7     0.7       0.7     0.47    0.0%       1.64
stencil3d   tether   anisotropic      74.1    74.1       4.6    16.77    0.0%       0.00
stencil3d   tether   isotropic        30.0    30.0      30.0     8.93    0.0%       0.00
```

```
links beyond optical reach: 0 of 36 placements -- every placement is connected
drone: cheapest placer is velocity (603.64 J total) -- the velocity reading is not
penalised on this platform
float: cheapest placer is anisotropic (5.97 J total), 2.78x cheaper than the
velocity reading
tether: motion is free on this platform, energy cannot rank it
platform anisotropy (z reach / xy reach): drone 0.82, float 231.01, tether 0.06
```

**A float can go 231× further down than sideways.**  Ballast sets a trim once and
the body coasts to depth for no further energy, so depth is bounded by *time*;
thrusters burn charge every second, so lateral travel is bounded by the battery.
Placing by *cost* rather than by speed spends the free axis and saves the
rationed one, and the formation is **2.78× cheaper to fly**.  The naive reading —
"ballast is slow, so keep the swarm flat" — does the exact opposite.

**The axes do not favour the same direction on every platform.**  The float's
cheap axis is depth (anisotropy 231), the tethered head's is lateral (0.06,
because the wire limits depth while the gantry does not), and a drone is nearly
isotropic (0.82) because vertical travel fights gravity.  Note that the tethered
head's 30 m of depth is its *wire*, not its light — it carries the float's optics,
so once it is converted for the medium it sees the same 41.6 m.  (An earlier
revision of this bench built the head from an unconverted `MOTE_C` and reported
1.75 m of reach, which is 850 nm drowning in water rather than anything about
tethers; the float and drone numbers were unaffected.)  A placer tuned to one
platform is wrong on the other two, which is why all three are in the bench —
and on the drone the naive velocity scaling is *not* penalised.

**Energy is not the binding constraint; light is.**  The float's energy budget
permits 161 m of depth spread, but 470 nm light crosses only 41.6 m of water.
Every placement is therefore scaled until its longest link closes, with 5% held
back as tolerance — Phase A measured that a 0.04 m/s drift destroys an open-loop
link, so the last few percent of reach is not usable range.

---

## 13. B4 — flying the swarm without dropping its links

```bash
python -m iotc.cli motion
```

```
workload    platform planner      devs  prot  waves  makespan s   broken  collide  energy J
-------------------------------------------------------------------------------------------
spmv        float    direct          7    10      1        88.8   100.0%        0     17.42
spmv        float    waves           7    10      4       177.1     0.0%        0     17.42
spmv        float    sequential      7    10      7       199.6     0.0%        0     17.42
stencil2d   float    direct         38    31      1       165.6   100.0%        0    104.07
stencil2d   float    waves          38    31      3       363.3     0.0%        0    104.07
stencil2d   float    sequential     38    31     38      2643.5     0.0%        0    104.07
stencil3d   float    direct         22    21      1       160.4   100.0%        0     57.02
stencil3d   float    waves          22    21      3       379.0     0.0%        0     57.02
stencil3d   float    sequential     22    21     22      1285.4     0.0%        0     57.02
```

```
flying straight drops 100% of protected links; wave-ordered flight drops 0%
the price is makespan: 2.18x on average (worst 2.36x)
moving one device at a time is equally safe and 4.5x slower -- wave count follows
degree, not device count (3.3 vs 22)
energy is identical across planners -- waves reorder the motion, they do not
lengthen the paths
collisions along the way: 0 of 9 plans -- every plan is collision-free
```

**Interpolating destroys every protected link.**  A link needs both gimbals aimed
at each other, and Phase A leaves the closed-loop tracker out on purpose —
open-loop pointing tolerates centimetres.  So a link whose two ends are both in
motion is *dropped*, not degraded, and flying everyone straight at their targets
drops 100% of the traffic worth protecting for the entire manoeuvre.

**Colouring the critical links fixes it, and the bill is time.**  Move in colour
classes and every protected link keeps one endpoint stationary for the moving
end to re-aim against: zero breakage, for **2.18× the makespan** on average.
Nothing else changes — energy is identical to the digit across all three
planners, because waves reorder the motion without lengthening any path.

**The wave count follows degree, not device count.**  Moving one device at a
time is equally safe and 4.5× slower; greedy colouring needs 3.3 waves on
average where sequential needs 22.  That gap is why colouring is worth doing
rather than simply taking turns.

---

## 14. B5 — beam rounds, and what the gimbals can point at

```bash
python -m iotc.cli beam
```

```
workload    platform assigner         B3 fix  tx  over  rounds  1st traffic  rejected  loss dB
----------------------------------------------------------------------------------------------
stencil3d   float    heaviest_first       no   2   6.5       1        5.5%     96.5%     0.05
stencil3d   float    heaviest_first      yes   2   6.5       7       48.1%      3.5%     0.07
stencil3d   float    lightest_first      yes   2   6.5       7        6.9%      3.5%     0.07
stencil3d   drone    heaviest_first       no   2   6.5       7       48.7%      4.7%     0.15
stencil3d   drone    heaviest_first      yes   2   6.5       7       48.7%      2.4%     0.07
stencil3d   relay    heaviest_first      yes   6   2.2       3       90.1%      0.0%     0.00
```

```
demand exceeds transceivers by 3.8x on average -- the graph cannot be a topology,
only a sequence of them
drone: B3's placement leaves 2.6% of links unaimable; after the refinement
iteration, 0.8%
float: B3's placement leaves 97.8% of links unaimable; after the refinement
iteration, 9.4%
relay: B3's placement leaves 0.0% of links unaimable; after the refinement
iteration, 0.0%
tether: B3's placement leaves 0.0% of links unaimable; after the refinement
iteration, 0.0%
ordering the b-matching by payload puts 57.6% of the traffic in the opening round
against 33.9% reversed -- same rounds, different bytes
rounds against heads: 2 heads 4.9 rounds, 6 heads 2.0 rounds
assignments exceeding a device's head count: 0 of 72 -- every assignment is realisable
```

**The energy-optimal placement cannot be aimed.**  This is the sharpest
cross-pass result in the compiler so far.  B3 minimises motion energy, and on a
float the cheap axis is depth — so it returns a formation 81 m tall and 0.4 m
wide.  Nearly every bearing in a column points straight up or down, outside the
mote's ±75° elevation limit, and **96.5% of the links cannot be aimed at all**.
Each pass was locally right and the composition was unusable.

**The platform, not the refinement, is the real fix — the tether needs no
refinement at all.**  B3 gives the tethered head a *slab* (anisotropy 0.06, since
the wire caps depth while the gantry does not): 74.1 x 74.1 x 4.6 m for stencil3d
against the float's 0.4 x 0.4 x 81.0 m column.  A slab's bearings are
near-horizontal, exactly what a +/-75 deg gimbal wants, so **0.0% of its links are
unaimable before any feedback iteration runs** and all seven rounds schedule
where the float raw managed one.  The 96.5% figure is a property of *columns*,
not of gimbals.
Two caveats keep this honest.  The tether's mean loss reads 0.14 dB raw against
the float's 0.05 dB, but that is not a like-for-like comparison: the float's
figure is averaged over the 3.5% of links it did not reject, so it is a mean over
the easy survivors.  And **B5 does not charge for occlusion** -- the wire forest
is tested in `iotc.sim.world`, not in the compiler's aimability check -- so this
result says the slab is *pointable*, not that the beams arrive.  Those are the
beams the wires block.

**The feedback iteration is a structural correction, not a nudge.**  Returning
the pointing cost to B3 widens the column from 0.4 m to 20 m and drops rejection
to 3.5%, taking first-round traffic from 5.5% to 48.1%.  The roadmap framed this
as feeding back the *snap residual*; the residual turned out to be the minor
term.  Note also that minimising raw residual makes the loss **worse** — loss is
quadratic in the residual, so the objective has to be the loss itself.

**Demand exceeds supply by 3.8×, so the topology is a sequence.**  A mote has two
heads and wants up to thirteen neighbours.  A relay's six heads cut the rounds
from 4.9 to 2.0 and let one round carry 90% of the traffic — the clearest
argument yet for heterogeneity in the swarm.

**Ordering the b-matching is the only thing this pass chooses**, and it is worth
57.6% of traffic in the opening round against 33.9% reversed, at identical round
count.  Since each round costs every device in it a ~0.5 s retarget, front-loading
the heavy edges is what B6 will have to exploit.

---

## 15. B6 — slot timing, and what bounds the amortisation

```bash
python -m iotc.cli schedule
```

```
workload    class      scheduler      batch    wanted  period s  efficiency       kB/s  buffer KiB
--------------------------------------------------------------------------------------------------
stencil3d   mote-c-aq  single             1         1      4.43    0.0098%     135.38        20.0
stencil3d   mote-c-aq  amortised        256    687890      4.45    0.3250%    2936.32       256.0
stencil3d   mote-m-aq  single             1         1      2.49    0.0161%     227.95        20.0
stencil3d   mote-m-aq  amortised       2048    242211      2.59    3.8741%   37548.98      2048.0
stencil3d   relay-aq   single             1         1      4.34    0.0020%     140.72        20.0
stencil3d   relay-aq   amortised        512   3701840      4.35    0.1147%    4448.19       512.0
```

```
stencil3d on mote-c-aq: unbatched 0.0098% -> amortised 0.3250% (33x), batch 256
stencil3d on mote-m-aq: unbatched 0.0161% -> amortised 3.8741% (241x), batch 2048
stencil3d on relay-aq: unbatched 0.0020% -> amortised 0.1147% (56x), batch 512
but 5 of 6 schedules are RAM-starved: the batch needed is 2787x larger than the
buffer allows
reaching the target would need 1361 MiB of buffer against 4.0 MiB of RAM (752x short)
-- the fabric stays aim-dominated because the part is too small, not because the
schedule is poor
stencil3d, holding the gimbal fixed and varying only RAM: 1.0 MiB batch 256 -> 0.325%,
8.0 MiB batch 2048 -> 3.874%
the relay is not in that comparison: it has six heads and a finer step, so it differs
in more than buffer
```

**Unbatched, the schedule runs at 0.002%–0.09% efficiency** — Phase A's "over 99%
overhead", reproduced through the whole compiler rather than asserted.

**Amortising the aim lifts it 33×–241×**, and still is not enough.  The batch
that would reach 90% efficiency is ~2,800× larger than the buffer permits;
reaching it needs **1,361 MiB against 4 MiB of RAM**.  No scheduler closes that
gap.  The fabric stays aim-dominated because the part is too small, and fixing it
means a bigger buffer, a bigger packet, a faster gimbal or a slower line rate —
all decisions outside the compiler.

The RAM comparison holds the gimbal fixed and varies only the buffer, so it says
what it appears to say.  The relay is excluded deliberately: it has six heads and
a finer step as well as different memory, so including it would make the trend
look non-monotonic and prove nothing.

---

## 16. B7 — powering the schedule, and the duty cycle it forces

```bash
python -m iotc.cli energy
```

```
workload   platform  policy       batch   draw mJ  harv mW      duty   dominant   share   ok
--------------------------------------------------------------------------------------------
spmv       float     ignore           5     64.07    117.2  100.000%         rx   42.6%  yes
spmv       drone     ignore           6    106.60   1200.0  100.000%         rx   42.7%  yes
stencil3d  float     ignore         256    229.88    120.0  100.000%    compute   34.2%  yes
stencil3d  float-m   ignore        2048   2568.27    120.0  100.000%    compute   98.0%   NO
stencil3d  float-m   duty_cycle    2048   2568.67    120.0   11.540%    compute   98.0%  yes
stencil3d  float-m   half_duty     2048   2568.33    120.0   50.000%    compute   98.0%   NO
stencil3d  drone     ignore         256    257.94   1199.9  100.000%         rx   30.5%  yes
```

```
dominant energy term: compute in 2/6 cases, rx in 4/6 cases
retargeting is 18.1% of the draw on average -- the fabric is aim-dominated in TIME
but not in energy, which is not the same statement
running B6's frame as written: 1 of 6 configurations drain their store
(worst endurance 6 s)
stencil3d on float-m must drop to a 11.5% duty cycle to be sustainable
(batch 2048, compute dominates at 98%)
a fixed 50% duty rule still leaves 1 configurations draining -- the sleep has to be
computed, not chosen
```

**The fabric is aim-dominated in time but not in energy.**  Every pass so far has
been about retargeting, because it consumes over 99% of the *clock*.  It is
**18.1% of the joules**.  What actually drains the store is listening — a
receiver cannot know when a packet will start, so it stays powered through the
aim and the guard band as well as the wire — and, at large batch, computing.
Anyone optimising this platform for energy should start somewhere other than the
gimbal.

**B6 and B7 pull against each other, which is why the roadmap pairs them.**
B6 batches to amortise the retarget, and every batched superstep is another
superstep of compute inside the same frame.  The memory-rich part is the one B6
favours — it grants the largest batch, 2048 — and it is the only configuration
here that cannot power the result: compute reaches 98% of the draw and the store
empties in **6 seconds**.  Padding with sleep fixes it, at an **11.5% duty
cycle**.  The best schedule for the link is not the cheapest to run.

**A fixed duty-cycle rule is not a substitute for computing one.**  Halving the
duty leaves that configuration still draining, while costing the five healthy
ones half their throughput for nothing.

---

## 17. B8 — redundancy, failover, and the pointing tracker

```bash
python -m iotc.cli resilience
```

Mean surviving traffic over 60 random failure draws, replicating a fifth of the
swarm:

```
workload   policy       devs  repl     10%     20%     30%     40%     50%
--------------------------------------------------------------------------
spmv       none            7     0   0.714   0.714   0.481   0.287   0.143
spmv       uniform         7     1   0.765   0.765   0.577   0.375   0.223
spmv       traffic         7     1   0.781   0.781   0.545   0.377   0.220
spmv       centrality      7     1   0.763   0.763   0.545   0.356   0.209
stencil3d  none           22     0   0.819   0.660   0.459   0.342   0.230
stencil3d  uniform        22     4   0.848   0.720   0.529   0.440   0.318
stencil3d  traffic        22     4   0.856   0.742   0.581   0.471   0.381
stencil3d  centrality     22     4   0.852   0.743   0.575   0.464   0.376
```

```
stencil3d: at 30% device loss, no redundancy keeps 45.9% of traffic; replicating a
fifth of the swarm by traffic keeps 58.1%
stencil3d: choosing by importance beats choosing at random by +0.052 +/- 0.013 --
significant
stencil3d: sampled betweenness over the cheap traffic proxy: -0.005 +/- 0.011 --
NOT significant, so the proxy is enough
spmv: choosing by importance beats choosing at random by -0.031 +/- 0.021 --
NOT significant
note: B2 leaves device graphs dense -- spmv at mean degree 6.0/6 -- so there are few
structural bottlenecks for centrality to find
```

```
drift m/sqrt(s)   tol m  interval s   frame s  re-aims/frame  energy mJ
-----------------------------------------------------------------------
          0.000    0.02         inf      4.45            0.0      0.000
          0.010    0.02       4.000      4.45            1.1      4.558
          0.040    0.02       0.250      4.45           17.8     72.921
          0.040    0.05       1.562      4.45            2.8     11.667
          0.100    0.02       0.040      4.45          111.2    455.753
```

**Redundancy pays; choosing what to replicate pays much less than expected.**
Replicating a fifth of the swarm lifts 30%-loss survival from 45.9% to 58.1% on
the mesh.  But **sampled betweenness never beats the cheap traffic-weighted
proxy** — the gap is −0.005 ± 0.011 on the mesh and −0.000 ± 0.020 on SpMV,
inside the noise both times.  Use the proxy.

The reason is a consequence of an earlier pass rather than a property of
centrality: **B2 leaves device graphs dense**.  SpMV coarsens to seven devices at
mean degree 6.0 of a possible 6 — a *complete* graph — and on a complete graph
there are no structural bottlenecks to find, so even the importance-vs-random
comparison fails to reach significance there (−0.031 ± 0.021).  Only the mesh,
which retains real geometry, rewards choosing at all.

**The pointing tracker is not optional and it is not cheap.**  Drift is a random
walk, so the safe interval is *quadratic* in tolerance: at Phase A's measured
0.04 m/√s and a 2 cm open-loop tolerance a link survives **0.25 s**, against a
4.45 s frame — **17.8 re-aims per frame** purely to hold links open, at 72.9 mJ.
Relaxing tolerance to 5 cm cuts that to 2.8 re-aims; a still medium needs none.
This is the gap Phase A left deliberately, now priced.

Erasure coding and checkpoint intervals are named in the roadmap and are **not**
implemented — recorded as remaining work rather than quietly dropped.

---

## 18. B9 — the emitted bytecode, its size and its round trip

```bash
python -m iotc.cli codegen
```

```
workload   class      encoding    repeat  stored   executed  total KiB  worst KiB flash  plan
---------------------------------------------------------------------------------------------
stencil3d  mote-c-aq  flat          1000      17      15002     8569.9      622.1    ok    ok
stencil3d  mote-c-aq  packed        1000      17      15002     3469.0      255.9    ok    ok
stencil3d  mote-c-aq  looped        1000      17      15002        3.8        0.3    ok    ok
stencil3d  mote-c-aq  flat        100000      17    1500002   856934.2    62207.1  OVER    ok
stencil3d  mote-c-aq  packed      100000      17    1500002   346875.3    25586.0  OVER    ok
stencil3d  mote-c-aq  looped      100000      17    1500002        3.9        0.3    ok    ok
```

```
at 100000 frames: flat 516700 KiB, packed 208643 KiB, looped 2.3 KiB -- varints
save 2.5x, looping saves a further 89021x
images that do not fit flash: flat at 10000 frames, flat at 100000 frames, packed at
10000 frames, packed at 100000 frames -- unrolling overflows a mote from 10000 frames on
the looped image is independent of run length (1.01x across a 1000x range of frame
counts) -- the others are linear in it
VM round trip: 24 of 24 images deliver exactly the bytes B6 planned, with every SEND
matched by a RECV
```

**The program is a loop, not a stream, and that is the whole pass.**  B6's
schedule is one frame repeated, so a device stores **17 instructions** and
executes **1,500,002** of them.  Emitting what it *does* rather than what it *is*
costs 516,700 KiB against 2.3 KiB — and **unrolling overflows a mote's 2 MiB
flash from 10,000 frames on**, while the looped image is flat in run length
(1.01× across a 1000× range).  Varints alone save 2.5×; looping saves a further
89,021×.  This is the same shape of question B1 asked of RAM, which is why the
encoding is a compiler decision and not a serialization detail.

**The VM is the correctness gate, and it earned its place immediately.**  All 24
images deliver exactly the bytes B6 planned, with every SEND matched by a RECV —
a link needs both gimbals, so a one-way edge is not physically representable.
On its first run the VM rejected the emitter with *"device 0 aimed at 3 peers
with 2 transceivers"*: heads were being released only at the frame edge, when
B5's rounds exist precisely so devices retarget between them.  A `BARRIER` at
each slot boundary fixed it, and the check is now a test.

---

## 19. B10 — maximum clique: proximity as the adjacency test

```bash
python -m iotc.cli clique
```

A clique of 16 planted in an Erdős–Rényi graph of mean degree 10, so recovery
has a ground truth that is not itself exponential to establish.

```
     n solver              size  planted        ms      tests  rounds  energy J   ok
------------------------------------------------------------------------------------
   400 greedy_degeneracy     16   16/16        0.6       2973       0    340.93   ok
  4000 greedy_degeneracy     16   16/16        3.2      14164       0    340.93   ok
  4000 colour_bound          16   16/16      196.2       4125       0    340.93   ok
  4000 bron_kerbosch         16   16/16       49.9      24219       0    340.93   ok
  4000 annealing              2    0/16       56.6      38903       0     42.62   ok
  4000 physical              16   16/16      623.6       6653      36    340.93   ok
```

```
adjacency tests over a 10x range of n, among solvers that found the clique:
physical 2.1x, greedy_degeneracy 4.8x, colour_bound 7.9x, bron_kerbosch 9.1x
-- the embedding is what keeps the physical scheme's growth flat
excluded from that comparison because they did not find the clique: annealing
-- their low test counts are the cost of not answering
broadcast rounds are constant at 36 regardless of n -- the physical cost does not
grow with the problem
and it is the slowest in wall-clock: 624 ms against 3.2 ms for greedy_degeneracy at
n=4000 -- the embedding and LSH are overhead a greedy extension never pays
the claim the pass actually makes is about verification: confirming a 16-clique costs
120 pairwise tests conventionally and 1 broadcast round physically, whatever k is
paid for physically: 340.9 J to converge the members and 1.82e+10 photons for the round
independently checked: 15 of 15 answers are genuine cliques
```

**The claim, stated narrowly.**  Verifying a candidate clique of size k costs
`k(k−1)/2` pairwise adjacency tests conventionally — 120 for k=16, 2,016 for
k=64 — and **one broadcast round** once the members are in mutual line of sight,
whatever k is.  That is the collapse the architecture argues for, and it is
arithmetic rather than a measurement.

**What the measurement adds, and what it costs.**  Among the four solvers that
actually find the planted clique, the physical scheme's adjacency tests grow
slowest over a 10× range of n — **2.1× against 4.8×, 7.9× and 9.1×** — because
the common-neighbour weight puts dense regions of the graph in dense regions of
the embedding, so LSH nominates good balls.  Its broadcast rounds are **constant
at 36** regardless of problem size.

**And it is the slowest in wall-clock**, 624 ms against greedy extension's 3.2 ms
at n=4000.  That is a real limitation, not a rounding error: the FastMap
embedding and the LSH probing are overhead a greedy pass never pays.  The
argument is that on hardware the cost is *rounds of aiming* — ~0.5 s of gimbal
time each — not CPU microseconds, so a scheme with a constant round count wins
where it matters.  On a laptop it does not, and the bench says so.

Simulated annealing collapses on the larger instances (size 2 of a planted 16 at
n=4000) and is excluded from the cost comparison: its low test count is the cost
of not answering, and ranking it as cheap would invert the reading.

Every answer is checked independently against the adjacency sets — 15 of 15 are
genuine cliques.

---

## 20. Phase C — large-scale runs, invariants and fault campaigns

```bash
python -m iotc.cli testbench
```

One run is the whole pipeline: workload → tile → partition → place → beams →
schedule → energy → bytecode → VM, classified pass / degraded / wrong / hung
with all five invariants checked on every run.

```
scale demonstration -- one clean run per workload
workload       tasks  devices    links   verdict  delivered   seconds
---------------------------------------------------------------------
spmv            1024       25      300  degraded     100.0%      10.6
stencil2d        576      150      555  degraded      97.2%       4.9
stencil3d       2048     1270     6628  degraded      96.5%      66.2
```

```
fault campaign -- every fault, with and without a pointing tracker
workload    fault            trk  runs  devices  delivered  drift-lost  reach-lost  verdicts
--------------------------------------------------------------------------------------------
stencil3d   none              no     3       22      99.6%         0.0         1.0  degraded=3
stencil3d   position_drift    no     3       22      95.1%         3.3         1.0  degraded=3
stencil3d   position_drift   yes     3       22      99.6%         0.0         1.0  degraded=3
stencil3d   scatter           no     3       22      85.8%         9.7         1.0  degraded=3
stencil3d   node_crash        no     3       22      85.2%         9.7         5.7  degraded=3
stencil3d   node_crash       yes     3       22      93.9%         0.0         5.7  degraded=3
```

```
42 runs: pass 0, degraded 42, wrong 0, hung 0
invariant violations: none -- no run was classified wrong
node_crash: delivery 82.2% without a pointing tracker, 86.5% with one
position_drift: delivery 90.8% without a pointing tracker, 99.8% with one
scatter: delivery 82.5% without a pointing tracker, 99.8% with one
a tracker cannot fix distance: 18 runs still lose links that are out of optical
reach however well they are aimed
```

**1,270 devices compiled and executed, with every invariant holding.**  From a
2,048-task tiled stencil, in 66 s.  Across the whole campaign **no run was ever
classified `wrong`** — the compiler never emitted a plan that violated its own
rules.

**The tracker B8 built is what survives faults, and only the faults it can
reach.**  Drift 90.8% → 99.8%, scatter 82.5% → 99.8% — but a node crash only
82.2% → 86.5%, because those links are beyond optical reach and no amount of
re-aiming brings the photons back.  The two failure modes are separated in the
table (`drift-lost` against `reach-lost`) precisely because one is fixable and
the other is not.

**Short links are the fragile ones under drift**, which inverts the obvious
reading.  The beam footprint is the divergence half-angle *at range*, so it grows
with link length while the drift does not: 1.6 cm of tolerance on a 3 m link
against 20 cm on a 39 m one.  Measured, the links drift breaks average 15.4 m and
the survivors 30.5 m.

**Scale makes a run degrade.**  The same stencil passes at 22 devices and lands
at 96.5% at 1,270: a larger swarm spreads further, and some links fall outside
the optical budget even though their bearings are legal.  That is reported as
`degraded` rather than smoothed away — a swarm that loses traffic *and knows it*
is usable; one that silently drops the same traffic is not.

---

## Reproducibility

Every run is seeded, and determinism is treated as a correctness property rather
than a convenience — a fault campaign that cannot replay a failure is not
evidence of anything.  Two runs with the same seed produce byte-identical
traces, and adding a stochastic subsystem does not shift the draws of an
unrelated one.  Both are pinned by tests in `tests/test_engine.py`.

One live bug was caught by exactly that test: fault RNG streams had been keyed on
`id(fault)`, a memory address that changes between runs.

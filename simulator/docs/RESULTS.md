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

## Reproducibility

Every run is seeded, and determinism is treated as a correctness property rather
than a convenience — a fault campaign that cannot replay a failure is not
evidence of anything.  Two runs with the same seed produce byte-identical
traces, and adding a stochastic subsystem does not shift the draws of an
unrelated one.  Both are pinned by tests in `tests/test_engine.py`.

One live bug was caught by exactly that test: fault RNG streams had been keyed on
`id(fault)`, a memory address that changes between runs.

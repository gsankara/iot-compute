# iot-compute

A **mobility-driven IoT computation substrate**: thousands of small,
energy-harvesting devices that compute by *interaction timing and coordination*
rather than by instruction streams or gate netlists.  It sits between gate-level
computation on an FPGA and general-purpose computation in a multi-tier service
deployment.

The devices are linked by Gbps free-space photonics on **quantized-step** gimbals
plus a slow omnidirectional radio, and they **move** — as drones in air, as
ballast-driven floats in a liquid, or as remote heads on wires positioned from
outside the medium.  Because they move, the interconnect topology is not a given:
it is something a compiler *produces* and the swarm then physically realizes.
That is the thesis — mobility buys dynamic proximity.

```
problem graph ──FastMap──▶ 3D placement ──▶ motion plan ──▶ beam assignment
                                                                  │
                                                            slot schedule
                                                                  │
   per-device bytecode ◀── energy & resilience passes ◀───────────┘
            │
            ▼
   simulator: medium ▸ beam ▸ attenuation ▸ turbulence ▸ pointing ▸ receiver
              ▸ modulation ▸ coding ▸ clocks ▸ energy ▸ faults
```

## Status

**Phase A (simulator) is complete and tested.**  Phase B is under way: **B0**
(the compiler IR and its twelve workload frontends, `iotc graph`), **B1**
(tiling to fit a target part, `iotc tile`), **B2** (grouping tasks onto devices,
`iotc partition`), **B3** (projecting onto physical coordinates, `iotc place`),
**B4** (flying there without dropping links, `iotc motion`), **B5** (beam rounds,
`iotc beam`) and **B6** (slot scheduling, `iotc schedule`) are implemented, tested
and benched.  The remaining compiler
passes and the large-scale testbench follow; see `docs/ROADMAP.md`.

## Install

```bash
pip install -e ".[dev]"
```

Only `numpy` is required.  `scipy` and `networkx` are used by the compiler
passes, `matplotlib` by the figures, and `pyyaml` by the config loader.

## 60-second tour

How far can these things talk, and in what?

```bash
iotc link --medium water --wavelength 470 --distance 20 --sweep
```

Run a heterogeneous swarm of ballast-driven floats:

```bash
iotc sim configs/liquid_swarm.yaml --connect --traffic 30 --until 20
```

Compare the platforms as a table rather than an argument:

```bash
iotc bench configs/platform_study.yaml --out results/platform --plot
```

Correctness with and without faults:

```bash
iotc bench configs/fault_campaign.yaml --out results/faults
```

Ask what a problem demands of the hardware before compiling it:

```bash
iotc graph stencil3d maxclique --class mote-c
```

Split it to fit a part, and see what the shape of the cut costs:

```bash
iotc tile stencil3d gemm --class mote-c
```

Group the tasks onto devices, and check the partitioner actually scales:

```bash
iotc partition --class mote-c
```

Put the swarm in space, and see what the formation costs to fly:

```bash
iotc place --platform float
```

Fly it there without dropping the links on the way:

```bash
iotc motion stencil3d
```

Assign the beams, and find out what the gimbals can actually point at:

```bash
iotc beam stencil3d
```

In Python:

```python
import numpy as np
from iotc.devices import MOTE_C, aquatic
from iotc.sim import Simulation, World

world = World(medium="water", topology="gradient", seed=0)
for p in np.random.default_rng(0).normal(scale=5, size=(24, 3)) - [0, 0, 10]:
    world.add_device(aquatic(MOTE_C), p)

sim = Simulation(world, dt=0.05)
sim.establish(0, 1)          # aim both heads; links are mutual
sim.run(2.0)
print(world.link_budget(0, 1).report())
print(sim.send(0, 1, 8192))
```

## What the simulator already tells you

These are outputs, not design intentions — each one falls out of the physics and
is pinned by a test.

| Finding | Where |
|---|---|
| 850 nm cannot cross 2 m of water; 470 nm reaches 42 m. Any liquid deployment must move to blue-green. | `examples/01_reach_vs_medium.py` |
| Seawater absorbs ~1700 dB/m at 2.4 GHz. A liquid deployment has no control radio unless it drops to tens of kHz. | `iotc.devices.aquatic` |
| An ambient-light drone cannot hover — it browns out and falls. Aerial platforms need a battery and have a finite endurance; floats and tethered heads do not. | `test_mobility.py` |
| A naive PD controller cannot fly a ballast actuator with 25 s of lag; it limit-cycles. Feedforward on terminal velocity plus lead compensation does. | `BuoyantKinematics` |
| For a PV-powered float, **z is slow but free and xy is fast but rationed** — the opposite of the obvious reading. | `examples/03_custom_plugin.py` |
| A tethered head is ~50× smaller than a float, moves 4× faster in z at zero node energy — and its own wire forest blocks the beams, cutting delivered traffic by ~3×. | `configs/platform_study.yaml` |
| With 8 kbit packets at 1 Gb/s, transmission takes microseconds and aiming takes ~0.5 s. The fabric is aim-dominated; the scheduler must amortise retargeting. | `iotc sim --traffic` |
| Without a re-aiming loop, a 0.04 m/√s current destroys every link. Open-loop pointing has a tolerance of centimetres. | `configs/fault_campaign.yaml` |

## Layout

```
iotc/
  units.py registry.py geom.py rng.py config.py cli.py
  devices/      device capability profiles and the stock class library
  sim/
    medium.py   vacuum, air (with fog), water, engineered index fluid
    links/      the seven-stage link budget, RF, and a memoizing surrogate
    mobility/   kinematics (drone, buoyant, tethered) and topology dynamics
    energy.py clocks.py world.py engine.py faults.py trace.py
  compiler/
    ir.py       the TaskGraph every pass consumes                    (B0)
    frontends/  twelve stock workloads and their graph generators    (B0)
    tile.py     split tasks to fit a part; the cut's shape is its cost (B1)
    fastmap.py  graph FastMap: embed into R^kappa, near-linear         (B2/B3)
    partition.py  group tasks onto devices, minimising the cut          (B2)
    place.py    project onto 3D; the cheap axis is not the fast one    (B3)
    motion.py   fly there in waves so links keep one end still         (B4)
    beam.py     b-matching onto transceivers; what a gimbal can aim at  (B5)
    schedule.py slot timing; amortise the aim until RAM stops you      (B6)
  bench/        design-space sweeps, fault campaigns, per-pass evaluation
configs/  examples/  tests/  docs/
```

## Two pluggable axes

Transmission physics and topology dynamics are both extensible without touching
simulator code — subclass, decorate, and the model is selectable by name from a
config file.  `examples/03_custom_plugin.py` adds one of each in 40 lines.  See
`docs/PLUGINS.md`.

```python
from iotc.registry import register
from iotc.sim.links.base import Turbulence

@register("turbulence", "bubble_curtain")
class BubbleCurtain(Turbulence):
    def sample_gain(self, geo, medium, rng): ...
```

High fidelity is the **default**.  Scale comes from `ChannelSurrogate`, which
memoizes the geometry-dependent half of the *same* model and keeps the noise
algebra exact — and `--verify-surrogate` measures what the interpolation costs
rather than assuming it is free.

## Docs

- `docs/RESULTS.md` — **example results**: every table above, verbatim, with the
  command that produces it
- `docs/ARCHITECTURE.md` — the engine, the plugin slots, and how data flows
- `docs/PHYSICS.md` — every equation, its assumptions and its validity range
- `docs/PLUGINS.md` — adding a channel model or a topology law
- `docs/ROADMAP.md` — what Phases B and C add

## Reproducing the results

```bash
bash scripts/regenerate_results.sh
```

About two minutes; writes CSV, JSON, PNG and console transcripts into `results/`,
which is gitignored.  The curated versions are committed in `docs/RESULTS.md`.

## Tests

```bash
python -m pytest -q
```

429 tests.  They are written as invariants rather than golden values — monotonic
link budgets, unit-mean fading, energy conservation, quantization bounds, seeded
determinism, five-point and seven-point stencil neighbourhoods, surface-to-volume
scaling, FLOP and external-traffic conservation under tiling, embedded distance
tracking graph distance — so a new plugin, frontend, tiler or partitioner has to
satisfy them too.

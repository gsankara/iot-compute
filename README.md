# iot-compute

A **mobility-driven IoT computation substrate**: thousands of small,
energy-harvesting devices that compute by *interaction timing and coordination*
rather than by instruction streams or gate netlists.  It sits between gate-level
computation on an FPGA and general-purpose computation in a multi-tier service
deployment.

The devices are linked by Gbps free-space photonics on **quantized-step** gimbals
plus a slow omnidirectional radio, and they **move** — as drones in air, as
ballast-driven floats in a liquid, or as remote optical heads on wires positioned
from outside the medium.  Because they move, the interconnect topology is not a
given: it is something a compiler *produces* and the swarm then physically
realizes.  That is the thesis — **mobility buys dynamic proximity**.

## Repository layout

| Path | Contents |
|---|---|
| [`simulator/`](simulator/) | Phase A — the simulator, its testbench and evaluation harness. Complete. |
| [`simulator/iotc/compiler/`](simulator/iotc/compiler/) | Phase B — problem graph to per-device bytecode, pass by pass. B0 (IR and frontends) and B1 (tiling) complete. |
| `testbench/` | Phase C — large-scale emulation and fault campaigns. Not started. |

Start at [`simulator/README.md`](simulator/README.md).

## Status

**Phase A is complete** and **Phase B is under way**: 291 tests passing.
Sequencing follows the project plan — simulator first, then the compiler built
pass by pass with each pass shipping its own testbench and evaluation, then the
large-scale testbench.  B0 (the compiler IR and its twelve workload frontends)
and B1 (tiling onto a target part) are done and benched.  See
[`simulator/docs/ROADMAP.md`](simulator/docs/ROADMAP.md).

## What the simulator already establishes

Measured outputs, not design intentions — every one is reproduced in
[`simulator/docs/RESULTS.md`](simulator/docs/RESULTS.md) and pinned by a test.

- 850 nm cannot cross 2 m of water; 470 nm reaches 42 m.  Any liquid deployment
  must move to blue-green.
- Seawater absorbs ~1700 dB/m at 2.4 GHz.  A liquid deployment has **no** control
  radio unless it drops the carrier to tens of kilohertz.
- An ambient-light drone **cannot hover** — it browns out and falls.  Aerial
  platforms need a battery and have finite endurance; floats and tethered heads
  hold station indefinitely.
- For a photovoltaic float, **z is slow but free and xy is fast but rationed** —
  the opposite of the obvious reading, and a correction to the original plan.
- A tethered remote head is ~53× smaller by volume than a float and changes depth
  4× faster at zero node energy — and its own forest of wires occludes the beams,
  cutting delivered traffic by about a third.
- With 8 kbit packets, sending takes microseconds and aiming takes ~0.5 s.  The
  fabric is **aim-dominated**; amortising retargeting is the central problem for
  the scheduler.
- Without a closed-loop pointing tracker, a 0.04 m/√s current destroys every
  link.  Open-loop pointing has a tolerance of centimetres.

## What the compiler establishes so far

- Every one of the twelve stock workloads is **aim-dominated** on the small part
  — the least affected still spends 92% of its link time slewing.  No tiling
  choice changes that; only the scheduler can.
- **How you cut a task is worth more than how finely you cut it.**  Added traffic
  is `halo × Σ(fᵢ − 1)`, so geometric cutting moves 18.9× less traffic than
  slicing on average and 133× in the best case, for the same fit.
- Some workloads **cannot be tiled onto a part at any granularity**: an FFT
  butterfly's exchange alone exceeds the RAM, so the answer is a bigger part
  rather than a finer cut.

## Quick start

```bash
cd simulator
pip install -e ".[dev]"
python -m pytest -q
python -m iotc.cli link --medium water --wavelength 470 --distance 20 --sweep
python -m iotc.cli sim configs/liquid_swarm.yaml --connect --traffic 30 --until 20
```

## Provenance

The compiler's placement pass is built on graph FastMap (Li & Kumar, *Revisiting
FastMap: New Applications*, arXiv:2503.11908), which embeds a graph into ℝ^κ in
near-linear time preserving shortest-path distance.  The NP-hard workload family
is the standard one — maximum clique, maximum independent set, graph colouring,
capacitated k-median and max-cut.

## Licence

BSD-3-Clause.  See [LICENSE](LICENSE).

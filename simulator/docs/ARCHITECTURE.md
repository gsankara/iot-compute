# Architecture

## The shape of the problem

Three things make this substrate different from a wireless sensor network, and
each one dictates a piece of the design.

1. **Links are directional, mutual and quantized.**  A link exists only when
   *both* gimbals are aimed at each other, the path is clear, and the budget
   closes.  Neither end can open one unilaterally, and neither can point exactly
   — there is an irreducible pointing loss set by the gimbal tick.  So a link is
   a jointly scheduled resource, not a property of a pair of nodes.

2. **The topology is an output, not an input.**  The devices move, so
   communication locality is *produced* by flying the right nodes next to each
   other.  Placement is therefore a compiler pass, and the simulator has to
   charge honestly for the travel it implies.

3. **Coordination is timing.**  Two nodes exchange data because both were told
   to point at each other during slot 17, not because either asked.  That only
   works if they agree on when slot 17 is, so clock drift, resync and guard
   bands are modelled explicitly rather than assumed away.

## Layers

```
    iotc.bench      sweeps, fault campaigns, aggregation, figures
        │
    iotc.compiler   tile ▸ partition ▸ place ▸ motion ▸ beam ▸ schedule    (Phase B)
        │           ▸ energy ▸ resilience ▸ codegen
        │
    iotc.sim        world, engine, links, mobility, energy, clocks, faults
        │
    iotc.devices    capability profiles: compute, gimbal, optics, radio, body
        │
    foundation      units, registry, geom, rng, config, cli
```

Each layer depends only downward.  `devices` is pure data with no behaviour, so
the simulator and the compiler physically cannot disagree about what a node can
do.

## The engine

Hybrid, because neither clock alone works.  Motion, energy, harvesting and
clock drift are continuous and integrate on a fixed outer tick (typically 20–50
ms).  Packets are discrete and go through an event queue.  A single clock would
either integrate the physics at nanosecond resolution or quantise a 1 Gb/s
transfer onto a 10 ms grid.

```
Simulation.step():
    world.step(dt)            # topology targets → kinematics → energy → clocks → faults
    drain events ≤ now        # packet deliveries, scheduled callbacks
```

**Determinism is a correctness requirement**, not a convenience: a fault campaign
that cannot replay a failure is not evidence of anything.  Every random draw
comes from a named substream (`iotc.rng.RandomSource`), and the event queue
breaks ties on a monotonic sequence number.  Two runs with the same seed produce
identical traces, and adding a stochastic subsystem does not shift the draws of
an unrelated one.  One live bug was caught by exactly this test: fault streams
were keyed on `id(fault)`, a memory address that changes between runs.

## Plugin slots

Resolved through `iotc.registry`.  A `kind` is an abstraction slot; a `name` is a
specific model.

| Kind | Ships with | Axis |
|---|---|---|
| `medium` | vacuum, air, water, index_fluid | physics |
| `beam` | gaussian, flat_top, bessel | physics |
| `attenuation` | beer_lambert, mie | physics |
| `turbulence` | none, lognormal, gamma_gamma, kolmogorov_ts | physics |
| `pointing` | deterministic, beckmann | physics |
| `receiver` | pin, apd, spad | physics |
| `modulation` | ook, pam4, ppm | physics |
| `coding` | none, rs, fountain | physics |
| `channel` | fso, rf_omni, surrogate | physics |
| `kinematics` | static, drone, buoyant, tethered | topology |
| `topology` | static, gradient, flock, gravity, annealed, formation_seq | topology |
| `harvester` | photovoltaic, rf_rectenna, constant, none | platform |
| `store` | supercap, battery | platform |
| `fault` | 9 models, node and link | platform |

## Data flow for one link evaluation

```
World.link_budget(i, j)
  ├ both endpoints operational?              → else DOWN
  ├ pair blacklisted by a link fault?        → else DOWN
  ├ line of sight (bodies + tether capsules) → else DOWN
  ├ World.geometry(i, j)
  │    distance, wavelength, tx power (× waveguide loss on a tethered head),
  │    tx misalignment  = live bearing vs where head i is actually pointing
  │    rx misalignment  = live bearing vs where head j is actually pointing
  └ channel.evaluate(geo, medium, rng)
       pointing → beam capture → attenuation → turbulence → coupling
       → rx power → receiver SNR → modulation BER → coding PER
```

Misalignment is recomputed from **live positions** every time, never cached.
That is deliberate: it means a link decays as the bodies drift, without anything
having "failed" — which is the dominant failure mode in a mobile fabric and the
thing an open-loop pointing solution is worst at.

## World state

`Device` holds motion, gimbals, energy, clock and health.  `World` holds the
devices, the medium, both channels, the topology law and the clock domain.

Health is split the way hardware actually fails, because conflating the cases
hides the recovery paths:

- `crashed` — permanent fail-stop
- `stalled_until` — transient hang; the node wakes up believing it is still in
  the slot it went to sleep in
- `laser_derate` — gradual power loss; margin erodes, nothing drops
- `blacklist` — a specific pair can never talk again; both nodes stay healthy,
  so the correct response is a reroute
- `Transceiver.stuck` — a frozen head keeps firing at its *last* angle, which is
  worse than a dead laser because the sender believes it is transmitting
- store lockout — brown-out, latched with hysteresis

The position held in `Device` is also the problem's state, not merely the node's
location; `STATE.md` sets out what that memory costs to write, to read and to
keep.

## Scaling

High fidelity is the default.  `ChannelSurrogate` memoizes the geometry-dependent
half of the *same* model — the beam integral and the extinction — over
`(distance, offset angle)`, and keeps the noise and coding algebra exact and
stochastic per packet.  It falls back to exact evaluation outside its tabulated
envelope, and `verify()` measures the interpolation error in dB rather than
assuming it away.

The current cost is roughly 1–3 s of wall time for a 24–31 device swarm over 25 s
of simulated time with traffic.  Phase C pushes this to 1k–10k devices.

## Configuration

A config is a plain mapping — YAML on disk or a dict in a script — and every
plugin slot resolves through the registry, so a user-written model is selectable
by name without touching library code.

`variant` composes: `[aquatic, tethered]` is a remote head that can actually see,
because `aquatic` fixes the *optics and radio* for a liquid while `tethered`
fixes the *platform*.  Writing `tethered` alone leaves a near-infrared
transceiver under water, where it cannot cross two metres.  A sweep that
overwrites one variant with the other silently loses the optics — which is a
mistake this codebase made once and now documents.

## Reading a run

`iotc sim ... --check` prints the reach report first, and it is worth reading
before anything else: if optical reach is 2 m and the layout spacing is 7 m,
nothing will ever link and every later number is a confusing zero.

`retarget_overhead` deserves attention.  With 8 kbit packets at 1 Gb/s, a
transmission takes microseconds and an aim takes about half a second, so the
fabric is overwhelmingly aim-dominated.  Amortising that — parking beams,
batching traffic per slot, ordering the schedule to minimise slew — is the
central job of the Phase B scheduler.

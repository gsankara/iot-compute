# Writing a plugin

Two axes are meant to be extended: **transmission physics** and **topology
dynamics**.  Neither requires editing anything inside `iotc`.  Subclass the base,
decorate it, and the model is selectable by name from a config file.

`examples/03_custom_plugin.py` adds one of each and runs them.

## The pattern

```python
from iotc.registry import register
from iotc.sim.links.base import Turbulence

@register("turbulence", "bubble_curtain", "bubbles")   # extra names are aliases
class BubbleCurtain(Turbulence):
    def __init__(self, density: float = 0.05, opacity: float = 0.02) -> None:
        self.density = float(density)
        self.opacity = float(opacity)

    def sample_gain(self, geo, medium, rng) -> float:
        p = -math.expm1(-self.density * geo.distance / 10.0)
        return self.opacity if rng.random() < p else 1.0
```

Then, in a config:

```yaml
channel:
  type: fso
  turbulence: {type: bubble_curtain, density: 0.3}
```

Constructor keyword arguments come straight from the mapping, so a plugin's
signature *is* its config schema.

Import the module that defines it before the config is parsed — putting it in
your experiment script is enough, since `@register` runs at import time.

## Rules

- **Take SI, return SI.**  Metres, seconds, watts, radians.
- **Draw randomness only from the `rng` you are handed.**  It is a named
  substream; reaching for `np.random` directly destroys reproducibility and the
  determinism test will catch it.
- **Do not keep state that outlives a run**, or provide a `reset()`.
  `KolmogorovTimeSeries` keeps per-link state keyed by `(i, j)` and is the model
  to copy if you need memory.
- **Say where your model stops being valid**, in the docstring, and clamp rather
  than extrapolate. `LogNormalFading` clamps at Rytov variance 1 for this reason.

---

## Transmission physics

Each stage is independent, so you can replace one without knowing about the
others.

| Base class | Method | Returns |
|---|---|---|
| `Beam` | `spot_radius(geo)`, `captured_fraction(geo, offset)` | m, fraction |
| `Attenuation` | `transmittance(geo, medium)` | fraction |
| `Turbulence` | `sample_gain(geo, medium, rng)` | scaling, **mean 1** |
| `Pointing` | `sample_offset(geo, rng, jitter_sigma)` | metres at the receiver |
| `Receiver` | `snr(rx_power, geo)` | electrical power ratio |
| `Modulation` | `ber(snr)`, `bits_per_symbol` | probability |
| `Coding` | `per(ber, payload_bits)`, `code_rate` | probability |

`geo` is a `LinkGeometry`: distance, wavelength, transmit power, divergence,
apertures, field of view, both misalignments, bandwidth.

A turbulence model **must have unit mean** — fading redistributes power, it does
not create it.  There is a test for this; if yours fails it, the link budget will
be biased.

To memoize an expensive stage, wrap the channel in `ChannelSurrogate` rather than
caching inside the stage.  It tabulates the geometry-dependent term and keeps the
noise algebra exact, and `verify()` will tell you what the interpolation costs.

### A whole channel

Implement `evaluate(geo, medium, rng, *, jitter_sigma, payload_bits, line_rate,
stochastic, key, now) -> LinkBudget` and `max_range(...)`.  Honour
`stochastic=False` by substituting means — the compiler's cost model and the
surrogate both need a repeatable number rather than one draw.

Fill in every field of `LinkBudget`, not just the verdict.  When a link fails it
should be obvious *which* term killed it.

---

## Topology dynamics

One method:

```python
from iotc.sim.mobility.topology import TopologyContext, TopologyDynamics

@register("topology", "helix")
class HelixTopology(TopologyDynamics):
    def targets(self, ctx: TopologyContext) -> np.ndarray:
        ...   # (N, 3) desired positions
```

`ctx` gives positions, velocities, time, compiler goals, an affinity matrix,
current links, liveness, radii, bounds and an rng.  Anything not listed is
deliberately out of reach.

Two things every law should do, because an embedding algorithm does not have to
worry about them and a swarm does:

- add `_separation_push(...)` so bodies cannot coincide (it handles the exactly
  coincident case deterministically);
- respect `ctx.bounds` via `_clip_to_bounds(...)`.

Convergence is bounded by terminal velocity and by the **energy budget**, not by
iteration count.  For a photovoltaic float those are very different limits: see
`examples/03_custom_plugin.py`, where the swarm covers 16 m in z but stalls at
2 m in xy because the thrusters cost more than the cell harvests.

## Kinematics

```python
def step(self, state, target, dt, medium, body, power, tether=None) -> MotionCost
```

Mutate `state` in place; return the energy spent and whether actuators ran.  Set
the class attribute `slow_z = True` if depth is rate-limited rather than
force-limited.

Handle `state.frozen` honestly — it means no power or no actuation.  A frozen
drone must fall; a neutrally trimmed float must simply stop.  The difference is a
real argument about platform choice, and burying it would throw that away.

## Other slots

`harvester` (`power(spec, position, medium, now, env)`), `store`
(subclass `EnergyStore`), and `fault` (subclass `FaultModel`, implement
`step(world, now, dt, rng)`) follow the same pattern.

A fault model should re-check `self._armed(now)` inside its per-device loop, not
just at the top — otherwise a single tick fires on every candidate before the
event cap is consulted again.

## Checking your plugin

```bash
python -m pytest -q                  # the invariants your model must not break
iotc plugins                         # confirm it registered
iotc sim your_config.yaml --check    # build and print the reach report
```

The suite includes cross-cutting properties — monotonic link budgets, unit-mean
fading, energy conservation, quantization bounds, seeded determinism — that a new
model has to satisfy too.

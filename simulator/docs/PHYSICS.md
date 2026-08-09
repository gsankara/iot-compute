# Physics

Every model, its assumptions, and where it stops being valid.  SI throughout:
metres, seconds, kilograms, watts, joules, radians.

A model outside its validity range is worse than no model, because it still
returns a number.  Each section therefore ends with the regime in which it
should not be trusted.

---

## 1. Media

`iotc/sim/medium.py`.  The medium couples the two halves of the design problem:
it sets how far light carries *and* how the bodies move.

### Optical extinction

Beer–Lambert, `T = exp(-k(λ) · d)`, with `k` interpolated in log space between
tabulated points — absorption spans four decades across the visible and near
infrared, so linear interpolation would be badly wrong mid-decade.

Pure-water absorption `k` [1/m]:

| λ [nm] | 400 | 470 | 520 | 550 | 650 | 700 | 850 | 1310 | 1550 |
|---|---|---|---|---|---|---|---|---|---|
| k | 0.0065 | 0.016 | 0.045 | 0.064 | 0.34 | 0.65 | 4.4 | 130 | 1100 |

`turbidity` adds a wavelength-flat scattering term (0.02 clear ocean, ~0.4
coastal).  **This is the single most consequential table in the package:** it is
why a liquid deployment must use blue-green optics, and the reason
`devices.aquatic` exists.

Clear-sky molecular extinction in air is 1–4 ×10⁻⁵ /m.  Aerosols use Kim's
empirical model:

```
β [1/km] = (3.912 / V) · (λ / 550 nm)^(-q)
q = 1.6            V > 50 km
    1.3            6 < V ≤ 50
    0.16 V + 0.34  1 < V ≤ 6
    V - 0.5        0.5 < V ≤ 1
    0              V ≤ 0.5
```

*Not valid for:* wavelengths outside 400–1550 nm (the tables clamp); water
containing chlorophyll or sediment beyond a flat scattering term; horizontal
paths through a strong thermocline.

### RF absorption

Good-conductor skin-depth limit, `α = √(π f μ₀ σ)` Np/m.  At σ = 4 S/m
(seawater) this gives ~1700 dB/m at 2.4 GHz and ~3.4 dB/m at 10 kHz.  Air has
σ = 0 and therefore no medium absorption.

*Not valid for:* fresh water at GHz frequencies where displacement current is not
negligible; any case where σ/(ωε) is not ≫ 1.

### Refractive index

Air: constant 1.000273.  Water: two-term Cauchy, `n = 1.3199 + 6.878e-3 / λ_µm²`,
adequate over 400–800 nm.  Sets link latency via `c/n`.

---

## 2. The optical link budget

`iotc/sim/links/`.  Seven independently swappable stages:

```
beam → attenuation → turbulence → pointing → aperture → receiver → modulation → coding
```

### Beam and capture

Gaussian spot radius at range: `w(d) = √(w₀² + (M² θ d)²)` with `w₀` the
transmit aperture radius and `θ` the 1/e² half-angle divergence.

On-axis captured fraction is exact:

```
A₀ = 1 - exp(-2 a² / w²)
```

Off axis, the exact Bessel integral is replaced by the Farid–Hranilovic
equivalent-width Gaussian:

```
P(r) ≈ A₀ · exp(-2 r² / w_eq²)
v = √(π/2) · a / w
w_eq² = w² · √π · erf(v) / (2 v exp(-v²))
```

*Not valid for:* `w < a`, where the aperture over-fills and the approximation
loses accuracy.  Every link in this system operates with `w > a`, and the
on-axis case is computed exactly regardless.

`flat_top` uses the exact two-disc intersection area; `bessel` models a
quasi-nondiffracting beam as a Gaussian with suppressed divergence over a stated
range, plus a fixed power penalty for the energy in the outer rings.

### Turbulence

Plane-wave Rytov variance:

```
σ_R² = 1.23 · Cn² · k^(7/6) · L^(11/6)
```

* `lognormal` — `I = exp(2X)`, `X ~ N(-σ_x², σ_x²)`, `σ_x² = σ_R²/4`.
  **Valid only for σ_R² ≲ 1**; the model clamps rather than extrapolating.
* `gamma_gamma` — product of two gamma variates with the standard α, β from
  σ_R²; mean 1 by construction.  Covers moderate to strong turbulence.
* `kolmogorov_ts` — log-normal amplitude driven through a first-order AR process
  with a coherence time.  Matters because independent per-packet draws overstate
  how well retries and interleaving work; real fades arrive in bursts.

All three have unit mean: fading redistributes power, it does not create it.

### Pointing

The gimbal quantizes, so there is an irreducible bias no transmit power removes.
`AngularGrid.snap` returns the true great-circle residual, which is *not* simply
half a step — near the poles an azimuth tick subtends less angle.  The worst case
is bounded by `½√(Δaz² + Δel²)`.

`beckmann` adds independent Gaussian jitter on both axes, making the radial error
Rician.  Keeping bias and jitter separate matters: the compiler can trade the
bias against gimbal cost, but not the jitter.

A link additionally requires **mutual** pointing and an unobstructed path.
Receive coupling falls off as `exp(-2 (θ_rx/θ_fov)²)` and is zero beyond the
field of view.

### Receivers

* `pin` — thermal-noise limited. `SNR = (R·P)² / (2q(R·P + I_d)B + 4kTF B/R_L)`
* `apd` — internal gain `M` with excess noise `F = k_eff M + (2 - 1/M)(1 - k_eff)`.
  Gain has an optimum: beyond it multiplied shot noise overtakes the thermal
  floor it was hiding.
* `spad` — photon counting with paralysable dead-time saturation. The right
  choice at femtowatts, which is where long hops in an absorbing medium land.

### Modulation and coding

`OOK`: `BER = Q(√SNR / 2)`.  `PAM4`: same Q divided by `(M-1)`, ×3/4.
`PPM(L)`: slot SNR boosted by `L`, orthogonal-signalling symbol-to-bit
conversion.  PPM beats OOK when photons are scarce; PAM4 wins only with spare
SNR.

`none` gives `PER = 1 - (1-BER)^n`.  `rs(n,k)` evaluates the block binomial
exactly for `t = (n-k)/2` correctable symbols.  `fountain` models a rateless
erasure code across a burst: it absorbs losses up to roughly its overhead and
then falls off a cliff — 25% redundancy cannot repair a 22% loss rate, and the
model does not pretend otherwise.

### The surrogate

`ChannelSurrogate` tabulates only the geometry-dependent term — the beam overlap
integral and the extinction — over `(distance, offset angle)`, and keeps the
noise, BER and coding algebra exact and stochastic per packet.  Outside the
tabulated envelope it falls back to the exact model rather than extrapolating.
`verify()` reports max and RMS error in dB against exact evaluation; the shipped
default holds under 0.5 dB.

---

## 3. RF

Friis with the free-space path loss **floored at 0 dB**.  Inside a wavelength the
formula goes negative and would hand out free gain — and that matters here,
because a subsea link runs at tens of kHz where the wavelength is kilometres and
every useful range is deep in the near field.

`Pr = Pt + 2G - FSPL - medium - shadowing - implementation`, log-normal
shadowing, thermal noise `kTB·NF`, coherent BPSK `BER = Q(√(2 SNR))`.

*Not valid for:* near-field magnetic coupling, which is how real subsea
low-frequency links actually work; this is a far-field model pressed into
near-field service and should be read as an optimistic bound.

---

## 4. Motion

`iotc/sim/mobility/kinematics.py`.

### Drag

Quadratic plus a Stokes term:

```
F = -(½ ρ Cd A |v| + 6π µ r) v
```

Both regimes are needed: a 6 cm float at 0.2 m/s sits at Re ~ 10⁴ where the
quadratic term dominates, but the slow start of a ballast manoeuvre passes
through the linear regime, where the quadratic law alone would let it accelerate
unphysically.

### Buoyancy

Neutral trim is `m_neutral = ρ V - m_body`.  Net force is
`ρ V g - (m_body + b) g`, and terminal velocity is `√(F/k)` with
`k = ½ ρ Cd A`.

**The ballast controller is feedforward, not PD, and that is not a detail.**
Full tank travel takes `capacity/rate` ≈ 25 s for the stock float.  A
proportional-derivative loop saturates the pump, arrives with the actuator still
hard over, and limit-cycles indefinitely.  Instead: choose a target velocity,
invert the steady-state force balance to get the ballast that sustains it, and
add a lead term `τ = 0.3 · lag` so the command unwinds while the body is still
approaching.

Every stock hull is checked to satisfy `0 < m_neutral < capacity`
(`test_devices.py`) — a tank that cannot reach neutral means the part can never
hold a depth, and the symptom otherwise surfaces three layers downstream as
"the links keep dropping".

### Tethered heads

Position is commanded from outside the medium, rate-limited by the gantry and
the winch, confined to the cylinder under the anchor.  The node pays no motion
energy.  The tether is a **capsule obstacle** for line of sight — clamped
segment-to-segment distance (Ericson) against each beam path — which is the real
cost of the architecture.

---

## 5. Energy

Photovoltaic power is `E(z) · A · η` with

```
E(z) = E_surface · exp(-k(550 nm) · depth)
```

using the *same* extinction coefficient as the optical links.  Descending
therefore costs link budget and income together.  Over the tens of metres a
swarm spans, air costs under 0.5% while clear water costs more than 95%.

Rectenna power is `EIRP/(4π d²) · A · η · L_pol`, attenuated by the medium — which
makes wireless charging useless under salt water at any range.

The store has an **undervoltage lockout with hysteresis**: it latches off at
`brownout_fraction` and will not restart until charge reaches
`recovery_multiple ×` that level.  Without the hysteresis a node that runs out
simply has its draws refused, the charge pins a hair above threshold, and it
reports itself alive forever while doing no work — which hides a whole class of
failure.

---

## 6. Clocks

Local time is `t(1 + drift) + offset`.  Drift is drawn **once per device** and
held, because a crystal's error is a manufacturing property, not noise.  Reads
add jitter.

A beacon resync corrects the offset but **not** the drift: the node learns what
time it is, not how fast its crystal runs, so the error starts growing again
immediately — which is what makes the resync interval a design choice rather
than a free parameter.

```
uncertainty(t) = |drift_ppm| · 1e-6 · (t - t_sync) + sync_error + 3·jitter
guard_band     = safety · 2 · max_uncertainty
```

The factor of two is because both ends can be wrong in opposite directions.
This is the number the slot scheduler pays, and a schedule whose slots are
shorter than the guard band they need is simply infeasible.

---

## 7. What is deliberately not modelled

Stating the gaps is part of the model.

- **Attitude.** Bodies are assumed to hold orientation, so gimbal angles are in
  the world frame.  A rolling float would add a whole control problem.
- **Multipath and scattered-light crosstalk.** `mie` recovers some forward
  scatter; there is no reflection off surfaces or off other bodies.
- **Acoustic communication**, which is what a real subsea deployment would
  actually use for its control plane.
- **Near-field magnetic coupling** for the low-frequency radio (see §3).
- **Thermal effects** on laser wavelength and detector dark current.
- **Tether mechanics** — the wire is a rigid vertical capsule with no catenary,
  no drag and no tension.
- **A closed-loop pointing tracker.** Pointing is open loop, which is why the
  drift fault is so destructive; the tracking loop belongs to the compiled
  runtime in Phase B.

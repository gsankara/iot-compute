"""Adding your own physics and your own topology law.

The two axes the brief asks to keep open are transmission physics and topology
dynamics.  Neither requires touching simulator code: subclass, decorate, and the
new model is selectable from a config file by name.

    python examples/03_custom_plugin.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (puts the checkout on sys.path)

import math

import numpy as np

from iotc.registry import create, register
from iotc.sim.links import LinkGeometry
from iotc.sim.links.base import Turbulence
from iotc.sim.mobility.topology import TopologyContext, TopologyDynamics


# --------------------------------------------------------------------------
# a new transmission model
# --------------------------------------------------------------------------
@register("turbulence", "bubble_curtain")
class BubbleCurtain(Turbulence):
    """Occasional total obscuration by a rising bubble.

    Not fading at all: an on/off process.  Worth having as a separate model
    because a channel that is perfect 95 percent of the time and absent for the
    rest defeats interleaving in a way that continuous scintillation does not.
    """

    def __init__(self, density: float = 0.05, opacity: float = 0.02) -> None:
        self.density = float(density)
        self.opacity = float(opacity)

    def sample_gain(self, geo: LinkGeometry, medium, rng) -> float:
        # More path length, more chances to intersect a bubble.
        p = -math.expm1(-self.density * geo.distance / 10.0)
        return self.opacity if rng.random() < p else 1.0


# --------------------------------------------------------------------------
# a new topology law
# --------------------------------------------------------------------------
@register("topology", "helix")
class HelixTopology(TopologyDynamics):
    """Park the swarm on a slowly rotating helix.

    A deliberately simple example: it shows that a topology plugin only has to
    answer "where should each device be right now", and everything else --
    kinematics, energy, link maintenance -- keeps working around it.
    """

    def __init__(self, radius: float = 12.0, pitch: float = 1.5,
                 rate: float = 0.05, centre=(0.0, 0.0, -10.0)) -> None:
        self.radius = float(radius)
        self.pitch = float(pitch)
        self.rate = float(rate)
        self.centre = np.asarray(centre, dtype=float)

    def targets(self, ctx: TopologyContext) -> np.ndarray:
        n = ctx.n
        if n == 0:
            return ctx.positions.copy()
        k = np.arange(n, dtype=float)
        theta = 2.0 * math.pi * k / max(n, 1) + self.rate * ctx.time
        out = np.stack([
            self.radius * np.cos(theta),
            self.radius * np.sin(theta),
            self.pitch * (k - (n - 1) / 2.0),
        ], axis=-1)
        return out + self.centre


# --------------------------------------------------------------------------
def main() -> None:
    from iotc.devices import MOTE_C, aquatic
    from iotc.sim import Simulation, World

    print("Both plugins are now selectable by name:\n")
    from iotc import registry

    print("  turbulence :", registry.available("turbulence"))
    print("  topology   :", registry.available("topology"))

    # --- use the custom channel model ---------------------------------
    water = create("medium", "water", turbidity=0.02)
    channel = create("channel", "fso", turbulence={"type": "bubble_curtain",
                                                   "density": 0.3})
    geo = LinkGeometry(
        distance=15.0, wavelength=470e-9, tx_power=1e-3, divergence=5e-3,
        tx_aperture_radius=1.5e-3, rx_aperture_radius=2.5e-3, rx_fov=15e-3,
    )
    rng = np.random.default_rng(0)
    outages = sum(
        0 if channel.evaluate(geo, water, rng, line_rate=1e9).feasible else 1
        for _ in range(2000)
    )
    print(f"\nbubble curtain: {outages / 2000:.1%} of packets hit an obscuration")

    # --- use the custom topology law ----------------------------------
    world = World(medium="water", topology={"type": "helix", "radius": 10.0},
                  seed=0)
    for _ in range(12):
        world.add_device(aquatic(MOTE_C), np.array([0.0, 0.0, -10.0]))
    sim = Simulation(world, dt=0.05)

    print("\nhelix topology, floats starting from one point:")
    print(f"  {'t [s]':>7s} {'span x,y,z [m]':>26s} {'charge':>8s}")
    for t in (60.0, 300.0, 900.0, 1800.0):
        sim.run(t)
        spread = world.positions.max(axis=0) - world.positions.min(axis=0)
        charge = float(np.mean([d.store.fraction for d in world.devices]))
        print(f"  {t:7.0f} {str(spread.round(2)):>26s} {charge:8.3f}")

    print(
        "\nThe swarm converges far more slowly than its thrusters could manage,\n"
        "and the charge column says why: a 10 cm^2 cell at 10 m depth harvests\n"
        "around 40 mW while a thruster draws 500, so the float cannot power\n"
        "continuous motion.  It moves in bursts and spends the rest of the time\n"
        "recharging.  That is a constraint the motion planner has to respect --\n"
        "travel time here is set by the energy budget, not by terminal velocity."
    )
    print("\nNeither model required editing anything inside iotc.")


if __name__ == "__main__":
    main()

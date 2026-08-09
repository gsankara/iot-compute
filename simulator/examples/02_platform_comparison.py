"""Float, fly, or hang off a wire?

The same 10 m depth change on the three platforms, measured rather than argued.
What comes out is that the three differ on *different* axes -- speed, energy,
and whether the thing can hold station at all -- so there is no single winner,
which is exactly why the platform is a plugin.

    python examples/02_platform_comparison.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (puts the checkout on sys.path)

import numpy as np

from iotc.devices import MOTE_C, aerial, aquatic, tethered
from iotc.registry import create
from iotc.sim import Simulation, World

PLATFORMS = [
    ("buoyant float", aquatic(MOTE_C), "water", -1.0),
    ("aerial drone", aerial(MOTE_C), "air", +1.0),
    ("tethered head", tethered(aquatic(MOTE_C)), "water", -1.0),
]


def travel(cls, medium: str, sign: float, span: float = 10.0, limit: float = 400.0):
    """Move 10 m along z, and report how long it took and what it cost."""
    world = World(medium=medium, topology="gradient", seed=0)
    start = np.array([0.0, 0.0, sign * 2.0])
    dev = world.add_device(cls, start)
    goal = np.array([0.0, 0.0, sign * (2.0 + span)])
    world.goals = goal[None, :]
    sim = Simulation(world, dt=0.02)

    t = 0.0
    while t < limit and abs(dev.position[2] - goal[2]) > 0.05:
        t += 0.5
        sim.run(t)
    return t, dev.power.totals["motion"], dev.store.fraction, dev.position[2]


def station_keep(cls, medium: str, sign: float, duration: float = 600.0):
    """Hold position for ten minutes and see what is left in the buffer."""
    world = World(medium=medium, topology="gradient", seed=0)
    z = sign * 5.0
    dev = world.add_device(cls, np.array([0.0, 0.0, z]))
    world.goals = np.array([[0.0, 0.0, z]])
    Simulation(world, dt=0.1).run(duration)
    return dev.store.fraction


def main() -> None:
    print("A 10 m change of depth or altitude\n")
    print(f"{'platform':16s} {'time [s]':>10s} {'node energy [J]':>17s} {'final z [m]':>12s}")
    print("-" * 58)
    for label, cls, medium, sign in PLATFORMS:
        t, energy, _, z = travel(cls, medium, sign)
        print(f"{label:16s} {t:10.1f} {energy:17.3f} {z:12.3f}")

    print("\nTen minutes of station keeping\n")
    print(f"{'platform':16s} {'charge left':>12s}")
    print("-" * 30)
    for label, cls, medium, sign in PLATFORMS:
        print(f"{label:16s} {station_keep(cls, medium, sign):12.3f}")

    print("\nImmersed body size, which sets how tightly the swarm can pack\n")
    for label, cls, _, _ in PLATFORMS:
        r = cls.body.radius
        print(f"  {label:16s} radius {r * 1e3:6.1f} mm   "
              f"volume {cls.body.volume * 1e6:9.2f} cm^3")

    print(
        "\nRead it as three different limits, not three scores.  The float is\n"
        "slow in z but sustainable; the drone is fast everywhere and running\n"
        "down a battery from the moment it lifts off; the tethered head is fast,\n"
        "free to the node and tiny -- and pays for it in the fibre it trails,\n"
        "which blocks other nodes' beams."
    )


if __name__ == "__main__":
    main()

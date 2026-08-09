"""How far can these devices talk, and in what?

The first question any deployment has to answer, and the one that decides the
wavelength.  Run it and the blue-green requirement for any liquid medium falls
straight out of the numbers.

    python examples/01_reach_vs_medium.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (puts the checkout on sys.path)

from iotc.registry import create
from iotc.sim.links import LinkGeometry

WAVELENGTHS = [(470e-9, "blue-green"), (550e-9, "green"),
               (850e-9, "near IR"), (1550e-9, "C band")]

MEDIA = [
    ("vacuum", create("medium", "vacuum")),
    ("air, clear", create("medium", "air", visibility_km=50.0)),
    ("air, fog", create("medium", "air", visibility_km=0.2)),
    ("water, clear", create("medium", "water", turbidity=0.02)),
    ("water, coastal", create("medium", "water", turbidity=0.4)),
    ("index fluid", create("medium", "index_fluid")),
]


def geometry(wavelength: float) -> LinkGeometry:
    """A stock mote-class transceiver: 1 mW, 5 mrad, 2.5 mm aperture."""
    return LinkGeometry(
        distance=1.0, wavelength=wavelength, tx_power=1e-3, divergence=5e-3,
        tx_aperture_radius=1.5e-3, rx_aperture_radius=2.5e-3, rx_fov=15e-3,
    )


def main() -> None:
    channel = create("channel", "fso", turbulence="none")

    header = f"{'medium':16s}" + "".join(f"{name:>13s}" for _, name in WAVELENGTHS)
    print("Maximum range [m] for a 1 mW, 1 Gb/s link\n")
    print(header)
    print("-" * len(header))
    for label, medium in MEDIA:
        cells = []
        for lam, _ in WAVELENGTHS:
            reach = channel.max_range(geometry(lam), medium, line_rate=1e9)
            cells.append(f"{reach:13.2f}")
        print(f"{label:16s}" + "".join(cells))

    print("\nExtinction coefficient [1/m]\n")
    print(header)
    print("-" * len(header))
    for label, medium in MEDIA:
        cells = [f"{medium.extinction(lam):13.4g}" for lam, _ in WAVELENGTHS]
        print(f"{label:16s}" + "".join(cells))

    print("\nRF absorption by the medium [dB over 10 m]\n")
    for label, medium in MEDIA:
        low = medium.rf_attenuation_db(30e3, 10.0)
        high = medium.rf_attenuation_db(2.4e9, 10.0)
        print(f"  {label:16s} 30 kHz {low:12.2f}    2.4 GHz {high:14.2f}")

    print(
        "\nThe near-infrared column is why a liquid deployment has to move to\n"
        "blue-green, and the 2.4 GHz column is why it has to drop the control\n"
        "radio to tens of kilohertz -- or do without one."
    )


if __name__ == "__main__":
    main()

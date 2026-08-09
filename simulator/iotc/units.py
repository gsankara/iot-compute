"""Physical constants and unit helpers.

Internally the whole package is strictly SI: metres, seconds, kilograms, watts,
joules, radians, bits.  These helpers exist so that scripts and YAML configs can
stay readable.

Mirrors ``fsphotonix.units`` so the two simulators agree on constants; the extra
entries here cover the things an IoT node cares about that a bench optics rig
does not -- byte capacities, energy budgets, oscillator drift.
"""

from __future__ import annotations

import math

C0 = 299_792_458.0  # speed of light in vacuum, m/s
H_PLANCK = 6.626_070_15e-34  # J s
Q_E = 1.602_176_634e-19  # C
K_B = 1.380_649e-23  # J/K
G_EARTH = 9.806_65  # m/s^2
T_ROOM = 293.15  # K

# ---------------------------------------------------------------- length
nm = 1e-9
um = 1e-6
mm = 1e-3
cm = 1e-2
m = 1.0
km = 1e3

# ---------------------------------------------------------------- time
ps = 1e-12
ns = 1e-9
us = 1e-6
ms = 1e-3
s = 1.0
minute = 60.0
hour = 3600.0
day = 86400.0

# ---------------------------------------------------------------- frequency / rate
Hz = 1.0
kHz = 1e3
MHz = 1e6
GHz = 1e9
THz = 1e12

bps = 1.0
kbps = 1e3
Mbps = 1e6
Gbps = 1e9

# ---------------------------------------------------------------- data
B = 1.0  # byte
KiB = 1024.0
MiB = 1024.0 ** 2
GiB = 1024.0 ** 3

# ---------------------------------------------------------------- power / energy
pW = 1e-12
nW = 1e-9
uW = 1e-6
mW = 1e-3
W = 1.0

pJ = 1e-12
nJ = 1e-9
uJ = 1e-6
mJ = 1e-3
J = 1.0

# A milliamp-hour at a given terminal voltage, in joules.
def mAh(capacity_mah: float, voltage: float = 3.3) -> float:
    """Battery/supercap rating [mAh] at ``voltage`` [V] -> stored energy [J]."""
    return capacity_mah * 1e-3 * hour * voltage


# ---------------------------------------------------------------- angle
deg = math.pi / 180.0
rad = 1.0
mrad = 1e-3
urad = 1e-6

# ---------------------------------------------------------------- misc
ppm = 1e-6  # oscillator drift
ppb = 1e-9

# Common transceiver wavelengths.
LAMBDA_850 = 850e-9  # cheap VCSEL, short reach
LAMBDA_O_BAND = 1310e-9
LAMBDA_C_BAND = 1550e-9  # eye-safe, best atmospheric window
LAMBDA_BLUE_GREEN = 470e-9  # the low-attenuation window in water


# ---------------------------------------------------------------- conversions
def dbm(p_watt: float) -> float:
    """Watts -> dBm."""
    return 10.0 * math.log10(max(p_watt, 1e-300) / 1e-3)


def from_dbm(p_dbm: float) -> float:
    """dBm -> watts."""
    return 10.0 ** (p_dbm / 10.0) * 1e-3


def db(ratio: float) -> float:
    """Power ratio -> dB."""
    return 10.0 * math.log10(max(ratio, 1e-300))


def from_db(value_db: float) -> float:
    """dB -> power ratio."""
    return 10.0 ** (value_db / 10.0)


def photon_energy(lam: float) -> float:
    """Energy of one photon at wavelength ``lam`` [J]."""
    return H_PLANCK * C0 / lam


def photons_per_second(power: float, lam: float) -> float:
    """Photon flux for ``power`` [W] at wavelength ``lam`` [m]."""
    return power / photon_energy(lam)


def wavelength_to_freq(lam: float) -> float:
    """Vacuum wavelength [m] -> frequency [Hz]."""
    return C0 / lam


def freq_to_wavelength(f: float) -> float:
    """Frequency [Hz] -> vacuum wavelength [m]."""
    return C0 / f


def speed_in_medium(index: float) -> float:
    """Propagation speed [m/s] in a medium of refractive index ``index``."""
    return C0 / index

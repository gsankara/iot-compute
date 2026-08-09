"""The medium the swarm is immersed in.

The medium couples the two halves of the design problem.  It sets how far light
carries (extinction, refractive index, turbulence), *and* it sets how the bodies
move (density for buoyancy, viscosity and density for drag).  Choosing water
instead of air is therefore not a parameter tweak, it changes which wavelength
is usable, whether RF works at all, and whether z is cheap or expensive.

Extinction is strongly wavelength dependent and the tables below are the reason
a liquid deployment must move to blue-green: at 850 nm pure water absorbs about
4.4 per metre, which kills a 10 m hop outright, while at 470 nm it absorbs about
0.016 per metre.

References for the tabulated values are given in ``docs/PHYSICS.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..registry import register

MU_0 = 4.0e-7 * math.pi  # vacuum permeability, H/m


class Medium:
    """Base class for a propagation and flotation medium."""

    #: Bulk density [kg/m^3]; drives buoyancy.
    density: float = 0.0
    #: Dynamic viscosity [Pa s]; only used for the low-Reynolds correction.
    viscosity: float = 0.0
    #: Refractive-index structure constant [m^-2/3]; drives scintillation.
    cn2: float = 0.0
    #: Electrical conductivity [S/m]; drives RF absorption.
    conductivity: float = 0.0
    #: Whether bodies in this medium can float (i.e. buoyancy is meaningful).
    supports_buoyancy: bool = False

    def refractive_index(self, wavelength: float) -> float:
        """Refractive index at the given vacuum wavelength."""
        raise NotImplementedError

    def extinction(self, wavelength: float) -> float:
        """Optical extinction coefficient [1/m] (absorption plus scattering)."""
        raise NotImplementedError

    # -- derived ----------------------------------------------------------
    def optical_transmittance(self, distance: float, wavelength: float) -> float:
        """Beer-Lambert survival fraction over ``distance`` metres."""
        return float(math.exp(-self.extinction(wavelength) * distance))

    def light_speed(self, wavelength: float) -> float:
        """Propagation speed [m/s], which sets the link latency."""
        from ..units import C0

        return C0 / self.refractive_index(wavelength)

    def rf_attenuation_db(self, frequency: float, distance: float) -> float:
        """Medium absorption of an RF wave [dB] over ``distance`` metres.

        Uses the good-conductor skin-depth result, which is the right limit for
        salt water and harmless (essentially zero) for air.
        """
        if self.conductivity <= 0.0:
            return 0.0
        alpha_np = math.sqrt(math.pi * frequency * MU_0 * self.conductivity)
        return 8.685889638 * alpha_np * distance

    def summary(self) -> str:  # pragma: no cover - reporting aid
        return (
            f"{self.plugin_name:12s} rho={self.density:7.1f} kg/m3  "
            f"mu={self.viscosity:.2e} Pa.s  Cn2={self.cn2:.1e}  "
            f"sigma={self.conductivity:.3g} S/m"
        )


def _interp_log(wavelength: float, table_nm: np.ndarray, table_val: np.ndarray) -> float:
    """Interpolate an extinction table in log space, clamped at the ends.

    Absorption spans several decades across the visible and near infrared, so
    linear interpolation between tabulated points would be badly wrong in the
    middle of a decade.
    """
    lam_nm = wavelength * 1e9
    lam_nm = min(max(lam_nm, table_nm[0]), table_nm[-1])
    return float(np.exp(np.interp(lam_nm, table_nm, np.log(table_val))))


# --------------------------------------------------------------------------
@register("medium", "vacuum")
class Vacuum(Medium):
    """Lossless free space; the control case for validating the link budget."""

    supports_buoyancy = False

    def __init__(self, cn2: float = 0.0) -> None:
        self.cn2 = float(cn2)

    def refractive_index(self, wavelength: float) -> float:
        return 1.0

    def extinction(self, wavelength: float) -> float:
        return 0.0


# --------------------------------------------------------------------------
_AIR_NM = np.array([400.0, 550.0, 850.0, 1310.0, 1550.0])
#: Clear-sky molecular extinction [1/m] at sea level.
_AIR_EXT = np.array([4.0e-5, 2.2e-5, 1.4e-5, 1.1e-5, 1.0e-5])


@register("medium", "air")
class Air(Medium):
    """Sea-level atmosphere, optionally with fog or haze.

    ``visibility`` (km) drives Kim's empirical aerosol model, which is what makes
    the difference between a clear night and a fog bank: at 0.5 km visibility a
    100 m hop loses tens of dB regardless of how much power the laser has.
    """

    supports_buoyancy = False

    def __init__(
        self,
        density: float = 1.225,
        viscosity: float = 1.81e-5,
        cn2: float = 1e-14,
        visibility_km: float = 50.0,
        conductivity: float = 0.0,
    ) -> None:
        self.density = float(density)
        self.viscosity = float(viscosity)
        self.cn2 = float(cn2)
        self.visibility_km = float(visibility_km)
        self.conductivity = float(conductivity)

    def refractive_index(self, wavelength: float) -> float:
        return 1.000273

    @staticmethod
    def _kim_q(visibility_km: float) -> float:
        v = visibility_km
        if v > 50.0:
            return 1.6
        if v > 6.0:
            return 1.3
        if v > 1.0:
            return 0.16 * v + 0.34
        if v > 0.5:
            return v - 0.5
        return 0.0

    def extinction(self, wavelength: float) -> float:
        molecular = _interp_log(wavelength, _AIR_NM, _AIR_EXT)
        if self.visibility_km <= 0.0 or not math.isfinite(self.visibility_km):
            return molecular
        q = self._kim_q(self.visibility_km)
        # Kim: beta [1/km] = (3.912 / V) * (lambda / 550 nm) ** -q
        beta_per_km = (3.912 / self.visibility_km) * (wavelength * 1e9 / 550.0) ** (-q)
        return molecular + beta_per_km / 1000.0


# --------------------------------------------------------------------------
_WATER_NM = np.array([400.0, 470.0, 520.0, 550.0, 650.0, 700.0, 850.0, 1310.0, 1550.0])
#: Pure-water absorption [1/m].  Note the four-decade swing.
_WATER_ABS = np.array([6.5e-3, 1.6e-2, 4.5e-2, 6.4e-2, 3.4e-1, 6.5e-1,
                       4.4e0, 1.3e2, 1.1e3])


@register("medium", "water", "seawater")
class Water(Medium):
    """Natural water, from clear ocean to coastal turbidity.

    ``turbidity`` adds a wavelength-flat scattering term on top of the pure-water
    absorption curve; ``salinity_conductivity`` is what makes RF useless, which
    in turn is why the control plane in a liquid deployment has to drop to a
    very low carrier frequency or lean entirely on the optical links.
    """

    supports_buoyancy = True

    def __init__(
        self,
        density: float = 1025.0,
        viscosity: float = 1.08e-3,
        cn2: float = 1e-9,
        turbidity: float = 0.02,
        conductivity: float = 4.0,
    ) -> None:
        self.density = float(density)
        self.viscosity = float(viscosity)
        self.cn2 = float(cn2)
        self.turbidity = float(turbidity)
        self.conductivity = float(conductivity)

    def refractive_index(self, wavelength: float) -> float:
        # Two-term Cauchy fit, adequate over 400-800 nm.
        lam_um = wavelength * 1e6
        return 1.3199 + 6.878e-3 / lam_um**2

    def extinction(self, wavelength: float) -> float:
        return _interp_log(wavelength, _WATER_NM, _WATER_ABS) + self.turbidity


# --------------------------------------------------------------------------
@register("medium", "index_fluid", "index-matched", "lab_fluid")
class IndexFluid(Medium):
    """An engineered flotation fluid: dense, quiet and optically clear.

    This is the medium the design-space study wants -- it keeps the buoyancy
    control of water while removing the absorption penalty, so it isolates the
    effect of *mobility* from the effect of *attenuation*.  Whether such a fluid
    is procurable at scale is exactly the kind of question the simulator exists
    to price.
    """

    supports_buoyancy = True

    def __init__(
        self,
        density: float = 1000.0,
        viscosity: float = 2.0e-3,
        cn2: float = 1e-12,
        extinction_per_m: float = 5e-3,
        index: float = 1.33,
        conductivity: float = 1e-4,
    ) -> None:
        self.density = float(density)
        self.viscosity = float(viscosity)
        self.cn2 = float(cn2)
        self._ext = float(extinction_per_m)
        self._index = float(index)
        self.conductivity = float(conductivity)

    def refractive_index(self, wavelength: float) -> float:
        return self._index

    def extinction(self, wavelength: float) -> float:
        return self._ext


__all__ = ["Medium", "Vacuum", "Air", "Water", "IndexFluid"]

"""Experimental 2D airfoil polar lookup tables.

Parses the NASA NACA 0012 experimental data and provides α → Cl, Cd
via linear interpolation.  Supports extrapolation beyond the measured
range using thin-airfoil fallback behaviour.

Usage
-----
>>> from polar import load_experimental_polar
>>> polar = load_experimental_polar("CD_CL(80,120,180roughness).dat")
>>> cl, cd = polar(alpha_rad)   # α in radians
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class PolarTable(NamedTuple):
    """Interpolated Cl(α), Cd(α) from experimental measurements.

    Parameters
    ----------
    alpha_rad : np.ndarray
        Angle of attack [rad] (sorted ascending).
    cl : np.ndarray
        Measured lift coefficient.
    cd : np.ndarray
        Measured drag coefficient.
    label : str
        Human-readable description (e.g. "80 grit roughness").
    """

    alpha_rad: NDArray[np.float64]
    cl: NDArray[np.float64]
    cd: NDArray[np.float64]
    label: str

    def __call__(
        self,
        alpha: float | NDArray[np.float64],
        alpha0_override: float | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Interpolate Cl(α) and Cd(α).

        Parameters
        ----------
        alpha : float or np.ndarray
            Angle of attack [rad].
        alpha0_override : float or None
            If given, shift Cl so that Cl(alpha0_override) = 0.  This is
            useful for adjusting the polar to match a different camber/
            zero-lift angle than the NACA 0012 (which is symmetric, α₀≈0).

        Returns
        -------
        cl, cd : np.ndarray
        """
        alpha_arr = np.atleast_1d(np.asarray(alpha, dtype=np.float64))

        # interpolation (kind="linear" is fine — experimental data is dense)
        cl_out = np.interp(alpha_arr, self.alpha_rad, self.cl)
        cd_out = np.interp(alpha_arr, self.alpha_rad, self.cd)

        # beyond measured range: fall back to thin-airfoil slope for Cl
        # and constant Cd at the edge value
        below = alpha_arr < self.alpha_rad[0]
        above = alpha_arr > self.alpha_rad[-1]

        if np.any(below):
            d_alpha = alpha_arr[below] - self.alpha_rad[0]
            cl_out[below] = self.cl[0] + 2.0 * np.pi * d_alpha
            # Cd: hold constant at edge value (best we can do)
        if np.any(above):
            # post-stall region — hold Cl and Cd at the last measured
            # value.  The experimental data already reaches deep stall
            # (Cd ~ 0.27 at α = 19°) and extrapolating further has no
            # physical basis.  Blade elements operating here produce
            # enormous drag regardless.
            cl_out[above] = self.cl[-1]
            # Cd: held at last value (already very high in stall)

        if alpha0_override is not None:
            # shift Cl linearly so Cl(alpha0_override) = 0
            cl0 = np.interp(alpha0_override, self.alpha_rad, self.cl)
            cl_out = cl_out - cl0

        # un-scalar
        if np.ndim(alpha) == 0:
            return cl_out.item(), cd_out.item()  # type: ignore[union-attr]
        return cl_out, cd_out


# ---------------------------------------------------------------------------
#  parser
# ---------------------------------------------------------------------------
def load_experimental_polar(
    path: str | Path,
    zone_label: str = "Roughness = 80 grit",
) -> PolarTable:
    """Parse the NASA experimental Cl-Cd polar file.

    The file contains multiple roughness configurations.  *zone_label*
    selects which one to return (default: 80 grit, the smoothest surface).

    Returns a PolarTable callable with signature (α_rad) → (cl, cd).
    """
    text = Path(path).read_text(encoding="utf-8")

    zones: dict[str, dict[str, list[float]]] = {}
    current_zone: str | None = None

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("TITLE") or line.startswith("variables"):
            continue

        m = re.match(r'zone,\s*t="(.+)"', line)
        if m:
            current_zone = m.group(1)
            zones[current_zone] = {"alpha": [], "cl": [], "cd": []}
            continue

        if current_zone is not None:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    a = float(parts[0])
                    cl = float(parts[1])
                    cd = float(parts[2])
                except ValueError:
                    continue
                zones[current_zone]["alpha"].append(a)
                zones[current_zone]["cl"].append(cl)
                zones[current_zone]["cd"].append(cd)

    if zone_label not in zones:
        available = list(zones.keys())
        raise KeyError(
            f"Zone {zone_label!r} not found. Available: {available}"
        )

    z = zones[zone_label]
    alpha_deg = np.array(z["alpha"])
    cl = np.array(z["cl"])
    cd = np.array(z["cd"])

    # sort by α (ascending)
    order = np.argsort(alpha_deg)
    alpha_rad = np.radians(alpha_deg[order])
    cl = cl[order]
    cd = cd[order]

    return PolarTable(alpha_rad, cl, cd, label=zone_label)


# ---------------------------------------------------------------------------
#  built-in polar for NACA 0012 (convenience)
# ---------------------------------------------------------------------------
def naca0012_polar(roughness: str = "80 grit") -> PolarTable:
    """Return a PolarTable for NACA 0012 from the bundled NASA data."""
    data_dir = Path(__file__).resolve().parent / "NACA0012_data" / "Experimental"
    path = data_dir / "CD_CL(80,120,180roughness).dat"
    if not path.exists():
        raise FileNotFoundError(
            f"Experimental data not found at {path}.  "
            f"Unzip NACA0012.zip first."
        )
    return load_experimental_polar(path, zone_label=f"Roughness = {roughness}")

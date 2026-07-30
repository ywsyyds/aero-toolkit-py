"""NACA 4-digit camber line.

Parses a code like 2412 → m=0.02, p=0.4 and returns yc(x), dyc/dx
at arbitrary x-stations (chord normalised to [0, 1]).

Standard formulas (x ∈ [0, 1]):

  0 ≤ x ≤ p:   yc = (m/p²) · (2p·x - x²)
               dyc/dx = (2m/p²) · (p - x)

  p ≤ x ≤ 1:   yc = (m/(1-p)²) · ((1-2p) + 2p·x - x²)
               dyc/dx = (2m/(1-p)²) · (p - x)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def parse_naca4(code: int | str) -> tuple[float, float]:
    """Parse a 4-digit NACA code into (m, p).

    Parameters
    ----------
    code : int or str
        NACA 4-digit code, e.g. 2412 or "2412".

    Returns
    -------
    m : float
        Maximum camber as fraction of chord (e.g. 0.02 for 2412).
    p : float
        Position of maximum camber as fraction of chord (e.g. 0.4 for 2412).

    Raises
    ------
    ValueError
        If the code is not a valid NACA 4-digit string/number.
    """
    s = str(int(code)).zfill(4)
    if len(s) != 4:
        raise ValueError(f"Expected a 4-digit NACA code, got {code!r}")
    m = int(s[0]) / 100.0
    p = int(s[1]) / 10.0
    # s[2:] = max thickness (not used for camber line)
    return m, p


def naca4_camber(
    code: int | str = 2412,
    x: NDArray[np.float64] | None = None,
    n_pts: int = 200,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return (x, yc, dyc_dx) for a NACA 4-digit camber line.

    Parameters
    ----------
    code : int or str
        NACA 4-digit code.
    x : np.ndarray or None
        Chordwise stations (0…1).  If None, a uniform grid of *n_pts*
        points is generated.
    n_pts : int
        Number of points when *x* is None.

    Returns
    -------
    x : np.ndarray   (n,)
    yc : np.ndarray  (n,)
    dyc_dx : np.ndarray  (n,)
    """
    m, p = parse_naca4(code)
    if m == 0.0:
        if x is None:
            x = np.linspace(0, 1, n_pts)
        return x, np.zeros_like(x), np.zeros_like(x)

    if x is None:
        x = np.linspace(0, 1, n_pts)

    yc = np.empty_like(x)
    dyc_dx = np.empty_like(x)

    # leading edge → max-camber point
    mask_fore = x <= p
    xf = x[mask_fore]
    yc[mask_fore] = (m / p ** 2) * (2 * p * xf - xf ** 2)
    dyc_dx[mask_fore] = (2 * m / p ** 2) * (p - xf)

    # max-camber point → trailing edge
    mask_aft = x > p
    xa = x[mask_aft]
    om = 1.0 - p
    yc[mask_aft] = (m / om ** 2) * ((1 - 2 * p) + 2 * p * xa - xa ** 2)
    dyc_dx[mask_aft] = (2 * m / om ** 2) * (p - xa)

    return x, yc, dyc_dx

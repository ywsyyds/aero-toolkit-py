"""2D Discrete Vortex Lattice Method for flat-plate and cambered airfoils.

Each panel i:
  - vortex    at x_v[i] = (i + 1/4) * dx   (1/4-chord of the panel)
  - colloc.   at x_c[i] = (i + 3/4) * dx   (3/4-chord of the panel)

Vortices and collocation points live on the x-axis (thin-airfoil
approximation).  Camber enters via the boundary-condition RHS:

  A · Γ = V∞ · [sin(α) - dyc/dx · cos(α)]

Influence coefficient A[i][j] is the y-velocity at control point i
induced by a unit-strength vortex at vortex point j.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class VLM2D:
    """Two-dimensional vortex-lattice solver for a thin airfoil.

    Parameters
    ----------
    chord : float
        Chord length (default 1.0).
    n_panels : int
        Number of panels (≥ 2).
    V_inf : float
        Freestream velocity magnitude.
    dyc_dx : np.ndarray or None
        Camber-line slope at each collocation point.  *None* (default)
        corresponds to a flat plate.
    """

    def __init__(
        self,
        chord: float = 1.0,
        n_panels: int = 20,
        V_inf: float = 1.0,
        dyc_dx: NDArray[np.float64] | None = None,
    ) -> None:
        if n_panels < 2:
            raise ValueError("n_panels must be >= 2")
        self.chord = chord
        self.n_panels = n_panels
        self.V_inf = V_inf
        self.dx = chord / n_panels

        # vortex positions  (1/4-chord of each panel)
        self.xv = (np.arange(n_panels) + 0.25) * self.dx
        # collocation points (3/4-chord of each panel)
        self.xc = (np.arange(n_panels) + 0.75) * self.dx

        # camber-line slope at collocation points
        if dyc_dx is None:
            self.dyc_dx: NDArray[np.float64] = np.zeros(n_panels, dtype=np.float64)
        else:
            if len(dyc_dx) != n_panels:
                raise ValueError(
                    f"dyc_dx must have length n_panels ({n_panels}), "
                    f"got {len(dyc_dx)}"
                )
            self.dyc_dx = np.asarray(dyc_dx, dtype=np.float64)

        # precompute influence matrix (independent of alpha)
        self.A = self._build_influence_matrix()

    # ------------------------------------------------------------------
    @classmethod
    def from_naca4(
        cls, code: int | str = 2412, chord: float = 1.0, n_panels: int = 20, V_inf: float = 1.0
    ) -> "VLM2D":
        """Factory: build a VLM2D instance from a NACA 4-digit camber line.

        Parameters
        ----------
        code : int or str
            NACA 4-digit designation (e.g. 2412 or "2412").
        chord, n_panels, V_inf
            Forwarded to ``__init__``.
        """
        from camberline import naca4_camber

        # temporary instance just to get xc positions
        tmp = cls(chord=chord, n_panels=n_panels, V_inf=V_inf)
        _, _, dyc = naca4_camber(code, x=tmp.xc)
        return cls(chord=chord, n_panels=n_panels, V_inf=V_inf, dyc_dx=dyc)

    # ------------------------------------------------------------------
    def _build_influence_matrix(self) -> NDArray[np.float64]:
        """Return the (n × n) influence-coefficient matrix.

        A[i][j] = tangential (y) velocity induced at control point i
                  by a unit-strength point vortex at vortex point j.
        """
        # dx = xc_i - xv_j  →  shape (n, n)
        dx = self.xc[:, np.newaxis] - self.xv[np.newaxis, :]  # (n,1) - (1,n) = (n,n)
        # Guard against zero (should not happen for the standard VLM layout)
        A = 1.0 / (2.0 * np.pi * dx)
        return A

    # ------------------------------------------------------------------
    def solve(self, alpha_deg: float) -> tuple[float, NDArray[np.float64]]:
        """Solve for the circulation distribution and return (Cl, gamma).

        Parameters
        ----------
        alpha_deg : float
            Angle of attack in degrees.

        Returns
        -------
        cl : float
            Section lift coefficient.
        gamma : np.ndarray
            Circulation strength of each vortex (length n_panels).
        """
        alpha = np.radians(alpha_deg)

        # boundary condition (thin-airfoil approximation):
        #   V∞·n + v_ind·n = 0   →   A·Γ = V∞·[sin(α) - dyc/dx·cos(α)]
        rhs = self.V_inf * (np.sin(alpha) - self.dyc_dx * np.cos(alpha))

        gamma = np.linalg.solve(self.A, rhs)

        cl = self._compute_cl(gamma)
        return cl, gamma

    # ------------------------------------------------------------------
    def _compute_cl(self, gamma: NDArray[np.float64]) -> float:
        """Kutta-Joukowski:  L' = rho * V_inf * sum(gamma)
            Cl = L' / (0.5 * rho * V_inf^2 * chord)
               = 2 * sum(gamma) / (V_inf * chord)
        """
        Gamma_total = np.sum(gamma)
        return 2.0 * Gamma_total / (self.V_inf * self.chord)

    # ------------------------------------------------------------------
    @property
    def expected_slope(self) -> float:
        """Thin-airfoil-theory slope: dCl/dα = 2π  (α in rad)."""
        return 2.0 * np.pi


# ----------------------------------------------------------------------
def thin_airfoil_cl(alpha_deg: float) -> float:
    """Thin-airfoil theory for a symmetric section: Cl = 2π·α  (α in rad)."""
    return 2.0 * np.pi * np.radians(alpha_deg)

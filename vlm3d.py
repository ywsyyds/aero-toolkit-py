"""3D Vortex Lattice Method — aerodynamic solver.

Biot-Savart finite-segment induction → influence matrix → solve Γ → CL, CDi.

Coordinate system:
  x  — streamwise       (+x downstream)
  y  — spanwise         (+y starboard / right wing)
  z  — vertical         (+z up, lift direction)

Wing lies in the z = 0 plane. Each spanwise panel hosts one horseshoe vortex:

  Panel j :  y ∈ [y_j, y_{j+1}]

  bound vortex:          (x_bound_j, y_j, 0)  →  (x_bound_j, y_{j+1}, 0)     [+y]
  left  trailing vortex: (x_far,    y_j, 0)  ←  (x_bound_j, y_j, 0)          [-x]
  right trailing vortex: (x_bound_j, y_{j+1}, 0) → (x_far,    y_{j+1}, 0)    [+x]
  control point:         (x_ctrl_j,  (y_j + y_{j+1})/2,  0)

Circulation sign (right-hand rule):
  Γ > 0  →  bound vortex points +y  →  lift in +z.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
Vec3 = NDArray[np.float64]       # shape (3,)
Mat = NDArray[np.float64]        # shape (n, n) or (n, m)


# ===================================================================
#  Biot-Savart core
# ===================================================================
def biot_savart_segment(
    P: Vec3,
    P1: Vec3,
    P2: Vec3,
    gamma: float = 1.0,
    cutoff: float = 1e-12,
) -> Vec3:
    """Induced velocity at **P** from a straight vortex segment **P1** → **P2**.

    Katz & Plotkin form:

        v = (Γ / 4π) · (r₁ × r₂) / |r₁ × r₂|²  ·  (r₀ · r̂₁ − r₀ · r̂₂)

    with  r₁ = P − P₁,  r₂ = P − P₂,  r₀ = P₂ − P₁.

    When |r₁ × r₂| < *cutoff* (point nearly on the vortex filament), the
    function returns **zero** to avoid the singular self-induction.
    """
    r1 = P - P1
    r2 = P - P2
    r0 = P2 - P1

    cross = np.cross(r1, r2)
    cross_norm_sq = np.dot(cross, cross)

    if cross_norm_sq < cutoff * cutoff:
        return np.zeros(3, dtype=np.float64)

    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2)

    # guard against degenerate segments
    if r1_norm < cutoff or r2_norm < cutoff:
        return np.zeros(3, dtype=np.float64)

    dot_term = np.dot(r0, r1) / r1_norm - np.dot(r0, r2) / r2_norm

    return (gamma / (4.0 * np.pi)) * cross / cross_norm_sq * dot_term


# ===================================================================
#  VLM3D  —  three-dimensional vortex-lattice solver
# ===================================================================
class VLM3D:
    """3D VLM solver for a wing with given spanwise chord distribution.

    Parameters
    ----------
    span : float
        Wing span *b*.
    chords : np.ndarray or float
        Chord length at each panel centre  (length = *n_span*).
        A scalar is broadcast to a constant-chord rectangular wing.
    n_span : int
        Number of spanwise panels.
    x_far_factor : float
        Trailing-vortex far-field distance  = *x_far_factor* × *span*.
    V_inf : float
        Freestream velocity magnitude.
    """

    def __init__(
        self,
        span: float,
        chords: NDArray[np.float64] | float,
        n_span: int = 40,
        x_far_factor: float = 20.0,
        V_inf: float = 1.0,
    ) -> None:
        if n_span < 2:
            raise ValueError("n_span must be >= 2")
        self.span = span
        self.n_span = n_span
        self.V_inf = V_inf
        self.dy = span / n_span

        # ---- panel y-stations -----------------------------------------------
        self.y_edges = np.linspace(-span / 2, span / 2, n_span + 1)
        self.y_ctr = 0.5 * (self.y_edges[:-1] + self.y_edges[1:])

        # ---- chord distribution ---------------------------------------------
        if isinstance(chords, (int, float)):
            self.chords: NDArray[np.float64] = np.full(n_span, float(chords))
        else:
            self.chords = np.asarray(chords, dtype=np.float64)
            if len(self.chords) != n_span:
                raise ValueError(f"chords length {len(self.chords)} != n_span {n_span}")

        self.S = float(np.sum(self.chords * self.dy))   # planform area
        self.AR = self.span ** 2 / self.S                # aspect ratio
        self.chord_ref = self.S / self.span              # mean aerodynamic chord

        # ---- x-stations (per panel) -----------------------------------------
        self.x_bound: NDArray[np.float64] = self.chords / 4.0            # 1/4-c
        self.x_ctrl:  NDArray[np.float64] = 3.0 * self.chords / 4.0     # 3/4-c
        self.x_far = x_far_factor * span

        # ---- cutoff for self-induction guard --------------------------------
        self._cutoff = 1e-8 * self.chord_ref

        # ---- cached influence matrix ----------------------------------------
        self._A: Mat | None = None
        self._matrix_valid = False

    # ------------------------------------------------------------------
    #  Factory constructors
    # ------------------------------------------------------------------
    @classmethod
    def rectangular(
        cls, span: float = 10.0, chord: float = 1.0,
        n_span: int = 40, x_far_factor: float = 20.0, V_inf: float = 1.0,
    ) -> "VLM3D":
        """Constant-chord rectangular wing."""
        return cls(span=span, chords=float(chord), n_span=n_span,
                   x_far_factor=x_far_factor, V_inf=V_inf)

    @classmethod
    def elliptic(
        cls, span: float = 10.0, AR: float = 8.0,
        n_span: int = 40, x_far_factor: float = 20.0, V_inf: float = 1.0,
    ) -> "VLM3D":
        """Elliptic planform  c(y) = c₀ · √(1 − (2y/b)²).

        Parameters
        ----------
        span : float
        AR : float
            Aspect ratio  b² / S.  Determines root chord via  S = π·b·c₀/4.
        """
        c0 = 4.0 * span / (np.pi * AR)   # root chord
        # chord at each panel centre
        r = 2.0 * np.abs(cls._y_ctr_uniform(span, n_span)) / span
        chords = c0 * np.sqrt(np.maximum(1.0 - r * r, 0.0))
        return cls(span=span, chords=chords, n_span=n_span,
                   x_far_factor=x_far_factor, V_inf=V_inf)

    @staticmethod
    def _y_ctr_uniform(span: float, n_span: int) -> NDArray[np.float64]:
        y_edges = np.linspace(-span / 2, span / 2, n_span + 1)
        return 0.5 * (y_edges[:-1] + y_edges[1:])

    # ------------------------------------------------------------------
    #  Per-panel geometry helpers  (indices 0 … n_span-1)
    # ------------------------------------------------------------------
    def _yL(self, i: int) -> float: return self.y_edges[i]
    def _yR(self, i: int) -> float: return self.y_edges[i + 1]

    def _bound_start(self, i: int) -> Vec3:
        return np.array([self.x_bound[i], self._yL(i), 0.0])

    def _bound_end(self, i: int) -> Vec3:
        return np.array([self.x_bound[i], self._yR(i), 0.0])

    def _bound_mid(self, i: int) -> Vec3:
        return np.array([self.x_bound[i], self.y_ctr[i], 0.0])

    def _left_trail_start(self, i: int) -> Vec3:
        return np.array([self.x_far, self._yL(i), 0.0])

    def _left_trail_end(self, i: int) -> Vec3:
        return np.array([self.x_bound[i], self._yL(i), 0.0])

    def _right_trail_start(self, i: int) -> Vec3:
        return np.array([self.x_bound[i], self._yR(i), 0.0])

    def _right_trail_end(self, i: int) -> Vec3:
        return np.array([self.x_far, self._yR(i), 0.0])

    def _ctrl_point(self, i: int) -> Vec3:
        return np.array([self.x_ctrl[i], self.y_ctr[i], 0.0])

    # ------------------------------------------------------------------
    #  Horseshoe vortex — full induced velocity at P (Γ = 1)
    # ------------------------------------------------------------------
    def _horseshoe_velocity(self, P: Vec3, j: int) -> Vec3:
        """Velocity induced at **P** by horseshoe vortex of panel *j* with Γ = 1."""
        v = np.zeros(3)
        v += biot_savart_segment(P, self._bound_start(j), self._bound_end(j), 1.0, self._cutoff)
        v += biot_savart_segment(P, self._left_trail_start(j), self._left_trail_end(j), 1.0, self._cutoff)
        v += biot_savart_segment(P, self._right_trail_start(j), self._right_trail_end(j), 1.0, self._cutoff)
        return v

    # ------------------------------------------------------------------
    #  Influence matrix
    # ------------------------------------------------------------------
    def _build_matrix(self) -> Mat:
        """Return A where  A[i][j] = z-velocity at ctrl *i* from horseshoe *j* (Γ=1)."""
        A = np.empty((self.n_span, self.n_span), dtype=np.float64)
        for i in range(self.n_span):
            P = self._ctrl_point(i)
            for j in range(self.n_span):
                v = self._horseshoe_velocity(P, j)
                A[i, j] = v[2]   # z-component
        return A

    @property
    def matrix(self) -> Mat:
        if self._A is None:
            self._A = self._build_matrix()
        return self._A

    # ------------------------------------------------------------------
    #  Solve
    # ------------------------------------------------------------------
    def solve(self, alpha_deg: float) -> tuple[float, float, NDArray[np.float64]]:
        """Return (CL, CDi, gamma) at the given angle of attack.

        Parameters
        ----------
        alpha_deg : float
            Angle of attack in degrees.

        Returns
        -------
        CL  : float           — lift coefficient
        CDi : float           — induced-drag coefficient
        gamma : np.ndarray    — circulation at each panel (length *n_span*)
        """
        alpha = np.radians(alpha_deg)
        A = self.matrix

        # boundary condition:  A · Γ = − V∞ · sin(α)
        rhs = -self.V_inf * np.sin(alpha) * np.ones(self.n_span)
        gamma = np.linalg.solve(A, rhs)

        CL = self._compute_CL(gamma)
        CDi = self._compute_CDi(gamma)
        return CL, CDi, gamma

    # ------------------------------------------------------------------
    def _compute_CL(self, gamma: NDArray[np.float64]) -> float:
        """Kutta-Joukowski spanwise integration.

        dL = ρ · V∞ · Γ(y) · dy
        CL = L / (½ ρ V∞² S) = 2 · Σ Γ_i · dy / (V∞ · S)
        """
        L = self.V_inf * np.sum(gamma) * self.dy          # L / ρ
        return 2.0 * L / (self.V_inf ** 2 * self.S)

    # ------------------------------------------------------------------
    def _compute_CDi(self, gamma: NDArray[np.float64]) -> float:
        """Induced drag from trailing-vortex downwash at bound-vortex midpoints.

        w_i = z-velocity at bound-vortex midpoint *i* induced by ALL
             trailing vortices (Γ-weighted).

        dD_i = − ρ · w_i · Γ_i · dy           (w_i is negative — downwash)

        CDi = D / (½ ρ V∞² S) = −2 · Σ w_i · Γ_i · dy / (V∞² · S)
        """
        w = np.zeros(self.n_span, dtype=np.float64)

        for i in range(self.n_span):
            P = self._bound_mid(i)
            v = np.zeros(3)
            for j in range(self.n_span):
                if abs(gamma[j]) < 1e-15:
                    continue
                v += biot_savart_segment(
                    P,
                    self._left_trail_start(j),
                    self._left_trail_end(j),
                    gamma[j],
                    self._cutoff,
                )
                v += biot_savart_segment(
                    P,
                    self._right_trail_start(j),
                    self._right_trail_end(j),
                    gamma[j],
                    self._cutoff,
                )
            w[i] = v[2]

        # w is the downwash (negative for positive Γ) → −w > 0
        D_rho = -np.sum(w * gamma) * self.dy             # D / ρ
        return 2.0 * D_rho / (self.V_inf ** 2 * self.S)

    # ------------------------------------------------------------------
    #  Theoretical references
    # ------------------------------------------------------------------
    @staticmethod
    def prandtl_CL_alpha(AR: float, a0: float = 2.0 * np.pi) -> float:
        """Prandtl lifting-line slope (per radian) for an untwisted elliptic wing.

        a = a₀ / (1 + a₀ / (π · AR))
        """
        return a0 / (1.0 + a0 / (np.pi * AR))

    @staticmethod
    def prandtl_CDi(CL: float, AR: float, e: float = 1.0) -> float:
        """Induced drag:  CDi = CL² / (π · AR · e)."""
        return CL ** 2 / (np.pi * AR * e)

    @staticmethod
    def thin_airfoil_CL_alpha() -> float:
        return 2.0 * np.pi

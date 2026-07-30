"""Blade Element Momentum Theory (BEMT) for fixed-pitch propellers.

Couples 2D airfoil aerodynamics (blade element theory) with actuator-disk
momentum theory to predict propeller thrust, torque, and efficiency.

Dimensionless groups:
  J  = V∞ / (n·D)          advance ratio
  CT = T  / (ρ·n²·D⁴)      thrust coefficient
  CP = P  / (ρ·n³·D⁵)      power coefficient
  η  = J·CT / CP            propulsive efficiency

n = ω/(2π) [rev/s], D = 2R [m].

Algorithm (per radial station, per advance ratio)
-------------------------------------------------
  1.  Guess  a = a' = 0.
  2.  φ = atan2( V∞(1+a),  ω·r·(1−a') )
  3.  α = θ(r) − φ
  4.  Cl, Cd  from 2D polar
  5.  Cn =  Cl·cos φ − Cd·sin φ,  Ct =  Cl·sin φ + Cd·cos φ
  6.  σ = B·c / (2πr),  F = Prandtl tip/hub loss
  7.  CT_local = dT / (½ρ·V∞²·2πr·dr) = σ·Cn·(1+a)² / sin²φ
  8.  From momentum:  CT_local = 4a(1+a)F       (a ≤ ac)
      Glauert:         CT_local = 4F(ac² + (1−2ac)·a)   (a > ac)
  9.  Solve for a.  Similarly for a' from CQ_local.
  10. Relax and iterate.

References
----------
Glauert, H.  "Airplane Propellers", *Aerodynamic Theory* (Durand, ed.), 1935.
Ning, S. A.  "A simple solution method for the blade element momentum
  equations …", Wind Energy 17, 2014.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  2D airfoil polar
# ---------------------------------------------------------------------------
def simple_polar(
    alpha_rad: NDArray[np.float64],
    cl_slope: float = 2.0 * np.pi,
    alpha0: float = 0.0,
    cd0: float = 0.010,
    k: float = 0.008,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cl = clα·(α−α₀),  Cd = Cd₀ + k·Cl²."""
    cl = cl_slope * (alpha_rad - alpha0)
    cd = cd0 + k * cl * cl
    return cl, cd


# ===================================================================
class BEMTSolver:
    """Blade-element momentum solver for a fixed-pitch propeller.

    Parameters
    ----------
    R : float
        Tip radius [m].
    Rh : float
        Hub radius [m] (default 0.15·R).
    B : int
        Number of blades.
    omega : float
        Rotation rate [rad/s].
    chord : float or array
        Chord length [m] — scalar for constant chord, array per station.
    pitch_angle : array or None
        Geometric pitch angle θ(r) [rad] measured from rotation plane.
        If None, built from `J_design` + `alpha_design`.
    J_design : float
        Design advance ratio (used when *pitch_angle* is None).
    alpha_design_deg : float
        Design angle of attack [deg] added to geometric pitch.
    rho : float
        Air density [kg/m³].
    n_stations : int
        Radial stations.
    relax : float
        Under-relaxation factor.
    max_iter, tol : convergence controls.
    """

    def __init__(
        self,
        R: float = 0.127,                     # ≈ 5-inch radius
        Rh: float | None = None,
        B: int = 2,
        omega: float = 1000.0,                # ≈ 9550 RPM
        chord: float | NDArray[np.float64] | None = None,
        pitch_angle: NDArray[np.float64] | None = None,
        J_design: float = 0.7,
        alpha_design_deg: float = 4.0,
        rho: float = 1.225,
        n_stations: int = 30,
        cl_slope: float = 2.0 * np.pi,
        alpha0: float = 0.0,
        cd0: float = 0.010,
        k_cd: float = 0.008,
        polar_table=None,  # PolarTable | None
        relax: float = 0.3,
        max_iter: int = 500,
        tol: float = 1e-6,
    ) -> None:
        self.R = R
        self.Rh = Rh if Rh is not None else 0.15 * R
        self.B = B
        self.omega = omega
        self.rho = rho
        self.n_stations = n_stations
        self.relax = relax
        self.max_iter = max_iter
        self.tol = tol

        # derived
        self.D = 2.0 * R
        self.A = np.pi * R ** 2
        self.n = omega / (2.0 * np.pi)              # rev / s

        # ---- polar parameters -----------------------------------------------
        self.cl_slope = cl_slope
        self.alpha0 = alpha0
        self.cd0 = cd0
        self.k_cd = k_cd
        self.polar_table = polar_table  # PolarTable | None

        # ---- radial discretisation (element midpoints) ----------------------
        dr = (R - self.Rh) / n_stations
        self.dr = dr
        self.r = np.linspace(self.Rh + dr / 2, R - dr / 2, n_stations)
        self._w = np.full(n_stations, dr)     # uniform integration weights

        # ---- blade geometry -------------------------------------------------
        if chord is None:
            self.chord = np.full(n_stations, 0.12 * R)   # ~12% of radius
        elif isinstance(chord, (int, float)):
            self.chord = np.full(n_stations, float(chord))
        else:
            self.chord = np.asarray(chord, dtype=np.float64)

        if pitch_angle is None:
            self.theta = self._design_pitch(J_design, alpha_design_deg)
        else:
            self.theta = np.asarray(pitch_angle, dtype=np.float64)

        self.sigma = self.B * self.chord / (2.0 * np.pi * self.r)

        # per-station induction state (re-used across solve calls)
        self.a: NDArray[np.float64] = np.zeros(n_stations)
        self.ap: NDArray[np.float64] = np.zeros(n_stations)

    # ------------------------------------------------------------------
    def _design_pitch(self, J_design: float, alpha_design_deg: float) -> NDArray[np.float64]:
        """Constant-pitch blade:  θ(r) = atan(J·R / (π·r)) + α_design."""
        alpha_rad = np.radians(alpha_design_deg)
        return np.arctan(J_design * self.R / (np.pi * self.r)) + alpha_rad

    # ------------------------------------------------------------------
    def _prandtl_loss(self, phi: NDArray[np.float64]) -> NDArray[np.float64]:
        """Prandtl tip- + hub-loss factor."""
        sin_phi = np.maximum(np.abs(np.sin(phi)), 1e-12)
        f_tip = (self.B / 2.0) * (self.R - self.r) / (self.r * sin_phi)
        F_tip = 2.0 / np.pi * np.arccos(np.clip(np.exp(-f_tip), -1.0, 1.0))
        f_hub = (self.B / 2.0) * (self.r - self.Rh) / (self.r * sin_phi)
        F_hub = 2.0 / np.pi * np.arccos(np.clip(np.exp(-f_hub), -1.0, 1.0))
        return np.clip(F_tip * F_hub, 1e-12, 1.0)

    # ------------------------------------------------------------------
    def _polar(self, alpha: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if self.polar_table is not None:
            return self.polar_table(alpha)
        return simple_polar(
            alpha, cl_slope=self.cl_slope, alpha0=self.alpha0,
            cd0=self.cd0, k=self.k_cd,
        )

    # ==================================================================
    def solve(self, V_inf: float) -> BEMTResult:
        """Iterate BEMT for a given freestream velocity.

        Returns a ``BEMTResult`` with CT, CP, η, and radial distributions.
        """
        n = self.n_stations
        r = self.r
        c = self.chord
        theta = self.theta
        sigma = self.sigma

        # initialise from previous solve (warm start) or zero
        a = np.zeros_like(self.a)          # *not* self.a — fresh start per V_inf
        ap = np.zeros_like(self.ap)

        omega_r = self.omega * r
        V_inf_sq = V_inf * V_inf
        ac = 1.0 / 3.0          # Glauert critical a

        for iteration in range(self.max_iter):
            a_prev = a
            ap_prev = ap

            # -- velocity triangle --------------------------------------------
            V_axial = V_inf * (1.0 + a)
            V_tang = omega_r * (1.0 - ap)
            W = np.sqrt(V_axial ** 2 + V_tang ** 2)
            phi = np.arctan2(V_axial, V_tang)

            # -- angle of attack & polar -------------------------------------
            alpha = theta - phi
            cl, cd = self._polar(alpha)

            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            Cn = cl * cos_phi - cd * sin_phi
            Ct = cl * sin_phi + cd * cos_phi

            # -- tip / hub loss ----------------------------------------------
            F_raw = self._prandtl_loss(phi)
            # Use a floor for the momentum update to avoid singular behaviour
            # near the tip (where F→0 ⇒ the momentum equations degenerate).
            # The force integration uses the original F_raw.
            F = np.maximum(F_raw, 1e-3)

            # ================================================================
            #  Axial induction  —  momentum equilibrium with Glauert
            # ================================================================
            # CT_local from BET side:  dT / (½ρ·V∞²·2πr·dr)
            ct_local = sigma * Cn * (1.0 + a) ** 2 / (sin_phi ** 2 + 1e-15)
            ct_local = np.maximum(ct_local, 0.0)

            # standard momentum:  CT_local = 4·a·(1+a)·F
            # →  a² + a − CT_local/(4F) = 0  →  a = ½(√(1+CT_local/F) − 1)
            # (the negative root is for wind turbines / decelerating flow;
            #  the positive root gives a ≥ 0 for a propeller)

            disc = np.maximum(1.0 + ct_local / (F + 1e-15), 0.0)
            a_std = 0.5 * (np.sqrt(disc) - 1.0)

            # Glauert correction:  CT_local = 4F·(ac² + (1−2ac)·a)  for a > ac
            # → a = (CT_local/(4F) − ac²) / (1−2ac)
            a_glauert = (ct_local / (4.0 * F + 1e-15) - ac ** 2) / (1.0 - 2.0 * ac)
            a_glauert = np.clip(a_glauert, ac, 1.0)

            # blending near ac (smooth transition from standard to Glauert)
            w = np.clip((a_std - ac) / 0.05, 0.0, 1.0)   # blend over ±0.05 around ac
            a_mt = (1.0 - w) * a_std + w * a_glauert

            # fallback for stations not producing thrust
            a_mt = np.where(Cn > 0.0, a_mt, 0.0)
            a_mt = np.clip(a_mt, 0.0, 1.0)

            # ================================================================
            #  Tangential induction
            # ================================================================
            # CQ_local from BET:  dQ / (½ρ·V∞·ω·r²·2πr·dr)  … well let's just
            # use the standard form:
            #   a'/(1−a')  =  σ·Ct / (4F·sinφ·cosφ)
            # → a' = 1 / (4F·sinφ·cosφ / (σ·Ct) + 1)

            denom_ap = 4.0 * F * sin_phi * cos_phi / (sigma * Ct + 1e-15)
            ap_mt = np.where(Ct > 1e-12, 1.0 / (denom_ap + 1.0), 0.0)
            ap_mt = np.clip(ap_mt, 0.0, 1.0)

            # ================================================================
            #  Relaxation
            # ================================================================
            a = a_prev + self.relax * (a_mt - a_prev)
            ap = ap_prev + self.relax * (ap_mt - ap_prev)

            # check convergence
            if (np.max(np.abs(a - a_prev)) < self.tol and
                    np.max(np.abs(ap - ap_prev)) < self.tol):
                break

        # ==================================================================
        #  Final force computation with converged induction factors
        # ==================================================================
        V_axial = V_inf * (1.0 + a)
        V_tang = omega_r * (1.0 - ap)
        W = np.sqrt(V_axial ** 2 + V_tang ** 2)
        phi = np.arctan2(V_axial, V_tang)
        alpha = theta - phi
        cl, cd = self._polar(alpha)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        Cn = cl * cos_phi - cd * sin_phi
        Ct = cl * sin_phi + cd * cos_phi

        # elemental forces (B blades)
        dT_dr = 0.5 * self.rho * self.B * c * W ** 2 * Cn
        dQ_dr = 0.5 * self.rho * self.B * c * W ** 2 * Ct * r

        # integrate (trapezoidal with end-correction weights)
        T = float(np.sum(dT_dr * self._w))
        Q = float(np.sum(dQ_dr * self._w))
        P = self.omega * Q

        # ---- dimensionless coefficients ------------------------------------
        n2D4 = self.n ** 2 * self.D ** 4
        n3D5 = self.n ** 3 * self.D ** 5
        CT = T / (self.rho * n2D4)
        CP = P / (self.rho * n3D5)
        J = V_inf / (self.n * self.D)
        eta = J * CT / CP if CP > 1e-15 else 0.0

        # cache for next warm-start
        self.a = a
        self.ap = ap

        return BEMTResult(
            J=J, CT=CT, CP=CP, eta=eta,
            T=T, Q=Q, P=P,
            r=r.copy(), a=a.copy(), ap=ap.copy(),
            phi=phi.copy(), alpha=alpha.copy(),
            cl=cl.copy(), cd=cd.copy(),
            dT_dr=dT_dr.copy(), dQ_dr=dQ_dr.copy(),
            iterations=iteration + 1,
        )


# ===================================================================
class BEMTResult:
    """Output of a single BEMT operating point."""

    __slots__ = (
        "J", "CT", "CP", "eta", "T", "Q", "P",
        "r", "a", "ap", "phi", "alpha", "cl", "cd",
        "dT_dr", "dQ_dr", "iterations",
    )

    def __init__(
        self,
        J: float, CT: float, CP: float, eta: float,
        T: float, Q: float, P: float,
        r: NDArray[np.float64],
        a: NDArray[np.float64], ap: NDArray[np.float64],
        phi: NDArray[np.float64], alpha: NDArray[np.float64],
        cl: NDArray[np.float64], cd: NDArray[np.float64],
        dT_dr: NDArray[np.float64], dQ_dr: NDArray[np.float64],
        iterations: int,
    ) -> None:
        self.J = J
        self.CT = CT
        self.CP = CP
        self.eta = eta
        self.T = T
        self.Q = Q
        self.P = P
        self.r = r
        self.a = a
        self.ap = ap
        self.phi = phi
        self.alpha = alpha
        self.cl = cl
        self.cd = cd
        self.dT_dr = dT_dr
        self.dQ_dr = dQ_dr
        self.iterations = iterations

    def __repr__(self) -> str:
        return (
            f"BEMTResult(J={self.J:.4f}, CT={self.CT:.4f}, CP={self.CP:.4f}, "
            f"η={self.eta:.4f}, T={self.T:.3f}, P={self.P:.2f}, "
            f"iters={self.iterations})"
        )

"""2D longitudinal flight dynamics — point-mass Euler integration.

Wires together:
  - WingAero  (VLM3D + 2D polar → CL(α), CD(α) lookup table)
  - PropulsionSystem  (throttle + pitch + airspeed → thrust)
  - Point-mass EOM with ground interaction

Usage
-----
>>> from flight import WingAero, Aircraft, FlightState, simulate
>>> from vlm3d import VLM3D
>>> from polar import naca0012_polar
>>> from propulsion import PropulsionSystem
>>>
>>> vlm = VLM3D.rectangular(span=1.5, chord=0.2, n_span=30)
>>> aero = WingAero.from_vlm(vlm, polar=naca0012_polar())
>>> prop = PropulsionSystem.standard()
>>> ac = Aircraft(aero=aero, propulsion=prop, mass=2.0)
>>>
>>> traj = simulate(ac, dt=0.02, t_max=30.0,
...                 control_fn=takeoff_control(ac))
>>> traj['z']  # altitude history
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from vlm3d import VLM3D

# ---------------------------------------------------------------------------
#  1. WingAero — pre-computed CL(α), CD(α) lookup table
# ---------------------------------------------------------------------------


@dataclass
class WingAero:
    """3D wing aerodynamic lookup table with stall.

    Built from VLM3D (pre-stall CL, CDi) merged with 2D section polar
    (profile drag, post-stall Cl shape).

    Fields
    ------
    alpha_deg : np.ndarray
        Lookup abscissa [deg] (sorted ascending).
    CL : np.ndarray
        Total lift coefficient (3D, with stall).
    CD : np.ndarray
        Total drag = CDi + CDp.
    S : float
        Planform area [m²].
    AR : float
        Aspect ratio.
    label : str
        Human-readable description.
    """

    alpha_deg: NDArray[np.float64]
    CL: NDArray[np.float64]
    CD: NDArray[np.float64]
    S: float
    AR: float
    label: str = ""

    # ------------------------------------------------------------------
    @classmethod
    def from_vlm(
        cls,
        vlm: VLM3D,
        polar=None,  # PolarTable | None
        alpha_range: tuple[float, float] = (-10.0, 25.0),
        n_pts: int = 71,
    ) -> "WingAero":
        """Build the lookup table from a VLM3D wing + optional 2D polar.

        Parameters
        ----------
        vlm : VLM3D
            Configured 3D vortex-lattice solver.
        polar : PolarTable or None
            2D section polar for profile drag and stall behaviour.
            If None: CD = CDi only (inviscid) and CL is purely VLM3D
            (linear, no stall).
        alpha_range : (float, float)
            (min, max) alpha sweep [deg].
        n_pts : int
            Number of sweep points.

        Returns
        -------
        WingAero
        """
        alpha_deg = np.linspace(alpha_range[0], alpha_range[1], n_pts)
        CL_vlm = np.empty(n_pts)
        CDi_vlm = np.empty(n_pts)

        for i, a in enumerate(alpha_deg):
            CL_vlm[i], CDi_vlm[i], _ = vlm.solve(float(a))

        if polar is not None:
            alpha_rad = np.radians(alpha_deg)
            _, cd_2d = polar(alpha_rad)

            # --- pre-stall CL (from VLM3D, 3D-corrected) -----------------
            # --- post-stall CL (from 2D polar shape, scaled to match) ----
            # identify stall angle from the 2D polar Cl peak
            polar_cl, _ = polar(alpha_rad)

            # VLM3D CL slope (linear region, fit over pre-stall range)
            mask_linear = alpha_deg <= 8.0  # well within linear range
            if np.sum(mask_linear) >= 4:
                slope_vlm = np.polyfit(
                    np.radians(alpha_deg[mask_linear]),
                    CL_vlm[mask_linear], 1,
                )[0]
            else:
                slope_vlm = CL_vlm[n_pts // 2] / np.radians(
                    max(abs(alpha_deg[n_pts // 2]), 1e-6)
                )

            # 2D polar Cl slope (fit over α ∈ [−2°, 6°])
            mask_polar_lin = (alpha_deg >= -2.0) & (alpha_deg <= 6.0)
            if np.sum(mask_polar_lin) >= 4:
                slope_polar = np.polyfit(
                    np.radians(alpha_deg[mask_polar_lin]),
                    polar_cl[mask_polar_lin], 1,
                )[0]
            else:
                slope_polar = 2.0 * np.pi

            # join point: where the 2D polar Cl starts to diverge from VLM3D CL
            # (use the α where polar_cl slope drops below 70% of its linear slope)
            scale = slope_vlm / max(slope_polar, 1e-6)
            CL = cls._blend_stall(alpha_deg, CL_vlm, polar_cl, scale)

            # total drag
            CD = CDi_vlm + cd_2d

            label = (
                f"{vlm.span:.1f}m span, AR={vlm.AR:.1f}, "
                f"polar={polar.label}"
            )
        else:
            CL = CL_vlm
            CD = CDi_vlm
            label = f"{vlm.span:.1f}m span, AR={vlm.AR:.1f}, inviscid"

        return cls(
            alpha_deg=alpha_deg,
            CL=CL,
            CD=CD,
            S=vlm.S,
            AR=vlm.AR,
            label=label,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _blend_stall(
        alpha_deg: NDArray[np.float64],
        cl_vlm: NDArray[np.float64],
        polar_cl: NDArray[np.float64],
        scale: float,
    ) -> NDArray[np.float64]:
        """Merge VLM3D pre-stall CL with 2D-polar post-stall shape.

        For α below the stall break, use VLM3D (3D-corrected).  For α
        beyond the break, follow the 2D polar's roll-off scaled to match
        the VLM3D lift-curve slope in the linear region.

        Parameters
        ----------
        alpha_deg, cl_vlm : ndarray
            VLM3D results (linear, no stall).
        polar_cl : ndarray
            2D section Cl at the same α stations.
        scale : float
            Ratio CL_vlm_slope / polar_cl_slope (applies 3D correction
            to the 2D stall shape).

        Returns
        -------
        CL : ndarray
            Blended lift curve.
        """
        n = len(alpha_deg)

        # find stall angle as where the 2D polar Cl reaches its maximum
        idx_stall = int(np.argmax(polar_cl))
        alpha_stall = alpha_deg[idx_stall]

        # pre-stall region: VLM3D dominates (more accurate 3D CL)
        # blend zone: ± 3° around stall, linear crossfade
        blend_half = 3.0  # degrees

        CL = np.empty(n)
        for i in range(n):
            a = alpha_deg[i]
            if a <= alpha_stall - blend_half:
                # pure VLM3D
                CL[i] = cl_vlm[i]
            elif a >= alpha_stall + blend_half:
                # pure 2D polar, 3D-scaled
                delta_polar = polar_cl[i] - polar_cl[idx_stall]
                CL[i] = cl_vlm[idx_stall] + scale * delta_polar
            else:
                # blend zone
                t = (a - (alpha_stall - blend_half)) / (2.0 * blend_half)
                t = np.clip(t, 0.0, 1.0)
                vlm_val = cl_vlm[i]
                delta_polar = polar_cl[i] - polar_cl[idx_stall]
                polar_val = cl_vlm[idx_stall] + scale * delta_polar
                CL[i] = (1.0 - t) * vlm_val + t * polar_val

        return CL

    # ------------------------------------------------------------------
    def __call__(self, alpha_deg: float) -> tuple[float, float]:
        """Lookup (CL, CD) at the given angle of attack [deg].

        Extrapolates beyond the table range using thin-airfoil slope
        for CL and edge-value hold for CD.
        """
        a = np.atleast_1d(np.asarray(alpha_deg, dtype=np.float64))

        cl = np.interp(a, self.alpha_deg, self.CL)
        cd = np.interp(a, self.alpha_deg, self.CD)

        # below range — thin-airfoil extrapolation for CL
        below = a < self.alpha_deg[0]
        if np.any(below):
            d_rad = np.radians(a[below] - self.alpha_deg[0])
            cl[below] = self.CL[0] + 2.0 * np.pi * d_rad
        # above range — hold edge values (already done by np.interp)

        if np.ndim(alpha_deg) == 0:
            return float(cl.item()), float(cd.item())
        return cl, cd  # type: ignore[return-value]

    @property
    def CL_max(self) -> float:
        return float(np.max(self.CL))

    @property
    def alpha_stall_deg(self) -> float:
        return float(self.alpha_deg[np.argmax(self.CL)])


# ---------------------------------------------------------------------------
#  2. FlightState
# ---------------------------------------------------------------------------


@dataclass
class FlightState:
    """State vector for 2D longitudinal point-mass flight dynamics.

    Primary (integrated) variables:  x, z, V, gamma.
    Derived:  theta (control), alpha = theta - gamma.
    """

    x: float = 0.0               # horizontal position [m]
    z: float = 0.0               # altitude [m]  (z <= 0 → on ground)
    V: float = 0.0               # speed [m/s]
    gamma: float = 0.0           # flight-path angle [rad] (+ = climbing)
    theta: float = 0.0           # pitch attitude [rad] (control input)
    throttle: float = 0.0        # current throttle [0, 1]
    time: float = 0.0            # elapsed time [s]

    @property
    def on_ground(self) -> bool:
        return self.z <= 0.0

    @property
    def alpha(self) -> float:
        """Angle of attack [rad]."""
        return self.theta - self.gamma

    @property
    def alpha_deg(self) -> float:
        return np.degrees(self.alpha)


# ---------------------------------------------------------------------------
#  3. Aircraft — parameter bundle
# ---------------------------------------------------------------------------


@dataclass
class Aircraft:
    """Bundles wing aero, propulsion, and mass parameters.

    Parameters
    ----------
    aero : WingAero
        Wing CL(α), CD(α) lookup table.
    propulsion : PropulsionSystem
        Thrust T(V, throttle, pitch_deg).
    mass : float
        Aircraft mass [kg].
    rho : float
        Air density [kg/m³].
    mu_roll : float
        Rolling friction coefficient.
    theta_ground_deg : float
        Pitch attitude of the aircraft resting on landing gear [deg].
    pitch_deg : float
        Propeller blade pitch [deg] (for variable-pitch; fixed default).
    """

    aero: WingAero
    propulsion: "PropulsionSystem"   # forward ref — avoid circular import
    mass: float
    rho: float = 1.225
    mu_roll: float = 0.03
    theta_ground_deg: float = 2.0
    pitch_deg: float = 4.0          # propeller blade pitch

    # set in __post_init__
    V_stall: float = field(init=False)

    def __post_init__(self) -> None:
        g = 9.81
        CL_max = self.aero.CL_max
        self.V_stall = float(
            np.sqrt(2.0 * self.mass * g / (self.rho * self.aero.S * max(CL_max, 0.1)))
        )

    @property
    def weight(self) -> float:
        return self.mass * 9.81


# ---------------------------------------------------------------------------
#  4. Simulation engine
# ---------------------------------------------------------------------------

# Type alias for the control callback
ControlFn = Callable[[FlightState], tuple[float, float]]
# control_fn(state) → (throttle [0,1], theta_command [rad])

# Minimum speed passed to the propulsion model — BEMT is singular at
# exactly V=0 (sin(phi) denominator).  0.01 m/s is negligibly different
# from zero for flight dynamics but keeps the solver well-conditioned.
_V_PROP_MIN = 0.01


def _thrust(ac: "Aircraft", V: float, throttle: float) -> float:
    """Safe thrust call — clamps V to avoid BEMT singularity at V=0."""
    return float(
        max(ac.propulsion.thrust(max(V, _V_PROP_MIN), throttle, ac.pitch_deg), 0.0)
    )


def simulate(
    aircraft: Aircraft,
    dt: float = 0.01,
    t_max: float = 60.0,
    initial_state: FlightState | None = None,
    control_fn: ControlFn | None = None,
    *,
    V_min: float = 0.1,
    progress: bool = False,
) -> dict[str, NDArray[np.float64]]:
    """Run a 2D longitudinal flight simulation (explicit Euler).

    Parameters
    ----------
    aircraft : Aircraft
        Bundled parameters.
    dt : float
        Fixed time step [s].
    t_max : float
        Total simulation time [s].
    initial_state : FlightState or None
        Starting state.  Defaults to rest on the ground with
        theta = theta_ground.
    control_fn : callable or None
        ``control_fn(state) → (throttle, theta_cmd_rad)``.
        If None, holds throttle=0 and theta at theta_ground.
    V_min : float
        Speed below which dγ/dt is clamped to zero (avoids 1/V
        singularity).
    progress : bool
        If True, print key events (liftoff, stall).

    Returns
    -------
    dict[str, np.ndarray]
        Trajectory arrays with keys:
        'time', 'x', 'z', 'V', 'gamma', 'theta', 'alpha',
        'throttle', 'L', 'D', 'T', 'on_ground', 'CL', 'CD'.
    """
    g = 9.81
    mass = aircraft.mass
    rho = aircraft.rho
    S = aircraft.aero.S

    # initial condition
    if initial_state is None:
        st = FlightState(theta=np.radians(aircraft.theta_ground_deg))
    else:
        st = FlightState(
            x=initial_state.x,
            z=initial_state.z,
            V=initial_state.V,
            gamma=initial_state.gamma,
            theta=initial_state.theta,
            throttle=initial_state.throttle,
            time=initial_state.time,
        )

    # initialise ground constraint
    airborne = False
    if st.z > 0.0:
        airborne = True

    # ---- pre-allocate trajectory storage -----------------------------------
    n_steps = int(t_max / dt) + 1
    keys = [
        "time", "x", "z", "V", "gamma", "theta", "alpha",
        "throttle", "L", "D", "T", "on_ground", "CL", "CD",
    ]
    traj: dict[str, NDArray[np.float64]] = {
        k: np.empty(n_steps) for k in keys
    }

    # ---- integration loop --------------------------------------------------
    step = 0
    while st.time < t_max + dt / 2:
        # -- record current state --
        if step >= n_steps:
            break

        # compute aero forces for recording
        CL_val, CD_val = aircraft.aero(st.alpha_deg)
        L_val = 0.5 * rho * st.V ** 2 * S * CL_val
        D_val = 0.5 * rho * st.V ** 2 * S * CD_val
        T_val = _thrust(aircraft, st.V, st.throttle)

        traj["time"][step] = st.time
        traj["x"][step] = st.x
        traj["z"][step] = st.z
        traj["V"][step] = st.V
        traj["gamma"][step] = st.gamma
        traj["theta"][step] = st.theta
        traj["alpha"][step] = st.alpha
        traj["throttle"][step] = st.throttle
        traj["L"][step] = L_val
        traj["D"][step] = D_val
        traj["T"][step] = T_val
        traj["on_ground"][step] = 1.0 if (not airborne and st.z <= 0.0) else 0.0
        traj["CL"][step] = CL_val
        traj["CD"][step] = CD_val

        # -- controls --------------------------------------------------------
        if control_fn is not None:
            thr_cmd, theta_cmd = control_fn(st)
            st.throttle = float(np.clip(thr_cmd, 0.0, 1.0))
        else:
            theta_cmd = np.radians(aircraft.theta_ground_deg)

        # ---- ground model --------------------------------------------------
        if not airborne and st.z <= 0.0:
            # enforce kinematic constraint — but allow nose-up rotation
            st.z = 0.0
            st.gamma = 0.0
            # honour pilot pitch command above the landing-gear minimum
            st.theta = max(theta_cmd, np.radians(aircraft.theta_ground_deg))

            # re-compute aero forces with constrained gamma
            CL_val, CD_val = aircraft.aero(st.alpha_deg)
            D_val = 0.5 * rho * st.V ** 2 * S * CD_val
            L_val = 0.5 * rho * st.V ** 2 * S * CL_val
            T_val = _thrust(aircraft, st.V, st.throttle)

            # normal force on ground
            N = aircraft.weight - L_val
            if N < 0.0:
                N = 0.0

            # ground friction
            friction = aircraft.mu_roll * N

            # forward acceleration
            net_forward = T_val * np.cos(st.alpha) - D_val - friction
            if st.V <= V_min and net_forward < 0.0:
                net_forward = 0.0  # static friction holds

            dVdt = net_forward / mass
            dgammadt = 0.0
            dzdt = 0.0
            dxdt = st.V * np.cos(st.gamma)

            # -- liftoff check --
            if L_val > aircraft.weight and st.V > aircraft.V_stall:
                airborne = True
                st.z = 0.01  # nudge clear of ground to avoid oscillation
                if progress:
                    print(
                        f"  LIFTOFF  t={st.time:.2f}s  V={st.V:.1f}m/s  "
                        f"L={L_val:.2f}N  W={aircraft.weight:.1f}N"
                    )

        else:
            # ---- airborne EOM ----------------------------------------------
            st.theta = theta_cmd  # pilot has full pitch authority in the air
            alpha = st.alpha

            CL_val, CD_val = aircraft.aero(st.alpha_deg)
            L_val = 0.5 * rho * st.V ** 2 * S * CL_val
            D_val = 0.5 * rho * st.V ** 2 * S * CD_val
            T_val = _thrust(aircraft, st.V, st.throttle)

            cos_a = np.cos(alpha)
            sin_a = np.sin(alpha)

            # speed rate (wind axes, point-mass)
            dVdt = (
                T_val * cos_a - D_val - aircraft.weight * np.sin(st.gamma)
            ) / mass

            # flight-path-angle rate (with 1/V guard)
            if st.V > V_min:
                dgammadt = (
                    T_val * sin_a + L_val - aircraft.weight * np.cos(st.gamma)
                ) / (mass * st.V)
            else:
                dgammadt = 0.0

            # kinematic rates
            dxdt = st.V * np.cos(st.gamma)
            dzdt = st.V * np.sin(st.gamma)

            # -- stall detection (for progress output) --
            if progress and st.alpha_deg > aircraft.aero.alpha_stall_deg:
                # only report the first time we cross stall
                prev_idx = max(step - 1, 0)
                if step == 0 or traj["alpha"][prev_idx] < np.radians(
                    aircraft.aero.alpha_stall_deg
                ):
                    print(
                        f"  STALL    t={st.time:.2f}s  α={st.alpha_deg:.1f}°  "
                        f"CL={CL_val:.3f}"
                    )

            # -- crash / ground-contact check --
            # Only re-engage ground model on a clear descent through
            # the surface (Vz < −1 m/s), not on the tiny oscillation
            # that happens right after liftoff.
            if st.z <= 0.0 and dzdt < -1.0:
                airborne = False
                if progress:
                    print(
                        f"  TOUCHDOWN  t={st.time:.2f}s  V={st.V:.1f}m/s  "
                        f"Vz={dzdt:.1f}m/s"
                    )

        # ---- Euler step ----------------------------------------------------
        st.time += dt
        st.x += dxdt * dt
        st.z += dzdt * dt
        st.V += dVdt * dt
        st.gamma += dgammadt * dt
        st.V = max(st.V, 0.0)  # prevent negative speed (no reverse thrust)

        step += 1

    # ---- trim arrays to actual length --------------------------------------
    for k in keys:
        traj[k] = traj[k][:step]

    if progress and step > 0:
        idx_last = step - 1
        print(
            f"\n  Final:  t={traj['time'][idx_last]:.1f}s  "
            f"x={traj['x'][idx_last]:.1f}m  z={traj['z'][idx_last]:.1f}m  "
            f"V={traj['V'][idx_last]:.1f}m/s"
        )

    return traj

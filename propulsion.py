"""Propulsion black-box: throttle + pitch + airspeed → thrust + power.

Wraps BEMT blade-element solver, experimental airfoil polar, and a
simple motor model behind a single-call interface suitable for a
flight-dynamics loop.

Usage
-----
>>> prop = PropulsionSystem.standard()          # NACA 0012, 10-inch, 2-blade
>>> op = prop.solve(V=15.0, throttle=0.7, pitch_deg=4.0)
>>> op.thrust   # N
>>> op.power    # W
>>> curve = prop.curve(V_range=(5, 30), n_pts=40, throttle=0.8, pitch_deg=4.0)
>>> curve.V, curve.T, curve.P   # arrays for plotting
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from bemt import BEMTSolver

if TYPE_CHECKING:
    from polar import PolarTable


# ═══════════════════════════════════════════════════════════════════
#  Specs (simple value objects)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MotorSpec:
    """Motor / engine characteristics.

    Attributes
    ----------
    max_rpm : float
        Shaft speed at full throttle [RPM].
    idle_rpm : float
        Shaft speed at zero throttle [RPM].
    max_power : float or None
        Maximum continuous shaft power [W].  ``None`` = unlimited.
    """

    max_rpm: float = 9550.0
    idle_rpm: float = 0.0
    max_power: float | None = None

    def rpm(self, throttle: float) -> float:
        """Throttle [0, 1] → shaft speed [RPM] (linear map)."""
        t = float(np.clip(throttle, 0.0, 1.0))
        return self.idle_rpm + t * (self.max_rpm - self.idle_rpm)


@dataclass
class PropGeom:
    """Propeller geometry.

    Attributes
    ----------
    diameter : float
        Tip diameter [m].
    n_blades : int
        Number of blades.
    chord : float or np.ndarray
        Blade chord [m] at each radial station (scalar → constant chord).
    J_design : float
        Design advance ratio; sets the ideal twist distribution
        ``θ(r) = arctan(J_design·R / (π·r)) + pitch_deg``.
    hub_diameter : float or None
        Hub diameter [m].  Defaults to 15 % of tip diameter.
    """

    diameter: float = 0.254          # ~10 inch
    n_blades: int = 2
    chord: float = 0.015
    J_design: float = 0.7
    hub_diameter: float | None = None

    @property
    def R(self) -> float:
        return self.diameter / 2.0

    @property
    def Rh(self) -> float:
        if self.hub_diameter is not None:
            return self.hub_diameter / 2.0
        return 0.15 * self.R


# ═══════════════════════════════════════════════════════════════════
#  Result types
# ═══════════════════════════════════════════════════════════════════

@dataclass
class OperatingPoint:
    """Single propeller operating point."""

    V: float
    throttle: float
    pitch_deg: float
    rpm: float
    J: float
    thrust: float             # [N]
    torque: float             # [N·m]
    power: float              # shaft power [W]
    efficiency: float         # [0, 1]
    CT: float
    CP: float
    converged: bool = True
    power_limited: bool = False   # True if motor power limit is hit

    def __repr__(self) -> str:
        return (
            f"OpPoint(V={self.V:.1f}, thr={self.throttle:.2f}, "
            f"rpm={self.rpm:.0f}, pitch={self.pitch_deg:.1f}°, "
            f"T={self.thrust:.3f} N, P={self.power:.1f} W, "
            f"η={self.efficiency * 100:.1f}%)"
        )


@dataclass
class PerformanceCurve:
    """Thrust & power vs airspeed at fixed throttle + pitch."""

    V: NDArray[np.float64]       # airspeed [m/s]
    thrust: NDArray[np.float64]  # [N]
    power: NDArray[np.float64]   # shaft power [W]
    torque: NDArray[np.float64]  # [N·m]
    efficiency: NDArray[np.float64]
    J: NDArray[np.float64]
    rpm: float                   # constant RPM across the sweep
    throttle: float
    pitch_deg: float

    # convenience
    def eta_pct(self) -> NDArray[np.float64]:
        return self.efficiency * 100.0


# ═══════════════════════════════════════════════════════════════════
#  Main class
# ═══════════════════════════════════════════════════════════════════

class PropulsionSystem:
    """Black-box propeller propulsion model.

    Composes a motor model, propeller geometry, BEMT solver, and 2D
    airfoil polar.  Exposes a three-knob interface (throttle, pitch,
    airspeed) → thrust + power.

    Parameters
    ----------
    prop : PropGeom
        Propeller geometry.
    motor : MotorSpec
        Motor / engine specification.
    polar : PolarTable or None
        Section Cl/Cd polar.  ``None`` uses the analytical
        ``Cd₀ + k·Cl²`` model built into BEMTSolver.
    rho : float
        Air density [kg/m³].
    n_stations : int
        Radial resolution for BEMT.
    """

    def __init__(
        self,
        prop: PropGeom | None = None,
        motor: MotorSpec | None = None,
        polar: PolarTable | None = None,
        *,
        rho: float = 1.225,
        n_stations: int = 40,
    ) -> None:
        self.prop = prop or PropGeom()
        self.motor = motor or MotorSpec()
        self._polar = polar
        self.rho = rho
        self.n_stations = n_stations

        # cache: one BEMTSolver per (throttle, pitch) key
        self._solver_cache: dict[tuple[float, float], BEMTSolver] = {}
        self._last_key: tuple[float, float] | None = None

    # ------------------------------------------------------------------
    @classmethod
    def standard(cls) -> "PropulsionSystem":
        """Factory: standard NACA 0012 10-inch propeller."""
        from polar import naca0012_polar

        return cls(
            prop=PropGeom(),
            motor=MotorSpec(),
            polar=naca0012_polar(),
        )

    # ------------------------------------------------------------------
    #  internal: solver management
    # ------------------------------------------------------------------
    def _get_solver(self, throttle: float, pitch_deg: float) -> BEMTSolver:
        """Return a BEMTSolver configured for the given throttle & pitch.

        Solvers are cached so that warm-start induction factors
        persist across ``solve()`` calls at different airspeeds.
        """
        key = (round(throttle, 6), round(pitch_deg, 6))
        if key not in self._solver_cache:
            rpm = self.motor.rpm(throttle)
            omega = rpm * (2.0 * np.pi / 60.0)           # RPM → rad/s

            solver = BEMTSolver(
                R=self.prop.R,
                Rh=self.prop.Rh,
                B=self.prop.n_blades,
                omega=omega,
                chord=self.prop.chord,
                J_design=self.prop.J_design,
                alpha_design_deg=pitch_deg,             # ← the control knob
                rho=self.rho,
                n_stations=self.n_stations,
                polar_table=self._polar,
                relax=0.3,
                max_iter=500,
                tol=1e-6,
            )
            self._solver_cache[key] = solver

        self._last_key = key
        return self._solver_cache[key]

    # ------------------------------------------------------------------
    #  single-point query  (the flight-loop entry point)
    # ------------------------------------------------------------------
    def solve(
        self,
        V: float,
        throttle: float,
        pitch_deg: float,
    ) -> OperatingPoint:
        """Evaluate propeller at a single flight condition.

        Parameters
        ----------
        V : float
            Freestream / flight velocity [m/s].
        throttle : float
            Throttle position [0, 1].
        pitch_deg : float
            Blade pitch angle [deg] (replaces the design α in the
            BEMT twist formula).

        Returns
        -------
        OperatingPoint
        """
        solver = self._get_solver(throttle, pitch_deg)
        rpm = self.motor.rpm(throttle)

        try:
            res = solver.solve(V)
        except Exception:
            return OperatingPoint(
                V=V, throttle=throttle, pitch_deg=pitch_deg,
                rpm=rpm, J=float("nan"),
                thrust=0.0, torque=0.0, power=0.0, efficiency=0.0,
                CT=0.0, CP=0.0,
                converged=False,
            )

        # motor power limit check
        power_limited = False
        power = res.P
        if self.motor.max_power is not None and power > self.motor.max_power:
            power_limited = True
            # scale thrust proportionally (crude but serviceable —
            # a proper fix would iterate with torque-limiting BC)
            scale = self.motor.max_power / power
            power = self.motor.max_power
            thrust = res.T * scale
            torque = res.Q * scale
        else:
            thrust = res.T
            torque = res.Q

        return OperatingPoint(
            V=V, throttle=throttle, pitch_deg=pitch_deg, rpm=rpm,
            J=res.J, thrust=thrust, torque=torque, power=power,
            efficiency=res.eta, CT=res.CT, CP=res.CP,
            converged=True, power_limited=power_limited,
        )

    # ------------------------------------------------------------------
    #  convenience short-hands
    # ------------------------------------------------------------------
    def thrust(self, V: float, throttle: float, pitch_deg: float) -> float:
        """Return thrust [N] at the given condition."""
        return self.solve(V, throttle, pitch_deg).thrust

    def power(self, V: float, throttle: float, pitch_deg: float) -> float:
        """Return shaft power [W] at the given condition."""
        return self.solve(V, throttle, pitch_deg).power

    # ------------------------------------------------------------------
    #  performance curve  (airspeed sweep at fixed throttle + pitch)
    # ------------------------------------------------------------------
    def curve(
        self,
        V_range: tuple[float, float] = (1.0, 30.0),
        n_pts: int = 50,
        throttle: float = 0.7,
        pitch_deg: float = 4.0,
    ) -> PerformanceCurve:
        """Sweep airspeed at fixed throttle + pitch → T(V), P(V) curves.

        Parameters
        ----------
        V_range : (V_min, V_max)
            Airspeed sweep bounds [m/s].
        n_pts : int
            Number of points in the sweep.
        throttle : float
            Throttle setting [0, 1].
        pitch_deg : float
            Blade pitch [deg].

        Returns
        -------
        PerformanceCurve
            Arrays V, thrust, power, torque, efficiency, J.
        """
        V_vals = np.linspace(V_range[0], V_range[1], n_pts)
        n = len(V_vals)

        T_arr = np.empty(n)
        P_arr = np.empty(n)
        Q_arr = np.empty(n)
        eta_arr = np.empty(n)
        J_arr = np.empty(n)

        rpm = self.motor.rpm(throttle)

        for i, V in enumerate(V_vals):
            op = self.solve(V, throttle, pitch_deg)
            T_arr[i] = op.thrust
            P_arr[i] = op.power
            Q_arr[i] = op.torque
            eta_arr[i] = op.efficiency
            J_arr[i] = op.J

        return PerformanceCurve(
            V=V_vals,
            thrust=T_arr,
            power=P_arr,
            torque=Q_arr,
            efficiency=eta_arr,
            J=J_arr,
            rpm=rpm,
            throttle=throttle,
            pitch_deg=pitch_deg,
        )

    # ------------------------------------------------------------------
    #  static thrust  (V = 0)
    # ------------------------------------------------------------------
    def static_thrust(self, throttle: float, pitch_deg: float) -> float:
        """Return thrust at V = 0 (static / take-off condition)."""
        return self.solve(V=0.01, throttle=throttle, pitch_deg=pitch_deg).thrust

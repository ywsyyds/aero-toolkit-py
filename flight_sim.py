"""2D longitudinal flight dynamics — demo scenarios.

Demonstrates the flight.py module with three scenarios:
  1) Takeoff — ground roll → rotation → climb
  2) Stall   — excessive pitch → CL cliff → sink
  3) Cruise  — steady-state trim verification
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flight import Aircraft, FlightState, WingAero, simulate
from propulsion import PropulsionSystem
from vlm3d import VLM3D
from polar import naca0012_polar

G = 9.81


# ===================================================================
#  Build the shared test aircraft
# ===================================================================
def make_test_aircraft() -> Aircraft:
    """Build a small RC-scale test aircraft.

    Wing:   rectangular, span=1.5 m, chord=0.2 m  →  S=0.3 m², AR=7.5
    Mass:   2.0 kg
    Prop:   10-inch, 2-blade  (PropulsionSystem.standard)
    """
    print("Building test aircraft …")
    print("  Wing:  rectangular, span=1.5 m, chord=0.2 m")

    vlm = VLM3D.rectangular(span=1.5, chord=0.2, n_span=30)
    polar = naca0012_polar()
    aero = WingAero.from_vlm(vlm, polar=polar)
    prop = PropulsionSystem.standard()

    ac = Aircraft(
        aero=aero,
        propulsion=prop,
        mass=2.0,
        rho=1.225,
        mu_roll=0.03,
        theta_ground_deg=2.0,
        pitch_deg=4.0,
    )

    print(f"  S={aero.S:.3f} m²,  AR={aero.AR:.1f}")
    print(f"  CL_max = {aero.CL_max:.3f}  at α = {aero.alpha_stall_deg:.1f}°")
    print(f"  V_stall = {ac.V_stall:.1f} m/s")
    print(f"  Weight = {ac.weight:.1f} N")
    print()

    return ac


# ===================================================================
#  Scenario 1 — Takeoff roll, rotate, climb
# ===================================================================
def scenario_takeoff(ac: Aircraft) -> dict:
    """Throttle up from rest, accelerate on runway, rotate at V_stall,
    then climb."""

    print("=" * 60)
    print("Scenario 1 — TAKEOFF & CLIMB")
    print("=" * 60)

    theta_ground = np.radians(ac.theta_ground_deg)
    theta_climb = np.radians(8.0)   # climb pitch attitude
    rotate_speed = ac.V_stall * 1.1  # rotate slightly above stall

    # state machine
    rotation_started = False
    rotation_t0 = 0.0
    rotation_duration = 1.5  # seconds to complete rotation

    def control(state: FlightState) -> tuple[float, float]:
        nonlocal rotation_started, rotation_t0

        # throttle: ramp to 0.9 over first 2 seconds
        if state.time < 2.0:
            thr = 0.15 + 0.75 * (state.time / 2.0)
        else:
            thr = 0.9

        # pitch: hold ground attitude until rotate speed, then rotate
        if state.on_ground and state.V < rotate_speed:
            theta_cmd = theta_ground
        else:
            if not rotation_started:
                rotation_started = True
                rotation_t0 = state.time
                print(f"  ROTATE   t={state.time:.2f}s  V={state.V:.1f}m/s")

            frac = min((state.time - rotation_t0) / rotation_duration, 1.0)
            theta_cmd = theta_ground + frac * (theta_climb - theta_ground)

        return thr, theta_cmd

    # start from rest on the ground
    st0 = FlightState(theta=theta_ground)

    traj = simulate(
        ac, dt=0.01, t_max=25.0, initial_state=st0,
        control_fn=control, progress=True,
    )
    return traj


# ===================================================================
#  Scenario 2 — Stall from cruise
# ===================================================================
def _find_trim(
    ac: Aircraft, V: float, throttle: float,
) -> tuple[float, float]:
    """Bisect alpha for level flight (L=W) at the given V and throttle.

    Returns (theta_trim_rad, gamma_trim).  For level flight γ=0,
    so θ_trim = α_trim.
    """
    W = ac.weight
    rho = ac.rho
    S = ac.aero.S
    CL_req = W / (0.5 * rho * V ** 2 * S)

    a_lo, a_hi = -5.0, ac.aero.alpha_stall_deg
    for _ in range(40):
        a_mid = (a_lo + a_hi) / 2.0
        cl_mid, _ = ac.aero(a_mid)
        if cl_mid < CL_req:
            a_lo = a_mid
        else:
            a_hi = a_mid

    alpha_trim_deg = (a_lo + a_hi) / 2.0
    _, cd = ac.aero(alpha_trim_deg)
    D = 0.5 * rho * V ** 2 * S * cd
    T = max(ac.propulsion.thrust(V, throttle, ac.pitch_deg), 0.0)

    return np.radians(alpha_trim_deg), T - D


def scenario_stall(ac: Aircraft) -> dict:
    """Start in steady cruise, then pitch up aggressively → stall."""

    print("\n" + "=" * 60)
    print("Scenario 2 — STALL")
    print("=" * 60)

    V_cruise = 13.0
    throttle_cruise = 0.46

    theta_trim, delta = _find_trim(ac, V_cruise, throttle_cruise)
    alpha_trim_deg = np.degrees(theta_trim)  # γ=0 in level flight
    print(f"  Trim: V={V_cruise:.1f} m/s  θ={alpha_trim_deg:.1f}°  "
          f"thr={throttle_cruise}  (T-D = {delta:+.3f} N)")

    pitch_up_start = 2.0      # start pitching up at t=2s
    pitch_up_duration = 3.0   # ramp to stall angle over 3s
    theta_stall = np.radians(22.0)   # well past stall

    def control(state: FlightState) -> tuple[float, float]:
        if state.time < pitch_up_start:
            theta_cmd = theta_trim
        else:
            frac = min(
                (state.time - pitch_up_start) / pitch_up_duration, 1.0,
            )
            theta_cmd = theta_trim + frac * (theta_stall - theta_trim)

        return throttle_cruise, theta_cmd

    # start at cruise altitude
    st0 = FlightState(
        V=V_cruise,
        z=50.0,
        theta=theta_trim,
        gamma=0.0,
        throttle=throttle_cruise,
    )

    print(f"  Pitching to {np.degrees(theta_stall):.0f}° "
          f"over {pitch_up_duration}s")

    traj = simulate(
        ac, dt=0.01, t_max=15.0, initial_state=st0,
        control_fn=control, progress=True,
    )
    return traj


# ===================================================================
#  Scenario 3 — Cruise trim (steady-state check)
# ===================================================================
def scenario_cruise(ac: Aircraft) -> dict:
    """Hold constant controls and verify steady flight."""

    print("\n" + "=" * 60)
    print("Scenario 3 — CRUISE (trim check)")
    print("=" * 60)

    V_cruise = 13.0
    throttle_cruise = 0.46

    theta_trim, delta = _find_trim(ac, V_cruise, throttle_cruise)
    alpha_trim_deg = np.degrees(theta_trim)

    def control(state: FlightState) -> tuple[float, float]:
        return throttle_cruise, theta_trim

    st0 = FlightState(
        V=V_cruise,
        z=50.0,
        theta=theta_trim,
        gamma=0.0,
        throttle=throttle_cruise,
    )

    print(f"  V={V_cruise:.1f} m/s  z=50 m  "
          f"θ={alpha_trim_deg:.1f}°  throttle={throttle_cruise}  "
          f"(T-D = {delta:+.3f} N)")

    traj = simulate(
        ac, dt=0.01, t_max=10.0, initial_state=st0,
        control_fn=control, progress=False,
    )

    # report drift
    V_vals = traj["V"]
    z_vals = traj["z"]
    print(f"  V drift: {V_vals[-1] - V_vals[0]:+.2f} m/s over {traj['time'][-1]:.0f}s")
    print(f"  z drift: {z_vals[-1] - z_vals[0]:+.2f} m")

    return traj


# ===================================================================
#  Plotting
# ===================================================================
C = {
    "blue": "#2C5F8A",
    "red": "#D4604E",
    "green": "#4CAF50",
    "orange": "#FF9800",
    "grey": "#999999",
}


def plot_takeoff(traj: dict, ac: Aircraft) -> None:
    """Takeoff scenario — 4-panel diagnostic figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax_traj, ax_state), (ax_forces, ax_polar) = axes  # type: ignore[misc]

    t = traj["time"]
    mask_air = ~traj["on_ground"].astype(bool) if "on_ground" in traj else np.zeros_like(t, dtype=bool)

    # — a) x-z flight path —
    ax_traj.plot(traj["x"], traj["z"], "-", color=C["blue"], lw=1.5)
    # mark liftoff
    if np.any(mask_air):
        idx_lo = int(np.argmax(mask_air))
        ax_traj.plot(traj["x"][idx_lo], traj["z"][idx_lo], "o",
                     color=C["red"], ms=8, label="liftoff")
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("z [m]")
    ax_traj.set_title("Flight Path (side view)")
    ax_traj.legend(fontsize="small")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_aspect("equal")

    # — b) V and α vs time —
    ax_v = ax_state
    ax_v.plot(t, traj["V"], "-", color=C["blue"], lw=1.5, label="V [m/s]")
    ax_v.axhline(ac.V_stall, color=C["red"], ls="--", lw=1,
                 label=f"V_stall = {ac.V_stall:.1f} m/s")
    ax_alpha = ax_v.twinx()
    ax_alpha.plot(t, np.degrees(traj["alpha"]), "-", color=C["orange"],
                  lw=1.2, label="α [deg]")
    ax_alpha.axhline(ac.aero.alpha_stall_deg, color=C["red"], ls=":",
                     lw=1, label=f"α_stall = {ac.aero.alpha_stall_deg:.1f}°")
    ax_v.set_xlabel("Time [s]")
    ax_v.set_ylabel("V [m/s]", color=C["blue"])
    ax_alpha.set_ylabel("α [deg]", color=C["orange"])
    ax_v.set_title("Speed & Angle of Attack")
    lines_v, labels_v = ax_v.get_legend_handles_labels()
    lines_a, labels_a = ax_alpha.get_legend_handles_labels()
    ax_v.legend(lines_v + lines_a, labels_v + labels_a,
                fontsize="x-small", loc="center right")
    ax_v.grid(True, alpha=0.3)

    # — c) Forces —
    ax_forces.plot(t, traj["L"], "-", color=C["blue"], lw=1.2, label="Lift L")
    ax_forces.plot(t, traj["D"], "-", color=C["red"], lw=1.2, label="Drag D")
    ax_forces.plot(t, traj["T"], "-", color=C["green"], lw=1.2, label="Thrust T")
    ax_forces.axhline(ac.weight, color=C["grey"], ls="--", lw=1,
                      label=f"Weight = {ac.weight:.1f} N")
    ax_forces.set_xlabel("Time [s]")
    ax_forces.set_ylabel("Force [N]")
    ax_forces.set_title("Forces")
    ax_forces.legend(fontsize="small")
    ax_forces.grid(True, alpha=0.3)

    # — d) CL vs α (operating point overlaid on static curve) —
    ax_polar.plot(ac.aero.alpha_deg, ac.aero.CL, "-", color=C["grey"],
                  lw=1.5, alpha=0.5, label="CL(α) static")
    ax_polar.plot(ac.aero.alpha_deg, ac.aero.CD * 10, "--", color=C["grey"],
                  lw=1, alpha=0.4, label="10×CD(α) static")
    # flight data (airborne only)
    if np.any(mask_air):
        alpha_air = np.degrees(traj["alpha"][mask_air])
        cl_air = traj["CL"][mask_air]
        points = np.linspace(0, len(alpha_air) - 1, min(100, len(alpha_air))).astype(int)
        ax_polar.scatter(alpha_air[points], cl_air[points], s=8,
                         c=t[mask_air][points], cmap="viridis", zorder=5)
    ax_polar.axvline(ac.aero.alpha_stall_deg, color=C["red"], ls=":",
                     lw=1, label=f"α_stall = {ac.aero.alpha_stall_deg:.1f}°")
    ax_polar.set_xlabel("α [deg]")
    ax_polar.set_ylabel("CL")
    ax_polar.set_title("CL(α) — Static Curve + Flight Trace")
    ax_polar.legend(fontsize="x-small")
    ax_polar.grid(True, alpha=0.3)

    fig.suptitle("Takeoff & Climb", fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig("flight_takeoff.png", dpi=150)
    print("  Figure saved → flight_takeoff.png")
    plt.close(fig)


def plot_stall(traj: dict, ac: Aircraft) -> None:
    """Stall scenario — 4-panel diagnostic figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax_traj, ax_state), (ax_forces, ax_polar) = axes  # type: ignore[misc]

    t = traj["time"]

    # — a) x-z trajectory —
    ax_traj.plot(traj["x"], traj["z"], "-", color=C["blue"], lw=1.5)
    # mark start
    ax_traj.plot(traj["x"][0], traj["z"][0], "o", color=C["green"], ms=8,
                 label="start")
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("z [m]")
    ax_traj.set_title("Flight Path (side view)")
    ax_traj.legend(fontsize="small")
    ax_traj.grid(True, alpha=0.3)

    # — b) V and α vs time —
    ax_v = ax_state
    ax_v.plot(t, traj["V"], "-", color=C["blue"], lw=1.5, label="V [m/s]")
    ax_alpha = ax_v.twinx()
    ax_alpha.plot(t, np.degrees(traj["alpha"]), "-", color=C["orange"],
                  lw=1.2, label="α [deg]")
    ax_alpha.axhline(ac.aero.alpha_stall_deg, color=C["red"], ls=":",
                     lw=1.5, label=f"α_stall = {ac.aero.alpha_stall_deg:.1f}°")
    ax_v.set_xlabel("Time [s]")
    ax_v.set_ylabel("V [m/s]", color=C["blue"])
    ax_alpha.set_ylabel("α [deg]", color=C["orange"])
    ax_v.set_title("Speed & Angle of Attack")
    lines_v, labels_v = ax_v.get_legend_handles_labels()
    lines_a, labels_a = ax_alpha.get_legend_handles_labels()
    ax_v.legend(lines_v + lines_a, labels_v + labels_a,
                fontsize="x-small")
    ax_v.grid(True, alpha=0.3)

    # — c) CL and CD vs time —
    ax_forces.plot(t, traj["CL"], "-", color=C["blue"], lw=1.5, label="CL")
    ax_forces.plot(t, traj["CD"] * 10, "--", color=C["red"], lw=1.2,
                   label="10×CD")
    ax_forces.axhline(ac.aero.CL_max, color=C["grey"], ls=":", lw=0.8,
                      label=f"CL_max = {ac.aero.CL_max:.2f}")
    ax_forces.set_xlabel("Time [s]")
    ax_forces.set_ylabel("Coefficient")
    ax_forces.set_title("CL & CD During Stall")
    ax_forces.legend(fontsize="small")
    ax_forces.grid(True, alpha=0.3)

    # — d) CL vs α with flight trace —
    ax_polar.plot(ac.aero.alpha_deg, ac.aero.CL, "-", color=C["grey"],
                  lw=1.5, alpha=0.5, label="CL(α) static")
    # flight points
    alpha_deg = np.degrees(traj["alpha"])
    cl = traj["CL"]
    skip = max(1, len(t) // 120)
    pts = np.linspace(0, len(t) - 1, 120).astype(int)
    sc = ax_polar.scatter(alpha_deg[pts], cl[pts], s=10,
                          c=t[pts], cmap="coolwarm", zorder=5)
    cbar = plt.colorbar(sc, ax=ax_polar)
    cbar.set_label("Time [s]")
    ax_polar.axvline(ac.aero.alpha_stall_deg, color=C["red"], ls=":",
                     lw=1, label=f"α_stall = {ac.aero.alpha_stall_deg:.1f}°")
    ax_polar.set_xlabel("α [deg]")
    ax_polar.set_ylabel("CL")
    ax_polar.set_title("CL(α) — Stall Trace")
    ax_polar.legend(fontsize="x-small", loc="upper right")
    ax_polar.grid(True, alpha=0.3)

    fig.suptitle("Stall from Cruise", fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig("flight_stall.png", dpi=150)
    print("  Figure saved → flight_stall.png")
    plt.close(fig)


def plot_cruise(traj: dict, ac: Aircraft) -> None:
    """Cruise trim — 2-panel verification."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    t = traj["time"]

    ax1.plot(t, traj["V"], "-", color=C["blue"], lw=1.5, label="V [m/s]")
    ax1.set_ylabel("V [m/s]", color=C["blue"])
    ax_z = ax1.twinx()
    ax_z.plot(t, traj["z"], "-", color=C["green"], lw=1.2, label="z [m]")
    ax_z.set_ylabel("z [m]", color=C["green"])
    ax1.set_xlabel("Time [s]")
    ax1.set_title("Speed & Altitude")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax_z.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize="small")
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, np.degrees(traj["gamma"]), "-", color=C["blue"],
             lw=1.2, label="γ [deg]")
    ax2.plot(t, np.degrees(traj["alpha"]), "-", color=C["orange"],
             lw=1.2, label="α [deg]")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Angle [deg]")
    ax2.set_title("Flight-Path & Angle of Attack")
    ax2.legend(fontsize="small")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Cruise Trim Check", fontweight="bold")
    fig.tight_layout()
    fig.savefig("flight_cruise.png", dpi=150)
    print("  Figure saved → flight_cruise.png")
    plt.close(fig)


# ===================================================================
#  main
# ===================================================================
def main() -> None:
    ac = make_test_aircraft()

    # --- Scenario 1: Takeoff ---
    traj_to = scenario_takeoff(ac)
    plot_takeoff(traj_to, ac)

    # --- Scenario 2: Stall ---
    traj_stall = scenario_stall(ac)
    plot_stall(traj_stall, ac)

    # --- Scenario 3: Cruise ---
    traj_cruise = scenario_cruise(ac)
    plot_cruise(traj_cruise, ac)

    print("\nAll scenarios complete.")


if __name__ == "__main__":
    main()

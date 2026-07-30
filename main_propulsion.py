"""Propulsion black-box demo — the interface a flight-dynamics loop consumes.

Shows:
  1) T(V), P(V) curves at fixed throttle + pitch  (the "engine map")
  2) Throttle sweep at fixed airspeed  (control authority)
  3) Pitch sweep  (variable-pitch behaviour)
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from propulsion import PropulsionSystem, MotorSpec, PropGeom


def main() -> None:
    # ---- build the black box -----------------------------------------------
    prop = PropulsionSystem.standard()
    print(f"Propeller:  D = {prop.prop.diameter * 1000:.0f} mm  "
          f"({prop.prop.diameter / 0.0254:.1f} in)")
    print(f"            B = {prop.prop.n_blades} blades")
    print(f"            c = {prop.prop.chord * 1000:.1f} mm (constant)")
    print(f"            J_design = {prop.prop.J_design}")
    print(f"Motor:      max {prop.motor.max_rpm:.0f} RPM  "
          f"({'unlimited' if prop.motor.max_power is None else f'{prop.motor.max_power:.0f} W'})")
    print()

    # ==================================================================
    #  1) Performance curves — the "engine map" for the flight loop
    # ==================================================================
    throttle = 0.7
    pitch = 4.0
    curve = prop.curve(
        V_range=(1.0, 30.0), n_pts=50,
        throttle=throttle, pitch_deg=pitch,
    )

    print(f"Throttle {throttle}, Pitch {pitch}° — Performance curve:")
    print(f"  Static thrust:  {curve.thrust[0]:.3f} N")
    print(f"  Peak efficiency: {np.max(curve.eta_pct()):.1f}%  "
          f"at V ≈ {curve.V[np.argmax(curve.efficiency)]:.1f} m/s")
    print(f"  Thrust at 15 m/s: {prop.thrust(V=15.0, throttle=throttle, pitch_deg=pitch):.3f} N")
    print(f"  Power  at 15 m/s: {prop.power(V=15.0, throttle=throttle, pitch_deg=pitch):.1f} W")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # — thrust curve —
    ax = axes[0]
    ax.plot(curve.V, curve.thrust, "o-", ms=3, color="#2C5F8A")
    ax.set_xlabel("Airspeed V [m/s]")
    ax.set_ylabel("Thrust [N]")
    ax.set_title(f"Thrust vs Airspeed (thr={throttle}, pitch={pitch}°)")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="grey", lw=0.8)

    # — power curve —
    ax = axes[1]
    ax.plot(curve.V, curve.power, "s-", ms=3, color="#D4604E")
    ax.set_xlabel("Airspeed V [m/s]")
    ax.set_ylabel("Shaft Power [W]")
    ax.set_title(f"Power vs Airspeed (thr={throttle}, pitch={pitch}°)")
    ax.grid(True, alpha=0.3)

    # — efficiency curve —
    ax = axes[2]
    ax.plot(curve.V, curve.eta_pct(), "o-", ms=3, color="#4CAF50")
    ax.set_xlabel("Airspeed V [m/s]")
    ax.set_ylabel("η [%]")
    ax.set_title(f"Efficiency vs Airspeed (thr={throttle}, pitch={pitch}°)")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Propulsion System — Performance Curves", fontweight="bold")
    fig.tight_layout()
    fig.savefig("prop_curves.png", dpi=150)
    print("\nFigure saved → prop_curves.png")
    plt.close(fig)

    # ==================================================================
    #  2) Throttle sweep at cruise speed
    # ==================================================================
    V_cruise = 15.0  # m/s (~54 km/h)
    throttles = np.linspace(0.1, 1.0, 10)

    T_thr = [prop.thrust(V=V_cruise, throttle=t, pitch_deg=pitch) for t in throttles]
    P_thr = [prop.power(V=V_cruise, throttle=t, pitch_deg=pitch) for t in throttles]
    rpm_thr = [prop.motor.rpm(t) for t in throttles]

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(throttles * 100, T_thr, "o-", ms=4, color="#2C5F8A")
    ax1.set_xlabel("Throttle [%]")
    ax1.set_ylabel("Thrust [N]")
    ax1.set_title(f"Throttle → Thrust at V = {V_cruise} m/s")
    ax1.grid(True, alpha=0.3)

    ax2.plot(throttles * 100, P_thr, "s-", ms=4, color="#D4604E")
    ax2.set_xlabel("Throttle [%]")
    ax2.set_ylabel("Shaft Power [W]")
    ax2.set_title(f"Throttle → Power at V = {V_cruise} m/s")
    ax2.grid(True, alpha=0.3)

    fig2.suptitle("Propulsion — Control Authority", fontweight="bold")
    fig2.tight_layout()
    fig2.savefig("prop_throttle.png", dpi=150)
    print("Figure saved → prop_throttle.png")
    plt.close(fig2)

    # ==================================================================
    #  3) Pitch sweep — T(V) family at constant throttle
    # ==================================================================
    pitches = [0.0, 4.0, 8.0, 12.0]
    colors = ["#2C5F8A", "#D4604E", "#4CAF50", "#FF9800"]

    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for p, color in zip(pitches, colors):
        c = prop.curve(
            V_range=(1.0, 35.0), n_pts=40,
            throttle=0.8, pitch_deg=p,
        )
        ax1.plot(c.V, c.thrust, "-", color=color, label=f"pitch = {p:.0f}°")
        ax2.plot(c.V, c.power, "-", color=color, label=f"pitch = {p:.0f}°")

    ax1.set_xlabel("Airspeed V [m/s]")
    ax1.set_ylabel("Thrust [N]")
    ax1.set_title("Thrust vs Airspeed — Pitch Sweep (thr=0.8)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="grey", lw=0.8)

    ax2.set_xlabel("Airspeed V [m/s]")
    ax2.set_ylabel("Shaft Power [W]")
    ax2.set_title("Power vs Airspeed — Pitch Sweep (thr=0.8)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig3.suptitle("Propulsion — Variable-Pitch Behaviour", fontweight="bold")
    fig3.tight_layout()
    fig3.savefig("prop_pitch_sweep.png", dpi=150)
    print("Figure saved → prop_pitch_sweep.png")
    plt.close(fig3)

    # ==================================================================
    #  4) Single-point queries (the flight-loop API)
    # ==================================================================
    print("\n--- Single-point queries (flight-loop API) ---")
    for V in [0.01, 5.0, 10.0, 15.0, 20.0, 25.0]:
        op = prop.solve(V=V, throttle=0.7, pitch_deg=4.0)
        print(f"  {op}")


if __name__ == "__main__":
    main()

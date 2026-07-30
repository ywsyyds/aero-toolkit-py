"""BEMT propeller validation — performance curves and radial distributions.

Sweeps advance ratio J and plots CT, CP, η, plus radial loading at
selected operating points.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bemt import BEMTSolver


def main() -> None:
    # ---- propeller definition -----------------------------------------------
    R = 0.127          # tip radius [m] (~5-inch)
    omega = 1000.0     # [rad/s]  (~9550 RPM)

    bem = BEMTSolver(
        R=R,
        B=2,
        omega=omega,
        n_stations=40,
        J_design=0.7,
        alpha_design_deg=4.0,
        chord=0.015,             # constant chord [m]
        cd0=0.010,
        k_cd=0.008,
        relax=0.3,
        max_iter=500,
        tol=1e-6,
    )

    prop = bem  # alias

    print(f"Propeller:  D = {prop.D * 1000:.0f} mm  ({prop.D / 0.0254:.1f} in)")
    print(f"            B = {prop.B} blades")
    print(f"            ω = {omega:.0f} rad/s  ({omega * 30 / np.pi:.0f} RPM)")
    print(f"            n = {prop.n:.1f} rev/s")
    print(f"            c = {prop.chord[0] * 1000:.1f} mm (constant)")
    print(f"            J_design = {0.7},  α_design = 4°")
    print(f"            polar:  Cd0 = {prop.cd0},  k = {prop.k_cd}")
    print()

    # ---- J sweep ------------------------------------------------------------
    J_vals = np.linspace(0.05, 1.00, 40)
    CT = np.empty_like(J_vals)
    CP = np.empty_like(J_vals)
    eta = np.empty_like(J_vals)
    T = np.empty_like(J_vals)
    P = np.empty_like(J_vals)

    print(f"  {'J':>6s}  {'CT':>8s}  {'CP':>8s}  {'η':>8s}  {'T [N]':>8s}  {'P [W]':>8s}  {'iters':>5s}")
    print("  " + "-" * 62)

    for idx, J in enumerate(J_vals):
        V_inf = J * prop.n * prop.D
        res = prop.solve(V_inf)
        CT[idx] = res.CT
        CP[idx] = res.CP
        eta[idx] = res.eta if res.CP > 1e-10 else np.nan
        T[idx] = res.T
        P[idx] = res.P
        print(f"  {J:6.3f}  {res.CT:8.4f}  {res.CP:8.4f}  "
              f"{res.eta if res.CP > 1e-10 else float('nan'):8.4f}  "
              f"{res.T:8.3f}  {res.P:8.1f}  {res.iterations:5d}")

    # ---- find peak efficiency -----------------------------------------------
    mask_pos = CP > 1e-10
    if np.any(mask_pos):
        idx_peak = np.nanargmax(np.where(mask_pos, eta, -1))
        print(f"\n  Peak efficiency:  η = {eta[idx_peak]:.4f}  at J = {J_vals[idx_peak]:.3f}")
        print(f"  Max CT:           {np.max(CT[mask_pos]):.4f}  at J = {J_vals[np.argmax(CT)]:.3f}")
        print(f"  J where CT → 0:   ~{J_vals[np.where(mask_pos)[0][-1]]:.3f}")

    # ==================================================================
    #  Figure 1 — performance curves
    # ==================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(J_vals, CT, "o-", ms=3, color="#2C5F8A", label="CT")
    ax1.plot(J_vals, CP, "s-", ms=3, color="#D4604E", label="CP")
    ax1.set_xlabel("J  (advance ratio)")
    ax1.set_ylabel("CT, CP")
    ax1.set_title("Thrust & Power Coefficients")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="grey", lw=0.8)

    ax2.plot(J_vals[mask_pos], eta[mask_pos] * 100, "o-", ms=4,
             color="#2C5F8A")
    ax2.set_xlabel("J  (advance ratio)")
    ax2.set_ylabel("η  [%]")
    ax2.set_title("Propulsive Efficiency")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color="grey", lw=0.8)

    fig.suptitle("BEMT Fixed-Pitch Propeller", fontweight="bold")
    fig.tight_layout()
    fig.savefig("bemt_curves.png", dpi=150)
    plt.close(fig)
    print("  Figure saved → bemt_curves.png")

    # ==================================================================
    #  Figure 2 — radial distributions at selected J
    # ==================================================================
    J_samples = [0.2, 0.4, 0.6, 0.8]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = ["#2C5F8A", "#D4604E", "#4CAF50", "#FF9800"]

    for J_s, color in zip(J_samples, colors):
        V = J_s * prop.n * prop.D
        res = prop.solve(V)
        r_norm = res.r / prop.R

        axes[0, 0].plot(r_norm, res.a, "-", color=color,
                        label=f"J = {J_s:.1f}")
        axes[0, 1].plot(r_norm, res.cl, "-", color=color,
                        label=f"J = {J_s:.1f}")
        axes[1, 0].plot(r_norm, np.degrees(res.alpha), "-", color=color,
                        label=f"J = {J_s:.1f}")
        axes[1, 1].plot(r_norm, res.dT_dr, "-", color=color,
                        label=f"J = {J_s:.1f}")

    axes[0, 0].set_ylabel("a  (axial induction)")
    axes[0, 0].set_title("Axial Induction Factor")
    axes[0, 0].legend(fontsize="small")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_ylabel("Cl")
    axes[0, 1].set_title("Section Lift Coefficient")
    axes[0, 1].legend(fontsize="small")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("r / R")
    axes[1, 0].set_ylabel("α  [deg]")
    axes[1, 0].set_title("Angle of Attack")
    axes[1, 0].legend(fontsize="small")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel("r / R")
    axes[1, 1].set_ylabel("dT/dr  [N/m]")
    axes[1, 1].set_title("Thrust per Unit Span")
    axes[1, 1].legend(fontsize="small")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("BEMT — Radial Distributions", fontweight="bold")
    fig.tight_layout()
    fig.savefig("bemt_radial.png", dpi=150)
    plt.close(fig)
    print("  Figure saved → bemt_radial.png")


if __name__ == "__main__":
    main()

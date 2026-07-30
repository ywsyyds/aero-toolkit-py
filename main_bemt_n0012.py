"""BEMT comparison: analytical polar vs experimental NACA 0012 polar.

Quantifies how much the "idealized Cd0/k" model overestimates efficiency
compared to real airfoil data (NASA, Re=6E6, 80-grit).
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bemt import BEMTSolver
from polar import naca0012_polar


def main() -> None:
    # ---- propeller definition (same as main_bemt.py) ------------------------
    R = 0.127
    omega = 1000.0

    # shared geometry
    common = dict(
        R=R,
        B=2,
        omega=omega,
        n_stations=40,
        J_design=0.7,
        alpha_design_deg=4.0,
        chord=0.015,
        relax=0.3,
        max_iter=500,
        tol=1e-6,
    )

    # analytical polar (baseline)
    bem_ideal = BEMTSolver(
        cd0=0.010,
        k_cd=0.008,
        cl_slope=2.0 * np.pi,
        alpha0=0.0,
        **common,
    )

    # experimental NACA 0012 polar
    polar_exp = naca0012_polar("80 grit")
    bem_exp = BEMTSolver(
        polar_table=polar_exp,
        **common,
    )

    # ---- J sweep ------------------------------------------------------------
    J_vals = np.linspace(0.05, 1.00, 40)

    results: dict[str, dict] = {"ideal": {}, "exp": {}}

    for label, bem in [("ideal", bem_ideal), ("exp", bem_exp)]:
        CT, CP, eta, T, P = [], [], [], [], []
        for J in J_vals:
            V_inf = J * bem.n * bem.D
            try:
                res = bem.solve(V_inf)
            except Exception:
                # convergence failure → NaN
                CT.append(np.nan)
                CP.append(np.nan)
                eta.append(np.nan)
                T.append(np.nan)
                P.append(np.nan)
                continue
            CT.append(res.CT)
            CP.append(res.CP)
            eta.append(res.eta if res.CP > 1e-10 else np.nan)
            T.append(res.T)
            P.append(res.P)

        for key, arr in [("CT", CT), ("CP", CP), ("eta", eta), ("T", T), ("P", P)]:
            results[label][key] = np.array(arr)

    # ---- print comparison ---------------------------------------------------
    print("Comparing BEMT performance: Ideal polar vs NACA 0012 experimental")
    print(f"  Propeller: D={R*2000:.1f} mm, B={common['B']}, ω={omega:.0f} rad/s")
    print()

    mask_ideal = np.isfinite(results["ideal"]["eta"])
    mask_exp = np.isfinite(results["exp"]["eta"])
    mask_both = mask_ideal & mask_exp

    if np.any(mask_ideal):
        idx_pk_i = np.nanargmax(np.where(mask_ideal, results["ideal"]["eta"], -1))
        print(f"  Ideal:   peak η = {results['ideal']['eta'][idx_pk_i]*100:.1f}% "
              f"at J = {J_vals[idx_pk_i]:.3f}")
    if np.any(mask_exp):
        idx_pk_e = np.nanargmax(np.where(mask_exp, results["exp"]["eta"], -1))
        print(f"  NACA0012: peak η = {results['exp']['eta'][idx_pk_e]*100:.1f}% "
              f"at J = {J_vals[idx_pk_e]:.3f}")

    if np.any(mask_both):
        eta_delta = (results["exp"]["eta"][mask_both]
                     - results["ideal"]["eta"][mask_both])
        print(f"  Mean Δη (exp − ideal) = {np.mean(eta_delta)*100:.1f} pp "
              f"over J ∈ [{J_vals[mask_both][0]:.2f}, {J_vals[mask_both][-1]:.2f}]")
        print(f"  Max  Δη               = {np.max(np.abs(eta_delta))*100:.1f} pp")

    # ---- plot ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    colors = {"ideal": "#2C5F8A", "exp": "#D4604E"}
    labels = {"ideal": "Analytical (Cd₀=0.01, k=0.008)",
              "exp": "NACA 0012 exp. (80 grit, Re=6×10⁶)"}

    # --- CT/CP ---
    ax = axes[0]
    for key in ["ideal", "exp"]:
        ok = np.isfinite(results[key]["CT"])
        ax.plot(J_vals[ok], results[key]["CT"][ok], "o-", ms=3,
                color=colors[key], label=f"CT ({labels[key]})")
        ax.plot(J_vals[ok], results[key]["CP"][ok], "s-", ms=3,
                color=colors[key], alpha=0.4, label=f"CP ({labels[key]})")
    ax.set_xlabel("J")
    ax.set_ylabel("CT, CP")
    ax.set_title("Thrust & Power Coefficients")
    ax.legend(fontsize="x-small")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="grey", lw=0.8)

    # --- efficiency ---
    ax = axes[1]
    for key in ["ideal", "exp"]:
        ok = np.isfinite(results[key]["eta"])
        ax.plot(J_vals[ok], results[key]["eta"][ok] * 100, "o-", ms=4,
                color=colors[key], label=labels[key])
    ax.set_xlabel("J")
    ax.set_ylabel("η [%]")
    ax.set_title("Propulsive Efficiency")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)

    # --- experimental Cl/Cd polar ---
    ax = axes[2]
    alpha_deg_plot = np.linspace(-6, 20, 200)
    alpha_rad_plot = np.radians(alpha_deg_plot)
    cl_exp, cd_exp = polar_exp(alpha_rad_plot)
    ax.plot(alpha_deg_plot, cl_exp, "-", color=colors["exp"], label="Cl (exp)")
    ax.plot(alpha_deg_plot, cd_exp * 20, "--", color=colors["exp"],
            label="Cd × 20 (exp)")

    # analytical for reference
    cl_ideal = 2.0 * np.pi * alpha_rad_plot
    cd_ideal = 0.010 + 0.008 * cl_ideal ** 2
    ax.plot(alpha_deg_plot, cl_ideal, ":", color=colors["ideal"], label="Cl (2π·α)")
    ax.plot(alpha_deg_plot, cd_ideal * 20, ":", color=colors["ideal"],
            label="Cd × 20 (Cd₀+k·Cl²)")

    ax.set_xlabel("α [deg]")
    ax.set_ylabel("Cl, 20×Cd")
    ax.set_title("NACA 0012 Polar (Re = 6×10⁶)")
    ax.legend(fontsize="x-small")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="grey", lw=0.8)

    fig.suptitle(
        "BEMT: Analytical vs Experimental Airfoil Polar",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig("bemt_polar_compare.png", dpi=150)
    print("\nFigure saved → bemt_polar_compare.png")

    # ---- radial distribution at design J ------------------------------------
    J_design = 0.7
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 9))

    for label, bem, color in [("ideal", bem_ideal, colors["ideal"]),
                               ("exp", bem_exp, colors["exp"])]:
        V = J_design * bem.n * bem.D
        res = bem.solve(V)
        rn = res.r / bem.R

        axes2[0, 0].plot(rn, res.cl, "-", color=color,
                         label=f"{labels[label]}")
        axes2[0, 1].plot(rn, res.cd, "-", color=color,
                         label=f"{labels[label]}")
        axes2[1, 0].plot(rn, np.degrees(res.alpha), "-", color=color,
                         label=f"{labels[label]}")
        axes2[1, 1].plot(rn, res.dT_dr, "-", color=color,
                         label=f"{labels[label]}")

        # report polar range
        alpha_min = np.min(np.degrees(res.alpha))
        alpha_max = np.max(np.degrees(res.alpha))
        print(f"  {label:>5s} at J={J_design}: α ∈ [{alpha_min:.1f}°, {alpha_max:.1f}°], "
              f"Cl ∈ [{np.min(res.cl):.2f}, {np.max(res.cl):.2f}], "
              f"Cd ∈ [{np.min(res.cd):.4f}, {np.max(res.cd):.4f}]")

    axes2[0, 0].set_ylabel("Cl")
    axes2[0, 0].set_title("Section Lift Coefficient")
    axes2[0, 0].legend(fontsize="x-small")
    axes2[0, 0].grid(True, alpha=0.3)

    axes2[0, 1].set_ylabel("Cd")
    axes2[0, 1].set_title("Section Drag Coefficient")
    axes2[0, 1].legend(fontsize="x-small")
    axes2[0, 1].grid(True, alpha=0.3)

    axes2[1, 0].set_xlabel("r / R")
    axes2[1, 0].set_ylabel("α [deg]")
    axes2[1, 0].set_title("Angle of Attack")
    axes2[1, 0].legend(fontsize="x-small")
    axes2[1, 0].grid(True, alpha=0.3)

    axes2[1, 1].set_xlabel("r / R")
    axes2[1, 1].set_ylabel("dT/dr [N/m]")
    axes2[1, 1].set_title("Thrust per Unit Span")
    axes2[1, 1].legend(fontsize="x-small")
    axes2[1, 1].grid(True, alpha=0.3)

    fig2.suptitle(
        f"BEMT Radial Distributions at J = {J_design}",
        fontweight="bold",
    )
    fig2.tight_layout()
    fig2.savefig("bemt_radial_compare.png", dpi=150)
    print("Figure saved → bemt_radial_compare.png")


if __name__ == "__main__":
    main()

"""Validate 2D VLM against thin-airfoil theory.

Compares:
  1) Flat plate — Cl = 2π·α
  2) NACA 2412 (positive camber) — zero-lift angle α₀ < 0, slope ≈ 2π
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vlm import VLM2D, thin_airfoil_cl
from camberline import naca4_camber, parse_naca4


def thin_airfoil_alpha0(code: int = 2412, n_theta: int = 500) -> float:
    """Thin-airfoil-theory zero-lift angle for a NACA 4-digit camber line.

    α₀ = (1/π) ∫₀^π (dyc/dx)(θ) · (1 - cos θ) dθ
    where x(θ) = ½(1 - cos θ), c = 1.
    """
    m, p = parse_naca4(code)
    if m == 0:
        return 0.0

    theta = np.linspace(1e-10, np.pi - 1e-10, n_theta)
    x = 0.5 * (1.0 - np.cos(theta))
    _, _, dyc = naca4_camber(code, x=x)

    integrand = dyc * (1.0 - np.cos(theta))
    integral = np.trapezoid(integrand, theta)
    return integral / np.pi


def find_zero_crossing(alphas_deg: np.ndarray, cl: np.ndarray) -> float:
    """Linear interpolation to find α where Cl = 0."""
    for i in range(len(cl) - 1):
        if cl[i] * cl[i + 1] <= 0:
            t = cl[i] / (cl[i] - cl[i + 1])
            return alphas_deg[i] + t * (alphas_deg[i + 1] - alphas_deg[i])
    return float("nan")


def main() -> None:
    chord = 1.0
    V_inf = 1.0
    n_panels = 20
    alphas_deg = np.linspace(-5.0, 15.0, 41)
    naca_code = 2412
    m, p = parse_naca4(naca_code)

    # --- solvers -------------------------------------------------------------
    vlm_flat = VLM2D(chord=chord, n_panels=n_panels, V_inf=V_inf)
    vlm_camber = VLM2D.from_naca4(code=naca_code, chord=chord,
                                  n_panels=n_panels, V_inf=V_inf)

    # --- sweep α ------------------------------------------------------------
    cl_flat = np.empty_like(alphas_deg)
    cl_camber = np.empty_like(alphas_deg)

    for i, a in enumerate(alphas_deg):
        cl_flat[i], _ = vlm_flat.solve(a)
        cl_camber[i], _ = vlm_camber.solve(a)

    # --- zero-lift angle & slope (cambered) ---------------------------------
    a0_vlm = find_zero_crossing(alphas_deg, cl_camber)
    a0_theory = np.degrees(thin_airfoil_alpha0(naca_code))

    # linear fit in small-α region
    mask = np.abs(alphas_deg - a0_vlm) <= 12.0  # same range for slope fit
    slope_camber = np.polyfit(np.radians(alphas_deg[mask]),
                              cl_camber[mask], 1)[0]
    slope_flat = np.polyfit(np.radians(alphas_deg),
                            cl_flat, 1)[0]
    slope_theory = 2.0 * np.pi

    # --- print --------------------------------------------------------------
    print(f"NACA {naca_code}  (m = {m:.2f},  p = {p:.2f})")
    print(f"  Panels:  {n_panels}")
    print()
    print(f"  {'':>20s}  {'flat plate':>12s}  {'cambered':>12s}  {'theory':>12s}")
    print(f"  {'α₀ [deg]':>20s}  {'0.0000':>12s}  {a0_vlm:12.4f}  {a0_theory:12.4f}")
    print(f"  {'dCl/dα [/rad]':>20s}  {slope_flat:12.4f}  {slope_camber:12.4f}  "
          f"{slope_theory:12.4f}")
    print(f"  {'Cl @ α=0°':>20s}  {'0.0000':>12s}  "
          f"{cl_camber[np.argmin(np.abs(alphas_deg))]:12.4f}  {'-':>12s}")
    print()

    theory_slope = 2.0 * np.pi
    print(f"  Camber slope vs 2π:        {slope_camber / theory_slope * 100:.2f}%")

    # --- table --------------------------------------------------------------
    print(f"\n{'α [°]':>8s}  {'Cl (flat)':>10s}  {'Cl (camber)':>12s}")
    print("-" * 38)
    for a, cf, cc in zip(alphas_deg, cl_flat, cl_camber):
        print(f"{a:8.2f}  {cf:10.5f}  {cc:12.5f}")

    # --- plot ---------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # -- left: flat plate --
    cl_theory_flat = thin_airfoil_cl(alphas_deg)
    ax1.plot(alphas_deg, cl_flat, "o-", ms=4, label="VLM (N=20)", color="#2C5F8A")
    ax1.plot(alphas_deg, cl_theory_flat, "--", lw=2,
             label=r"thin-airfoil: $2\pi\alpha$", color="#D4604E")
    ax1.set_xlabel(r"$\alpha$ [deg]")
    ax1.set_ylabel(r"$C_l$")
    ax1.set_title("Flat Plate")
    ax1.legend(fontsize="small")
    ax1.grid(True, alpha=0.3)

    # -- right: cambered --
    ax2.plot(alphas_deg, cl_camber, "o-", ms=4,
             label=f"VLM NACA {naca_code}", color="#2C5F8A")
    ax2.plot(alphas_deg, thin_airfoil_cl(alphas_deg - a0_theory),
             "--", lw=2,
             label=rf"thin-airfoil: $2\pi(\alpha-\alpha_0)$, $\alpha_0$={a0_theory:.2f}°",
             color="#D4604E")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.axvline(a0_vlm, color="grey", ls=":", alpha=0.5,
                label=f"VLM α₀ = {a0_vlm:.2f}°")
    ax2.set_xlabel(r"$\alpha$ [deg]")
    ax2.set_ylabel(r"$C_l$")
    ax2.set_title(f"NACA {naca_code}")
    ax2.legend(fontsize="small")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("2D VLM Validation", fontweight="bold")
    fig.tight_layout()
    fig.savefig("cl_alpha.png", dpi=150)
    print("\nFigure saved → cl_alpha.png")


if __name__ == "__main__":
    main()

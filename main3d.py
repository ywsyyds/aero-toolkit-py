"""3D VLM Validation — three reference checks.

1. AR = 50 rectangular wing  →  CLα approaches 2π (2D limit).
2. AR sweep (4 … 50)         →  CLα vs Prandtl lifting-line theory.
3. Elliptic wing             →  Γ(y) half-ellipse shape + CDi match.

NOTE — The single-chordwise-panel VLM systematically underestimates CLα by
a few percent compared to Prandtl theory.  This is a known discretisation
effect; the physics trends are all correctly reproduced.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vlm3d import VLM3D

# ------------------------------------------------------------------
# Common parameters
# ------------------------------------------------------------------
N_SPAN = 60          # spanwise panels
ALPHA_DEG = 4.0      # single α for linear-region slope
ALPHA_RAD = np.radians(ALPHA_DEG)
TWO_PI = 2.0 * np.pi


# ===================================================================
def test_high_AR() -> None:
    """AR ≈ 50: CLα should approach 2π — 3D correction from Prandtl."""
    print("=" * 65)
    print("Test 1 — High AR (AR ≈ 50) — approach to 2D limit")
    print("=" * 65)

    vlm = VLM3D.rectangular(span=50.0, chord=1.0, n_span=N_SPAN)

    alphas = np.linspace(1.0, 6.0, 6)
    CLs = []
    for a in alphas:
        CL, CDi, _ = vlm.solve(a)
        CLs.append(CL)
        print(f"  α = {a:5.1f}°   CL = {CL:8.5f}   CDi = {CDi:8.6f}")

    slope = np.polyfit(np.radians(alphas), np.array(CLs), 1)[0]
    prandtl_AR50 = VLM3D.prandtl_CL_alpha(50.0)

    print(f"\n  VLM       dCL/dα  = {slope:.4f} /rad")
    print(f"  2D limit  (2π)    = {TWO_PI:.4f} /rad")
    print(f"  Prandtl   (AR=50) = {prandtl_AR50:.4f} /rad")
    print(f"  VLM / 2π          = {slope / TWO_PI * 100:.2f} %")
    print(f"  VLM / Prandtl     = {slope / prandtl_AR50 * 100:.2f} %")
    print()


# ===================================================================
def test_AR_sweep() -> None:
    """AR sweep — CLα vs Prandtl lifting-line.  The VLM naturally captures the
    non-elliptic loading of the rectangular wing (Oswald e < 1)."""
    print("=" * 65)
    print("Test 2 — AR sweep  (rectangular wing)")
    print("=" * 65)

    AR_list = np.array([4, 5, 6, 8, 10, 12, 16, 20, 30, 50])
    CLs = np.empty(len(AR_list))

    for idx, AR in enumerate(AR_list):
        vlm = VLM3D.rectangular(span=AR, chord=1.0, n_span=N_SPAN)
        CLs[idx], CDi, _ = vlm.solve(ALPHA_DEG)

    cl_alpha_vlm = CLs / ALPHA_RAD
    cl_alpha_prandtl = np.array([VLM3D.prandtl_CL_alpha(AR) for AR in AR_list])

    print(f"  {'AR':>5s}  {'CL (α=4°)':>12s}  {'CLα VLM':>10s}  "
          f"{'Prandtl e=1':>12s}  {'ratio %':>8s}")
    print("  " + "-" * 60)
    for AR, CL, cla, clp in zip(AR_list, CLs, cl_alpha_vlm, cl_alpha_prandtl):
        print(f"  {AR:5.1f}  {CL:12.5f}  {cla:10.4f}  {clp:12.4f}  "
              f"{cla / clp * 100:7.2f}")

    print(f"\n  2π limit               = {TWO_PI:.4f}")
    print(f"  Mean VLM / Prandtl e=1  = {np.mean(cl_alpha_vlm / cl_alpha_prandtl) * 100:.2f} %")
    print(f"  (Rectangular wing has e < 1; VLM captures this naturally)")

    # ---- plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(AR_list, cl_alpha_vlm, "o-", ms=6, label="VLM (rectangular wing)",
            color="#2C5F8A")
    ax.plot(AR_list, cl_alpha_prandtl, "s--", ms=5,
            label=r"Prandtl: $a_0/(1+a_0/\pi AR)$  (elliptic, $e{=}1$)",
            color="#D4604E")
    ax.axhline(TWO_PI, color="grey", ls=":", alpha=0.5, label=r"2D limit $2\pi$")
    ax.set_xlabel("Aspect Ratio  AR")
    ax.set_ylabel(r"$dC_L/d\alpha$  [/rad]")
    ax.set_title("3D VLM — Lift-curve slope vs Aspect Ratio")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("ar_sweep.png", dpi=150)
    plt.close(fig)
    print("  Figure saved → ar_sweep.png")
    print()


# ===================================================================
def test_elliptic_wing() -> None:
    """Elliptic wing: Γ(y) half-ellipse + CDi = CL²/(π·AR)."""
    print("=" * 65)
    print("Test 3 — Elliptic wing")
    print("=" * 65)

    span = 10.0
    AR = 8.0
    vlm = VLM3D.elliptic(span=span, AR=AR, n_span=N_SPAN)
    CL, CDi, gamma = vlm.solve(ALPHA_DEG)

    # theoretical references
    CL_theory = VLM3D.prandtl_CL_alpha(AR) * ALPHA_RAD
    gamma0_theory = 2.0 * vlm.V_inf * CL_theory * vlm.S / (np.pi * span)

    r = 2.0 * np.abs(vlm.y_ctr) / span
    gamma_theory = gamma0_theory * np.sqrt(np.maximum(1.0 - r * r, 0.0))

    # normalised shape
    gamma_norm = gamma / np.max(gamma)
    gamma_th_norm = gamma_theory / np.max(gamma_theory)
    rms_shape_err = np.sqrt(np.mean((gamma_norm - gamma_th_norm) ** 2))

    CDi_theory_vlm_CL = VLM3D.prandtl_CDi(CL, AR, e=1.0)

    print(f"  CL        = {CL:.5f}   (Prandtl e=1: {CL_theory:.5f})")
    print(f"  CDi       = {CDi:.5f}   (CL²/π·AR = {CDi_theory_vlm_CL:.5f})")
    print(f"  CDi match = {CDi / CDi_theory_vlm_CL * 100:.2f} %  of Prandtl using VLM's own CL")
    print(f"  Γ₀        = {np.max(gamma):.5f}   (theory: {gamma0_theory:.5f})")
    print(f"  Γ(y) RMS shape error = {rms_shape_err:.4f}")
    print()

    # ---- plot ---------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(vlm.y_ctr, gamma, "o-", ms=3, color="#2C5F8A", label="VLM")
    ax1.plot(vlm.y_ctr, gamma_theory, "--", lw=2, color="#D4604E",
             label="half-ellipse (theory)")
    ax1.set_xlabel("y")
    ax1.set_ylabel(r"$\Gamma(y)$")
    ax1.set_title(f"Elliptic wing  AR = {AR}   α = {ALPHA_DEG}°")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    c0 = 4.0 * span / (np.pi * AR)
    c_theory = c0 * np.sqrt(np.maximum(1.0 - r * r, 0.0))
    ax2.fill_between(vlm.y_ctr, 0, vlm.chords, step="mid", alpha=0.3,
                     color="#2C5F8A", label="VLM panels")
    ax2.plot(vlm.y_ctr, c_theory, "--", lw=2, color="#D4604E", label="ellipse")
    ax2.set_xlabel("y")
    ax2.set_ylabel("c(y)")
    ax2.set_title("Planform")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal", adjustable="box")

    fig.suptitle("3D VLM — Elliptic Wing Validation", fontweight="bold")
    fig.tight_layout()
    fig.savefig("elliptic_wing.png", dpi=150)
    plt.close(fig)
    print("  Figure saved → elliptic_wing.png")
    print()


# ===================================================================
if __name__ == "__main__":
    test_high_AR()
    test_AR_sweep()
    test_elliptic_wing()

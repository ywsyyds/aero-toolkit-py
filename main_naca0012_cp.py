"""Validate 2D VLM pressure distribution against NASA NACA 0012 experiment.

Compares VLM Cp(x) against experimental pressure-coefficient data at
α = 0°, 10°, 15° (M=0.15, Re=2.88×10⁶, NASA low-speed wind tunnel).

The experimental data is for the UPPER surface only.  NACA 0012 is
symmetric, so at α=0° the upper and lower surfaces coincide; at α≠0°
the experimental data covers only the suction side.

Key formula (thin-airfoil approximation):
    Cp_upper(x) = −γ(x) / V∞
    where γ(x) ≈ Γ_i / dx   (distributed vorticity from point-vortex
                              strength Γ_i and panel width dx).

The goal is NOT to "match" the experiment — a 12%-thick airfoil cannot
be exactly reproduced by a zero-thickness model.  The gap QUANTIFIES
where thickness effects matter, which informs when a panel method
(or CFD) is needed instead of VLM.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from vlm import VLM2D

# ---------------------------------------------------------------------------
#  paths
# ---------------------------------------------------------------------------
DATA_DIR = Path("NACA0012_data")
CP_EXP_PATH = DATA_DIR / "Experimental" / "CP(0,10,15AOA).dat"


# ---------------------------------------------------------------------------
#  experimental data parser
# ---------------------------------------------------------------------------
def load_experimental_cp(path: str | Path) -> dict[str, tuple[NDArray, NDArray]]:
    """Parse the NASA NACA 0012 experimental Cp file.

    Returns
    -------
    dict[str, tuple[NDArray, NDArray]]
        Keys are angle labels ("0", "10", "15").  Values are (x_c, Cp)
        arrays.  Cp sign convention: negative = suction (standard).
    """
    text = Path(path).read_text(encoding="utf-8")

    zones: dict[str, tuple[list[float], list[float]]] = {}
    current_angle: str | None = None

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("TITLE") or line.startswith("variables") or line.startswith("#"):
            continue

        # zone header:  zone, t="angle of attack=0"
        m = re.match(r'zone,\s*t="angle of attack=(\d+)"', line, re.IGNORECASE)
        if m:
            current_angle = m.group(1)
            zones[current_angle] = ([], [])
            continue

        # data line
        if current_angle is not None:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    cp = float(parts[1])
                except ValueError:
                    continue
                zones[current_angle][0].append(x)
                zones[current_angle][1].append(cp)

    return {k: (np.array(v[0]), np.array(v[1])) for k, v in zones.items()}


# ---------------------------------------------------------------------------
#  thin-airfoil theory (analytical, for reference)
# ---------------------------------------------------------------------------
def thin_airfoil_cp_upper(
    x_c: NDArray[np.float64],
    alpha_deg: float,
) -> NDArray[np.float64]:
    """Analytical Cp on the upper surface from thin-airfoil theory.

    For a symmetric section at incidence α:
        γ(θ) = 2 α V∞ (1 + cos θ) / sin θ
        Cp_upper = −γ / V∞ = −2 α (1 + cos θ) / sin θ
    where x/c = ½ (1 − cos θ),  θ ∈ (0, π).

    The leading-edge singularity (Cp → −∞ as x→0) is the classical
    thin-airfoil artifact; real viscous flow has a finite suction peak.
    """
    alpha = np.radians(alpha_deg)
    # clip to avoid singularity at θ=0,π
    eps = 1e-10
    x = np.clip(np.asarray(x_c), eps, 1.0 - eps)
    theta = np.arccos(1.0 - 2.0 * x)
    return -2.0 * alpha * (1.0 + np.cos(theta)) / np.sin(theta)


# ---------------------------------------------------------------------------
#  VLM → Cp
# ---------------------------------------------------------------------------
def vlm_to_cp_upper(vlm: VLM2D, gamma: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert VLM point-vortex strengths to upper-surface Cp.

    Cp_upper = −γ_continuous / V∞
             = −(Γ_i / dx) / V∞
             = −Γ_i / (V∞ · dx)

    Parameters
    ----------
    vlm : VLM2D
        Configured solver instance (provides dx, V_inf).
    gamma : np.ndarray
        Point-vortex strengths (length n_panels).

    Returns
    -------
    cp : np.ndarray
        Upper-surface Cp at each panel's collocation point (xc).
    """
    return -gamma / (vlm.V_inf * vlm.dx)


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------
def main() -> None:
    # ---- load experimental data -------------------------------------------
    exp_data = load_experimental_cp(CP_EXP_PATH)

    # ---- VLM setup --------------------------------------------------------
    n_panels = 120  # fine resolution for smooth Cp curves
    chord = 1.0
    V_inf = 1.0

    # NACA 0012 is symmetric → dyc_dx = 0 everywhere
    vlm = VLM2D(chord=chord, n_panels=n_panels, V_inf=V_inf, dyc_dx=None)

    # ---- plot -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    target_angles = ["0", "10", "15"]

    for idx, a_str in enumerate(target_angles):
        ax = axes[idx]
        alpha = float(a_str)

        # — VLM solution —
        cl_vlm, gamma_vlm = vlm.solve(alpha)
        cp_vlm = vlm_to_cp_upper(vlm, gamma_vlm)
        xc = vlm.xc

        # — analytical thin-airfoil Cp —
        x_fine = np.linspace(0.001, 0.999, 400)
        cp_ta = thin_airfoil_cp_upper(x_fine, alpha)

        # — experimental data —
        if a_str in exp_data:
            x_exp, cp_exp = exp_data[a_str]
            ax.plot(x_exp, cp_exp, "o", ms=4, mfc="white", mec="#D4604E",
                    mew=1.2, label="Experiment (NASA)", zorder=5)

        ax.plot(xc, cp_vlm, "s-", ms=3, lw=1.2, color="#2C5F8A",
                label=f"VLM (N={n_panels})", zorder=4)
        ax.plot(x_fine, cp_ta, "--", lw=1.5, color="grey", alpha=0.7,
                label="Thin-airfoil theory", zorder=3)

        ax.set_title(rf"$\alpha = {alpha:.0f}°$" + f"\nVLM Cl = {cl_vlm:.4f}")
        ax.set_xlabel(r"$x/c$")
        if idx == 0:
            ax.set_ylabel(r"$C_p$ (upper surface)")
        ax.invert_yaxis()  # suction up (standard convention)
        ax.legend(fontsize="x-small", loc="lower left")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.02, 1.02)

    fig.suptitle(
        "NACA 0012 — VLM vs Experiment: Pressure Distribution",
        fontweight="bold", fontsize=13,
    )
    fig.tight_layout()
    fig.savefig("cp_naca0012.png", dpi=150)
    print("Figure saved → cp_naca0012.png")

    # ---- quantitative deviation -------------------------------------------
    print("\n--- Cp RMS deviation: VLM vs Experiment ---")
    print(f"  {'AoA':>5s}  {'RMS (full)':>11s}  {'RMS (x/c>0.05)':>15s}"
          f"  {'max|ΔCp|':>10s}  {'at x/c':>8s}")
    print(f"  {'-'*5}  {'-'*11}  {'-'*15}  {'-'*10}  {'-'*8}")
    for a_str in target_angles:
        if a_str not in exp_data:
            continue
        alpha = float(a_str)
        _, gamma_vlm = vlm.solve(alpha)
        cp_vlm = vlm_to_cp_upper(vlm, gamma_vlm)

        x_exp, cp_exp = exp_data[a_str]
        cp_vlm_interp = np.interp(x_exp, vlm.xc, cp_vlm)

        rms_full = np.sqrt(np.mean((cp_vlm_interp - cp_exp) ** 2))

        # exclude leading edge (first 5% chord) — where the thin-airfoil
        # singularity dominates and thickness effects are strongest
        mask_mid = x_exp > 0.05
        rms_mid = (np.sqrt(np.mean((cp_vlm_interp[mask_mid] - cp_exp[mask_mid]) ** 2))
                   if np.any(mask_mid) else float("nan"))

        max_dev = np.max(np.abs(cp_vlm_interp - cp_exp))
        x_max = x_exp[np.argmax(np.abs(cp_vlm_interp - cp_exp))]

        print(f"  {a_str:>5s}°  {rms_full:11.4f}  {rms_mid:15.4f}  "
              f"{max_dev:10.3f}  {x_max:8.3f}")

    # ---- Cp at x/c = 0 (stagnation) ---------------------------------------
    print("\n--- Leading-edge behaviour ---")
    print("  Thin-airfoil theory has a Cp → −∞ singularity at the LE.")
    print("  Real flow (experiment) has a finite suction peak limited by")
    print("  thickness, viscosity, and the stagnation-point relocation.")
    print("  VLM inherits the thin-airfoil singularity (discretised).")
    for a_str in target_angles:
        if a_str not in exp_data:
            continue
        x_exp, cp_exp = exp_data[a_str]
        print(f"  α={a_str}°: exp Cp at x/c≈{x_exp[0]:.4f} → Cp={cp_exp[0]:.3f}")


if __name__ == "__main__":
    main()

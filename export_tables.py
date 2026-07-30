"""Export all Python lookup tables → JSON/CSV for C# interpolation.

Produces two files per table (JSON + CSV) in an ``export/`` directory:

  export/
    wing_aero.json      — CL(α), CD(α) lookup + metadata
    wing_aero.csv       — same, one row per α
    propulsion.json     — T(V, throttle), P(V, throttle) 2D grids
    propulsion.csv      — same, flattened rows

The JSON schemas are intentionally flat (no nested objects) so that
Unity's built-in JsonUtility can deserialise them without Newtonsoft.

Usage
-----
    python export_tables.py

To change the wing or propeller, edit the ``build_*`` functions.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vlm3d import VLM3D
from polar import naca0012_polar
from propulsion import PropulsionSystem, PropGeom, MotorSpec
from flight import WingAero

# ---------------------------------------------------------------------------
#  Output directory
# ---------------------------------------------------------------------------
EXPORT_DIR = Path(__file__).resolve().parent / "export"


# ===================================================================
#  Wing aerodynamics table
# ===================================================================
def build_wing_table() -> dict[str, Any]:
    """Build the wing CL(α), CD(α) lookup table + metadata."""
    print("Building wing aero table …")

    vlm = VLM3D.rectangular(span=1.5, chord=0.2, n_span=40)
    polar = naca0012_polar()
    aero = WingAero.from_vlm(vlm, polar=polar, alpha_range=(-10.0, 25.0), n_pts=91)

    # compute V_stall for the reference mass
    mass = 2.0
    g = 9.81
    rho = 1.225
    V_stall = np.sqrt(2.0 * mass * g / (rho * aero.S * max(aero.CL_max, 0.1)))

    print(f"  AR={aero.AR:.1f}  S={aero.S:.3f} m²  CL_max={aero.CL_max:.3f}  "
          f"α_stall={aero.alpha_stall_deg:.1f}°  V_stall={V_stall:.1f} m/s")

    return {
        # ---- metadata ----
        "span": round(float(vlm.span), 4),
        "area": round(float(aero.S), 6),
        "AR": round(float(aero.AR), 4),
        "CL_max": round(float(aero.CL_max), 4),
        "alpha_stall_deg": round(float(aero.alpha_stall_deg), 2),
        "V_stall": round(float(V_stall), 2),
        "mass_ref_kg": mass,
        "rho": rho,
        "n_pts": int(len(aero.alpha_deg)),
        # ---- table data ----
        "alpha_deg": _float_list(aero.alpha_deg),
        "CL": _float_list(aero.CL),
        "CD": _float_list(aero.CD),
    }


# ===================================================================
#  Propulsion table  —  2D grid  T(V, throttle), P(V, throttle)
# ===================================================================
def build_propulsion_table(
    V_range: tuple[float, float] = (0.5, 35.0),
    n_V: int = 36,
    n_throttle: int = 21,
    pitch_deg: float = 4.0,
) -> dict[str, Any]:
    """Build the 2D propulsion lookup grids at fixed blade pitch.

    Sweeps V × throttle and records thrust [N] and shaft power [W].

    Returns a dict with axes ``V``, ``throttle``, and 2D arrays
    ``thrust``, ``power`` stored in row-major (C) order:
    ``thrust[i][j]`` = thrust at V[i], throttle[j].
    """
    print("Building propulsion table …")

    prop = PropulsionSystem.standard()

    V_vals = np.linspace(V_range[0], V_range[1], n_V)
    thr_vals = np.linspace(0.0, 1.0, n_throttle)

    T_grid: NDArray[np.float64] = np.empty((n_V, n_throttle))
    P_grid: NDArray[np.float64] = np.empty((n_V, n_throttle))

    for i, V in enumerate(V_vals):
        for j, thr in enumerate(thr_vals):
            op = prop.solve(V=float(V), throttle=float(thr), pitch_deg=pitch_deg)
            # clamp negative thrust/power — windmilling is physically real
            # but C# flight loops typically don't model it; zero is safer
            T_grid[i, j] = max(op.thrust, 0.0)
            P_grid[i, j] = max(op.power, 0.0)

    print(f"  grid: {n_V} V points × {n_throttle} throttle points  "
          f"({n_V * n_throttle} cells)")
    print(f"  V ∈ [{V_vals[0]:.1f}, {V_vals[-1]:.1f}] m/s  "
          f"throttle ∈ [0, 1]  pitch={pitch_deg}°")
    print(f"  T range: [{np.min(T_grid):.2f}, {np.max(T_grid):.2f}] N")
    print(f"  P range: [{np.min(P_grid):.1f}, {np.max(P_grid):.1f}] W")

    return {
        # ---- metadata ----
        "diameter_m": round(prop.prop.diameter, 4),
        "n_blades": prop.prop.n_blades,
        "pitch_deg": pitch_deg,
        "max_rpm": round(prop.motor.max_rpm, 1),
        "idle_rpm": round(prop.motor.idle_rpm, 1),
        "rho": prop.rho,
        "n_V": n_V,
        "n_throttle": n_throttle,
        "V_min": round(float(V_vals[0]), 2),
        "V_max": round(float(V_vals[-1]), 2),
        # ---- axes ----
        "V": _float_list(V_vals),
        "throttle": _float_list(thr_vals),
        # ---- 2D data (row-major: V × throttle) ----
        "thrust": [_float_list(row) for row in T_grid],
        "power": [_float_list(row) for row in P_grid],
    }


# ===================================================================
#  Serialisation helpers
# ===================================================================
def _float_list(arr: NDArray[np.float64]) -> list[float]:
    """ndarray → list[float] with reasonable decimal rounding."""
    return [round(float(x), 6) for x in arr]


def write_json(data: dict[str, Any], stem: str) -> None:
    """Write a dict as indented JSON."""
    path = EXPORT_DIR / f"{stem}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    size_kb = path.stat().st_size / 1024
    print(f"  → {path.name}  ({size_kb:.1f} KB)")


def write_csv_wing(data: dict[str, Any]) -> None:
    """Write wing aero table as CSV (one row per α)."""
    path = EXPORT_DIR / "wing_aero.csv"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["alpha_deg", "CL", "CD"])
        for a, cl, cd in zip(data["alpha_deg"], data["CL"], data["CD"]):
            w.writerow([a, cl, cd])
    size_kb = path.stat().st_size / 1024
    print(f"  → {path.name}  ({size_kb:.1f} KB)")


def write_csv_propulsion(data: dict[str, Any]) -> None:
    """Write propulsion table as CSV (one row per V × throttle cell)."""
    path = EXPORT_DIR / "propulsion.csv"
    V_vals = data["V"]
    thr_vals = data["throttle"]
    thrust = data["thrust"]
    power = data["power"]

    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["V", "throttle", "thrust", "power"])
        for i, V in enumerate(V_vals):
            for j, thr in enumerate(thr_vals):
                w.writerow([V, thr, thrust[i][j], power[i][j]])
    size_kb = path.stat().st_size / 1024
    print(f"  → {path.name}  ({size_kb:.1f} KB)")


# ===================================================================
#  main
# ===================================================================
def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- wing aero ----
    wing = build_wing_table()
    write_json(wing, "wing_aero")
    write_csv_wing(wing)

    print()

    # ---- propulsion ----
    prop = build_propulsion_table()
    write_json(prop, "propulsion")
    write_csv_propulsion(prop)

    print(f"\nAll tables exported → {EXPORT_DIR.resolve()}/")
    print("Ready for C# linear interpolation.")


if __name__ == "__main__":
    main()

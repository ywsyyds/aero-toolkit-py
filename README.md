# 本项目由claude code整理而成.md

Aerodynamic analysis toolkit: 2D airfoil → 3D wing → propeller → flight dynamics.

## Dependencies

- `numpy`, `matplotlib` (required)
- Experimental data in `NACA0012_data/` requires unzipping `NACA0012.zip`:
  ```bash
  unzip -o NACA0012.zip -d NACA0012_data/
  ```

## Commands

```bash
# 2D VLM validation (flat plate + NACA 2412 camber)
python main.py

# 3D VLM validation (AR sweep + elliptic wing)
python main3d.py

# Cp validation vs NASA NACA 0012 experiment
python main_naca0012_cp.py

# BEMT propeller performance curves
python main_bemt.py

# BEMT analytical vs experimental polar comparison
python main_bemt_n0012.py

# Propulsion black-box demo
python main_propulsion.py

# 3D horseshoe-vortex geometry visualisation
python wing3d.py

# Flight dynamics: takeoff, stall, cruise
python flight_sim.py

# Export lookup tables → export/*.json, *.csv (for C#/Unity)
python export_tables.py
```

## Project Architecture & Data Flow

```
                    ┌──────────────────────────────────────────┐
                    │           2D AIRFOIL                     │
                    │                                          │
      camberline.py ──► VLM2D (vlm.py)                        │
       (NACA camber    .solve(α) → Cl, Γ                       │
        line math)    .from_naca4() factory                    │
                    └────────────────────┬─────────────────────┘
                                         │ Cl, Γ
                    ┌────────────────────▼─────────────────────┐
                    │        3D WING (VLM)                     │
                    │                                          │
      wing3d.py ────► VLM3D (vlm3d.py)                         │
      (geometry     .solve(α) → CL, CDi, Γ(y)                 │
       viz only)    .rectangular(), .elliptic() factories      │
                    │  biot_savart_segment() core              │
                    └───────────┬─────────────────────────────┘
                                │ CL(α) pre-stall
                                ▼
                    ┌──────────────────────────────────────────┐
      polar.py ────►│ WingAero (flight.py)                     │
      (NACA 0012    │ .from_vlm(vlm, polar) → CL(α), CD(α)     │
       exp. lookup) │   blends VLM pre-stall + polar post-stall│
                    └───────────┬──────────────────────────────┘
                                │ CL(α), CD(α)
                                ▼
                    ┌──────────────────────────────────────────┐
      simple_polar  │         PROPELLER (BEMT)                 │
       (analytical) │                                          │
                    │ BEMTSolver (bemt.py)                      │
                    │   .solve(V_inf) → BEMTResult             │
                    │     (CT, CP, η + radial distributions)    │
                    │                                          │
      polar_table ──► PropulsionSystem (propulsion.py)          │
      (exp. polar)  │   .solve(V, throttle, pitch) → OpPoint   │
                    │   .curve(V_range, ...) → PerformanceCurve │
                    │   MotorSpec + PropGeom specs             │
                    └───────────┬──────────────────────────────┘
                                │ Thrust T(V, throttle)
                                ▼
                    ┌──────────────────────────────────────────┐
                    │    FLIGHT DYNAMICS                       │
                    │                                          │
      Aircraft (flight.py)  bundles WingAero + PropulsionSystem│
        .V_stall, .weight                                     │
                                                               │
      simulate(ac, dt, t_max, control_fn) → trajectory dict    │
        Explicit Euler, ground model, liftoff/stall detection  │
                    └──────────────────────────────────────────┘
```

### Validation Scripts

| Script | Compares | Key Concept |
|--------|----------|-------------|
| `main.py` | VLM2D vs thin-airfoil theory (`Cl = 2π·α` + `α₀` for camber) | 2D validation |
| `main3d.py` | VLM3D vs Prandtl lifting-line theory | 3D validation |
| `main_naca0012_cp.py` | VLM Cp(x) vs NASA NACA 0012 pressure data | Thickness-effect quantification |
| `main_bemt_n0012.py` | BEMT analytical polar vs NACA 0012 exp. polar | Polar-fidelity impact on efficiency |

### Module Details

**2D Airfoil Analysis** — `vlm.py` (`VLM2D`): point-vortex lattice on a thin airfoil. Each panel has a vortex at 1/4-chord and collocation at 3/4-chord. Supports camber via `dyc_dx` RHS. Factory `VLM2D.from_naca4()` builds from NACA 4-digit code. `camberline.py` provides `parse_naca4()` and `naca4_camber()` for camber-line geometry.

**3D Wing** — `vlm3d.py` (`VLM3D`): multi-panel 3D VLM solver. Each spanwise panel hosts one horseshoe vortex with finite-segment Biot-Savart induction (`biot_savart_segment()`). Builds influence matrix `A[i][j]` = z-velocity at ctrl point *i* from horseshoe *j*, solves `A·Γ = −V∞·sin(α)`. Single-chordwise-panel VLM systematically underestimates CLα ~5% — a known discretisation effect.

**Propeller** — `bemt.py` (`BEMTSolver`): couples 2D airfoil polars with actuator-disk momentum theory (Glauert, Prandtl tip/hub loss). Solves axial + tangential induction iteratively. `simple_polar()` provides analytical Cd₀ + k·Cl² drag model. Blade pitch defined by design advance ratio `J_design` + `α_design`.

**Experimental Data** — `polar.py` (`PolarTable`): callable Cl(α), Cd(α) lookup from NASA NACA 0012 data (Re=6×10⁶, 80/120/180-grit roughness). Factory `naca0012_polar()`. Linear interpolation with thin-airfoil fallback beyond measurement range.

**Propulsion Black-Box** — `propulsion.py` (`PropulsionSystem`): composes `MotorSpec` (throttle→RPM), `PropGeom` (diameter, blades, chord), and `BEMTSolver`. `PropulsionSystem.standard()` gives a NACA 0012 10-inch 2-blade prop. `curve()` caches BEMTSolver per (throttle, pitch) for warm-started sweeps.

**Flight Dynamics** — `flight.py`: `WingAero` merges VLM3D pre-stall CL with 2D polar post-stall shape via `_blend_stall()`. `Aircraft` bundles WingAero + PropulsionSystem + mass/geometry. `simulate()` runs explicit Euler integration with ground-interaction model (rolling friction, rotation, liftoff, touchdown, crash).

**C# / Unity Export** — `export_tables.py` writes flat JSON (Unity JsonUtility-compatible) + CSV to `export/`. `export/README.md` documents the schemas and includes C# bilinear-interpolation code for the propulsion grid.

### NACA0012_data/ Directory

Experimental validation data from NASA for the NACA 0012 airfoil:
- `Experimental/CP(0,10,15AOA).dat` — pressure coefficient at α=0°, 10°, 15°
- `Experimental/CD_CL(80,120,180roughness).dat` — Cl/Cd polars (3 roughness levels)
- `Computational/` — CFD reference solutions (structured/unstructured)
- `Geometries/NACA0012.igs` — CAD geometry
- `Grid/` — CGNS mesh files

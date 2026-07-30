"""3D Horseshoe Vortex Wing — Geometry visualization (Step 1, no aerodynamics).

Generates the vortex-lattice geometry for a rectangular wing and plots it
in 3D so the horseshoe-vortex topology can be verified by eye.

Coordinate system (standard flight-dynamics convention):
  x  — streamwise       (+x downstream, freestream direction)
  y  — spanwise          (+y starboard / right wing)
  z  — vertical          (+z up, lift direction)

Wing lies in the z = 0 plane,  x ∈ [0, c],  y ∈ [-b/2, b/2].

Each spanwise panel hosts one horseshoe vortex:

  Panel i :  y ∈ [y_i, y_{i+1}]

  bound vortex          (c/4, y_i, 0)  →  (c/4, y_{i+1}, 0)        [+y]
  left  trailing vortex (c/4, y_i, 0)  ←  (x_far, y_i, 0)         [-x]
  right trailing vortex (c/4, y_{i+1}, 0) → (x_far, y_{i+1}, 0)   [+x]
  (starting vortex at x_far closes the loop — not drawn.)

  control point         (3c/4,  (y_i + y_{i+1})/2,  0)

Circulation sign convention (right-hand rule):
  Γ > 0  →  bound vortex points in +y (left-to-right across the wing).
  Then, looking from upstream, the left trailing vortex runs upstream
  (toward the wing), the bound vortex crosses the wing rightward, and
  the right trailing vortex runs downstream — producing an upward (+z)
  induced velocity behind the bound segment.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers the '3d' projection


# ---------------------------------------------------------------------------
# 3-element vector type
# ---------------------------------------------------------------------------
Vec3 = NDArray[np.float64]  # shape (3,), dtype float64


# ===================================================================
class Wing3D:
    """Rectangular-wing geometry for the 3D horseshoe-vortex method.

    Parameters
    ----------
    span : float
        Wing span *b*.
    chord : float
        Constant chord length *c*.
    n_span : int
        Number of spanwise panels.
    x_far_factor : float
        Trailing-vortex far-field distance as a multiple of *span*
        (default 20).
    """

    def __init__(
        self,
        span: float = 10.0,
        chord: float = 1.0,
        n_span: int = 8,
        x_far_factor: float = 20.0,
    ) -> None:
        if n_span < 1:
            raise ValueError("n_span must be >= 1")
        self.span = span
        self.chord = chord
        self.n_span = n_span
        self.dy = span / n_span

        # ---- panel edges & centres in y -----------------------------------
        self.y_edges = np.linspace(-span / 2, span / 2, n_span + 1)
        self.y_ctr = 0.5 * (self.y_edges[:-1] + self.y_edges[1:])

        # ---- x-stations ----------------------------------------------------
        self.x_bound = chord / 4.0       # 1/4-chord  (vortex)
        self.x_ctrl = 3.0 * chord / 4.0  # 3/4-chord  (collocation)
        self.x_far = x_far_factor * span  # far downstream cutoff

    # ------------------------------------------------------------------
    #  Per-panel geometry accessors
    # ------------------------------------------------------------------
    def _yleft(self, i: int) -> float:
        return self.y_edges[i]

    def _yright(self, i: int) -> float:
        return self.y_edges[i + 1]

    def panel_bound_vortex(self, i: int) -> tuple[Vec3, Vec3]:
        """Return (start, end) of the bound vortex segment of panel *i*.

        Direction: -y → +y  (left → right), positive-Γ convention.
        """
        start = np.array([self.x_bound, self._yleft(i), 0.0])
        end = np.array([self.x_bound, self._yright(i), 0.0])
        return start, end

    def panel_left_trailing(self, i: int) -> tuple[Vec3, Vec3]:
        """Return (start, end) of the *left* trailing vortex.

        Direction: +x → -x  (far-field upstream toward the wing).
        """
        y = self._yleft(i)
        start = np.array([self.x_far, y, 0.0])   # far field
        end = np.array([self.x_bound, y, 0.0])   # wing
        return start, end

    def panel_right_trailing(self, i: int) -> tuple[Vec3, Vec3]:
        """Return (start, end) of the *right* trailing vortex.

        Direction: -x → +x  (wing downstream toward far field).
        """
        y = self._yright(i)
        start = np.array([self.x_bound, y, 0.0])  # wing
        end = np.array([self.x_far, y, 0.0])      # far field
        return start, end

    def panel_control_point(self, i: int) -> Vec3:
        """Return the (x, y, z) collocation point of panel *i*."""
        return np.array([self.x_ctrl, self.y_ctr[i], 0.0])

    # ------------------------------------------------------------------
    #  Visualisation
    # ------------------------------------------------------------------
    def plot_geometry(
        self, fname: str = "wing3d_geometry.png", dpi: int = 150
    ) -> None:
        """Draw the wing outline, all horseshoe vortices, and control points.

        Colour coding
        -------------
        • red   — bound vortex segments (with direction arrow)
        • blue  — trailing vortex filaments (with direction arrow)
        • green — control (collocation) points
        • grey  — wing outline
        """
        fig = plt.figure(figsize=(12, 7))
        ax: Axes3D = fig.add_subplot(111, projection="3d")

        # --- wing outline (semi-transparent) --------------------------------
        xw = np.array([0, self.chord, self.chord, 0, 0])
        yw = np.array([-1, -1, 1, 1, -1]) * self.span / 2
        zw = np.zeros_like(xw)
        ax.plot(xw, yw, zw, color="grey", lw=1.5, alpha=0.6, label="wing outline")
        # chord-line
        ax.plot([0, self.chord], [0, 0], [0, 0], color="grey", lw=0.8, ls="--", alpha=0.4)
        # quarter-chord line
        ax.plot(
            [self.x_bound, self.x_bound],
            [-self.span / 2, self.span / 2],
            [0, 0],
            color="grey", lw=0.8, ls=":", alpha=0.4,
        )

        # --- vortices & control points --------------------------------------
        arrow_len = 0.3  # fraction of segment length for quiver arrow

        for i in range(self.n_span):
            # --- bound vortex ---
            b_start, b_end = self.panel_bound_vortex(i)
            bx = [b_start[0], b_end[0]]
            by = [b_start[1], b_end[1]]
            bz = [b_start[2], b_end[2]]
            # first panel → label, rest → no label
            lbl = "bound vortex (1/4 c, +y)" if i == 0 else None
            ax.plot(bx, by, bz, color="#D4604E", lw=2, label=lbl)
            # direction arrow at midpoint
            b_mid = 0.5 * (b_start + b_end)
            b_dir = b_end - b_start
            b_arrow = b_dir / np.linalg.norm(b_dir) * arrow_len * np.linalg.norm(b_dir)
            ax.quiver(
                b_mid[0], b_mid[1], b_mid[2],
                b_arrow[0], b_arrow[1], b_arrow[2],
                color="#D4604E", linewidth=1.2, arrow_length_ratio=0.25,
            )

            # --- left trailing ---
            lt_start, lt_end = self.panel_left_trailing(i)
            lx = [lt_start[0], lt_end[0]]
            ly = [lt_start[1], lt_end[1]]
            lz = [lt_start[2], lt_end[2]]
            lbl = "trailing vortex (-x)" if i == 0 else None
            ax.plot(lx, ly, lz, color="#2C5F8A", lw=1.2, label=lbl)
            lt_mid = 0.5 * (lt_start + lt_end)
            lt_dir = lt_end - lt_start
            lt_arrow = lt_dir / np.linalg.norm(lt_dir) * arrow_len * np.linalg.norm(lt_dir)
            ax.quiver(
                lt_mid[0], lt_mid[1], lt_mid[2],
                lt_arrow[0], lt_arrow[1], lt_arrow[2],
                color="#2C5F8A", linewidth=1.0, arrow_length_ratio=0.25,
            )

            # --- right trailing ---
            rt_start, rt_end = self.panel_right_trailing(i)
            rx = [rt_start[0], rt_end[0]]
            ry = [rt_start[1], rt_end[1]]
            rz = [rt_start[2], rt_end[2]]
            lbl = "trailing vortex (+x)" if i == 0 else None
            ax.plot(rx, ry, rz, color="#3A7CA5", lw=1.2, ls="--", label=lbl)
            rt_mid = 0.5 * (rt_start + rt_end)
            rt_dir = rt_end - rt_start
            rt_arrow = rt_dir / np.linalg.norm(rt_dir) * arrow_len * np.linalg.norm(rt_dir)
            ax.quiver(
                rt_mid[0], rt_mid[1], rt_mid[2],
                rt_arrow[0], rt_arrow[1], rt_arrow[2],
                color="#3A7CA5", linewidth=1.0, arrow_length_ratio=0.25,
            )

            # --- control point ---
            cp = self.panel_control_point(i)
            lbl = "control point (3/4 c)" if i == 0 else None
            ax.scatter(*cp, color="#4CAF50", s=40, marker="o", zorder=5, label=lbl)

        # --- axes -----------------------------------------------------------
        ax.set_xlabel("x (streamwise)")
        ax.set_ylabel("y (spanwise)")
        ax.set_zlabel("z (vertical)")
        ax.set_title(
            f"3D Horseshoe Vortex Geometry  —  "
            f"span = {self.span}, chord = {self.chord}, N = {self.n_span}",
            fontweight="bold",
        )

        # equal aspect ratio on the geometry
        max_range = max(self.span, self.chord)
        ax.set_xlim(-0.1 * max_range, self.x_far + 0.1 * max_range)
        ax.set_ylim(-self.span / 2 - 0.5, self.span / 2 + 0.5)
        ax.set_zlim(-self.span / 4, self.span / 4)

        ax.legend(loc="upper left", fontsize="small", ncol=2)
        fig.tight_layout()
        fig.savefig(fname, dpi=dpi)
        plt.close(fig)
        print(f"Figure saved → {fname}")

    # ------------------------------------------------------------------
    def summary(self) -> None:
        """Print a text summary of all panels."""
        print(f"Wing3D:  span = {self.span},  chord = {self.chord},  "
              f"N_span = {self.n_span}")
        print(f"  x_bound = {self.x_bound:.4f}  (1/4 chord)")
        print(f"  x_ctrl  = {self.x_ctrl:.4f}  (3/4 chord)")
        print(f"  x_far   = {self.x_far:.4f}  (far-field cutoff)")
        print(f"  dy      = {self.dy:.4f}")
        print()
        hdr = f"{'panel':>6s}  {'y_left':>8s}  {'y_right':>8s}  "
        hdr += f"{'y_ctrl':>8s}  {'Γ direction':>14s}"
        print(hdr)
        print("-" * len(hdr))
        for i in range(self.n_span):
            yl = self._yleft(i)
            yr = self._yright(i)
            yc = self.y_ctr[i]
            b_start, b_end = self.panel_bound_vortex(i)
            if b_end[1] > b_start[1]:
                direction = "+y (left→right)"
            else:
                direction = "-y (right→left)"
            print(f"{i:6d}  {yl:8.4f}  {yr:8.4f}  {yc:8.4f}  {direction:>14s}")


# ===================================================================
def main() -> None:
    wing = Wing3D(span=10.0, chord=1.0, n_span=8)

    wing.summary()

    # print detailed vortex segment endpoints
    print(f"\n{'='*70}")
    print("Per-panel vortex segment endpoints")
    print(f"{'='*70}")
    for i in range(wing.n_span):
        b_s, b_e = wing.panel_bound_vortex(i)
        lt_s, lt_e = wing.panel_left_trailing(i)
        rt_s, rt_e = wing.panel_right_trailing(i)
        cp = wing.panel_control_point(i)
        print(f"\n--- Panel {i} ---")
        print(f"  Bound:       ({b_s[0]:8.4f}, {b_s[1]:8.4f}, {b_s[2]:8.4f})"
              f"  →  ({b_e[0]:8.4f}, {b_e[1]:8.4f}, {b_e[2]:8.4f})")
        print(f"  Left trail:  ({lt_s[0]:8.4f}, {lt_s[1]:8.4f}, {lt_s[2]:8.4f})"
              f"  →  ({lt_e[0]:8.4f}, {lt_e[1]:8.4f}, {lt_e[2]:8.4f})")
        print(f"  Right trail: ({rt_s[0]:8.4f}, {rt_s[1]:8.4f}, {rt_s[2]:8.4f})"
              f"  →  ({rt_e[0]:8.4f}, {rt_e[1]:8.4f}, {rt_e[2]:8.4f})")
        print(f"  Ctrl point:  ({cp[0]:8.4f}, {cp[1]:8.4f}, {cp[2]:8.4f})")

    wing.plot_geometry("wing3d_geometry.png")


if __name__ == "__main__":
    main()

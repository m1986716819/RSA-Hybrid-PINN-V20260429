"""Paper-level figure: coupling_alpha decision behaviour visualization.

Generates two core figures for the RHP-PINN paper:

Figure 1: Path overlay with alpha-weighted colouring
  - Scene top-down view (obstacles, start, goal)
  - RHP-PINN rollout path coloured by coupling_alpha
  - Colour bar: blue (PINN dominant) → red (RSA dominant)
  - Annotated phase regions: "Trap Entry", "Narrow Passage", "Open Area"

Figure 2: Alpha vs trajectory progress curve
  - x-axis: path progress (0% → 100%)
  - y-axis: coupling_alpha (0.0 → 1.0)
  - Horizontal dashed lines at alpha=0.5
  - Shaded regions for "RSA dominant" (alpha>0.5) and "PINN dominant" (alpha<0.5)
  - Key scene annotations at critical decision points
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

try:
    plt.rcParams["font.family"] = "Times New Roman"
except Exception:
    pass

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

_ALPHA_CMAP = LinearSegmentedColormap.from_list(
    "alpha_coupling",
    ["#2166AC", "#F7F7F7", "#B2182B"],  # blue-white-red
    N=256,
)


def _smooth_alpha(alpha: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving average for alpha trajectory."""
    if alpha.size < window:
        return alpha
    kernel = np.ones(window) / window
    return np.convolve(alpha, kernel, mode="same")


def _annotate_phase(ax, x_start: float, x_end: float, y_pos: float,
                    label: str, color: str = "gray"):
    ax.axvspan(x_start, x_end, alpha=0.08, color=color, zorder=1)
    ax.text((x_start + x_end) * 0.5, y_pos, label,
            ha="center", va="bottom", fontsize=9,
            color=color, fontstyle="italic")


class DecisionPlotter:
    """Generate coupling_alpha decision behaviour figures."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, name: str) -> Path:
        path = self.output_dir / name
        fig.savefig(path)
        plt.close(fig)
        svg_path = path.with_suffix(".svg")
        fig.savefig(svg_path, format="svg")
        plt.close(fig)
        return path

    def plot_decision_path(
        self,
        scene_grid: np.ndarray,
        rhp_path: np.ndarray,
        alpha_seq: np.ndarray,
        start_px: Tuple[int, int],
        goal_px: Tuple[int, int],
        rsa_path: Optional[np.ndarray] = None,
        ref_path: Optional[np.ndarray] = None,
        scene_name: str = "",
        save_name: str = "figure1_decision_path.pdf",
    ) -> Path:
        """Figure 1: Path overlay with alpha-weighted colouring.

        Args:
            scene_grid: (H, W) uint8, 0=free, 1=obstacle
            rhp_path: (N, 2) array of rollout waypoints in pixel coords
            alpha_seq: (M,) array of coupling_alpha values per rollout step
            start_px: (row, col) start point
            goal_px: (row, col) goal point
            rsa_path: optional (K, 2) RSA reference path in pixel coords
            ref_path: optional (L, 2) RSA high-res ref path
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))

        grid_display = np.where(scene_grid == 1, 1.0, 0.0)
        ax.imshow(grid_display, cmap="gray_r", origin="upper",
                  extent=[0, scene_grid.shape[1], scene_grid.shape[0], 0],
                  alpha=0.35, zorder=0)

        # RSA reference path (faint gray)
        if ref_path is not None and ref_path.shape[0] >= 2:
            ax.plot(ref_path[:, 1], ref_path[:, 0],
                    linewidth=2.5, color="gray", alpha=0.4,
                    linestyle="--", label="RSA Reference", zorder=2)

        # RHP-PINN path with alpha colouring
        if rhp_path.shape[0] >= 2 and alpha_seq.size > 0:
            alpha_smooth = _smooth_alpha(alpha_seq, window=3)
            n_seg = min(rhp_path.shape[0] - 1, alpha_smooth.size)
            for i in range(n_seg):
                y1, x1 = rhp_path[i]
                y2, x2 = rhp_path[i + 1]
                a = float(alpha_smooth[i])
                ax.plot([x1, x2], [y1, y2],
                        linewidth=2.8, color=_ALPHA_CMAP(a),
                        solid_capstyle="round", zorder=4)

            # Colour bar
            sm = plt.cm.ScalarMappable(
                cmap=_ALPHA_CMAP,
                norm=plt.Normalize(0, 1),
            )
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(r"Coupling $\alpha$ (0 = PINN, 1 = RSA)",
                           fontsize=11)
            cbar.ax.tick_params(labelsize=9)

        # Start and goal markers
        ax.plot(start_px[1], start_px[0], "o", color="#2ECC71",
                markersize=12, markeredgecolor="white",
                markeredgewidth=1.5, zorder=6, label="Start")
        ax.plot(goal_px[1], goal_px[0], "X", color="#E74C3C",
                markersize=14, markeredgecolor="white",
                markeredgewidth=1.5, zorder=6, label="Goal")

        # Phase annotations (auto-detected from alpha)
        if alpha_seq.size > 10:
            n = alpha_seq.size
            rsa_dominated = np.where(alpha_seq > 0.6)[0]
            pinn_dominated = np.where(alpha_seq < 0.4)[0]
            y_annot = scene_grid.shape[0] * 0.02

            if rsa_dominated.size > 0:
                start_idx = max(0, rsa_dominated[0] - 2)
                end_idx = min(n, rsa_dominated[-1] + 2)
                seg_len = rhp_path.shape[0] / n
                x_s = start_idx * seg_len
                x_e = end_idx * seg_len
                _annotate_phase(ax, x_s, x_e, y_annot,
                                "RSA\nDominant", "#B2182B")

            if pinn_dominated.size > 0:
                start_idx = max(0, pinn_dominated[0] - 2)
                end_idx = min(n, pinn_dominated[-1] + 2)
                seg_len = rhp_path.shape[0] / n
                x_s = start_idx * seg_len
                x_e = end_idx * seg_len
                _annotate_phase(ax, x_s, x_e, y_annot,
                                "PINN\nDominant", "#2166AC")

        ax.set_xlim(-2, scene_grid.shape[1] + 2)
        ax.set_ylim(scene_grid.shape[0] + 2, -2)
        ax.set_aspect("equal")
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        title = f"RHP-PINN Decision Behaviour"
        if scene_name:
            title += f" — {scene_name}"
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

        return self._save(fig, save_name)

    def plot_alpha_trajectory(
        self,
        alpha_seq: np.ndarray,
        path_length_px: float,
        scene_name: str = "",
        save_name: str = "figure2_alpha_trajectory.pdf",
    ) -> Path:
        """Figure 2: Alpha vs path progress curve with phase annotations."""
        fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))

        n = alpha_seq.size
        progress = np.linspace(0, 100, n)

        # Shaded dominance regions
        ax.axhspan(0.5, 1.0, alpha=0.06, color="#B2182B", label="RSA dominant")
        ax.axhspan(0.0, 0.5, alpha=0.06, color="#2166AC", label="PINN dominant")
        ax.axhline(y=0.5, color="gray", linewidth=0.8, linestyle="--",
                   alpha=0.7, label=r"$\alpha=0.5$ threshold")

        # Smoothed alpha curve
        alpha_smooth = _smooth_alpha(alpha_seq, window=5)
        ax.plot(progress, alpha_smooth, linewidth=2.0, color="#333333",
                label=r"Coupling $\alpha$ (smoothed)", zorder=3)

        # Raw alpha scatter
        ax.scatter(progress[::3], alpha_seq[::3], s=4, color="#666666",
                   alpha=0.3, zorder=2, label="Raw per-step")

        # Running mean overlay
        window = max(n // 20, 3)
        running = np.convolve(alpha_seq, np.ones(window) / window, mode="valid")
        rp = np.linspace(0, 100, running.size)
        ax.plot(rp, running, linewidth=1.2, color="#E67E22", linestyle=":",
                alpha=0.8, label=f"Running mean (w={window})")

        ax.set_xlabel("Path Progress (%)", fontsize=12)
        ax.set_ylabel(r"Coupling $\alpha$", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-2, 102)
        title = r"RHP-PINN Gating Decision Trajectory"
        if scene_name:
            title += f" — {scene_name}"
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9,
                  ncol=2)
        ax.grid(True, alpha=0.2)

        # Annotations
        mean_a = float(np.mean(alpha_seq))
        rsa_frac = float(np.mean(alpha_seq > 0.5))
        ax.text(0.98, 0.02,
                rf"$\bar{{\alpha}}$={mean_a:.2f}  "
                rf"RSA%={rsa_frac:.0%}",
                transform=ax.transAxes, fontsize=10,
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor="gray", alpha=0.8))

        return self._save(fig, save_name)
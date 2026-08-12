"""
Player Position Heatmap Generator for FootballVision.

Accumulates normalized (x, y) player positions over a game and
renders a PNG heatmap on request using matplotlib + numpy.
"""

import io
import numpy as np
from typing import List, Tuple, Optional


class HeatmapGenerator:
    """
    Accumulates player positions and renders heatmap PNGs.

    Example:
        hm = HeatmapGenerator()
        hm.add_positions([(0.3, 0.5), (0.6, 0.5)])
        png_bytes = hm.render()
    """

    def __init__(self, grid_w: int = 96, grid_h: int = 54):
        """
        Args:
            grid_w: Width of the accumulation grid in cells.
            grid_h: Height of the accumulation grid in cells.
        """
        self.grid_w = grid_w
        self.grid_h = grid_h
        self._grid = np.zeros((grid_h, grid_w), dtype=np.float32)
        self._total_frames = 0

    def add_positions(self, positions: List[Tuple[float, float]]):
        """
        Accumulate a list of normalized (x, y) positions ∈ [0, 1].

        Args:
            positions: List of (x_norm, y_norm) tuples.
        """
        for x, y in positions:
            xi = int(np.clip(x, 0, 0.9999) * self.grid_w)
            yi = int(np.clip(y, 0, 0.9999) * self.grid_h)
            self._grid[yi, xi] += 1
        self._total_frames += 1

    def add_tracks(self, tracks: list, frame_w: int = 1280, frame_h: int = 720):
        """
        Convenience wrapper — accepts Track objects directly.

        Args:
            tracks: List of Track objects with .center property.
            frame_w: Frame pixel width.
            frame_h: Frame pixel height.
        """
        positions = [
            (t.center[0] / frame_w, t.center[1] / frame_h)
            for t in tracks
            if hasattr(t, "center")
        ]
        self.add_positions(positions)

    def render(self, title: str = "Player Position Heatmap") -> bytes:
        """
        Render the heatmap as a PNG and return raw bytes.

        Args:
            title: Plot title.

        Returns:
            PNG image bytes.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap
        except ImportError:
            return self._fallback_png()

        # Smooth with gaussian filter
        try:
            from scipy.ndimage import gaussian_filter
            grid = gaussian_filter(self._grid, sigma=2.0)
        except ImportError:
            grid = self._grid

        fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0a0a14")
        ax.set_facecolor("#0a0a14")

        # Custom colormap: dark → green → yellow → red
        colors = ["#0a0a14", "#003322", "#00d4aa", "#ffb547", "#ff6b35", "#ff4757"]
        cmap = LinearSegmentedColormap.from_list("fv", colors)

        im = ax.imshow(grid, cmap=cmap, aspect="auto", origin="upper",
                       extent=[0, 1, 1, 0])

        # Field lines overlay
        ax.axhline(0.5, color="rgba(255,255,255,0.3)", linewidth=1, linestyle="--")
        ax.set_xticks([])
        ax.set_yticks([])

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
        cbar.set_label("Density", color="white", fontsize=9)

        ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=10)
        ax.text(0.01, 0.97, "← Offense →", transform=ax.transAxes,
                color="rgba(255,255,255,0.4)", fontsize=8, va="top")

        plt.tight_layout(pad=0.5)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120,
                    facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    def reset(self):
        """Clear accumulated data."""
        self._grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self._total_frames = 0

    @property
    def total_frames(self) -> int:
        return self._total_frames

    def _fallback_png(self) -> bytes:
        """Return a minimal 1×1 transparent PNG if matplotlib is unavailable."""
        return (
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
            b'\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89'
            b'\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01'
            b'\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
        )

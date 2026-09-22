"""Shared matplotlib styling for every figure script in this directory.

Every `fig_*.py` script imports `apply_theme()` and the palette constants
from here instead of picking colors or fonts ad hoc, so the figures across
all chapters read as one visual system. Figures are rendered once as static
PNGs into `docs/images/` and committed - Quarto does not execute Python at
render time (see `scripts/figures/README.md` for the rationale).

Palette matches the one used in this author's other two Quarto books
(optimization-lab, modern-ai-systems-and-methods) for a consistent look
across all three public sites.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# Fixed categorical order - always index into this rather than letting
# matplotlib auto-cycle, so a given role (e.g. "Gaussian", "Laplace") keeps
# the same color if it appears in more than one figure.
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# Sequential (light -> dark blue), for heatmaps / density / magnitude surfaces.
SEQUENTIAL_BLUE = [
    "#eaf2fc", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef",
    "#6da7ec", "#5598e7", "#3987e5", "#1f6dc9", "#0d366b",
]

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

FIGURE_DPI = 200  # retina-sharp static PNGs


def apply_theme() -> None:
    """Call once at the top of every fig_*.py before creating any figure."""
    plt.rcParams.update(
        {
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRIDLINE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.grid": True,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif",
            ],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "600",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "mathtext.fontset": "cm",
        }
    )


def savefig(fig, path: str) -> None:
    """Save with the shared tight-layout/padding convention."""
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {path}")

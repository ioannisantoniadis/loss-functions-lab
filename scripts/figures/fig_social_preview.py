"""Widescreen banner for the README hero image and the GitHub social / link preview
(2560x1280, 2x GitHub's recommended 1280x640).

This figure makes visible, at thumbnail size, what the book is: one spine of 18
chapters in reading order, each colored by where its loss comes from, with the
cross-links that tie later objectives back to earlier derivations.

CHAPTERS, FAMILY_COLOR and CROSS_LINKS are copied from fig_taxonomy_map.py (which
renders its own figure on import, so it cannot be imported here). Keep them in sync.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, CATEGORICAL, INK, MUTED, SURFACE

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

FOUNDATION, LIKELIHOOD, DECISION = "foundation", "likelihood", "decision theory"
SURROGATE, MODERN, SYNTHESIS = "surrogate / structured", "modern / hybrid", "synthesis"

CHAPTERS = {
    1: ("Learning\nproblem", FOUNDATION),
    2: ("Prob. to\nobjective", LIKELIHOOD),
    3: ("MLE, MAP,\nreg.", LIKELIHOOD),
    4: ("Gaussian\nto MSE", LIKELIHOOD),
    5: ("Laplace,\nHuber", LIKELIHOOD),
    6: ("Bernoulli/\ncat. to CE", LIKELIHOOD),
    7: ("Entropy,\nCE, KL", LIKELIHOOD),
    8: ("Decision\ntheory", DECISION),
    9: ("Surrogate\nlosses", SURROGATE),
    10: ("Ranking\nfoundations", SURROGATE),
    11: ("RankNet", SURROGATE),
    12: ("ListMLE", LIKELIHOOD),
    13: ("Optimizing\nrank metrics", SURROGATE),
    14: ("Contrastive\nobjectives", MODERN),
    15: ("LM\nobjectives", LIKELIHOOD),
    16: ("Preference\n/ DPO", MODERN),
    17: ("Loss &\noptimization", SYNTHESIS),
    18: ("Designing\na new loss", SYNTHESIS),
}

FAMILY_COLOR = {
    FOUNDATION: MUTED,
    LIKELIHOOD: CATEGORICAL[0],
    DECISION: CATEGORICAL[1],
    SURROGATE: CATEGORICAL[2],
    MODERN: CATEGORICAL[4],
    SYNTHESIS: CATEGORICAL[6],
}

CROSS_LINKS = [
    (3, 4), (3, 5), (1, 8), (6, 7), (6, 9), (9, 11),
    (11, 16), (12, 15), (6, 14), (15, 16), (3, 16),
]

W, H = 12.8, 6.4
X0, X1 = 0.75, 12.05   # x of chapter 1 and chapter 18 (inches = data units)
Y_SPINE = 5.0         # y grows downward


def x_of(c: int) -> float:
    return X0 + (c - 1) * (X1 - X0) / 17


def main() -> None:
    apply_theme()
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_axis_off()

    ax.text(0.55, 0.45, "loss-functions-lab", fontsize=25, fontweight="bold", color=INK,
            ha="left", va="top")
    ax.text(0.55, 1.12, "Where cross-entropy, MSE, hinge loss, RankNet and DPO actually come "
            "from: derived from first principles, not catalogued", fontsize=12, color=MUTED,
            ha="left", va="top")

    # Origin key, directly under the subtitle.
    x = 0.55
    for family, color in FAMILY_COLOR.items():
        ax.plot([x + 0.08], [1.78], "o", color=color, markersize=8)
        t = ax.text(x + 0.22, 1.78, family, fontsize=9.5, color=MUTED, ha="left", va="center")
        fig.canvas.draw()
        bb = t.get_window_extent().transformed(ax.transData.inverted())
        x = bb.x1 + 0.38

    # Spine in reading order.
    ax.plot([x_of(1), x_of(18)], [Y_SPINE, Y_SPINE], color="#d8d6cf", linewidth=3,
            solid_capstyle="round", zorder=1)

    # Cross-links as arcs above the spine; long links are relatively flatter so the
    # tallest arc stays inside the band below the key.
    for a, b in CROSS_LINKS:
        dist = abs(b - a)
        rad = -(0.6 - 0.017 * dist)
        ax.add_patch(FancyArrowPatch(
            (x_of(a), Y_SPINE), (x_of(b), Y_SPINE), connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>", mutation_scale=11, color="#5b5a55", linewidth=1.2,
            alpha=0.6, shrinkA=13, shrinkB=13, zorder=2,
        ))

    for c, (label, family) in CHAPTERS.items():
        ax.scatter([x_of(c)], [Y_SPINE], s=640, color=FAMILY_COLOR[family], zorder=4,
                   edgecolor=SURFACE, linewidth=2)
        ax.text(x_of(c), Y_SPINE, str(c), ha="center", va="center", fontsize=10.5,
                color="white", fontweight="bold", zorder=5)
        ax.text(x_of(c), Y_SPINE + 0.36, label, ha="center", va="top", fontsize=7.7,
                color=INK, linespacing=1.15, zorder=5)

    out = Path(__file__).resolve().parents[2] / "docs" / "images" / "social_preview.png"
    fig.savefig(str(out), dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

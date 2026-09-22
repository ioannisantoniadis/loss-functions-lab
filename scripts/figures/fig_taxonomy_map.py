import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK, INK_SECONDARY, MUTED

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

apply_theme()

# Chapter number -> (short label, family). Family drives node color.
FOUNDATION = "foundation"
LIKELIHOOD = "likelihood"
DECISION = "decision theory"
SURROGATE = "surrogate / structured"
MODERN = "modern / hybrid"
SYNTHESIS = "synthesis"

CHAPTERS = {
    1:  ("Learning\nproblem", FOUNDATION),
    2:  ("Prob. to\nobjective", LIKELIHOOD),
    3:  ("MLE, MAP,\nreg.", LIKELIHOOD),
    4:  ("Gaussian\nto MSE", LIKELIHOOD),
    5:  ("Laplace,\nHuber", LIKELIHOOD),
    6:  ("Bernoulli/cat.\nto CE", LIKELIHOOD),
    7:  ("Entropy,\nCE, KL", LIKELIHOOD),
    8:  ("Decision\ntheory", DECISION),
    9:  ("Surrogate\nlosses", SURROGATE),
    10: ("Ranking\nfoundations", SURROGATE),
    11: ("RankNet", SURROGATE),
    12: ("ListMLE", LIKELIHOOD),
    13: ("Optimizing\nrank metrics", SURROGATE),
    14: ("Contrastive\nobjectives", MODERN),
    15: ("LM\nobjectives", LIKELIHOOD),
    16: ("Preference /\nDPO", MODERN),
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

# Main spine: the book's reading order (drawn as a straight connecting line).
SPINE = [(i, i + 1) for i in range(1, 18)]

# Curated cross-links: the explicit callbacks named in each chapter's
# "Connections" section -- not exhaustive, just the most illuminating ones,
# drawn as arcs above the spine so the linear progression stays legible.
CROSS_LINKS = [
    (3, 4), (3, 5),        # Gaussian/Laplace: prior (Ch3) vs likelihood (Ch4/5)
    (1, 8),                 # Ch8 resolves Ch1's opening figure
    (6, 7),                 # cross-entropy formalized as H(P,Q)
    (6, 9),                 # BCE vs logistic-surrogate reconciliation
    (9, 11),                # logistic loss reused as RankNet's pairwise loss
    (11, 16),                # Bradley-Terry reused verbatim in DPO
    (12, 15),                # chain-of-categorical-choices, ranking vs sequences
    (6, 14),                 # InfoNCE as categorical cross-entropy
    (15, 16),                # pretraining feeds preference tuning
    (3, 16),                 # KL/regularization-strength theme echoed by beta
]

fig, ax = plt.subplots(figsize=(17, 7.5))

xs = {c: c for c in CHAPTERS}
y_spine = 0.0

# Spine (reading order)
for a, b in SPINE:
    ax.plot([xs[a], xs[b]], [y_spine, y_spine], color="#d8d6cf", linewidth=2.4,
             zorder=1, solid_capstyle="round")

# Cross-links as arcs above the spine, height ~ proportional to distance
for a, b in CROSS_LINKS:
    dist = abs(xs[b] - xs[a])
    rad = -(0.25 + 0.012 * dist)  # negative rad => arcs bow upward, into empty space
    arrow = FancyArrowPatch((xs[a], y_spine), (xs[b], y_spine),
                              connectionstyle=f"arc3,rad={rad}",
                              arrowstyle="-|>", mutation_scale=10,
                              color=INK_SECONDARY, linewidth=1.1, alpha=0.55,
                              zorder=2)
    ax.add_patch(arrow)

# Nodes
for c, (label, family) in CHAPTERS.items():
    color = FAMILY_COLOR[family]
    ax.scatter([xs[c]], [y_spine], s=520, color=color, zorder=4,
                edgecolor="white", linewidth=1.5)
    ax.text(xs[c], y_spine, str(c), ha="center", va="center", zorder=5,
             fontsize=9.5, color="white", fontweight="bold")
    ax.text(xs[c], y_spine - 0.62, label, ha="center", va="top", zorder=5,
             fontsize=8.2, color=INK)

ax.set_xlim(0, 19)
ax.set_ylim(-1.1, 3.3)
ax.axis("off")
ax.set_title("The book's spine (chapter order) and its main cross-links", pad=10, y=1.02)

# Legend for families
handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=10,
                        markerfacecolor=color, markeredgecolor="white",
                        label=family)
            for family, color in FAMILY_COLOR.items()]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.16),
           ncol=6, fontsize=8.6, frameon=False)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "taxonomy_map.png"
savefig(fig, str(out))

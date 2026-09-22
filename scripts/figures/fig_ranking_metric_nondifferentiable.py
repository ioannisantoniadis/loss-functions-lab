import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK_SECONDARY

import numpy as np
import matplotlib.pyplot as plt

apply_theme()

# Five items with fixed relevance grades. Item A's score is swept
# continuously; the other four items keep fixed scores. Everything else
# about the list is held constant, so any change in NDCG is caused purely
# by A's score crossing one of the other four fixed scores and changing
# the sort order.
names = ["A", "B", "C", "D", "E"]
relevance = np.array([3, 2, 1, 0, 2], dtype=float)
fixed_scores = np.array([np.nan, 1.5, 0.5, -1.0, 1.0])

ideal_order = np.sort(relevance)[::-1]
idcg = np.sum((2 ** ideal_order - 1) / np.log2(np.arange(2, len(ideal_order) + 2)))


def ndcg_at_k(scores, relevance, k=5):
    order = np.argsort(-scores)
    ranked_rel = relevance[order][:k]
    gains = (2 ** ranked_rel - 1) / np.log2(np.arange(2, len(ranked_rel) + 2))
    return gains.sum() / idcg


a_scores = np.linspace(-3, 3, 2000)
ndcg_values = np.empty_like(a_scores)
for i, a in enumerate(a_scores):
    scores = fixed_scores.copy()
    scores[0] = a
    ndcg_values[i] = ndcg_at_k(scores, relevance)

fig, ax = plt.subplots(figsize=(7.6, 4.6))
ax.plot(a_scores, ndcg_values, color=CATEGORICAL[0], linewidth=2.2)

for thresh in [1.5, 1.0, 0.5, -1.0]:
    ax.axvline(thresh, color=INK_SECONDARY, linewidth=0.7, linestyle=":", alpha=0.6)

ax.set_xlabel(r"score of item A, $s_A$ (all other items' scores fixed)")
ax.set_ylabel(r"$\mathrm{NDCG@5}$")
ax.set_title("A ranking metric as a function of one item's score")
ax.set_ylim(ndcg_values.min() - 0.03, 1.03)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "ranking_metric_nondifferentiable.png"
savefig(fig, str(out))

print("NDCG plateaus (value, count):",
      sorted(set(np.round(ndcg_values, 4))))

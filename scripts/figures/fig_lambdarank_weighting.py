import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK_SECONDARY

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

apply_theme()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dcg(ordered_rel):
    ordered_rel = np.asarray(ordered_rel, dtype=float)
    ranks = np.arange(1, len(ordered_rel) + 1)
    return np.sum((2 ** ordered_rel - 1) / np.log2(ranks + 1))


# Five items: relevance grades (ground truth) and current, partly
# misordered model scores.
items = [1, 2, 3, 4, 5]
relevance = {1: 3, 2: 2, 3: 2, 4: 1, 5: 0}
score = {1: 0.5, 2: 2.0, 3: -0.5, 4: 1.0, 5: -1.5}

current_order = sorted(items, key=lambda i: -score[i])
ideal_order = sorted(items, key=lambda i: -relevance[i])
idcg = dcg([relevance[i] for i in ideal_order])
current_ndcg = dcg([relevance[i] for i in current_order]) / idcg


def ndcg_after_swap(order, a, b):
    order = order.copy()
    ia, ib = order.index(a), order.index(b)
    order[ia], order[ib] = order[ib], order[ia]
    return dcg([relevance[i] for i in order]) / idcg


# All pairs (i, j) where i should outrank j (relevance_i > relevance_j) --
# exactly the pairs RankNet would train on.
pairs = [(i, j) for i, j in combinations(items, 2) if relevance[i] != relevance[j]]
pairs = [(i, j) if relevance[i] > relevance[j] else (j, i) for i, j in pairs]

records = []
for i, j in pairs:
    plain_grad = sigmoid(-(score[i] - score[j]))  # RankNet pull magnitude on s_i
    delta_ndcg = abs(ndcg_after_swap(current_order, i, j) - current_ndcg)
    records.append((rf"${i}\succ{j}$", plain_grad, delta_ndcg, plain_grad * delta_ndcg))

records.sort(key=lambda r: -r[3])
labels = [r[0] for r in records]
plain = np.array([r[1] for r in records])
weighted = np.array([r[3] for r in records])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.4, 6.2), sharex=True)

x = np.arange(len(labels))
ax1.bar(x, plain, color=CATEGORICAL[0])
ax1.set_ylabel("plain RankNet\npairwise gradient")
ax1.set_title("Same pairs, before metric-aware weighting")

ax2.bar(x, weighted, color=CATEGORICAL[1])
ax2.set_ylabel(r"$\times\,|\Delta\mathrm{NDCG}|$" + "\n(LambdaRank-style)")
ax2.set_xlabel("pair (ground-truth preference)")
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_title("After weighting by the NDCG change from swapping the pair")

fig.suptitle("Pairs near the top of the list get amplified; deep pairs shrink", y=1.02)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "lambdarank_weighting.png"
savefig(fig, str(out))

print(f"current NDCG@5 = {current_ndcg:.4f}")
for r in records:
    print(f"{r[0]}: plain={r[1]:.3f} |dNDCG|={r[2]:.3f} weighted={r[3]:.3f}")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK_SECONDARY, MUTED

import numpy as np
import matplotlib.pyplot as plt

apply_theme()

# Four items with fixed scores; the *observed* ranking (B, A, D, C) is
# deliberately not the same as sorting by score (which would give
# A, B, C, D) so the sequential construction has something nontrivial to
# show at each step, and the resulting ListMLE loss is not zero.
items = ["A", "B", "C", "D"]
scores = {"A": 2.0, "B": 1.0, "C": 0.5, "D": -1.0}
observed_order = ["B", "A", "D", "C"]

remaining = list(items)
step_probs = []
picked_probs = []

fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.6), sharey=True)

for step, ax in enumerate(axes):
    s = np.array([scores[i] for i in remaining])
    p = np.exp(s) / np.exp(s).sum()
    pick = observed_order[step]
    pick_idx = remaining.index(pick)
    step_probs.append(dict(zip(remaining, p)))
    picked_probs.append(p[pick_idx])

    colors = [CATEGORICAL[2] if i == pick_idx else MUTED for i in range(len(remaining))]
    ax.bar(remaining, p, color=colors)
    for i, val in enumerate(p):
        ax.text(i, val + 0.03, f"{val:.2f}", ha="center", fontsize=8.5, color=INK_SECONDARY)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"step {step+1}: pick \"{pick}\"", fontsize=11)
    if step == 0:
        ax.set_ylabel("softmax probability\nover remaining items")

    remaining.remove(pick)

fig.suptitle(
    "ListMLE's sequential construction: a chain of categorical choices",
    y=1.05, fontsize=12.5,
)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "listmle_construction.png"
savefig(fig, str(out))

nll = -sum(np.log(p) for p in picked_probs)
print("observed order:", observed_order)
print("per-step picked probabilities:", [f"{p:.4f}" for p in picked_probs])
print(f"ListMLE loss (-log P(pi)) = {nll:.4f}")

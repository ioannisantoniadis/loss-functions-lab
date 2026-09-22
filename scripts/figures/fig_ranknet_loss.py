import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK_SECONDARY

import numpy as np
import matplotlib.pyplot as plt

apply_theme()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


delta = np.linspace(-6, 6, 500)  # Delta = s_i - s_j, for an observed i > j
loss = np.log1p(np.exp(-delta))          # -log sigmoid(delta)
pull_on_si = sigmoid(-delta)             # -d(loss)/d(s_i) = sigmoid(-delta)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.4))

ax1.plot(delta, loss, color=CATEGORICAL[0], linewidth=2.2)
ax1.axvline(0, color=INK_SECONDARY, linewidth=0.8, alpha=0.5)
ax1.set_xlabel(r"score difference $\Delta = s_i - s_j$")
ax1.set_ylabel(r"loss $= -\log\sigma(\Delta)$")
ax1.set_title("RankNet loss for an observed $i \\succ j$")

ax2.plot(delta, pull_on_si, color=CATEGORICAL[2], linewidth=2.2)
ax2.axvline(0, color=INK_SECONDARY, linewidth=0.8, alpha=0.5)
ax2.axhline(0.5, color=INK_SECONDARY, linewidth=0.6, linestyle=":", alpha=0.5)
ax2.set_xlabel(r"score difference $\Delta = s_i - s_j$")
ax2.set_ylabel(r"upward pull on $s_i$, $-\partial\,\text{loss}/\partial s_i$")
ax2.set_title("Gradient: how hard the pair pulls $s_i$ up")
ax2.set_ylim(-0.02, 1.02)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "ranknet_loss.png"
savefig(fig, str(out))

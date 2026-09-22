import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK_SECONDARY

import numpy as np
import matplotlib.pyplot as plt

apply_theme()

p = np.linspace(1e-4, 1 - 1e-4, 500)
loss_y1 = -np.log(p)        # loss paid when the true label is y=1
loss_y0 = -np.log(1 - p)    # loss paid when the true label is y=0

fig, ax = plt.subplots(figsize=(7.4, 4.8))
ax.plot(p, loss_y1, color=CATEGORICAL[0], linewidth=2.4, label=r"$y=1$: $-\log p$")
ax.plot(p, loss_y0, color=CATEGORICAL[7], linewidth=2.4, label=r"$y=0$: $-\log(1-p)$")

ax.axvline(0.5, color=INK_SECONDARY, linewidth=0.8, alpha=0.5)
ax.set_xlabel(r"predicted probability $p = \sigma(z)$")
ax.set_ylabel("binary cross-entropy loss")
ax.set_ylim(0, 5)
ax.set_title("BCE punishes confident, wrong predictions steeply")
ax.legend(loc="upper center", fontsize=10)

ax.annotate("confidently wrong\n(y=1, p near 0)", xy=(0.082, 2.5),
            xytext=(0.22, 1.65), fontsize=8.5, color=INK_SECONDARY,
            arrowprops=dict(arrowstyle="->", color=INK_SECONDARY, lw=1))
ax.annotate("confidently wrong\n(y=0, p near 1)", xy=(0.918, 2.5),
            xytext=(0.56, 1.65), fontsize=8.5, color=INK_SECONDARY,
            arrowprops=dict(arrowstyle="->", color=INK_SECONDARY, lw=1))

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "bce_curve.png"
savefig(fig, str(out))

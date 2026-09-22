import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK_SECONDARY

import numpy as np
import matplotlib.pyplot as plt

apply_theme()

m = np.linspace(-3, 3, 400)

zero_one = (m < 0).astype(float)
hinge = np.maximum(0, 1 - m)
logistic = np.log1p(np.exp(-m)) / np.log(2)  # normalized to pass through (0,1) like hinge

fig, ax = plt.subplots(figsize=(7.4, 4.8))

ax.plot(m, zero_one, color=INK_SECONDARY, linewidth=2.0, linestyle="--",
        label="0–1 loss: $\\mathbb{1}\\{m<0\\}$")
ax.plot(m, hinge, color=CATEGORICAL[0], linewidth=2.4,
        label="hinge: $\\max(0,\\,1-m)$")
ax.plot(m, logistic, color=CATEGORICAL[7], linewidth=2.4,
        label="logistic: $\\log_2(1+e^{-m})$")

ax.axvline(0, color=INK_SECONDARY, linewidth=0.7, alpha=0.5)
ax.axvline(1, color=CATEGORICAL[0], linewidth=0.7, alpha=0.4, linestyle=":")
ax.set_xlabel(r"margin $m = y \cdot z$")
ax.set_ylabel("loss")
ax.set_ylim(-0.15, 3.2)
ax.set_title("Two convex surrogates for a non-convex target")
ax.legend(loc="upper right", fontsize=9.5)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "surrogate_losses.png"
savefig(fig, str(out))

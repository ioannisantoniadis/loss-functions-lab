import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, SEQUENTIAL_BLUE, INK_SECONDARY, INK

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

apply_theme()

n = 400
p1 = np.linspace(1e-3, 1 - 1e-3, n)
p2 = np.linspace(1e-3, 1 - 1e-3, n)
P1, P2 = np.meshgrid(p1, p2)

# Valid region: p1 + p2 <= 1 (so p3 = 1 - p1 - p2 >= 0) -- this triangle
# *is* the probability simplex for 3 classes, parameterized by its first
# two coordinates.
valid = (P1 + P2) <= 1.0

# Cross-entropy loss when the true class is class 1: -log(p1). Masked
# outside the simplex.
loss = -np.log(P1)
loss_masked = np.ma.array(loss, mask=~valid)

cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)

fig, ax = plt.subplots(figsize=(6.6, 5.6))
levels = np.linspace(0, 5, 21)
cf = ax.contourf(P1, P2, np.clip(loss_masked, 0, 5), levels=levels, cmap=cmap, extend="max")
cbar = fig.colorbar(cf, ax=ax, shrink=0.85)
cbar.set_label(r"loss $= -\log p_1$ (true class is class 1)", color=INK_SECONDARY)

# Draw the simplex boundary (the hypotenuse p1+p2=1) and axes.
ax.plot([0, 1], [1, 0], color=INK, linewidth=1.3)
ax.plot([0, 0], [0, 1], color=INK, linewidth=1.3)
ax.plot([0, 1], [0, 0], color=INK, linewidth=1.3)

ax.scatter([1], [0], color=INK, zorder=5, s=25)
ax.annotate("certain, correct\n(loss = 0)", xy=(1, 0), xytext=(0.68, 0.12),
            fontsize=8.5, color=INK_SECONDARY,
            arrowprops=dict(arrowstyle="->", color=INK_SECONDARY, lw=1))
ax.annotate("$p_1 \\to 0$: loss blows up", xy=(0.02, 0.55), xytext=(0.08, 0.75),
            fontsize=8.5, color=INK_SECONDARY,
            arrowprops=dict(arrowstyle="->", color=INK_SECONDARY, lw=1))

ax.set_xlabel(r"$p_1$")
ax.set_ylabel(r"$p_2$   (with $p_3 = 1-p_1-p_2$)")
ax.set_title("Cross-entropy over the 3-class simplex")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "categorical_cross_entropy_surface.png"
savefig(fig, str(out))

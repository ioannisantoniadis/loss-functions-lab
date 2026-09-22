import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK_SECONDARY

import numpy as np
import matplotlib.pyplot as plt

apply_theme()

delta = 1.0
r = np.linspace(-3, 3, 600)

squared = 0.5 * r ** 2
absolute = np.abs(r)
huber = np.where(np.abs(r) <= delta, 0.5 * r ** 2, delta * (np.abs(r) - 0.5 * delta))

grad_squared = r
grad_absolute = np.sign(r)
grad_huber = np.clip(r, -delta, delta)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4), sharex=True)

for curve, color, label in [
    (squared, CATEGORICAL[0], r"squared: $\frac{1}{2}r^2$"),
    (absolute, CATEGORICAL[1], r"absolute: $|r|$"),
    (huber, CATEGORICAL[2], rf"Huber ($\delta={delta:g}$)"),
]:
    ax1.plot(r, curve, color=color, linewidth=2.2, label=label)

for curve, color, label in [
    (grad_squared, CATEGORICAL[0], "squared"),
    (grad_absolute, CATEGORICAL[1], "absolute"),
    (grad_huber, CATEGORICAL[2], "Huber"),
]:
    ax2.plot(r, curve, color=color, linewidth=2.2, label=label)

ax1.axvline(delta, color=INK_SECONDARY, linestyle=":", linewidth=1, alpha=0.6)
ax1.axvline(-delta, color=INK_SECONDARY, linestyle=":", linewidth=1, alpha=0.6)
ax1.set_xlabel(r"residual $r$")
ax1.set_ylabel("loss")
ax1.set_title("Loss vs. residual")
ax1.legend(loc="upper center", fontsize=9)
ax1.set_ylim(-0.1, 4.5)

ax2.axvline(delta, color=INK_SECONDARY, linestyle=":", linewidth=1, alpha=0.6)
ax2.axvline(-delta, color=INK_SECONDARY, linestyle=":", linewidth=1, alpha=0.6)
ax2.set_xlabel(r"residual $r$")
ax2.set_ylabel(r"$d(\text{loss})/dr$")
ax2.set_title("Gradient vs. residual")
ax2.legend(loc="upper left", fontsize=9)

fig.suptitle(r"MSE's gradient grows without bound; MAE's is constant "
             r"but kinked at 0; Huber is bounded and smooth",
             y=1.04, fontsize=11.5, color=INK_SECONDARY)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "mse_mae_huber_comparison.png"
savefig(fig, str(out))

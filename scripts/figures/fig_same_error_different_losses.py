import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK_SECONDARY

import numpy as np
import matplotlib.pyplot as plt

apply_theme()

r = np.linspace(-3, 3, 400)

squared = r ** 2
absolute = np.abs(r)
# tau = 0.8 quantile (pinball) loss: penalizes under-prediction 4x harder
# than over-prediction. Same residual, very different penalty, and not even
# symmetric -- exactly the point of this figure.
tau = 0.8
quantile = np.where(r >= 0, tau * r, (tau - 1) * r)
zero_one = (np.abs(r) > 0.75).astype(float) * 1.6  # rescaled for display

fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.plot(r, squared, color=CATEGORICAL[0], linewidth=2.4, label=r"squared: $r^2$")
ax.plot(r, absolute, color=CATEGORICAL[1], linewidth=2.4, label=r"absolute: $|r|$")
ax.plot(r, quantile, color=CATEGORICAL[2], linewidth=2.4,
        label=r"asymmetric ($\tau=0.8$ pinball)")
ax.plot(r, zero_one, color=CATEGORICAL[7], linewidth=2.0, linestyle="--",
        label="0–1 (rescaled): only 'close enough' matters")

ax.axvline(0, color=INK_SECONDARY, linewidth=0.8, alpha=0.6)
ax.set_xlabel(r"residual $r = y - \hat y$")
ax.set_ylabel("loss")
ax.set_ylim(-0.2, 4.5)
ax.set_title("The same residual, penalized four different ways")
ax.legend(loc="upper center", ncol=2, fontsize=8.5)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "same_error_different_losses.png"
savefig(fig, str(out))

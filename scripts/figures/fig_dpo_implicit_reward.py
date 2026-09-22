import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK_SECONDARY

import numpy as np
import matplotlib.pyplot as plt

apply_theme()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dpo_loss(margin):
    # -log sigmoid(margin), same logistic-loss family as Chapters 9 and 11.
    return np.log1p(np.exp(-margin))


# Panel 1: the DPO loss as a function of the beta-scaled margin
#   m = beta * [logratio(y+) - logratio(y-)]
# -- deliberately the same curve shape as Chapter 9's logistic surrogate
# and Chapter 11's RankNet loss, now with the "score" being a difference
# of log-probability ratios instead of a learned scalar score.
m = np.linspace(-6, 6, 500)
loss_m = dpo_loss(m)

# Panel 2: fix the RAW log-ratio difference
#   Delta = logratio(y+) - logratio(y-)
# and vary beta, showing beta rescale that same fixed Delta into a
# steeper or shallower effective margin beta*Delta -- i.e. beta controls
# how hard the loss pushes for a given amount of raw preference signal.
delta = np.linspace(-4, 4, 500)
betas = [0.1, 0.5, 1.0, 3.0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6))

ax1.plot(m, loss_m, color=CATEGORICAL[0], linewidth=2.2)
ax1.axvline(0, color=INK_SECONDARY, linewidth=0.8, alpha=0.5)
ax1.set_xlabel(r"margin $\beta\left[\log\frac{\pi_\theta(y^+|x)}{\pi_{\mathrm{ref}}(y^+|x)} - \log\frac{\pi_\theta(y^-|x)}{\pi_{\mathrm{ref}}(y^-|x)}\right]$")
ax1.set_ylabel(r"$\mathcal{L}_{\mathrm{DPO}}$")
ax1.set_title("Same logistic-loss-vs-margin shape as Ch. 9 and Ch. 11")

for i, beta in enumerate(betas):
    ax2.plot(delta, dpo_loss(beta * delta), color=CATEGORICAL[i], linewidth=2.2,
              label=rf"$\beta={beta}$")
ax2.axvline(0, color=INK_SECONDARY, linewidth=0.8, alpha=0.5)
ax2.set_xlabel(r"raw log-ratio difference $\Delta = \log\frac{\pi_\theta(y^+|x)}{\pi_{\mathrm{ref}}(y^+|x)} - \log\frac{\pi_\theta(y^-|x)}{\pi_{\mathrm{ref}}(y^-|x)}$")
ax2.set_ylabel(r"$\mathcal{L}_{\mathrm{DPO}}$")
ax2.set_title(r"Same raw $\Delta$, varying $\beta$: steepness, not shape, changes")
ax2.legend(loc="upper right", fontsize=9)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "dpo_implicit_reward.png"
savefig(fig, str(out))

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK_SECONDARY

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

apply_theme()

# Three assumed noise levels for the same Gaussian conditional model
# Y | X=x ~ N(f_theta(x), sigma^2). Only sigma changes; the resulting
# per-residual loss is always a parabola, but a wider assumed noise level
# flattens it -- the model is told to tolerate bigger errors there.
sigmas = [0.6, 1.0, 1.8]
colors = [CATEGORICAL[0], CATEGORICAL[2], CATEGORICAL[7]]

r = np.linspace(-4, 4, 400)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4))

for sigma, color in zip(sigmas, colors):
    density = stats.norm.pdf(r, loc=0, scale=sigma)
    ax1.plot(r, density, color=color, linewidth=2.2, label=rf"$\sigma={sigma}$")

    # NLL(r) up to the additive constant (1/2)log(2*pi*sigma^2), which does
    # not depend on the residual and so does not affect the loss's *shape*.
    nll = r ** 2 / (2 * sigma ** 2)
    ax2.plot(r, nll, color=color, linewidth=2.2, label=rf"$\sigma={sigma}$")

ax1.set_xlabel(r"residual $r = y - f_\theta(x)$")
ax1.set_ylabel(r"density $p(r \mid \sigma)$")
ax1.set_title("Assumed noise: Gaussian densities")
ax1.legend(loc="upper right", fontsize=9)

ax2.set_xlabel(r"residual $r = y - f_\theta(x)$")
ax2.set_ylabel(r"$r^2 / (2\sigma^2)$  (loss, up to a constant)")
ax2.set_title("Resulting per-residual loss")
ax2.set_ylim(0, 10)
ax2.legend(loc="upper center", fontsize=9)

fig.suptitle("A wider assumed noise level flattens the loss it implies",
             y=1.03, fontsize=12, color=INK_SECONDARY)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "gaussian_to_mse.png"
savefig(fig, str(out))

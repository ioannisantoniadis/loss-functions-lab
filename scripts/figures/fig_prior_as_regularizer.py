import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK_SECONDARY

import numpy as np
import matplotlib.pyplot as plt

apply_theme()

# A single scalar weight w, estimated from n=3 noisy observations with
# sample mean y_bar=0.7 and known noise variance sigma^2=1 -- deliberately
# weak evidence, so the prior's pull is visible. Every number below is
# chosen so the three minimizers land at clean, exactly-computable values
# (see chapter 3 for the closed forms).
n, sigma2 = 3, 1.0
y_bar = 0.7
tau2 = 1.0     # Gaussian prior variance
b = 0.3        # Laplace prior scale

A = n / sigma2  # precision of the likelihood term

w = np.linspace(-0.6, 1.6, 500)

nll = 0.5 * A * (w - y_bar) ** 2
gaussian_neg_log_post = nll + w ** 2 / (2 * tau2)
laplace_neg_log_post = nll + np.abs(w) / b

w_mle = y_bar
w_map_gauss = y_bar * A / (A + 1 / tau2)
threshold = 1 / (b * A)
w_map_laplace = np.sign(y_bar) * max(abs(y_bar) - threshold, 0.0)

fig, ax = plt.subplots(figsize=(7.6, 4.8))

for curve, color, label, w_star in [
    (nll, CATEGORICAL[0], "likelihood only (MLE)", w_mle),
    (gaussian_neg_log_post, CATEGORICAL[2], r"+ Gaussian prior (MAP, $\ell_2$)", w_map_gauss),
    (laplace_neg_log_post, CATEGORICAL[7], r"+ Laplace prior (MAP, $\ell_1$)", w_map_laplace),
]:
    ax.plot(w, curve, color=color, linewidth=2.2, label=label)
    y_star = np.interp(w_star, w, curve)
    ax.plot([w_star], [y_star], marker="o", color=color, markersize=7, zorder=5)

ax.axvline(0, color=INK_SECONDARY, linewidth=0.8, alpha=0.5)
ax.set_xlabel(r"candidate weight $w$")
ax.set_ylabel(r"negative log-posterior (up to a constant)")
ax.set_title(r"Same weak evidence ($n=3$), three different estimates")
ax.legend(loc="upper center", fontsize=9)
ax.set_ylim(-0.1, 3.2)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "prior_as_regularizer.png"
savefig(fig, str(out))

print(f"w_MLE = {w_mle:.3f}")
print(f"w_MAP (Gaussian prior) = {w_map_gauss:.3f}")
print(f"threshold = 1/(b*A) = {threshold:.3f}, |y_bar| = {abs(y_bar):.3f}")
print(f"w_MAP (Laplace prior) = {w_map_laplace:.3f}")

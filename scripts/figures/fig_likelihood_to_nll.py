import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK_SECONDARY

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

apply_theme()
rng = np.random.default_rng(3)

TRUE_MEAN = 2.0
SIGMA = 1.5
N = 30

x = rng.normal(TRUE_MEAN, SIGMA, size=N)
m_grid = np.linspace(0.5, 3.5, 400)

# Likelihood of the data as a function of a candidate mean m, sigma fixed.
# log-likelihood first (numerically safe), then exponentiate for display.
log_lik = np.array([np.sum(stats.norm.logpdf(x, loc=m, scale=SIGMA)) for m in m_grid])
lik = np.exp(log_lik)
nll = -log_lik

m_hat = m_grid[np.argmax(log_lik)]  # == sample mean, up to grid resolution
sample_mean = x.mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4), sharex=True)

ax1.plot(m_grid, lik, color=CATEGORICAL[0], linewidth=2.2)
ax1.axvline(sample_mean, color=INK_SECONDARY, linestyle="--", linewidth=1)
ax1.set_yscale("log")
ax1.set_xlabel(r"candidate mean $m$")
ax1.set_ylabel(r"likelihood $\mathcal{L}(m) = \prod_i p(x_i \mid m)$")
ax1.set_title(f"Raw likelihood: a product of {N} densities")
ax1.text(sample_mean + 0.08, lik.max() * 0.15, "argmax",
          color=INK_SECONDARY, fontsize=9)

ax2.plot(m_grid, nll, color=CATEGORICAL[7], linewidth=2.2)
ax2.axvline(sample_mean, color=INK_SECONDARY, linestyle="--", linewidth=1)
ax2.set_xlabel(r"candidate mean $m$")
ax2.set_ylabel(r"$\mathrm{NLL}(m) = -\sum_i \log p(x_i \mid m)$")
ax2.set_title("Negative log-likelihood: a sum")
ax2.text(sample_mean + 0.08, nll.min() + (nll.max() - nll.min()) * 0.08,
          "argmin", color=INK_SECONDARY, fontsize=9)

fig.suptitle(f"Same optimum ($m={sample_mean:.2f}$), very different numbers ({N} points)",
             y=1.03, fontsize=12, color=INK_SECONDARY)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "likelihood_to_nll.png"
savefig(fig, str(out))

# Print the underflow motivation as a concrete number for the chapter prose.
lik_at_peak = lik.max()
print(f"peak raw likelihood value: {lik_at_peak:.3e} (n={N})")
for n_demo in [10, 50, 200, 800]:
    x_demo = rng.normal(TRUE_MEAN, SIGMA, size=n_demo)
    ll = np.sum(stats.norm.logpdf(x_demo, loc=x_demo.mean(), scale=SIGMA))
    print(f"n={n_demo:4d}: peak likelihood ~ {np.exp(ll):.3e}, NLL = {-ll:.2f}")

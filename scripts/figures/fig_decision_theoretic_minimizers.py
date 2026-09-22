import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK_SECONDARY

import numpy as np
import matplotlib.pyplot as plt

apply_theme()
rng = np.random.default_rng(11)

# A visibly right-skewed distribution: a mixture of two Gaussians. Mean,
# median, and the 0.8-quantile will all land at clearly different points.
N = 300_000
component = rng.random(N) < 0.7
y = np.where(component, rng.normal(2.0, 1.0, N), rng.normal(8.0, 1.8, N))

TAU = 0.8
a_grid = np.linspace(-1, 12, 400)


def pinball(r, tau):
    return np.where(r >= 0, tau * r, (tau - 1) * r)


risk_sq = np.array([np.mean((y - a) ** 2) for a in a_grid])
risk_abs = np.array([np.mean(np.abs(y - a)) for a in a_grid])
risk_pin = np.array([np.mean(pinball(y - a, TAU)) for a in a_grid])

a_mean = a_grid[np.argmin(risk_sq)]
a_median = a_grid[np.argmin(risk_abs)]
a_quantile = a_grid[np.argmin(risk_pin)]

true_mean = y.mean()
true_median = np.median(y)
true_quantile = np.quantile(y, TAU)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6))

ax1.hist(y, bins=120, color=CATEGORICAL[0], alpha=0.55, density=True)
for val, color, label in [
    (true_mean, CATEGORICAL[0], "mean"),
    (true_median, CATEGORICAL[2], "median"),
    (true_quantile, CATEGORICAL[7], f"{TAU:.0%} quantile"),
]:
    ax1.axvline(val, color=color, linewidth=2, linestyle="--")
    ax1.text(val, ax1.get_ylim()[1] * 0.92, f" {label}", color=color,
              fontsize=8.5, rotation=90, va="top")
ax1.set_xlabel("$y$")
ax1.set_ylabel("density")
ax1.set_title("A skewed, bimodal $Y$")

# Normalize each risk curve to [0, 1] so all three fit on shared axes and
# the *location* of each minimum, not its scale, is what's compared.
def norm(r):
    return (r - r.min()) / (r.max() - r.min())

for risk, color, label, a_star in [
    (risk_sq, CATEGORICAL[0], "squared loss", a_mean),
    (risk_abs, CATEGORICAL[2], "absolute loss", a_median),
    (risk_pin, CATEGORICAL[7], f"pinball ($\\tau={TAU}$)", a_quantile),
]:
    ax2.plot(a_grid, norm(risk), color=color, linewidth=2.2, label=label)
    ax2.plot([a_star], [norm(risk)[np.argmin(risk)]], marker="o", color=color,
              markersize=7, zorder=5)

ax2.set_xlabel(r"candidate decision $a$")
ax2.set_ylabel("risk $\\mathbb{E}[\\ell(a,Y)]$ (rescaled to [0,1])")
ax2.set_title("Three losses, three different minimizers")
ax2.legend(loc="upper center", fontsize=9)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "decision_theoretic_minimizers.png"
savefig(fig, str(out))

print(f"mean={true_mean:.3f} (argmin sq={a_mean:.3f})")
print(f"median={true_median:.3f} (argmin abs={a_median:.3f})")
print(f"{TAU:.0%} quantile={true_quantile:.3f} (argmin pinball={a_quantile:.3f})")

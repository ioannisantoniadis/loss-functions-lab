import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK_SECONDARY

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

apply_theme()
rng = np.random.default_rng(11)

# A real (if tiny) experiment: fit a line to noisy 1-D data with one strong
# outlier, minimizing three different losses over the same (slope, intercept).
N = 18
x = np.linspace(0, 10, N)
true_slope, true_intercept = 1.4, 1.0
y = true_slope * x + true_intercept + rng.normal(0, 0.7, size=N)

# One strong outlier: a point whose y is far below the trend.
outlier_idx = N - 3
y[outlier_idx] -= 12.0

X = np.column_stack([x, np.ones_like(x)])


def mse_fit(x, y, X):
    # Closed-form OLS.
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coeffs


def mae_fit(x, y, X):
    def objective(w):
        r = y - (w[0] * x + w[1])
        return np.sum(np.abs(r))
    res = minimize(objective, x0=[0.0, 0.0], method="Nelder-Mead",
                    options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 5000})
    return res.x


def huber_fit(x, y, X, delta=1.5):
    def objective(w):
        r = y - (w[0] * x + w[1])
        abs_r = np.abs(r)
        quad = 0.5 * r ** 2
        lin = delta * (abs_r - 0.5 * delta)
        return np.sum(np.where(abs_r <= delta, quad, lin))
    res = minimize(objective, x0=[0.0, 0.0], method="Nelder-Mead",
                    options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 5000})
    return res.x


w_mse = mse_fit(x, y, X)
w_mae = mae_fit(x, y, X)
w_huber = huber_fit(x, y, X)

fig, ax = plt.subplots(figsize=(7.6, 5.0))

is_outlier = np.zeros(N, dtype=bool)
is_outlier[outlier_idx] = True
ax.scatter(x[~is_outlier], y[~is_outlier], s=32, color=INK_SECONDARY,
           zorder=5, label="data")
ax.scatter(x[is_outlier], y[is_outlier], s=90, color=CATEGORICAL[7],
           marker="X", zorder=6, label="outlier")

x_line = np.linspace(0, 10, 100)
for w, color, label in [
    (w_mse, CATEGORICAL[0], "MSE fit"),
    (w_mae, CATEGORICAL[2], "MAE fit"),
    (w_huber, CATEGORICAL[4], r"Huber fit ($\delta=1.5$)"),
]:
    ax.plot(x_line, w[0] * x_line + w[1], color=color, linewidth=2.3, label=label)

ax.plot(x_line, true_slope * x_line + true_intercept, color=INK_SECONDARY,
        linestyle=":", linewidth=1.5, label="true trend (no outlier)")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("One outlier, three fitted lines")
ax.legend(loc="upper left", fontsize=8.5)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "outlier_robustness.png"
savefig(fig, str(out))

print(f"MSE fit:   slope={w_mse[0]:.3f}, intercept={w_mse[1]:.3f}")
print(f"MAE fit:   slope={w_mae[0]:.3f}, intercept={w_mae[1]:.3f}")
print(f"Huber fit: slope={w_huber[0]:.3f}, intercept={w_huber[1]:.3f}")
print(f"true:      slope={true_slope:.3f}, intercept={true_intercept:.3f}")

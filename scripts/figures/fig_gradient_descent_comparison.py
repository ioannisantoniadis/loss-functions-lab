import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK_SECONDARY

import numpy as np
import matplotlib.pyplot as plt

apply_theme()
rng = np.random.default_rng(11)

# A tiny 1D problem: fit a single scalar a to n points clustered near a
# true value, plus one strong outlier -- same flavor as Chapter 5's
# fig_outlier_robustness.py, but the question here is optimization
# behavior under a *fixed* learning rate, not the final fitted value.
true_a = 3.0
y = np.concatenate([rng.normal(true_a, 0.4, size=14), [15.0]])  # one outlier

def mse_grad(a):
    return np.mean(a - y)          # d/da of mean squared error

def huber_grad(a, delta=1.5):
    r = a - y
    g = np.where(np.abs(r) <= delta, r, delta * np.sign(r))
    return np.mean(g)

LR = 0.3
N_STEPS = 40
a0 = 0.0

def run(grad_fn):
    a = a0
    trace = [a]
    for _ in range(N_STEPS):
        a = a - LR * grad_fn(a)
        trace.append(a)
    return np.array(trace)

trace_mse = run(mse_grad)
trace_huber = run(lambda a: huber_grad(a, delta=1.5))

fig, ax = plt.subplots(figsize=(7.4, 4.8))
ax.plot(trace_mse, color=CATEGORICAL[0], linewidth=2.2, marker="o", markersize=3,
        label="MSE objective")
ax.plot(trace_huber, color=CATEGORICAL[2], linewidth=2.2, marker="o", markersize=3,
        label=r"Huber objective ($\delta=1.5$)")
ax.axhline(true_a, color=INK_SECONDARY, linestyle=":", linewidth=1.2, label="true value")
ax.set_xlabel("gradient descent step")
ax.set_ylabel(r"parameter estimate $a$")
ax.set_title(f"Same step size ({LR}), same data (with one outlier at y=15)")
ax.legend(loc="center right", fontsize=9)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "gradient_descent_comparison.png"
savefig(fig, str(out))

print(f"MSE final estimate: {trace_mse[-1]:.3f}")
print(f"Huber final estimate: {trace_huber[-1]:.3f}")
print(f"true value: {true_a}")

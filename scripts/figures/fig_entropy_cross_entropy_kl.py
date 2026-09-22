import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK_SECONDARY

import numpy as np
import matplotlib.pyplot as plt

apply_theme()

# A fixed target distribution P over 4 outcomes -- deliberately not one-hot,
# so H(P) > 0 is visible as a real, nonzero baseline (the "not every target
# is one-hot" point the chapter makes about label smoothing / soft targets).
P = np.array([0.40, 0.30, 0.20, 0.10])


def entropy(p):
    return -np.sum(p * np.log(p))


def kl(p, q):
    return np.sum(p * np.log(p / q))


def cross_entropy(p, q):
    return -np.sum(p * np.log(q))


H_P = entropy(P)

# A sequence of Q's moving progressively farther from P.
Qs = {
    "Q = P\n(exact match)": np.array([0.40, 0.30, 0.20, 0.10]),
    "Q close to P": np.array([0.35, 0.30, 0.22, 0.13]),
    "Q moderately off": np.array([0.28, 0.27, 0.25, 0.20]),
    "Q = uniform": np.array([0.25, 0.25, 0.25, 0.25]),
    "Q far from P": np.array([0.10, 0.15, 0.30, 0.45]),
}

labels = list(Qs.keys())
kl_vals = np.array([kl(P, q) for q in Qs.values()])
ce_vals = np.array([cross_entropy(P, q) for q in Qs.values()])
# Sanity check: cross_entropy should equal H_P + KL for every Q.
assert np.allclose(ce_vals, H_P + kl_vals)

x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(8.4, 5.0))

ax.bar(x, np.full_like(kl_vals, H_P), color=CATEGORICAL[0], label=r"$H(P)$ (fixed)")
ax.bar(x, kl_vals, bottom=H_P, color=CATEGORICAL[7],
       label=r"$D_{\mathrm{KL}}(P\,\|\,Q)$ (grows with $Q$'s distance from $P$)")

for xi, (h, total) in enumerate(zip([H_P] * len(x), ce_vals)):
    ax.text(xi, total + 0.03, f"{total:.2f}", ha="center", fontsize=9, color=INK_SECONDARY)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel(r"nats")
ax.set_title(r"$H(P,Q) = H(P) + D_{\mathrm{KL}}(P\,\|\,Q)$, made visible")
ax.legend(loc="upper left", fontsize=9)
ax.set_ylim(0, ce_vals.max() * 1.25)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "entropy_cross_entropy_kl.png"
savefig(fig, str(out))

print(f"H(P) = {H_P:.4f}")
for label, q in Qs.items():
    print(f"{label!r}: CE={cross_entropy(P, q):.4f}, KL={kl(P, q):.4f}, H(P)+KL={H_P + kl(P, q):.4f}")

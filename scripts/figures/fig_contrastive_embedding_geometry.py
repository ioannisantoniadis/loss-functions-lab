import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK_SECONDARY

import numpy as np
import matplotlib.pyplot as plt

apply_theme()
rng = np.random.default_rng(7)

# Three synthetic classes, n points each. Embeddings are initialized at
# random in 2D -- deliberately with NO structure tying position to class,
# so the "before" panel looks like noise and any class geometry in the
# "after" panel is entirely a consequence of the gradient steps below, not
# of the initialization.
n_per_class = 12
n_classes = 3
labels = np.repeat(np.arange(n_classes), n_per_class)
n = len(labels)
z = rng.uniform(-3.0, 3.0, size=(n, 2))
z0 = z.copy()

# Contrastive-style objective, evaluated directly on the point coordinates:
# same-class pairs are pulled together (squared distance), different-class
# pairs are pushed apart up to a margin (hinge on distance), exactly the
# contrastive loss of Hadsell, Chopra & LeCun (2006) applied to a raw 2D
# embedding instead of a learned encoder's output.
margin = 4.0
same_mask = labels[:, None] == labels[None, :]
diff_mask = ~same_mask
np.fill_diagonal(same_mask, False)


def contrastive_grad(z):
    """Gradient of the pairwise contrastive loss w.r.t. every point."""
    diff = z[:, None, :] - z[None, :, :]          # (n, n, 2), z_i - z_j
    dist = np.sqrt((diff ** 2).sum(-1) + 1e-12)    # (n, n)

    grad = np.zeros_like(z)

    # Pull term: d/dz_i sum_j same (||z_i-z_j||^2) = 2 * sum_j same (z_i-z_j)
    pull = 2.0 * (diff * same_mask[:, :, None]).sum(axis=1)

    # Push term: d/dz_i sum_j diff max(0, margin-||z_i-z_j||)^2
    #          = -2*(margin-dist)*(diff/dist) summed over active (dist<margin) pairs
    shortfall = np.clip(margin - dist, 0, None)
    active = diff_mask & (shortfall > 0)
    push_coeff = np.where(active, -2.0 * shortfall / dist, 0.0)
    push = (push_coeff[:, :, None] * diff).sum(axis=1)

    grad += pull + push
    return grad


lr = 0.005
n_steps = 20
for _ in range(n_steps):
    g = contrastive_grad(z)
    z = z - lr * g

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 5.0))

for c in range(n_classes):
    mask = labels == c
    ax1.scatter(z0[mask, 0], z0[mask, 1], color=CATEGORICAL[c], s=40,
                edgecolor="white", linewidth=0.6, label=f"class {c}")
    ax2.scatter(z[mask, 0], z[mask, 1], color=CATEGORICAL[c], s=40,
                edgecolor="white", linewidth=0.6, label=f"class {c}")

ax1.set_title("Before: random 2D embedding")
ax2.set_title(f"After {n_steps} gradient steps on a contrastive objective")
for ax in (ax1, ax2):
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    ax.set_aspect("equal")
ax1.legend(loc="upper right", fontsize=8.5)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "contrastive_embedding_geometry.png"
savefig(fig, str(out))

# Sanity check printed to stdout: mean same-class and mean different-class
# distance should move apart after training.
def mean_dists(z):
    diff = z[:, None, :] - z[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1) + 1e-12)
    same = dist[same_mask].mean()
    diffm = dist[diff_mask].mean()
    return same, diffm

s0, d0 = mean_dists(z0)
s1, d1 = mean_dists(z)
print(f"before: mean same-class dist={s0:.2f}, mean diff-class dist={d0:.2f}")
print(f"after:  mean same-class dist={s1:.2f}, mean diff-class dist={d1:.2f}")

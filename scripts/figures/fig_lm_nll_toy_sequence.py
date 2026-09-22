import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _theme import apply_theme, savefig, CATEGORICAL, INK_SECONDARY, MUTED

import numpy as np
import matplotlib.pyplot as plt

apply_theme()

# A genuine (if tiny) character-level bigram language model: count
# character-to-character transitions in a short, repetitive toy corpus,
# normalize into P(next char | current char), then evaluate that model's
# per-token NLL on a held-out sequence -- some of it drawn from the same
# pattern (should be near-zero NLL) and some of it a genuine surprise
# (should spike).
train_corpus = "the cat sat on the mat. " * 40
test_seq = "the cat sat on the map"   # last word deliberately breaks the pattern

vocab = sorted(set(train_corpus + test_seq))
idx = {c: i for i, c in enumerate(vocab)}
V = len(vocab)

# Count bigram transitions with add-one (Laplace) smoothing, so no
# transition ever has exactly zero probability.
counts = np.ones((V, V))  # smoothing pseudo-count of 1
for a, b in zip(train_corpus[:-1], train_corpus[1:]):
    counts[idx[a], idx[b]] += 1
probs = counts / counts.sum(axis=1, keepdims=True)  # each row a categorical dist

# Per-token NLL on the test sequence: -log P(x_t | x_{t-1}), t = 2..T.
# (the very first character has no predecessor under a bigram model, so it
# is skipped -- exactly analogous to how a real LM's first token has an
# empty context.)
nlls = []
chars = []
for a, b in zip(test_seq[:-1], test_seq[1:]):
    p = probs[idx[a], idx[b]]
    nlls.append(-np.log(p))
    chars.append(b)

nlls = np.array(nlls)
avg_nll = nlls.mean()
perplexity = np.exp(avg_nll)

fig, ax = plt.subplots(figsize=(11.5, 4.6))
positions = np.arange(len(nlls))
colors = [CATEGORICAL[2] if v < avg_nll else CATEGORICAL[7] for v in nlls]
ax.bar(positions, nlls, color=colors, width=0.65)
ax.set_xticks(positions)
ax.set_xticklabels([repr(c)[1:-1] if c != " " else "␣" for c in chars], fontsize=10)
ax.axhline(avg_nll, color=INK_SECONDARY, linestyle="--", linewidth=1.1,
           label=f"mean NLL = {avg_nll:.2f} nats  (perplexity $\\approx$ {perplexity:.2f})")
ax.set_xlabel(r"predicted character $x_t$ (context: preceding character $x_{t-1}$)")
ax.set_ylabel(r"per-token NLL $-\log p(x_t \mid x_{t-1})$  (nats)")
ax.set_title('Bigram character LM trained on "the cat sat on the mat." — evaluated on "...the cat sat on the map"')
ax.legend(loc="upper left", fontsize=9.5)

out = Path(__file__).resolve().parents[2] / "docs" / "images" / "lm_nll_toy_sequence.png"
savefig(fig, str(out))

print("chars:", chars)
print("nlls:", np.round(nlls, 3))
print(f"mean NLL = {avg_nll:.3f} nats, perplexity = {perplexity:.3f}")

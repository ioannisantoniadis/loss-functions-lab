# Figure scripts

Each `fig_*.py` script is a small, self-contained Python program. Most of
them are also the numerical experiment the chapter is describing: they
compute something real (a maximum-likelihood fit, a RankNet-style gradient,
a Plackett-Luce sample, a few epochs of gradient descent on a toy model) and
render the result as a static PNG in `docs/images/`. Quarto does not execute
Python at render time; these images are pre-generated and checked into the
repo like any other asset.

Setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r scripts/figures/requirements.txt
```

Regenerate a single figure:

```bash
python scripts/figures/fig_gaussian_to_mse.py
```

Regenerate everything:

```bash
for f in scripts/figures/fig_*.py; do python "$f"; done
```

Every script imports `apply_theme()` and the shared palette from `_theme.py`
so all figures across all chapters look like one system, and reuses the
palette from this author's other two Quarto books
(`optimization-lab`, `modern-ai-systems-and-methods`) so all three sites
read as one visual family. New figure scripts should do the same rather than
styling matplotlib ad hoc.

`fig_social_preview.py` is the odd one out: it renders a 1280x640 (2:1)
banner with the book's taxonomy and title baked in, for use as the README
hero image and the repo's GitHub social-preview / link-preview thumbnail.

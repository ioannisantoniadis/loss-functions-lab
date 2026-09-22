# Authoring conventions for loss-functions-lab

This file is the single source of truth for how chapters in this book are
written. It exists so that chapters written at different times (or by
different passes) read as one coherent book rather than a stitched-together
collection. Read `docs/appendix-notation.qmd` first — it defines every
symbol used below — then this file for structure and mechanics.

Not part of the rendered book. Not referenced by `_quarto.yml`.

## What this book is, in one paragraph

Not a catalogue of loss functions. A chain of reasoning: learning problem →
risk / empirical risk → probabilistic model → likelihood → negative
log-likelihood → MLE/MAP → classical losses → decision theory / proper
scoring rules → surrogate objectives → ranking / structured objectives →
modern ML/LLM objectives → a reusable recipe for designing the next one.
Every loss introduced must be explicitly classified by where it actually
came from (see "Origin tags" below) — that classification is a core feature
of the book, not decoration.

## Chapter file and frontmatter

`docs/chapters/NN-slug.qmd`, minimal frontmatter:

```yaml
---
title: "Human-Readable Chapter Title"
---
```

No author/date fields (book-level metadata in `_quarto.yml` covers that).

## The chapter template

Most chapters derive one coherent objective (or a tight family, e.g. MSE +
weighted/heteroscedastic variants) and follow this shape. Deviate when a
chapter's role is different (survey/framing chapters like the map, chapter 1,
chapter 10, or synthesis chapters like 17–18) — but **never drop the closing
"Connections" section**; it's what makes the book feel like one object
instead of independent essays.

1. **Opening paragraph(s), no header.** Motivate the problem in plain
   language before any notation. Say what's being predicted/decided and why
   it's not already solved by the previous chapter.
2. **`## The problem`** — state precisely what quantity is being predicted,
   what a "good" prediction means here, and what's still missing.
3. **`## Assumptions`** — the probabilistic / decision-theoretic assumptions
   about to be introduced, stated explicitly as assumptions (not asserted as
   fact). Name what would break if the assumption were wrong.
4. **`## Derivation`** — the actual math, step by step. This is the heart of
   the chapter. Show the algebra that matters; skip algebra a reader at this
   level can fill in, but never skip a *conceptual* step (e.g. "why does the
   log turn a product into a sum" is worth one sentence even though it's
   easy, because it's load-bearing intuition, not filler).
5. **The resulting objective**, boxed:
   ```
   ::: {.callout-note title="Definition — <Name>"}
   $$ ... $$
   :::
   ```
6. **Origin tag**, immediately after the definition box:
   ```
   ::: {.callout-tip title="Origin: <one of the six labels>"}
   One or two sentences on *why* this label applies.
   :::
   ```
   (`callout-tip`'s built-in green/lightbulb styling is what visually
   distinguishes it from a `callout-note` Definition box — Quarto strips
   unrecognized custom classes off callout divs, so don't add one.)
   The six labels, used verbatim: `derived from likelihood`,
   `derived from Bayesian/MAP reasoning`, `derived from decision theory`,
   `surrogate/relaxation`, `heuristic/design choice`, `hybrid`. If a chapter
   introduces more than one objective, each gets its own definition box +
   origin tag pair at the point it's introduced, not batched at the end.
7. **`## Interpretation`** — what does minimizing this actually give you?
   (conditional mean? median? a calibrated probability? a margin?) Say it
   plainly.
8. **`## Behavior and edge cases`** — gradient shape, curvature, what happens
   at perfect prediction, confidently-wrong prediction, outliers, extreme
   logits, etc. — whichever are relevant to *this* loss.
9. **`## Limitations`** — what the assumptions get wrong, when this breaks.
10. **`## Optimization implications`** — brief (2–4 sentences): gradient
    saturation, conditioning, convexity. Do not turn this into a full
    optimization treatment — that's [Chapter 17](chapters/17-losses-and-optimization.qmd)'s
    job; a chapter earning a comparison there should say so
    ("see Chapter 17 for how this compares to ...") rather than repeat it.
11. **`## Connections`** — 3–6 sentences or a short bulleted list, explicitly
    naming 2–4 neighboring chapters and *why* they're related (not just a
    bare link). Model this on the "How this connects" sections in
    `modern-ai-systems-and-methods` chapters, but keep it tighter.

Word count target per chapter: roughly **1200–2200 words** of prose (figures,
code, and equations don't count toward this). This book prioritizes depth
per objective over page count — resist the urge to pad.

## Critical distinctions to never blur

Restated from the spec because getting these wrong is the one way to break
the book's central thesis: loss vs. likelihood, likelihood vs. probability,
NLL vs. cross-entropy, cross-entropy vs. KL divergence, MLE vs. MAP,
regularization vs. Bayesian prior (don't claim every regularizer is "really"
a prior — say so only when the equivalence is exact), metric vs. loss,
target objective vs. surrogate, statistical justification vs. optimization
justification. Use the notation appendix's `\ell` vs `\mathcal{L}` split
consistently — never use `L` for both loss and likelihood in the same
chapter.

## Math and code mechanics

- Inline math `$...$`, display math `$$...$$`, using only the symbols in
  `docs/appendix-notation.qmd`. If you need a new symbol, add it there too.
- **Quarto does not execute code in this project** (no `jupyter:` engine is
  configured in `_quarto.yml`). Fenced ` ```python ` blocks are for
  *display* only — short, illustrative pseudocode (e.g. "this is what
  computing the NLL looks like"), never a code cell expected to run at
  render time. Actual computation lives in `scripts/figures/fig_*.py` and
  produces a static image.
- Citations: `[@bibkey]` against `docs/references.bib`. Add a new BibTeX
  entry there (accurate, real, verifiable) if a chapter needs a source not
  already present — never fabricate a citation.
- Cross-references to other chapters: standard Markdown links,
  `[Chapter N title](chapters/NN-slug.qmd)`, using the exact filenames in
  the table below.

## Figures

- One script per figure: `scripts/figures/fig_<slug>.py`, self-contained,
  starts with:
  ```python
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).resolve().parent))
  from _theme import apply_theme, savefig, CATEGORICAL, INK_SECONDARY, ...

  apply_theme()
  ```
  and ends by calling `savefig(fig, str(<repo_root>/docs/images/<slug>.png))`.
  Use only `numpy`, `scipy`, `matplotlib` (already in
  `scripts/figures/requirements.txt`) — no new heavy dependencies without a
  strong reason.
- Every figure answers a specific mathematical question — never decorative.
  Before writing one, be able to finish the sentence "this figure makes
  visible that ...".
- Embed with an id and a caption that states the lesson, not just what's
  plotted:
  ```markdown
  ![One sentence describing the plot, then one sentence stating the lesson it teaches.](../images/<slug>.png){#fig-<slug>}
  ```
  Reference it in prose via `@fig-<slug>` before or after it appears.
- Run the script after writing it (`python scripts/figures/fig_<slug>.py`
  from the repo root) so the PNG actually exists before the chapter is
  considered done.

## The full table of contents

Chapter files live in `docs/chapters/`. This is the final, fixed structure
— do not rename files or reorder without updating `docs/_quarto.yml`.

- `00-map.qmd` — The Map (taxonomy figure + short framing, not a full
  derivation chapter)
- Part I: `01-the-learning-problem.qmd`
- Part II: `02-probability-to-objective.qmd`, `03-mle-map-regularization.qmd`
- Part III: `04-gaussian-squared-error.qmd`, `05-laplace-robust-losses.qmd`
- Part IV: `06-bernoulli-categorical-cross-entropy.qmd`,
  `07-entropy-cross-entropy-kl.qmd`
- Part V: `08-decision-theory-proper-scoring.qmd`
- Part VI: `09-surrogate-losses.qmd`
- Part VII: `10-ranking-foundations.qmd`, `11-ranknet.qmd`, `12-listmle.qmd`,
  `13-optimizing-ranking-metrics.qmd`
- Part VIII: `14-contrastive-objectives.qmd`,
  `15-language-model-objectives.qmd`, `16-preference-objectives.qmd`
- Part IX (synthesis): `17-losses-and-optimization.qmd`,
  `18-designing-a-new-loss.qmd`
- Appendices: `appendix-notation.qmd` (done), `appendix-further-reading.qmd`
  (done)

Chapters 1–3 and the notation/further-reading appendices are already
written and should be treated as the binding style precedent for everything
downstream, particularly for how much the derivations spell out vs. assume.

# Research log

Sources consulted for this book and what was checked in each. Add an entry whenever a new source
is used; record what could not be verified as well as what was.

## 2026-10-02: audit and remediation pass

**Bibliography identifiers.** Every non-book entry in `docs/references.bib` now carries a DOI,
arXiv id or URL. Each DOI was confirmed against its Crossref record (title, authors, year, venue);
for classic papers the *original* publication DOI was used, not a later reprint. Each arXiv id was
confirmed against its arXiv abstract page. One error was corrected: `rafailov2023direct` had
Manning and Ermon in the wrong order.

**Claims checked against primary sources.**

| Claim | Source | Result |
|---|---|---|
| LambdaRank's λᵢⱼ = σ(−Δ)·\|ΔNDCG\| (Ch 13) | Burges, MSR-TR-2010-82, Eq. 6 | ok (σ = 1); Origin tag added, quoting the report |
| InfoNCE bound (Ch 14) | van den Oord et al. 2018, §2.3 and appendix | reworded: minimizing the loss *maximizes* a lower bound, capped at log(K+1) |
| Brier/Good attribution for properness (Ch 8) | Gneiting & Raftery 2007, §2 | ok |
| DPO derivation (Ch 16) | Rafailov et al. 2023, §4 | ok |
| DPO "increases the preferred completion's probability" (Ch 16) | Razin et al. 2025 (ICLR), arXiv 2410.08847 | corrected: the loss widens a margin; the preferred likelihood often falls |
| DPO vs. RLHF "not provably identical", "more stable" (Ch 16) | Azar et al. 2023, arXiv 2310.12036 | corrected: same optimum in the infinite-data limit, different finite-sample estimators |
| Outlier-fit slopes and GD estimates quoted in captions (Ch 5, 17) | fresh runs of the figure scripts | ok, exact |

**Not yet checked:** Chapters 3, 7 and 10 were not sampled in the audit.

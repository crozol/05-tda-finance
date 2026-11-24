# Crisis detection: analysis plan and amendments

The plan of the first study (README, section 5), fixed on 2026-10-17 before any feature was
computed on market data and amended three times before any result. Each version is archived on
Zenodo (versions 0.1.0–0.1.2); the record at the end links every step to its commit. Nothing in the
plan was changed after it was run on market data; what was added afterwards is listed at the end,
declared before it was computed.

## The plan (version 0.1.2)

- **Data.** Daily adjusted closes of the seven tickers in `src/data.py` (SPY, QQQ, XLF, XLE, XLK,
  XLY, XLV), 2005-01-03 to 2026-11-06, downloaded with yfinance 1.8.0 on 2026-11-10 and frozen as
  `data/prices_2026-11-10.csv` (SHA-256
  `10535bf43cf79624dd9c81f2284d4520e49235981fa25061e9431c7aa1d6ae29`). Yahoo revises adjusted prices
  retroactively, so a new download need not match it.
- **Episodes.** Six crisis periods fixed in advance in `src/data.py`. A window is positive when its
  last day falls inside one: 24 windows for 2008–09, 9 for 2011, 2 for 2015, 13 for Q4 2018, 5 for
  2020 and 39 for 2022 (1 092 windows in all). 2015 is reported but not counted (5 counted
  episodes). With so few episodes the conclusion is qualitative; no p-value is computed across
  episodes.
- **Primary comparison.** Topological features plus realised volatility and the correlation
  spectrum, against volatility and the spectrum alone. Reported, not counted: three other controls
  for the topology (point 11) and the settings of Gidea & Katz (point 13).
- **Splits.** Leave one crisis out: the test block of an episode is its positive windows plus 12
  normal windows (60 trading days) on each side; the model is trained on every other window except
  a 7-window purge on each side. Walk-forward: the same test blocks, trained only on the windows
  that end before the block, minus the purge; 2008–09 has no earlier crisis to learn from, so the
  walk-forward scores four counted episodes.
- **Seeds.** Every random forest is fitted with seeds 0–9; every ROC-AUC and average precision is
  the mean over them. The alarm rule uses the probability averaged over the 10 forests.
- **Success criterion.** For each held-out episode, ΔAUC is the ROC-AUC of the combined model minus
  that of the baselines, averaged over the 10 seeds; an episode is a win when that mean exceeds the
  standard deviation of ΔAUC across seeds. Success: at least 4 of the 5 counted episodes under
  leave-one-crisis-out and at least 3 of the 4 scorable under the walk-forward. If every episode
  were a coin flip, the two together would pass with probability at most 0.19: the criterion is a
  filter, not evidence on its own.
- **"Detects earlier", defined.** For the combined model, the baselines alone and volatility alone:
  a threshold at the 95th percentile of the score on the normal windows of a calibration block (the
  last 20 % of the training fold, purged); the alarm on the window that completes a run of 3 above
  it, searched from 60 trading days before the crisis; the lead time in trading days to the drawdown
  minimum. No alarm before the minimum is a miss.
- **Fixed parameters.** Window 40 days, step 5, homology dimensions 0 and 1; for each dimension the
  L¹ and L² norms of the persistence landscape summed over all layers (Gidea & Katz, 2018, eq. 2.3),
  on giotto-tda's default grid (100 points per dimension). Random forest of 300 trees (logistic
  regression as a check). Realised volatility: the mean over the seven assets of the standard
  deviation of their daily log returns in the window; spectrum: the seven eigenvalues of the
  window's correlation matrix divided by seven; kurtosis: Mardia's (1970) multivariate kurtosis.

## Calibration before any market data

Each cell: the number of 20 synthetic datasets in which the whole criterion passed
(`scripts/calibrate_plan.py`). Crises double volatility and raise correlation from 0.3 to 0.7, with
Gaussian days (null), heavy-tailed days (Student t, 3 degrees of freedom), or a change in the tails
only (positive control).

| Comparison | Gaussian (null) | Heavy-tailed | Tails only |
|---|---|---|---|
| topology + spectrum + volatility vs spectrum + volatility (primary) | 3/20 | 0/20 | 7/20 |
| topology + spectrum vs spectrum | 6/20 | 1/20 | 11/20 |
| primary, with Mardia's kurtosis on both sides | 1/20 | 1/20 | 2/20 |
| topology + leading eigenvalue vs eigenvalue (version 0.1.1) | 8/20 | 5/20 | 20/20 |
| primary, settings of Gidea & Katz: H1, 50-day windows | 0/20 | – | – |
| primary, settings of Gidea & Katz: H1, 100-day windows | 0/20 | – | – |

## Amendments before any result

**First amendment (2026-11-10).** Four problems, and one definition that needed to be precise. None
of the changes uses a feature or a score computed on market data.

1. *Threshold.* The 95th percentile of the score on the training fold's normal windows is
   degenerate for a random forest, which scores its own training windows near 0; out-of-bag scores
   do not fix it, since every out-of-bag window has in-bag neighbours sharing 35 of its 40 days. The
   threshold now comes from a purged calibration block, and the three alarm scores are of the same
   kind (model probabilities).
2. *Success criterion.* "Higher ROC-AUC in at least 4 of 5 episodes", by any margin, passes by chance
   with probability 0.19, and the forest's seed alone can flip a small difference: now the mean over
   10 seeds and a margin equal to the spread between them.
3. *Positive windows per episode,* stated (the 2020 episode has 5).
4. *Frozen data* with its SHA-256.
5. *Alarm time.* The alarm is the third window of the run, the first at which the run is known.

**Second amendment (2026-11-16).** Writing `scripts/run_plan.py` showed four points the plan left
open; only a synthetic dry run had been executed.

6. *Test blocks:* 12 normal windows on each side (60 trading days), the stretch of the alarm search.
7. *Walk-forward:* the leave-one-crisis-out test blocks with past-only training; 3 of 4 needed
   (5/16 ≈ 0.31 by chance).
8. *One alarm per episode:* the probability averaged over the seeds.
9. *Landscape norms:* summed over all layers as in Gidea & Katz (giotto-tda's default keeps only the
   first layer).

**Third amendment (2026-11-19).** The criterion was run on synthetic data before any market data
(table above).

10. *Primary comparison.* Topology plus the leading eigenvalue share against that share alone passed
    in 8 of 20 null datasets (95 % interval 19–64 %), above the 0.19 stated for chance. For
    independent Gaussian days, the topology of a standardised window depends on the correlation
    matrix only through its eigenvalues (README, section 4), so the full spectrum is the complete
    correlation control; with it the rate fell only to 6 of 20. Standardising removes the
    volatility level of a window, not a change inside it, and near the start of a crisis the label
    is almost a function of the fraction of turbulent days in the window. With volatility and the
    spectrum on both sides the criterion passed in 3 of 20 null datasets (3–38 %), in line with
    chance.
11. *Reported comparisons:* the spectrum alone; the primary comparison with Mardia's kurtosis on both
    sides (for elliptical days the topology depends on the law of the days' radii, whose leading
    summary is the kurtosis); and the leading eigenvalue share, the primary comparison of version
    0.1.1. None is counted.
12. *Positive control.* With crises only in the tails, topology plus the leading eigenvalue share
    passed in 20 of 20 datasets: the topological features see the tails. Volatility takes much of
    that signal and the kurtosis nearly all of it: the primary comparison passed in 7 of 20.
13. *Settings of Gidea & Katz, reported:* dimension 1 only, windows of 50 and 100 days, step 5; on
    the null the primary comparison passed in 0 of 20 datasets with either window.

## Added after the result

The plan was run once on market data on 2026-11-19. Nothing in it was changed afterwards. Four
additions, each declared in the code before it was computed:

14. *Monte Carlo tests of four statistics* (2026-11-19) and the rule to read them (2026-11-20, after
    the synthetic runs and before any statistic was computed on market data). They do not change the
    verdict.
15. *Correction to point 12* (2026-11-20). "Its power is real but modest" overstated it: 7 passes in
    20 against 3 in 20 on the null is no significant difference (Fisher's exact test, two-sided
    p = 0.27).
16. *Direction diagnostic* (2026-11-20), declared with a prediction before it was computed: the
    prediction does not hold, and the mechanism accounts for three of the five crises.
17. *Hypothesis for 2008–09* (2026-11-21), declared with two predictions and their thresholds before
    it was computed: both hold.

## Full Monte Carlo tables

Generated by `python -m scripts.post_result --tables` from `results/post_result.json`. Power: datasets
with p ≤ 0.05 out of 20, with an exact 95 % interval; p: Monte Carlo p-value of the market statistic.

**Plan's setting** (99 null datasets)

| Scheme | Comparison | Statistic | Size | Heavy-tailed | Tails only | Clayton | Market S | p |
|---|---|---|---|---|---|---|---|---|
| leave one crisis out | primary | dAUC | 0.04 | 3/20 (0.03–0.38) | 10/20 (0.27–0.73) | 2/20 (0.01–0.32) | +0.0292 | 0.020 |
| leave one crisis out | primary | dLL | 0.04 | 5/20 (0.09–0.49) | 12/20 (0.36–0.81) | 4/20 (0.06–0.44) | +0.0242 | 0.350 |
| leave one crisis out | primary | dBrier | 0.04 | 5/20 (0.09–0.49) | 13/20 (0.41–0.85) | 3/20 (0.03–0.38) | −0.0027 | 0.960 |
| leave one crisis out | primary | dAUC\|vol | 0.04 | 3/20 (0.03–0.38) | 2/20 (0.01–0.32) | 1/20 (0.00–0.25) | −0.0506 | 0.810 |
| leave one crisis out | kurtosis | dAUC | 0.04 | 5/20 (0.09–0.49) | 5/20 (0.09–0.49) | 0/20 (0.00–0.17) | +0.0690 | 0.010 |
| leave one crisis out | kurtosis | dLL | 0.04 | 7/20 (0.15–0.59) | 2/20 (0.01–0.32) | 5/20 (0.09–0.49) | +0.0878 | 0.010 |
| leave one crisis out | kurtosis | dBrier | 0.04 | 9/20 (0.23–0.68) | 3/20 (0.03–0.38) | 3/20 (0.03–0.38) | +0.0123 | 0.040 |
| leave one crisis out | kurtosis | dAUC\|vol | 0.04 | 4/20 (0.06–0.44) | 0/20 (0.00–0.17) | 2/20 (0.01–0.32) | −0.0147 | 0.570 |
| walk-forward | primary | dAUC | 0.04 | 1/20 (0.00–0.25) | 11/20 (0.32–0.77) | 3/20 (0.03–0.38) | +0.0384 | 0.010 |
| walk-forward | primary | dLL | 0.04 | 7/20 (0.15–0.59) | 9/20 (0.23–0.68) | 3/20 (0.03–0.38) | +0.0411 | 0.240 |
| walk-forward | primary | dBrier | 0.04 | 2/20 (0.01–0.32) | 12/20 (0.36–0.81) | 4/20 (0.06–0.44) | +0.0029 | 0.620 |
| walk-forward | primary | dAUC\|vol | 0.04 | 4/20 (0.06–0.44) | 3/20 (0.03–0.38) | 2/20 (0.01–0.32) | +0.0076 | 0.480 |
| walk-forward | kurtosis | dAUC | 0.04 | 1/20 (0.00–0.25) | 5/20 (0.09–0.49) | 1/20 (0.00–0.25) | +0.0288 | 0.020 |
| walk-forward | kurtosis | dLL | 0.04 | 12/20 (0.36–0.81) | 2/20 (0.01–0.32) | 6/20 (0.12–0.54) | +0.0642 | 0.020 |
| walk-forward | kurtosis | dBrier | 0.04 | 4/20 (0.06–0.44) | 1/20 (0.00–0.25) | 4/20 (0.06–0.44) | +0.0088 | 0.090 |
| walk-forward | kurtosis | dAUC\|vol | 0.04 | 3/20 (0.03–0.38) | 0/20 (0.00–0.17) | 1/20 (0.00–0.25) | −0.0096 | 0.530 |

**Gidea & Katz, 50-day windows** (20 null datasets)

| Scheme | Comparison | Statistic | Size | Heavy-tailed | Tails only | Market S | p |
|---|---|---|---|---|---|---|---|
| leave one crisis out | primary | dAUC | 0.05 | 1/20 (0.00–0.25) | 10/20 (0.27–0.73) | +0.0254 | 0.048 |
| leave one crisis out | primary | dLL | 0.05 | 10/20 (0.27–0.73) | 13/20 (0.41–0.85) | −0.0172 | 0.952 |
| leave one crisis out | primary | dBrier | 0.05 | 4/20 (0.06–0.44) | 11/20 (0.32–0.77) | −0.0008 | 0.857 |
| leave one crisis out | primary | dAUC\|vol | 0.05 | 3/20 (0.03–0.38) | 4/20 (0.06–0.44) | +0.0493 | 0.143 |
| leave one crisis out | kurtosis | dAUC | 0.05 | 6/20 (0.12–0.54) | 6/20 (0.12–0.54) | +0.0067 | 0.095 |
| leave one crisis out | kurtosis | dLL | 0.05 | 10/20 (0.27–0.73) | 5/20 (0.09–0.49) | +0.0157 | 0.048 |
| leave one crisis out | kurtosis | dBrier | 0.05 | 8/20 (0.19–0.64) | 5/20 (0.09–0.49) | +0.0044 | 0.048 |
| leave one crisis out | kurtosis | dAUC\|vol | 0.05 | 4/20 (0.06–0.44) | 1/20 (0.00–0.25) | +0.0393 | 0.095 |
| walk-forward | primary | dAUC | 0.05 | 1/20 (0.00–0.25) | 9/20 (0.23–0.68) | −0.0079 | 0.810 |
| walk-forward | primary | dLL | 0.05 | 8/20 (0.19–0.64) | 11/20 (0.32–0.77) | −0.0014 | 0.667 |
| walk-forward | primary | dBrier | 0.05 | 7/20 (0.15–0.59) | 15/20 (0.51–0.91) | +0.0001 | 0.667 |
| walk-forward | primary | dAUC\|vol | 0.05 | 0/20 (0.00–0.17) | 1/20 (0.00–0.25) | −0.0172 | 0.714 |
| walk-forward | kurtosis | dAUC | 0.05 | 1/20 (0.00–0.25) | 4/20 (0.06–0.44) | −0.0016 | 0.667 |
| walk-forward | kurtosis | dLL | 0.05 | 6/20 (0.12–0.54) | 3/20 (0.03–0.38) | +0.0060 | 0.333 |
| walk-forward | kurtosis | dBrier | 0.05 | 6/20 (0.12–0.54) | 5/20 (0.09–0.49) | +0.0006 | 0.238 |
| walk-forward | kurtosis | dAUC\|vol | 0.05 | 1/20 (0.00–0.25) | 1/20 (0.00–0.25) | −0.0083 | 0.667 |

**Gidea & Katz, 100-day windows** (20 null datasets)

| Scheme | Comparison | Statistic | Size | Heavy-tailed | Tails only | Market S | p |
|---|---|---|---|---|---|---|---|
| leave one crisis out | primary | dAUC | 0.05 | 0/20 (0.00–0.17) | 3/20 (0.03–0.38) | −0.0370 | 0.762 |
| leave one crisis out | primary | dLL | 0.05 | 1/20 (0.00–0.25) | 3/20 (0.03–0.38) | −0.0577 | 0.952 |
| leave one crisis out | primary | dBrier | 0.05 | 0/20 (0.00–0.17) | 1/20 (0.00–0.25) | +0.0014 | 0.476 |
| leave one crisis out | primary | dAUC\|vol | 0.05 | 0/20 (0.00–0.17) | 0/20 (0.00–0.17) | −0.1275 | 1.000 |
| leave one crisis out | kurtosis | dAUC | 0.05 | 3/20 (0.03–0.38) | 1/20 (0.00–0.25) | −0.0611 | 0.952 |
| leave one crisis out | kurtosis | dLL | 0.05 | 9/20 (0.23–0.68) | 1/20 (0.00–0.25) | −0.0265 | 0.762 |
| leave one crisis out | kurtosis | dBrier | 0.05 | 6/20 (0.12–0.54) | 3/20 (0.03–0.38) | +0.0001 | 0.619 |
| leave one crisis out | kurtosis | dAUC\|vol | 0.05 | 1/20 (0.00–0.25) | 1/20 (0.00–0.25) | −0.0005 | 0.333 |
| walk-forward | primary | dAUC | 0.05 | 0/20 (0.00–0.17) | 5/20 (0.09–0.49) | +0.0727 | 0.048 |
| walk-forward | primary | dLL | 0.05 | 0/20 (0.00–0.17) | 10/20 (0.27–0.73) | +0.0306 | 0.286 |
| walk-forward | primary | dBrier | 0.05 | 0/20 (0.00–0.17) | 0/20 (0.00–0.17) | +0.0097 | 0.381 |
| walk-forward | primary | dAUC\|vol | 0.05 | 0/20 (0.00–0.17) | 2/20 (0.01–0.32) | −0.0539 | 0.667 |
| walk-forward | kurtosis | dAUC | 0.05 | 0/20 (0.00–0.17) | 0/20 (0.00–0.17) | +0.0459 | 0.095 |
| walk-forward | kurtosis | dLL | 0.05 | 3/20 (0.03–0.38) | 1/20 (0.00–0.25) | +0.0252 | 0.429 |
| walk-forward | kurtosis | dBrier | 0.05 | 0/20 (0.00–0.17) | 1/20 (0.00–0.25) | +0.0072 | 0.429 |
| walk-forward | kurtosis | dAUC\|vol | 0.05 | 0/20 (0.00–0.17) | 0/20 (0.00–0.17) | +0.0195 | 0.286 |

## Record

| Date | Step | Commit | Archive |
|---|---|---|---|
| 2026-10-17 | Plan fixed before any result | `bb57e7e` | |
| 2026-11-10 | First amendment; frozen prices | `874fada`, `be8e290` | v0.1.0 (doi:10.5281/zenodo.24095681) |
| 2026-11-16 | Second amendment; norms summed over all layers | `23e3c90`, `c730476` | v0.1.1 (doi:10.5281/zenodo.24289753) |
| 2026-11-18 | Calibration on synthetic data | `cf7bbfa`, `25edfd5` | |
| 2026-11-19 | Third amendment | `98a1987` | v0.1.2 (doi:10.5281/zenodo.24356109) |
| 2026-11-19 | First and only run on market data | `caf7aaa` | |
| 2026-11-19 | Monte Carlo tests declared | `a69d18e` | |
| 2026-11-20 | Rule to read them, before the market statistics | `7bb229f` | |
| 2026-11-20 | Market statistics and summary | `c662159` | v0.2.0 (doi:10.5281/zenodo.24404808) |
| 2026-11-20 | Direction diagnostic declared, then run | `d76881d`, `4e083ed` | v0.2.1 (doi:10.5281/zenodo.24408764) |
| 2026-11-21 | Hypothesis for 2008–09 declared, then run | `95fced8`, `3320597` | v0.2.2 (doi:10.5281/zenodo.24424111) |
| 2026-11-24 | Early-warning plan (docs/early-warning-plan.md) | `ae1a91b` | |

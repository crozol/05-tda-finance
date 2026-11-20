# TDA on financial time series — regime change detection

[![CI](https://img.shields.io/github/actions/workflow/status/crozol/05-tda-finance/tests.yml?label=CI&style=flat-square)](https://github.com/crozol/05-tda-finance/actions/workflows/tests.yml)
[![Portfolio page](https://img.shields.io/badge/Portfolio-page-7C3AED?style=flat-square)](https://crozol.github.io/projects/05-tda-finance.html)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24095680.svg)](https://doi.org/10.5281/zenodo.24095680)

**Status: in progress.** The analysis plan, including its success
criterion, was fixed before any result (version 0.1.2,
[doi:10.5281/zenodo.24356109](https://doi.org/10.5281/zenodo.24356109)) and
was run on market data on 2026-11-19; its result, with the analyses added
after it, is being written up. A negative control in the tests shows that
shuffled k-fold gives ROC-AUC ≈ 0.97 to an unrelated feature, against ≈ 0.5
with purged folds.

Using **persistent homology** on sliding windows of multi-asset returns to
detect market regime changes (the 2008 crisis, the 2020 crash, normal periods).

## Motivation

Classical methods for detecting regime changes in finance rely on variance,
rolling correlations and hidden Markov models. Topological Data Analysis
offers a complementary angle: treat each window of daily returns as a point
cloud in `ℝ^d` (one point per day, one axis per asset) and summarise its
*shape* with persistence diagrams, which are stable under small perturbations
of the data.

## Hypothesis

Gidea & Katz (2018) found that the `L^p` norms of persistence landscapes of
sliding windows of four US equity indices grow in the run-up to the 2000 and
2008 crashes. The question this project tests is narrower and harder: **do
topological features detect crisis regimes better than realised volatility
and the eigenvalues of the correlation matrix together, evaluated on the same
leakage-free folds?** A crisis raises volatility and makes assets move
together, and both effects also change a persistence diagram, so an
improvement over a model with no baselines would not show that topology adds
anything.

## Evaluation design

- **Labels fixed in advance.** Crisis periods are listed in `src/data.py`
  (`CRISIS_PERIODS`: 2008–09, 2011, 2015, Q4 2018, 2020, 2022) with the event that
  bounds each one, and are not tuned afterwards. A window is labelled by its
  *last* day, so a window counts as "crisis" only once the crisis has begun.
- **No leakage between overlapping windows.** With a 40-day window and a
  5-day step, neighbouring windows share 35 days, so shuffled k-fold puts
  near-copies of every test window in the training set (López de Prado, 2018).
  `src/validation.py` provides two schemes that avoid it:
  - **purged walk-forward**: train on the past, test on the next block, and
    drop the 7 windows in between, so no training window shares a day with a
    test window;
  - **leave one crisis out**: hold out each crisis with its surrounding
    normal windows, train on the rest minus a purge gap on both sides.
- **Baselines on identical folds.** Realised volatility and the eigenvalues
  of the correlation matrix, as shares of their sum (the first one is the
  one-component absorption ratio of Kritzman et al., 2011), are scored with
  the same model and splits as the topological features, alone and combined.
- **Scale removed from the topology.** Each window is standardised per asset
  before computing persistence, so the diagrams describe the shape of the
  joint distribution rather than re-measuring its volatility. Standardising
  removes the volatility level of a window, not a change of volatility inside
  it: a window that mixes calm and turbulent days is still a mixture of two
  scales, and the diagrams can see it (point 10).
- **Success criterion.** Defined in the analysis plan below, before any
  result.

`tests/test_validation.py` checks that the purged splits share no day between
training and test windows (and that one window less of purge would), that
every crisis is held out once, and demonstrates the leakage on synthetic
data: a feature with no relation to the label, but slowly varying in time,
reaches a ROC-AUC near 0.97 under shuffled k-fold and about 0.5 under purged
walk-forward.

## Analysis plan (fixed before any result)

Committed before any feature is computed on market data (`bb57e7e`,
2026-10-17) and amended three times, still before any result, on
2026-11-10, 2026-11-16 and 2026-11-19 (lists at the end of this section). Any
change made after the first result will be listed there too, with its date and
reason.

- **Data.** Daily adjusted closes of the seven tickers in `src/data.py`,
  2005-01-03 to 2026-11-06, downloaded with yfinance 1.8.0 on 2026-11-10 and
  frozen as `data/prices_2026-11-10.csv` (SHA-256
  `10535bf43cf79624dd9c81f2284d4520e49235981fa25061e9431c7aa1d6ae29`). Every
  result uses this file; Yahoo revises adjusted prices retroactively, so a new
  download need not match it.
- **Episodes.** Six crisis periods (`src/data.py`). A window is positive when
  its last day falls inside one; on the frozen data that gives 24 positive
  windows for 2008–09, 9 for 2011, 2 for 2015, 13 for Q4 2018, 5 for 2020 and
  39 for 2022 (1 092 windows in all). The 2015 episode is reported but not
  counted, so the effective number of episodes is 5. The 2020 AUC rests on
  5 positive windows and counts as much as the 2022 one, which rests on 39;
  every episode is reported with its number of positive windows. With so few
  episodes, the conclusion is qualitative; no p-value is computed across
  episodes.
- **Primary comparison.** Topological features plus realised volatility and
  the correlation spectrum, against volatility and the spectrum alone. Reported,
  not counted: three other controls for the topology (point 11) and the primary
  comparison with the settings of Gidea & Katz (point 13).
- **Splits.** Under leave-one-crisis-out, the test block of an episode is its
  positive windows plus 12 normal windows (60 trading days) on each side, the
  stretch over which the alarm is searched; the model is trained on every
  other window except the 7-window purge on each side. The walk-forward uses
  the same test blocks but trains only on the windows that end before the
  block, minus the purge, as a detector running in real time would. The
  2008–09 episode has no earlier crisis to learn from and cannot be scored
  this way, so the walk-forward scores four counted episodes: 2011, Q4 2018,
  2020 and 2022.
- **Seeds.** Every random forest is fitted with 10 seeds (0–9); every ROC-AUC
  and average precision is the mean over them. The alarm rule needs one score
  per window, so it uses the probability averaged over the 10 forests, which
  is itself a forest of 3 000 trees: one threshold and one alarm per score and
  episode.
- **Success criterion.** For each held-out episode, ΔAUC is the ROC-AUC of
  the combined model minus that of the baselines alone, averaged over the 10
  seeds. An episode counts as a win only if that mean exceeds the standard
  deviation of ΔAUC across the seeds. The combined model succeeds if it wins
  in at least 4 of the 5 counted episodes under leave-one-crisis-out and in
  at least 3 of the 4 that the walk-forward can score. If topology added
  nothing and each episode were a coin flip, 4 of 5 would still happen with
  probability 6/32 ≈ 0.19 and 3 of 4 with probability 5/16 ≈ 0.31, so the two
  together pass by chance with probability at most 0.19: the criterion is a
  filter, not evidence on its own. On synthetic data in which crises change
  only volatility and correlation it passed in 3 of 20 datasets, and in
  7 of 20 when crises change only the tails (points 10 and 12). Average
  precision is reported alongside.
- **"Detects earlier", defined.** For each episode and each score — the
  random-forest probability of the combined model, of the baselines alone and
  of volatility alone — with the same rule for all three:
  - threshold: the last 20 % of the windows of the training fold form a
    calibration block, separated from the rest by the same 7-window purge. The
    model is fitted on the rest, the threshold is the 95th percentile of its
    score on the normal windows of the calibration block, and that same model
    scores the held-out episode;
  - alarm: the first window, from 60 trading days before the start of the
    crisis, that completes a run of 3 consecutive windows above the
    threshold: the alarm is raised on the third window of the run, the first
    at which the run is known, with no look-ahead;
  - lead time: trading days from the alarm to the drawdown minimum (the end
    date of the episode). No alarm before the minimum counts as a miss.
- **Fixed parameters.** Window 40 days, step 5, homology dimensions 0 and 1.
  For each dimension, the `L¹` and `L²` norms of the persistence landscape
  summed over all its layers, `‖λ‖_p = (Σ_k ‖λ_k‖_p^p)^{1/p}`, as in Gidea &
  Katz (2018, eq. 2.3): four features. Gidea & Katz use dimension 1 only,
  windows of 50 and 100 days and a step of one day. The integrals are Riemann
  sums on giotto-tda's default grid, 100 points per dimension from the
  smallest birth to the largest death over all windows: the grid uses no
  label, but its range includes later windows. A random forest of 300 trees
  as the main model (logistic regression as a check). Realised volatility is
  the mean over the seven assets of the standard deviation of their daily log
  returns in the window; the spectrum, the seven eigenvalues of the window's
  correlation matrix in decreasing order, divided by seven; the kurtosis,
  Mardia's (1970) multivariate kurtosis of the window. These are the defaults
  in `src/` and are not tuned on the held-out episodes.

**Amendments before any result (2026-11-10).** An external review of the plan
found four problems, and one definition needed to be made precise; all are
fixed above, and none of them uses a feature or a score computed on market
data.

1. *Threshold.* The 95th percentile of the score on the normal windows of the
   training fold is degenerate for a random forest, which scores its own
   training windows near 0. Out-of-bag scores do not fix it: the bootstrap
   ignores time, so every out-of-bag window has in-bag neighbours that share
   35 of its 40 days, the leakage the negative control demonstrates. The
   threshold now comes from a purged calibration block inside the training
   fold, and the three scores are now of the same kind (model probabilities),
   with the combined model in place of topology alone, as in the primary
   comparison.
2. *Success criterion.* "Higher ROC-AUC in at least 4 of 5 episodes", by any
   margin, passes by chance with probability 0.19, and the random-forest seed
   alone can flip a small difference. Now: the mean over 10 seeds, a margin
   equal to the spread between seeds, and the 0.19 stated.
3. *Positive windows per episode,* added: the 2020 episode has 5.
4. *Frozen data* with its SHA-256, since yfinance is unofficial and adjusted
   prices change retroactively.
5. *Alarm time.* "The first window after which the score exceeds the threshold
   for 3 consecutive windows" could be read as the first window of the run,
   which is only known two windows later. The alarm is now the third window
   of the run.

**Second amendment before any result (2026-11-16).** Writing
`scripts/run_plan.py`, which runs the plan end to end, showed four points the
plan left open. They are fixed above. Only a synthetic dry run has been
executed; no feature or score has been computed on market data.

6. *Test blocks.* "The surrounding normal windows" of a held-out episode had
   no number: now 12 on each side (60 trading days), the stretch the alarm
   search covers.
7. *Walk-forward.* "By the same rule" cannot apply as written: trained only on
   the past, 2008–09 has no earlier crisis to learn from, which leaves four
   counted episodes. The walk-forward now uses the leave-one-crisis-out test
   blocks with past-only training and needs 3 of 4 (5/16 ≈ 0.31 by chance).
8. *One alarm per episode.* With 10 seeds the rule gave ten thresholds and ten
   alarms per score; it now uses the probability averaged over the seeds.
9. *Landscape norms.* The plan named the norms but not how they are computed.
   giotto-tda's default keeps only the first landscape layer, while the norm
   of Gidea & Katz (2018), whose finding the hypothesis tests, sums over all
   layers; the features now use their definition (`c730476`). The grid of the
   Riemann sums stays at giotto-tda's default and is stated above.

**Third amendment before any result (2026-11-19).** Before the plan was run on
market data, its criterion was run on synthetic data
(`scripts/calibrate_plan.py`, one file per dataset in `results/`): datasets
with the crisis periods of `src/data.py`, in which a crisis, and the 30 days
before it, doubles volatility and raises the correlation between assets from
0.3 to 0.7, with Gaussian days, so that topology has no shape of its own to
detect (the null); the same with heavy-tailed crisis days (multivariate
Student t, 3 degrees of freedom, same correlation); and a positive control in
which crises change only the tails. No market data were used. Each cell is the
number of 20 datasets in which the whole criterion passed:

| Comparison | Gaussian crises (null) | Heavy-tailed crises | Tails only (positive control) |
|---|---|---|---|
| topology + spectrum + volatility vs spectrum + volatility (primary) | 3/20 | 0/20 | 7/20 |
| topology + spectrum vs spectrum | 6/20 | 1/20 | 11/20 |
| primary, with Mardia's kurtosis on both sides | 1/20 | 1/20 | 2/20 |
| topology + leading eigenvalue vs eigenvalue (plan v0.1.1) | 8/20 | 5/20 | 20/20 |
| primary, Gidea & Katz settings: H1, 50-day windows | 0/20 | – | – |
| primary, Gidea & Katz settings: H1, 100-day windows | 0/20 | – | – |

10. *Primary comparison.* Topology plus the leading eigenvalue share against
    that share alone passed in 8 of the 20 null datasets (95 % interval
    19–64 %), above the 0.19 stated for chance. Two channels were left open.
    Correlation: for independent Gaussian days, the distances between the
    days of a standardised window, all that persistence uses, depend on the
    window's correlation matrix only through its eigenvalues
    (`src/features.py`), so the full spectrum is the complete correlation
    control; with it the rate fell only to 6 of 20.
    Volatility: standardising removes the volatility level of a window, not a
    change inside it, and near the start of a crisis the label is almost a
    function of the fraction of turbulent days in the window (ROC-AUC
    0.96–0.99 inside the held-out blocks of the counted episodes). The
    hypothesis names volatility as a baseline, but the primary comparison left
    it out. With volatility and the spectrum on both sides the criterion
    passed in 3 of the 20 null datasets (3–38 %), in line with the 0.19
    stated for chance.
11. *Reported comparisons.* Three other controls are reported with the primary
    comparison and count for nothing: the spectrum alone; the primary
    comparison with Mardia's multivariate kurtosis on both sides; and the
    leading eigenvalue share, the primary comparison of version 0.1.1. For
    independent elliptical days, the persistence of a standardised window
    depends on the correlation matrix only through its spectrum and otherwise
    on the law of the days' distances from the centre, whose leading summary
    is the kurtosis: a gain that survives it needs structure beyond the fourth
    moment. The hypothesis does not name the kurtosis, so it is not counted.
12. *Positive control.* With volatility and correlation unchanged and the
    crises only in the tails, a criterion with power should pass. Topology
    plus the leading eigenvalue share passed in 20 of 20 datasets, so the
    topological features see the tails. Realised volatility takes much of
    that signal, since a heavy-tailed window has an unusual standard
    deviation, and the kurtosis nearly all of it: the primary comparison
    passed in 7 of 20 (15–59 %), against 3 on the null, and with the
    kurtosis on both sides in 2. Its power is real but modest. When
    heavy-tailed crises also change volatility and correlation, the baselines
    alone separate them and it passed in 0 of 20.
13. *Settings of Gidea & Katz, reported.* The analysis is also run as Gidea &
    Katz (2018) set it up: homology in dimension 1 only, windows of 50 and 100
    days (`--setting gidea-katz-50` and `gidea-katz-100`). The step stays at
    5 days, and the purge grows with the window (9 and 19 windows). On the
    null data the primary comparison passed in none of the 20 datasets with
    either window, and topology plus the leading eigenvalue share in 0 and 2.
    These runs count for nothing.

## Window features on market data

![Window features on market data, 2005–2026](figures/features_timeseries.png)

Realised volatility, the leading eigenvalue share and the landscape `L¹`
norms in dimensions 0 and 1 of every window, 2005–2026, crisis periods
shaded (`scripts/plot_features.py`). Descriptive: it tests nothing.

## Stack

- Python 3.11+
- giotto-tda (persistent homology, landscapes)
- pandas, NumPy
- yfinance (market data)
- scikit-learn (classifiers, metrics)
- Matplotlib

## Structure

```
05-tda-finance/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt      # the pinned environment of the analysis plan
├── src/
│   ├── data.py           # prices, log returns, crisis periods and labels
│   ├── persistence.py    # sliding windows, standardisation, Vietoris-Rips persistence
│   ├── features.py       # landscapes, landscape norms, volatility and eigenvalue baselines
│   ├── validation.py     # purged walk-forward, leave-one-crisis-out, fold scoring, alarm rule
│   └── classify.py       # models, seed average and feature-set comparison on shared folds
├── scripts/
│   ├── run_plan.py       # the analysis plan end to end: synthetic dry run by default, --data frozen
│   ├── calibrate_plan.py # how often the criterion passes on synthetic data
│   ├── post_result.py    # analyses added after the first run on market data
│   └── plot_features.py  # the window features over time, crisis periods shaded
├── figures/              # features_timeseries.png
├── results/              # the run on market data (plan*.json), calibration summaries and one file per synthetic dataset
└── tests/
    ├── test_validation.py
    ├── test_features.py
    ├── test_plan.py
    └── test_post_result.py
```

## Roadmap

- [x] Crisis labels fixed in advance; windows labelled by their last day.
- [x] Leakage-free validation: purged walk-forward and leave-one-crisis-out, with tests.
- [x] Baselines: realised volatility and the eigenvalue spectrum of the correlation matrix.
- [x] Success criterion run on synthetic data before any market data: null, heavy tails, positive control.
- [x] Daily prices of sector ETFs and indices, frozen with their SHA-256 (analysis plan); log returns.
- [x] Persistent homology (dimensions 0 and 1) on standardised windows.
- [x] Landscape norms as features.
- [x] Compare feature sets on both split schemes.
- [x] Time series of the features annotated with the crisis periods.

## References

- Gidea, M. & Katz, Y. (2018). *Topological data analysis of financial time
  series: Landscapes of crashes.* Physica A 491, 820–834.
- Bubenik, P. (2015). *Statistical topological data analysis using
  persistence landscapes.* Journal of Machine Learning Research 16, 77–102.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*, ch. 7
  (cross-validation in finance: purging and embargo). Wiley.
- Mardia, K. V. (1970). *Measures of multivariate skewness and kurtosis with
  applications.* Biometrika 57(3), 519–530.
- Kritzman, M., Li, Y., Page, S. & Rigobon, R. (2011). *Principal components
  as a measure of systemic risk.* Journal of Portfolio Management 37(4), 112–126.
- Tauzin, G. et al. (2021). *giotto-tda: A topological data analysis toolkit
  for machine learning and data exploration.* Journal of Machine Learning
  Research 22(39), 1–6.

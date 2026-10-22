# TDA on financial time series — regime change detection

[![CI](https://img.shields.io/github/actions/workflow/status/crozol/05-tda-finance/tests.yml?label=CI&style=flat-square)](https://github.com/crozol/05-tda-finance/actions/workflows/tests.yml)
[![Portfolio page](https://img.shields.io/badge/Portfolio-page-7C3AED?style=flat-square)](https://crozol.github.io/projects/05-tda-finance.html)

**Status: in progress.** The data pipeline, features and evaluation design are
implemented and tested; no result has been computed yet, and this document
reports none. The analysis plan, including the success criterion, was fixed
before any result (section "Analysis plan"); a negative control in the tests
shows that shuffled k-fold gives ROC-AUC ≈ 0.97 to an unrelated feature,
against ≈ 0.5 with purged folds.

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
and the leading eigenvalue of the correlation matrix, evaluated on the same
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
- **Baselines on identical folds.** Realised volatility and the share of
  variance in the first principal component of the correlation matrix
  (the one-component absorption ratio, Kritzman et al., 2011) are scored with
  the same model and splits as the topological features, alone and combined.
- **Scale removed from the topology.** Each window is standardised per asset
  before computing persistence, so the diagrams describe the shape of the
  joint distribution rather than re-measuring its volatility.
- **Success criterion.** Topology plus baselines must beat baselines alone,
  in ROC-AUC and average precision, under both split schemes.

`tests/test_validation.py` checks that the purged splits share no day between
training and test windows (and that one window less of purge would), that
every crisis is held out once, and demonstrates the leakage on synthetic
data: a feature with no relation to the label, but slowly varying in time,
reaches a ROC-AUC near 0.97 under shuffled k-fold and about 0.5 under purged
walk-forward.

## Analysis plan (fixed before any result)

Committed before any feature is computed on market data. Any change made
after the first result will be listed at the end of this section, with its
date and reason.

- **Episodes.** Six crisis periods (`src/data.py`). The 2015 episode lasts
  ten trading days, about two windows at a 5-day step, so it is reported but
  not counted: the effective number of episodes is 5 (2008–09, 2011, Q4 2018,
  2020, 2022). With so few, the conclusion is qualitative; no p-value is
  computed across episodes.
- **Primary comparison.** Topological features plus the leading eigenvalue
  share against the leading eigenvalue share alone. Realised volatility is a
  secondary baseline.
- **Success criterion.** The combined model has a higher ROC-AUC than the
  eigenvalue alone in at least 4 of the 5 held-out episodes under
  leave-one-crisis-out, and a higher ROC-AUC under purged walk-forward.
  Average precision is reported alongside.
- **"Detects earlier", defined.** For each episode and each score
  (topology, eigenvalue, volatility), with the same rule for all three:
  - threshold: the 95th percentile of the score on the normal windows of the
    training fold;
  - alarm: the first window, from 60 trading days before the start of the
    crisis, after which the score exceeds the threshold for 3 consecutive
    windows;
  - lead time: trading days from the alarm to the drawdown minimum (the end
    date of the episode). No alarm before the minimum counts as a miss.
- **Fixed parameters.** Window 40 days, step 5, homology dimensions 0 and 1,
  `L¹` and `L²` landscape norms, a random forest of 300 trees as the main
  model (logistic regression as a check). These are the defaults in `src/`
  and are not tuned on the held-out episodes.

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
├── requirements.txt
├── src/
│   ├── data.py           # prices, log returns, crisis periods and labels
│   ├── persistence.py    # sliding windows, standardisation, Vietoris-Rips persistence
│   ├── features.py       # landscapes, landscape norms, volatility and eigenvalue baselines
│   ├── validation.py     # purged walk-forward, leave-one-crisis-out, fold scoring
│   └── classify.py       # models and feature-set comparison on shared folds
└── tests/
    └── test_validation.py
```

## Roadmap

- [x] Crisis labels fixed in advance; windows labelled by their last day.
- [x] Leakage-free validation: purged walk-forward and leave-one-crisis-out, with tests.
- [x] Baselines: realised volatility and leading eigenvalue of the correlation matrix.
- [ ] Download daily prices of sector ETFs and indices; log returns.
- [ ] Persistent homology (dimensions 0 and 1) on standardised windows.
- [ ] Landscape norms and landscapes as features.
- [ ] Compare feature sets on both split schemes.
- [ ] Time series of the features annotated with the crisis periods.

## References

- Gidea, M. & Katz, Y. (2018). *Topological data analysis of financial time
  series: Landscapes of crashes.* Physica A 491, 820–834.
- Bubenik, P. (2015). *Statistical topological data analysis using
  persistence landscapes.* Journal of Machine Learning Research 16, 77–102.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*, ch. 7
  (cross-validation in finance: purging and embargo). Wiley.
- Kritzman, M., Li, Y., Page, S. & Rigobon, R. (2011). *Principal components
  as a measure of systemic risk.* Journal of Portfolio Management 37(4), 112–126.
- Tauzin, G. et al. (2021). *giotto-tda: A topological data analysis toolkit
  for machine learning and data exploration.* Journal of Machine Learning
  Research 22(39), 1–6.

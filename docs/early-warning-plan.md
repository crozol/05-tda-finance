# Early warning: analysis plan

Version 0.3.0, fixed on 2026-11-24 before any feature is computed on the market data it uses.
Code: `src/early_warning.py`, `scripts/early_warning.py`; tests: `tests/test_early_warning.py`.

## Question

Gidea & Katz (2018) report that the norms of persistence landscapes of four US indices, and the
variance and low-frequency spectral density of those norms, rise in the months before the crashes
of 2000 and 2008. The question here: **do the windows that end shortly before the peak of a large
drawdown differ from calm windows in their topology more than they do in volatility and
correlation?** Rising variance before crashes is documented without topology (Guttal et al., 2016),
so the topology is scored against those baselines, on held-out episodes, with a test calibrated on
synthetic markets before the market data are used.

## Data

- Daily closes of the S&P 500, Dow Jones Industrial Average, Nasdaq Composite and Russell 2000
  (`^GSPC`, `^DJI`, `^IXIC`, `^RUT`), the four indices of Gidea & Katz, downloaded with yfinance
  1.8.0 on 2026-11-24 and frozen as `data/indices_2026-11-24.csv`
  (SHA-256 `45e86aaa8581ff96e9d4aa2dd0d0218076376963a0ed492f99c8d8675503cbcf`).
- 1992-01-02 to 2026-11-23, 8 786 days: Yahoo's Dow Jones series starts in 1992 (Gidea & Katz
  start in December 1987).
- Daily log returns.

## Episodes

A rule on the S&P 500 closes, with no date chosen by hand:

- a **peak** is a close that is the highest of the previous 252 trading days and is followed by a
  close at least 15 % below it before any higher close;
- each peak starts an **episode**, which ends at the lowest close before the next peak
  (the **trough**). A rally inside a long bear market sets no 52-week high and starts nothing new.

| Peak | Trough | Drawdown |
|---|---|---|
| 1998-07-17 | 1998-08-31 | −19.3 % |
| 2000-03-24 | 2002-10-09 | −49.1 % |
| 2007-10-09 | 2009-03-09 | −56.8 % |
| 2010-04-23 | 2010-07-02 | −16.0 % |
| 2011-04-29 | 2011-10-03 | −19.4 % |
| 2018-09-20 | 2018-12-24 | −19.8 % |
| 2020-02-19 | 2020-03-23 | −33.9 % |
| 2022-01-03 | 2022-10-12 | −25.4 % |
| 2025-02-19 | 2025-04-08 | −18.9 % |

## Windows and labels

- Windows of 50 trading days (as Gidea & Katz), step 5; each window is dated by its last day.
- **Positive**: the window ends 1 to 60 trading days before a peak (12 windows per episode).
- **Negative**: the window ends at least 120 trading days from every peak and every trough, before
  or after it, and outside every episode.
- **Left out**: everything else, the drawdowns and the rebounds after them included. The features
  ignore the sign of the returns; a label that depends on it (inside a drawdown or after it) is
  not asked of them.
- Counts: 1 748 windows, 108 positive, 1 023 negative.
- What is known when a window ends: that a peak follows within 60 trading days is known only
  later, which is the point of a prediction; the label of a negative window depends on events up
  to 120 trading days after its last day. Neither enters a feature.

## Features

- **Topology (Gidea & Katz)**: Vietoris–Rips persistence of the window of raw log returns, 50
  points in R⁴, in dimension 1; the L¹ and L² norms of its persistence landscape summed over all
  layers (their eq. 2.3). In closed form, ‖λ‖ₚᵖ = Σ (d − b)^(p+1) / (2ᵖ (p + 1)) over the points
  (b, d) of the diagram (`src.features.total_persistence_norms`): no grid.
- **Baselines**: realised volatility (mean over the four indices of the standard deviation of
  their returns in the window), the eigenvalues of the correlation matrix divided by 4, and the
  eigenvalues of the covariance matrix. For independent Gaussian days, the topology of a raw window
  depends on everything else only through its covariance spectrum (the raw counterpart of the
  proposition checked in `scripts/spectrum_check.py`): with the covariance spectrum among the
  baselines, topology can add something only through structure a Gaussian does not have.

## Model and held-out blocks

- Random forest, 300 trees, minimum 5 samples per leaf, balanced classes; seeds 0–9.
- Every window belongs to the block of the episode whose peak is nearest to its last day.
- **Leave one episode out** (primary): the labelled windows of one block are held out; the model
  is fitted on the labelled windows of the other blocks, minus 9 windows on each side of the held-out
  block, so that no training window shares a day with a test window.
- **Walk-forward** (reported): the same test blocks, fitted only on the windows that end before the
  block (8 of 9 blocks are scorable). The labels of its last training windows use events after the
  start of the test block (look-ahead in the labels, not in the features).

## Primary statistic and test

- For a held-out block and a seed: the class-balanced log-loss,
  ½ [mean over positives of −log p + mean over negatives of −log(1 − p)], probabilities clipped to
  [0.001, 0.999], of a forest on the baselines minus that of a forest on topology plus the
  baselines. Positive when topology helps.
- **S** = the mean over the 9 blocks of that difference, averaged over the 10 seeds.
- **Monte Carlo test** (Barnard, 1963): p = (1 + #{S_null ≥ S}) / (N + 1) against N = 99 synthetic
  null datasets. Size checked by leaving each null dataset out in turn.

## Synthetic markets

Four one-factor return series on the market's trading days, with the market's episodes and the
same windows and labels. Regimes (daily volatility, correlation between the indices):

| Regime | Days | Volatility | Correlation |
|---|---|---|---|
| normal | everything else | 1.0 % | 0.80 |
| pre-peak | the 109 days before each peak (every day of a positive window) | 1.1 % | 0.82 |
| crisis | peak to trough | 2.0 % | 0.90 |
| rebound | the 120 days after each trough | 1.5 % | 0.85 |

- **Null** (99 datasets, seeds 0–98): every day Gaussian, so topology has nothing to add to the
  covariance spectrum except through windows that mix two regimes, as in the market. The pre-peak
  volatility and correlation were set on synthetic data only, so that the baselines reach a ROC-AUC
  near 0.7 (0.68 in the dry run on seed 0).
- **Heavy-tailed** (20 datasets, seeds 100–119): as the null, with the pre-peak days multivariate
  Student t with 3 degrees of freedom at the same covariance. The realistic alternative.
- **Tails only** (20 datasets, seeds 100–119): the positive control; pre-peak days keep the
  normal volatility and correlation, and only their tails change.

## Reading, fixed before the market data

- **Power**: the share of the 20 heavy-tailed datasets whose p-value against the 99 nulls is at
  most 0.05. The test has power when that share exceeds 0.05 by an exact one-sided binomial test at
  level 0.05 (at least 4 of 20).
- Without power: *the test cannot answer the question*, whatever p is on market data.
- With power: p ≤ 0.05 on market data is *evidence that the topology adds early warning to the
  baselines*; p > 0.05 is *no evidence*, reported with the power.

## Reported, counting for nothing

- ΔAUC per block (mean ± standard deviation over the seeds) and the number of blocks where the
  mean exceeds the spread.
- The walk-forward scheme (Monte Carlo p against the first 20 null datasets).
- The plan's topology of version 0.1: standardised windows, dimensions 0 and 1, L¹ and L² norms in
  closed form, against volatility and the correlation spectrum, its complete Gaussian control
  (Monte Carlo p against the first 20 null datasets).
- Logistic regression in place of the forest (market data only).
- The ROC-AUC of the baselines on market data, against its 5–95 % range on the null datasets: how
  hard the market's question is compared with the synthetic one.

## Replication of Gidea & Katz, descriptive

Computed after this plan is fixed and counting for nothing:

- L¹ and L² norms of dimension-1 landscapes of raw log returns, windows of 50 and 100 days, step 1,
  1992–2016 (their period from the first day Yahoo has all four indices).
- On rolling windows of 500 days: the variance of the norm series, its lag-1 autocorrelation and its
  mean spectral density at low frequencies (the periodogram of the demeaned segment, averaged over
  frequencies up to 0.05 cycles per day; Gidea & Katz do not specify the band).
- Kendall's τ of each statistic over the 250 trading days before 2000-03-10 and 2008-09-15, their
  dates; they report τ = 0.89 and 1.00 for the spectral density of the L¹ norm with 50-day windows.
- The same statistics computed on realised volatility, and the share of all 250-day stretches of
  1992–2016 in which τ reaches the value found before each crash.

## Limits

- The market data were not unseen: the features of seven sector ETFs over 2005–2026 are published
  in this project, and the finding of Gidea & Katz on 2000 and 2008 is public. What has not been
  computed is this test — topology beyond the covariance spectrum, on this label, these indices and
  these blocks — so it is confirmatory for this test only. A fully clean test can only be
  prospective, on drawdowns after 2026-11-23.
- Nine episodes, with 12 positive windows each.

## References

- Barnard, G. A. (1963). Discussion of *The spectral analysis of point processes* by M. S. Bartlett.
  JRSS B 25(2), 294.
- Cohen-Steiner, D., Edelsbrunner, H., Harer, J. & Mileyko, Y. (2010). Lipschitz functions have
  Lp-stable persistence. *Foundations of Computational Mathematics* 10(2), 127–139.
- Gidea, M. & Katz, Y. (2018). Topological data analysis of financial time series: Landscapes of
  crashes. *Physica A* 491, 820–834.
- Guttal, V., Raghavendra, S., Goel, N. & Hoarau, Q. (2016). Lack of critical slowing down suggests
  that financial meltdowns are not critical transitions, yet rising variability could signal
  systemic risk. *PLoS ONE* 11(1), e0144198.

# TDA on financial time series — regime change detection

Using **persistent homology** on sliding windows of multi-asset price series to detect market regime changes (e.g. the 2008 crisis, the 2020 crash, normal periods).

## Motivation

Classical methods for detecting regime changes in finance rely on variance, rolling correlations, and HMMs. Topological Data Analysis (TDA) offers a complementary angle: look at the **shape** of the return cloud in return space and extract topological descriptors (persistence bars) that are invariant under smooth transformations.

This project applies persistent homology to sliding windows of a multi-asset index and shows that the topological descriptors spike just before / during crises.

## Stack

- Python 3.11+
- Giotto-TDA (or Ripser)
- Pandas, NumPy
- yfinance (market data)
- scikit-learn (classifier over topological features)
- Matplotlib

## Structure

```
05-tda-finance/
├── README.md
├── requirements.txt
├── src/
│   ├── data.py           # Download and normalize prices
│   ├── persistence.py    # Persistent homology over sliding windows
│   ├── features.py       # Persistence landscapes / images
│   └── classify.py       # Regime classifier
└── notebooks/
```

## Roadmap

- [ ] Download daily prices of indices and sectors (SP500, NASDAQ, sector ETFs).
- [ ] Compute log returns and normalize.
- [ ] Sliding windows → point clouds in ℝ^d.
- [ ] Persistent homology (dim 0 and 1) per window.
- [ ] Convert diagrams into persistence landscapes / images.
- [ ] Train classifier (Random Forest) normal vs. crisis.
- [ ] Temporal plot annotated with real macro events.

## Expected results

Temporal "topological anomaly score" curve where peaks coincide with: subprime crisis (2008), European default (2011), China selloff (2015), COVID (2020). Classifier with AUC > 0.85 separating normal regime from crisis.

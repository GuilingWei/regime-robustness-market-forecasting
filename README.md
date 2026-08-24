# Regime Robustness in Market Forecasting

A scoped empirical study of how deep learning models for market prediction
degrade under a regime shift (calm → volatile market conditions), and whether
model simplicity affects that degradation.

## Motivation

Recent literature on limit-order-book forecasting (e.g. Zhang, Zohren & Roberts'
DeepLOB, and the more recent TLOB paper) repeatedly notes that models trained
under one market condition generalize poorly to another. This project runs a
direct, honest test of that claim on public crypto market data, comparing a
simple MLP-based architecture (MLPLOB) against classical baselines (GARCH,
XGBoost/Ridge).

## Research questions

1. How much does model performance degrade when trained on one volatility
   regime and tested, unmodified, on a different one?
2. Does a simpler model (MLPLOB) degrade more or less than classical baselines
   under this shift?
3. What does the failure look like — which features/conditions drive the
   degradation, and can a lightweight fix reduce it?

## Data

- Source: [Binance public data archive](https://data.binance.vision)
- Symbols: BTCUSDT, ETHUSDT (spot)
- Interval: 1h klines
- Period: Jan 2023 – Aug 2026
- Not committed to this repo — regenerate via `src/data_loader.py` (see below).

## Assumptions and limitations

- OHLCV-derived features are used as a proxy for true limit-order-book input;
  this is an adaptation of MLPLOB, not a faithful reproduction on its original
  input type.
- Regimes are defined via a rolling (168h) realized-volatility threshold —
  calm = below the 25th percentile, stress = above the 85th percentile.
  Final windows: calm = 2023-07-17 to 2023-08-17 (~31 days), stress =
  2025-03-01 to 2025-03-17 (~16 days). See `src/regime_analysis.py` and
  `results/regime_volatility.png` for the full analysis. The calm window is
  short (~570 usable hourly rows after feature warmup), which is a genuine
  data-scarcity limitation worth keeping in mind when interpreting model
  performance.
- Crypto market dynamics differ from traditional equities/futures; findings
  here should be read as evidence about the *mechanism* (regime-shift
  degradation), not as claims that transfer numerically to other asset classes.
- Results are based on a single train/test regime split, not a repeated or
  statistically bootstrapped experiment.

## Repo structure

```
src/            reusable code: data loading, features, validation, models
notebooks/      exploratory analysis and the regime experiment writeup
results/        saved metrics, plots, tables
tests/          unit tests, especially for the validation splitter
```

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Results

_(to be filled in as the project progresses)_

| Model    | Same-regime score | Cross-regime score | Degradation |
|----------|-------------------|---------------------|-------------|
| GARCH    | -                 | -                   | -           |
| XGBoost  | -                 | -                   | -           |
| MLPLOB   | -                 | -                   | -           |

## What I'd try next

_(to be filled in)_

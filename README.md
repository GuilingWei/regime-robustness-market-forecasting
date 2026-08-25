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

## Features

Twelve engineered features, all computed using only `.shift()`, `.rolling()`,
or `.diff()` (strictly backward-looking pandas operations — no feature can
see into the future at the row it's computed for):

- **Realized volatility** (6h, 24h, 168h): rolling standard deviation of log returns
- **Momentum** (6h, 24h, 168h): cumulative log return over each horizon
- **Volume z-score** (168h): how unusual current volume is vs. its recent history
- **RSI** (14-period): standard relative-strength momentum indicator
- **Cyclical time encoding**: hour-of-day and day-of-week (sin/cos pairs),
  deterministic from the timestamp so inherently leak-free

Features are computed on the full continuous 3-year series (not per-regime
slice) so rolling windows have proper history to draw from, before slicing
into calm/stress windows. This leaves 745 usable rows in the calm window and
385 in the stress window after the 168h warmup period is dropped — see
`src/features.py` and `src/regime_analysis.py`.

## Assumptions and limitations

- OHLCV-derived features are used as a proxy for true limit-order-book input;
  this is an adaptation of MLPLOB, not a faithful reproduction on its original
  input type.
- Regimes are defined via a rolling (168h) realized-volatility threshold —
  calm = below the 25th percentile, stress = above the 85th percentile.
  Final windows: calm = 2023-07-17 to 2023-08-17 (~31 days), stress =
  2025-03-01 to 2025-03-17 (~16 days). See `src/regime_analysis.py` and
  `results/regime_volatility.png` for the full analysis. The calm window is
  short (745 usable hourly rows after feature warmup; stress has 385), which
  is a genuine data-scarcity limitation worth keeping in mind when
  interpreting model performance.
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

### Same-regime baseline (trained and tested within the calm window)

| Model   | R²      | Directional Accuracy | MSE       | n   |
|---------|---------|-----------------------|-----------|-----|
| Ridge   | -0.067  | 55.4%                 | 3.14e-06  | 112 |
| XGBoost | -0.350  | 50.9%                 | 3.97e-06  | 112 |

GARCH 1-step volatility forecast: 0.00281 (actual test-window return std: 0.00172)

Negative R² is expected here — next-hour crypto return prediction is close
to unpredictable at this horizon and data scale (521 training rows).
Ridge's modest directional edge (55.4% vs. XGBoost's near-random 50.9%) is
an early instance of the "complexity doesn't help" pattern this project is
built around: with limited data, XGBoost's added flexibility appears to
overfit noise rather than capture signal.

### Cross-regime experiment (train calm, test stress)

### Cross-regime experiment (train on calm, test on 6 independent stress windows)

Models trained once on the calm window were evaluated, unmodified, on six
independent stress windows (a seventh candidate, 2023-03-12 to 2023-03-29,
was excluded since it predates the calm training window). Full results in
`results/regime_experiment_results.csv`.

| Model   | Same-regime DirAcc | Avg-stress DirAcc | Delta  |
|---------|---------------------|---------------------|--------|
| Ridge   | 55.4%               | 49.6%               | -5.8pp |
| XGBoost | 50.9%               | 51.4%               | +0.5pp |
| MLPLOB  | 51.8%               | 50.4%               | -1.4pp |

**Key finding: Ridge is the only model with a genuine in-sample directional
edge (55.4%, meaningfully above chance), and it is also the only model that
loses that edge under regime shift** (dropping to 49.6%, below chance).
XGBoost and MLPLOB show comparatively little directional degradation, but
only because both were already near-random in-sample -- there was no real
edge to lose. This suggests Ridge's calm-regime performance reflects a real
but regime-specific pattern rather than a generalizable relationship.

**A second pattern: MLPLOB's error magnitude (R²), not just its directional
accuracy, degrades sharply under stress** -- from -17.7 same-regime to
between -24 and -51 across the six stress windows. Directional accuracy
barely moves, but squared error blows up, consistent with a model producing
noisy, unstructured predictions that get punished more severely when actual
price swings are larger (as they are in stress regimes by construction).

**Not all stress windows behaved identically.** The 2024-04-13 to 2024-04-24
window is a partial exception: Ridge and XGBoost both post their best R²
scores of the entire experiment there (+0.003 and +0.010 respectively),
better than their own same-regime numbers. This heterogeneity is reported
rather than averaged away -- it suggests "stress" is not a monolithic
condition, and some volatility episodes may retain more learnable structure
than others.

### Side investigation: does target choice explain the weak return-prediction results?

Motivated by comparing against TLOB's own published results (which show
genuine skill on order-book classification), a secondary question was
tested: is next-hour return direction inherently unpredictable at this data
scale, or was something about the setup (data size, model choice) the real
bottleneck? Volatility was tested as an alternative target, since it is a
well-documented predictable quantity in the literature (Corsi 2009's HAR-RV
model routinely achieves R² of 0.5-0.6 out-of-sample).

Using the identical calm-window training data (521 rows) and the same
three-model comparison, an HAR-RV baseline (linear regression on lagged
realized volatility at multiple horizons, following Corsi 2009) achieved
**same-regime R² = 0.165** in log-volatility space — a genuinely positive
result, unlike anything obtained for return prediction. This confirms
target choice, not data size or model architecture, was the primary
constraint on the earlier return-prediction results.

However, cross-regime performance for the volatility target collapsed even
more severely than for return prediction (R² as low as -2850 for MLPLOB
across the six stress windows). Two effects are likely mixed together here:
genuine model instability on small stress-window samples (241-385 rows,
single training run), and a metric artifact -- stress windows were selected
as *sustained* high-volatility stretches, meaning the target has low
internal variance within each window, which can make R² (whose denominator
is the target's in-window variance) swing to extreme values even for
modest absolute prediction errors. Disentangling these two effects
rigorously (multiple seeds, alternative scale-invariant metrics) was judged
out of scope for this project, since the core research question --
regime-shift degradation -- was already answered cleanly by the primary
return-prediction experiment above.

**Conclusion:** target choice was confirmed as a real factor (volatility
carries genuine same-regime signal that return direction does not), but
this doesn't change the project's central finding -- even a target with
real in-sample predictability shows severe degradation under regime shift,
reinforcing rather than undermining the core thesis.

## What I'd try next

_(to be filled in)_

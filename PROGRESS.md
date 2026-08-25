# Progress Tracker

## Day 1: Data pull + regime definition
- [x] Download BTCUSDT 1h klines
- [x] Download ETHUSDT 1h klines
- [x] Sanity-check both files (correct row count, no gaps, chronological)
- [x] Decide regime definition (rolling volatility threshold vs event window) — document reasoning in notebook

## Day 2: EDA + leak-free feature engineering
- [x] Plot price/volume for both symbols, confirm regimes look visually distinct
- [x] Implement `add_features()` in `src/features.py`
- [x] Verify no lookahead leakage in any feature

## Day 3: Walk-forward validation harness
- [x] Implement `walk_forward_split()` in `src/validation.py`
- [x] Implement `assert_no_leakage()` + write the unit test in `tests/test_validation.py`
- [x] Confirm tests pass — 5/5 passing (`pytest tests/ -v`)

## Day 4: Baseline models
- [x] Implement GARCH baseline
- [x] Implement XGBoost/Ridge baseline
- [x] Record baseline metrics on same-regime split

## Day 5-6: MLPLOB implementation
- [x] Implement `MLPLOB` architecture in `src/models/mlplob.py`
- [x] Get it training on same-regime split, comparable to baseline

## Day 7: Regime-generalization experiment
- [x] Train on calm regime, test on stress regime (all models)
- [x] Extended to 6 independent stress windows (not just one) for statistical robustness
- [x] Record degradation for each model — Ridge loses its only real edge under
      regime shift; MLPLOB's error magnitude, not direction, blows up under stress

## Day 7.5: Side investigation — target choice diagnostic
- [x] Tested volatility (HAR-RV) as an alternative target on the same data
- [x] Confirmed target choice was a real factor (positive same-regime R²,
      unlike return prediction)
- [x] Observed severe cross-regime collapse, likely a mix of genuine
      instability and an R² metric artifact — documented, not chased further
           
## Day 8: Diagnose + propose fix
- [ ] Identify which features shifted most between regimes
- [ ] Propose one concrete fix
- [ ] (Optional) implement and test the fix

## Day 9: Write-up + rehearsal
- [ ] Fill in README results table
- [ ] Write "what I'd try next" section
- [ ] Practice explaining the project out loud, under 3 minutes

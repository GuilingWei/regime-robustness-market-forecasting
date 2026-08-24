# Progress Tracker

## Day 1: Data pull + regime definition
- [x] Download BTCUSDT 1h klines
- [x] Download ETHUSDT 1h klines
- [x] Sanity-check both files (correct row count, no gaps, chronological)
- [ ] Decide regime definition (rolling volatility threshold vs event window) — document reasoning in notebook

## Day 2: EDA + leak-free feature engineering
- [ ] Plot price/volume for both symbols, confirm regimes look visually distinct
- [ ] Implement `add_features()` in `src/features.py`
- [ ] Verify no lookahead leakage in any feature

## Day 3: Walk-forward validation harness
- [ ] Implement `walk_forward_split()` in `src/validation.py`
- [ ] Implement `assert_no_leakage()` + write the unit test in `tests/test_validation.py`
- [ ] Confirm tests pass

## Day 4: Baseline models
- [ ] Implement GARCH baseline
- [ ] Implement XGBoost/Ridge baseline
- [ ] Record baseline metrics on same-regime split

## Day 5-6: MLPLOB implementation
- [ ] Implement `MLPLOB` architecture in `src/models/mlplob.py`
- [ ] Get it training on same-regime split, comparable to baseline

## Day 7: Regime-generalization experiment
- [ ] Train on calm regime, test on stress regime (all models)
- [ ] Record degradation for each model

## Day 8: Diagnose + propose fix
- [ ] Identify which features shifted most between regimes
- [ ] Propose one concrete fix
- [ ] (Optional) implement and test the fix

## Day 9: Write-up + rehearsal
- [ ] Fill in README results table
- [ ] Write "what I'd try next" section
- [ ] Practice explaining the project out loud, under 3 minutes

"""
Day 7 (extended): regime-generalization experiment across multiple stress
windows, not just one.

Trains Ridge, XGBoost, and MLPLOB once on the calm window (same train/val
split used in Days 4-6), then evaluates all three, unmodified, on the
same-regime test set plus six independent stress windows identified in
src/regime_analysis.py.

Usage:
    python -m src.train_regime_experiment
"""

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.features import get_feature_columns
from src.train_baseline import build_dataset, slice_calm_window, CALM_WINDOW
from src.models.baseline import fit_ridge_baseline, fit_xgboost_baseline
from src.train_mlplob import train_mlplob, to_tensor
from src.evaluate import compute_metrics
from src.validation import walk_forward_split, regime_train_test_split

STRESS_WINDOWS = [
    ("2024-03-01", "2024-03-12"),
    ("2024-03-15", "2024-03-27"),
    ("2024-04-13", "2024-04-24"),
    ("2024-08-05", "2024-08-19"),
    ("2025-03-01", "2025-03-17"),
    ("2026-02-03", "2026-02-13"),
]


def evaluate_all_models(ridge, xgb, mlplob, scaler, X, y, label: str) -> dict:
    row = {"window": label, "n": len(y)}

    ridge_metrics = compute_metrics(y, ridge.predict(X))
    row["ridge_r2"] = ridge_metrics["r2"]
    row["ridge_diracc"] = ridge_metrics["directional_accuracy"]

    xgb_metrics = compute_metrics(y, xgb.predict(X))
    row["xgb_r2"] = xgb_metrics["r2"]
    row["xgb_diracc"] = xgb_metrics["directional_accuracy"]

    X_scaled = scaler.transform(X)
    mlplob.eval()
    with torch.no_grad():
        mlplob_preds = mlplob(to_tensor(X_scaled)).numpy()
    mlplob_metrics = compute_metrics(y, mlplob_preds)
    row["mlplob_r2"] = mlplob_metrics["r2"]
    row["mlplob_diracc"] = mlplob_metrics["directional_accuracy"]

    return row


def main():
    df = build_dataset("BTCUSDT")
    calm = slice_calm_window(df)
    train_df, val_df, test_df = walk_forward_split(calm, train_frac=0.7, val_frac=0.15)
    feature_cols = get_feature_columns()

    X_train, y_train = train_df[feature_cols], train_df["target_next_return"]
    X_val, y_val = val_df[feature_cols], val_df["target_next_return"]

    print("Fitting Ridge and XGBoost on calm-window training data...")
    ridge = fit_ridge_baseline(X_train, y_train)
    xgb = fit_xgboost_baseline(X_train, y_train)

    print("Fitting MLPLOB (same architecture/training as Day 5-6)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    mlplob = train_mlplob(X_train_scaled, y_train.values, X_val_scaled, y_val.values,
                           input_dim=len(feature_cols))

    results = []

    X_test, y_test = test_df[feature_cols], test_df["target_next_return"].values
    results.append(evaluate_all_models(ridge, xgb, mlplob, scaler, X_test, y_test,
                                         "same-regime (calm test)"))

    for start, end in STRESS_WINDOWS:
        _, stress_df = regime_train_test_split(df, CALM_WINDOW[0], CALM_WINDOW[1], start, end)
        if stress_df.empty:
            print(f"  [skip] {start} -> {end}: no rows found")
            continue
        X_s = stress_df[feature_cols]
        y_s = stress_df["target_next_return"].values
        results.append(evaluate_all_models(ridge, xgb, mlplob, scaler, X_s, y_s,
                                             f"{start} -> {end}"))

    results_df = pd.DataFrame(results)
    pd.set_option("display.width", 160)
    print("\n" + results_df.to_string(index=False))

    results_df.to_csv("results/regime_experiment_results.csv", index=False)
    print("\nSaved to results/regime_experiment_results.csv")

    same_regime = results_df.iloc[0]
    stress_only = results_df.iloc[1:]
    print("\n--- Average directional accuracy across stress windows vs same-regime ---")
    for model in ["ridge", "xgb", "mlplob"]:
        same = same_regime[f"{model}_diracc"]
        avg_stress = stress_only[f"{model}_diracc"].mean()
        print(f"  {model}: same-regime={same:.3f}  avg-stress={avg_stress:.3f}  "
              f"delta={avg_stress - same:+.3f}")


if __name__ == "__main__":
    main()

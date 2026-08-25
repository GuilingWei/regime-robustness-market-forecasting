"""
Day 7.5: diagnostic test -- does the same data/pipeline produce genuine
predictive power on volatility (a documented predictable target, per
Corsi 2009's HAR-RV model), or does it fail here too?

Target: next 24h realized volatility (realized_vol_24h shifted back by 24
rows, so it represents FUTURE volatility relative to each row's features).

HAR-RV baseline follows Corsi (2009): regress future realized volatility
on lagged realized volatility at multiple horizons.

Usage:
    python -m src.train_volatility
"""

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.features import add_features, get_feature_columns
from src.models.baseline import fit_xgboost_baseline
from src.models.mlplob import MLPLOB
from src.train_mlplob import train_mlplob, to_tensor
from src.evaluate import compute_metrics
from src.validation import walk_forward_split, regime_train_test_split

CALM_WINDOW = ("2023-07-17", "2023-08-17")
STRESS_WINDOWS = [
    ("2024-03-01", "2024-03-12"),
    ("2024-03-15", "2024-03-27"),
    ("2024-04-13", "2024-04-24"),
    ("2024-08-05", "2024-08-19"),
    ("2025-03-01", "2025-03-17"),
    ("2026-02-03", "2026-02-13"),
]
HAR_FEATURES = ["log_realized_vol_6h", "log_realized_vol_24h", "log_realized_vol_168h"]

def build_dataset(symbol: str = "BTCUSDT") -> pd.DataFrame:
    df = pd.read_csv(f"data/processed/{symbol}_1h.csv", parse_dates=["open_time"])
    df = add_features(df)
    for w in [6, 24, 168]:
        df[f"log_realized_vol_{w}h"] = np.log(df[f"realized_vol_{w}h"])
    df["target_next_vol_24h"] = np.log(df["realized_vol_24h"].shift(-24))  # log, not raw

    feature_cols = get_feature_columns()
    required_cols = feature_cols + ["target_next_vol_24h"]
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    return df


def slice_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df[(df["open_time"] >= start) & (df["open_time"] <= end)].reset_index(drop=True)


def fit_har_rv(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train[HAR_FEATURES], y_train)
    return model


def evaluate_all(har, xgb, mlplob, scaler, feature_cols, X, y, label: str) -> dict:
    row = {"window": label, "n": len(y)}

    har_preds = har.predict(X[HAR_FEATURES])
    row["har_r2"] = compute_metrics(y, har_preds)["r2"]

    xgb_preds = xgb.predict(X[feature_cols])
    row["xgb_r2"] = compute_metrics(y, xgb_preds)["r2"]

    X_scaled = scaler.transform(X[feature_cols])
    mlplob.eval()
    with torch.no_grad():
        mlplob_preds = mlplob(to_tensor(X_scaled)).numpy()
    row["mlplob_r2"] = compute_metrics(y, mlplob_preds)["r2"]

    return row


def main():
    df = build_dataset("BTCUSDT")
    calm = slice_window(df, *CALM_WINDOW)
    train_df, val_df, test_df = walk_forward_split(calm, train_frac=0.7, val_frac=0.15)
    feature_cols = get_feature_columns()

    print(f"Calm window: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    X_train, y_train = train_df, train_df["target_next_vol_24h"]
    X_val, y_val = val_df, val_df["target_next_vol_24h"]

    print("\nFitting HAR-RV (Corsi 2009 baseline)...")
    har = fit_har_rv(X_train, y_train)

    print("Fitting XGBoost...")
    xgb = fit_xgboost_baseline(X_train[feature_cols], y_train)

    print("Fitting MLPLOB...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[feature_cols])
    X_val_scaled = scaler.transform(X_val[feature_cols])
    mlplob = train_mlplob(X_train_scaled, y_train.values, X_val_scaled, y_val.values,
                           input_dim=len(feature_cols))

    results = []
    X_test, y_test = test_df, test_df["target_next_vol_24h"].values
    results.append(evaluate_all(har, xgb, mlplob, scaler, feature_cols, X_test, y_test,
                                  "same-regime (calm test)"))

    for start, end in STRESS_WINDOWS:
        _, stress_df = regime_train_test_split(df, CALM_WINDOW[0], CALM_WINDOW[1], start, end)
        if stress_df.empty:
            continue
        y_s = stress_df["target_next_vol_24h"].values
        results.append(evaluate_all(har, xgb, mlplob, scaler, feature_cols, stress_df, y_s,
                                      f"{start} -> {end}"))

    results_df = pd.DataFrame(results)
    pd.set_option("display.width", 160)
    print("\n" + results_df.to_string(index=False))
    results_df.to_csv("results/volatility_experiment_results.csv", index=False)
    print("\nSaved to results/volatility_experiment_results.csv")

    print("\n--- Diagnostic verdict ---")
    same_r2 = results_df.iloc[0]["har_r2"]
    if same_r2 > 0.1:
        print(f"HAR-RV same-regime R^2 = {same_r2:.3f} -- genuinely positive, consistent with")
        print("the literature. This confirms: the earlier near-zero/negative R^2 results were")
        print("about the TARGET (return direction, near-random-walk), not the data size or model.")
    else:
        print(f"HAR-RV same-regime R^2 = {same_r2:.3f} -- still weak even on a documented-")
        print("predictable target. This suggests data scale is a bigger factor for THIS")
        print("dataset than the broader literature would predict.")


if __name__ == "__main__":
    main()

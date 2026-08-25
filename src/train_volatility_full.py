"""
Full-dataset volatility prediction (separate from the regime-shift
experiment, which deliberately uses the small calm window).

This answers a different question: given a realistic amount of data
(~31,000 hourly rows across 3+ years), how well can HAR-RV, XGBoost, and
MLPLOB forecast next-24h log-volatility? Uses a standard walk-forward
split across the FULL series, not the artificially small/homogeneous
calm window.

Usage:
    python -m src.train_volatility_full
"""

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.features import add_features, get_feature_columns
from src.models.baseline import fit_xgboost_baseline
from src.train_mlplob import train_mlplob, to_tensor
from src.evaluate import compute_metrics
from src.validation import walk_forward_split

HAR_FEATURES = ["log_realized_vol_6h", "log_realized_vol_24h", "log_realized_vol_168h"]


def build_full_dataset(symbol: str = "BTCUSDT") -> pd.DataFrame:
    df = pd.read_csv(f"data/processed/{symbol}_1h.csv", parse_dates=["open_time"])
    df = add_features(df)
    for w in [6, 24, 168]:
        df[f"log_realized_vol_{w}h"] = np.log(df[f"realized_vol_{w}h"])
    df["target_next_vol_24h"] = np.log(df["realized_vol_24h"].shift(-24))

    feature_cols = get_feature_columns()
    required_cols = feature_cols + ["target_next_vol_24h"]
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    return df


def fit_har_rv(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train[HAR_FEATURES], y_train)
    return model


def main():
    df = build_full_dataset("BTCUSDT")
    print(f"Full dataset: {len(df)} rows")

    train_df, val_df, test_df = walk_forward_split(df, train_frac=0.7, val_frac=0.15)
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    print(f"Train period: {train_df['open_time'].min()} -> {train_df['open_time'].max()}")
    print(f"Test period:  {test_df['open_time'].min()} -> {test_df['open_time'].max()}")

    feature_cols = get_feature_columns()
    y_train = train_df["target_next_vol_24h"]
    y_val = val_df["target_next_vol_24h"]
    y_test = test_df["target_next_vol_24h"].values

    print("\nFitting HAR-RV on full dataset...")
    har = fit_har_rv(train_df, y_train)
    har_preds = har.predict(test_df[HAR_FEATURES])
    har_metrics = compute_metrics(y_test, har_preds)
    print(f"HAR-RV:  R2={har_metrics['r2']:.4f}  MSE={har_metrics['mse']:.6e}")

    print("\nFitting XGBoost on full dataset...")
    xgb = fit_xgboost_baseline(train_df[feature_cols], y_train)
    xgb_preds = xgb.predict(test_df[feature_cols])
    xgb_metrics = compute_metrics(y_test, xgb_preds)
    print(f"XGBoost: R2={xgb_metrics['r2']:.4f}  MSE={xgb_metrics['mse']:.6e}")

    print("\nFitting MLPLOB on full dataset (this will take longer -- more data per epoch)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[feature_cols])
    X_val_scaled = scaler.transform(val_df[feature_cols])
    X_test_scaled = scaler.transform(test_df[feature_cols])
    mlplob = train_mlplob(X_train_scaled, y_train.values, X_val_scaled, y_val.values,
                           input_dim=len(feature_cols),epochs=300, patience=30)
    mlplob.eval()
    with torch.no_grad():
        mlplob_preds = mlplob(to_tensor(X_test_scaled)).numpy()
    mlplob_metrics = compute_metrics(y_test, mlplob_preds)
    print(f"MLPLOB:  R2={mlplob_metrics['r2']:.4f}  MSE={mlplob_metrics['mse']:.6e}")

    results = pd.DataFrame([
        {"model": "HAR-RV", **har_metrics},
        {"model": "XGBoost", **xgb_metrics},
        {"model": "MLPLOB", **mlplob_metrics},
    ])
    results.to_csv("results/volatility_full_dataset_results.csv", index=False)
    print("\nSaved to results/volatility_full_dataset_results.csv")


if __name__ == "__main__":
    main()

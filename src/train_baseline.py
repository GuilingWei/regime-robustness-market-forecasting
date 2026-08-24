"""
Day 4: baseline models on the calm regime.

Trains Ridge and XGBoost to predict next-hour log return, and GARCH to
forecast volatility, all on a walk-forward split within the calm window
(2023-07-17 to 2023-08-17). These are the reference numbers MLPLOB needs to
be compared against later.

Usage:
    python -m src.train_baseline
"""

import pandas as pd

from src.features import add_features, get_feature_columns
from src.models.baseline import fit_garch, forecast_garch_volatility, fit_ridge_baseline, fit_xgboost_baseline
from src.evaluate import compute_metrics, print_metrics
from src.validation import walk_forward_split

CALM_WINDOW = ("2023-07-17", "2023-08-17")


def build_dataset(symbol: str = "BTCUSDT") -> pd.DataFrame:
    df = pd.read_csv(f"data/processed/{symbol}_1h.csv", parse_dates=["open_time"])
    df = add_features(df)
    df["target_next_return"] = df["log_return"].shift(-1)

    feature_cols = get_feature_columns()
    required_cols = feature_cols + ["target_next_return"]
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    return df


def slice_calm_window(df: pd.DataFrame) -> pd.DataFrame:
    start, end = CALM_WINDOW
    return df[(df["open_time"] >= start) & (df["open_time"] <= end)].reset_index(drop=True)


def main():
    df = build_dataset("BTCUSDT")
    calm = slice_calm_window(df)
    print(f"Calm window rows available for training/eval: {len(calm)}")

    train_df, val_df, test_df = walk_forward_split(calm, train_frac=0.7, val_frac=0.15)
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    feature_cols = get_feature_columns()
    X_train, y_train = train_df[feature_cols], train_df["target_next_return"]
    X_test, y_test = test_df[feature_cols], test_df["target_next_return"]

    ridge = fit_ridge_baseline(X_train, y_train)
    ridge_preds = ridge.predict(X_test)
    print_metrics("Ridge (same-regime test)", compute_metrics(y_test, ridge_preds))

    xgb = fit_xgboost_baseline(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    print_metrics("XGBoost (same-regime test)", compute_metrics(y_test, xgb_preds))

    garch_fitted = fit_garch(train_df["log_return"])
    garch_vol_forecast = forecast_garch_volatility(garch_fitted, horizon=1)
    actual_test_vol = test_df["log_return"].std()
    print(f"\nGARCH 1-step volatility forecast: {garch_vol_forecast:.6f}")
    print(f"Actual test-window return std:    {actual_test_vol:.6f}")


if __name__ == "__main__":
    main()

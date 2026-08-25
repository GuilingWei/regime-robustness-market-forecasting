"""
Day 5-6: train MLPLOB on the calm regime, compare against baselines.

Uses the same feature set, target, and walk-forward split as
src/train_baseline.py, so results are directly comparable.

Usage:
    python -m src.train_mlplob
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.features import add_features, get_feature_columns
from src.models.mlplob import MLPLOB
from src.evaluate import compute_metrics, print_metrics
from src.validation import walk_forward_split

CALM_WINDOW = ("2023-07-17", "2023-08-17")
EPOCHS = 1000
PATIENCE = 50
LEARNING_RATE = 1e-3
SEED = 42


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


def to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32)


def train_mlplob(X_train, y_train, X_val, y_val, input_dim: int,epochs: int = EPOCHS, patience: int = PATIENCE) -> MLPLOB:
    torch.manual_seed(SEED)
    model = MLPLOB(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    loss_fn = nn.MSELoss()

    X_train_t, y_train_t = to_tensor(X_train), to_tensor(y_train)
    X_val_t, y_val_t = to_tensor(X_val), to_tensor(y_val)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = loss_fn(preds, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = loss_fn(val_preds, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:3d}  train_loss={loss.item():.6e}  val_loss={val_loss:.6e}")

        if patience_counter >= patience:
            print(f"  early stopping at epoch {epoch} (best val_loss={best_val_loss:.6e})")
            break

    model.load_state_dict(best_state)
    return model


def main():
    df = build_dataset("BTCUSDT")
    calm = slice_calm_window(df)
    train_df, val_df, test_df = walk_forward_split(calm, train_frac=0.7, val_frac=0.15)

    feature_cols = get_feature_columns()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_val = scaler.transform(val_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])

    y_train = train_df["target_next_return"].values
    y_val = val_df["target_next_return"].values
    y_test = test_df["target_next_return"].values

    print(f"Training MLPLOB on {len(X_train)} rows, validating on {len(X_val)}...")
    model = train_mlplob(X_train, y_train, X_val, y_val, input_dim=len(feature_cols))

    model.eval()
    with torch.no_grad():
        test_preds = model(to_tensor(X_test)).numpy()

    print(f"\ny_test stats: min={y_test.min():.6f}  max={y_test.max():.6f}  std={y_test.std():.6f}")
    print(f"preds stats:  min={test_preds.min():.6f}  max={test_preds.max():.6f}  std={test_preds.std():.6f}")

    metrics = compute_metrics(y_test, test_preds)
    print_metrics("\nMLPLOB (same-regime test)", metrics)


if __name__ == "__main__":
    main()

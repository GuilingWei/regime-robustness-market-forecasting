"""Shared evaluation metrics for comparing models on the same footing."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def compute_metrics(y_true, y_pred) -> dict:
    """
    Standard regression metrics for return prediction, plus directional
    accuracy (did the model get the sign of the return right?) -- often more
    interpretable than R^2 for noisy financial targets.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    direction_correct = np.sign(y_true) == np.sign(y_pred)
    directional_accuracy = direction_correct.mean()

    return {
        "mse": mse,
        "r2": r2,
        "directional_accuracy": directional_accuracy,
        "n": len(y_true),
    }


def print_metrics(name: str, metrics: dict):
    print(f"{name}: MSE={metrics['mse']:.6e}  R2={metrics['r2']:.4f}  "
          f"DirAcc={metrics['directional_accuracy']:.3f}  n={metrics['n']}")
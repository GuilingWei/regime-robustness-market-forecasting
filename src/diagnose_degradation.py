"""
Day 8: diagnose the regime-shift degradation.

Focuses on Ridge, since it's the only model with genuine same-regime
directional edge (55.4%) that collapsed under regime shift (49.6% avg
stress) -- the most informative case to understand, since there was
something real to lose.

Two-part diagnosis:
1. Feature distribution shift: which features differ most between the calm
   training window and the stress windows (measured in standard deviations
   of the calm distribution)?
2. Ridge's own learned weights: which features does Ridge rely on most
   heavily? Cross-referencing this against (1) identifies which of Ridge's
   heavily-weighted features are also the most unstable across regimes --
   the likely mechanism behind the edge loss.

Usage:
    python -m src.diagnose_degradation
"""

import numpy as np
import pandas as pd

from src.features import get_feature_columns
from src.train_baseline import build_dataset, slice_calm_window, CALM_WINDOW
from src.models.baseline import fit_ridge_baseline
from src.validation import walk_forward_split, regime_train_test_split
from src.train_regime_experiment import STRESS_WINDOWS


def compute_feature_shift(calm_df: pd.DataFrame, stress_df: pd.DataFrame,
                            feature_cols: list) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        calm_mean = calm_df[col].mean()
        calm_std = calm_df[col].std()
        stress_mean = stress_df[col].mean()

        shift_in_std = (stress_mean - calm_mean) / calm_std if calm_std > 0 else np.nan

        rows.append({
            "feature": col,
            "calm_mean": calm_mean,
            "calm_std": calm_std,
            "stress_mean": stress_mean,
            "shift_in_std": shift_in_std,
        })
    return pd.DataFrame(rows).sort_values("shift_in_std", key=abs, ascending=False)


def main():
    df = build_dataset("BTCUSDT")
    calm = slice_calm_window(df)
    train_df, val_df, test_df = walk_forward_split(calm, train_frac=0.7, val_frac=0.15)
    feature_cols = get_feature_columns()

    ridge = fit_ridge_baseline(train_df[feature_cols], train_df["target_next_return"])

    print("=== Ridge's learned feature weights (sorted by |coefficient|) ===")
    weights = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": ridge.coef_,
    }).sort_values("coefficient", key=abs, ascending=False)
    print(weights.to_string(index=False))

    print("\n=== Feature distribution shift: calm training window vs. each stress window ===")
    all_shifts = []
    for start, end in STRESS_WINDOWS:
        _, stress_df = regime_train_test_split(df, CALM_WINDOW[0], CALM_WINDOW[1], start, end)
        if stress_df.empty:
            continue
        shift_df = compute_feature_shift(train_df, stress_df, feature_cols)
        shift_df["stress_window"] = f"{start} -> {end}"
        all_shifts.append(shift_df)

    combined = pd.concat(all_shifts, ignore_index=True)

    print("\n=== Average |shift_in_std| per feature across all stress windows ===")
    avg_shift = (
        combined.groupby("feature")["shift_in_std"]
        .apply(lambda x: x.abs().mean())
        .sort_values(ascending=False)
    )
    print(avg_shift.to_string())

    print("\n=== Cross-reference: Ridge's top-weighted features vs. their avg shift ===")
    weights["abs_coef"] = weights["coefficient"].abs()
    weights = weights.set_index("feature")
    weights["avg_abs_shift"] = avg_shift
    weights["risk_score"] = weights["abs_coef"] * weights["avg_abs_shift"]
    risk_ranked = weights.sort_values("risk_score", ascending=False)
    print(risk_ranked[["coefficient", "avg_abs_shift", "risk_score"]].to_string())

    risk_ranked.to_csv("results/degradation_diagnosis.csv")
    print("\nSaved to results/degradation_diagnosis.csv")

    top_risk_feature = risk_ranked.index[0]
    print(f"\n=== Diagnosis ===")
    print(f"Highest risk_score feature: '{top_risk_feature}'")
    print("This feature combines (a) a large weight in Ridge's learned model and")
    print("(b) a large average distributional shift between calm and stress regimes --")
    print("making it the most likely driver of Ridge's edge loss under regime shift.")


if __name__ == "__main__":
    main()

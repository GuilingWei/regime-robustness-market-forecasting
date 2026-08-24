"""
Classical baselines.

- GARCH: forecasts volatility (not return) -- the standard econometric
  reference point that every deep learning finance paper benchmarks against.
- Ridge / XGBoost: forecast next-period return directly from engineered
  features -- the same prediction task MLPLOB will be trained on.
"""

import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor


def fit_garch(returns: pd.Series, p: int = 1, q: int = 1):
    scaled_returns = returns.dropna() * 1000  # was *100; arch recommends scale between 1-1000
    model = arch_model(scaled_returns, vol="Garch", p=p, q=q, mean="Zero")
    fitted = model.fit(disp="off")
    return fitted

def forecast_garch_volatility(fitted_model, horizon: int = 1) -> float:
    """Return the model's h-step-ahead volatility forecast, rescaled back to log-return units."""
    forecast = fitted_model.forecast(horizon=horizon, reindex=False)
    variance = forecast.variance.values[-1, -1]
    vol = np.sqrt(variance) / 1000  # undo the *1000 scaling
    return vol


def fit_ridge_baseline(X_train: pd.DataFrame, y_train: pd.Series, alpha: float = 1.0) -> Ridge:
    """Fit a Ridge-regularized linear model on engineered features to predict next-period return."""
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def fit_xgboost_baseline(X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> XGBRegressor:
    """Fit XGBoost on engineered features to predict next-period return."""
    defaults = dict(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    defaults.update(kwargs)
    model = XGBRegressor(**defaults)
    model.fit(X_train, y_train)
    return model
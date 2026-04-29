"""Statistical helpers for pairs trading."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


def calculate_cointegration_p_value(series_a: pd.Series, series_b: pd.Series) -> float:
    """Compute Engle-Granger cointegration p-value."""
    _, p_value, _ = coint(series_a, series_b)
    return float(p_value)


def calculate_hedge_ratio(log_a: pd.Series, log_b: pd.Series) -> float:
    """Estimate hedge ratio via OLS: log_a = alpha + beta * log_b."""
    model = sm.OLS(log_a, sm.add_constant(log_b)).fit()
    return float(model.params.iloc[1])


def calculate_spread(log_a: pd.Series, log_b: pd.Series, beta: float) -> pd.Series:
    """Compute spread for the pair."""
    return log_a - beta * log_b


def calculate_z_score(spread: pd.Series) -> float:
    """Compute latest z-score from spread values."""
    std = spread.std(ddof=0)
    if std == 0 or np.isnan(std):
        return 0.0
    return float((spread.iloc[-1] - spread.mean()) / std)


def calculate_half_life(spread: pd.Series) -> float:
    """Estimate half-life of mean reversion using OU approximation.

    The regression form is:
    delta_spread_t = alpha + phi * spread_{t-1} + epsilon_t
    Half-life = -ln(2) / phi, if phi < 0.
    """
    lagged = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    aligned = pd.concat([delta, lagged], axis=1).dropna()
    if aligned.empty:
        return float("inf")

    delta_aligned = aligned.iloc[:, 0]
    lagged_aligned = aligned.iloc[:, 1]
    model = sm.OLS(delta_aligned, sm.add_constant(lagged_aligned)).fit()
    phi = float(model.params.iloc[1])

    if phi >= 0:
        return float("inf")

    half_life = -np.log(2) / phi
    if np.isnan(half_life) or half_life <= 0:
        return float("inf")
    return float(half_life)

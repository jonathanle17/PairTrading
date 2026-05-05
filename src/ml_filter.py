"""Machine-learning gatekeeper for validating trade entries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
try:
    import pandas_ta as ta
except ImportError:  # pragma: no cover - optional dependency in some runtimes
    ta = None

import config


class TradeGatekeeper:
    """Validates entries by scoring technical state with a classifier."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = Path(model_path or config.MODEL_PATH)
        self._model: Any | None = None

    def _load_model(self) -> Any | None:
        if self._model is not None:
            return self._model
        if not self.model_path.exists():
            return None
        self._model = joblib.load(self.model_path)
        return self._model

    def generate_features(self, prices_a: pd.Series, prices_b: pd.Series) -> pd.DataFrame:
        """Build technical features for both assets and their relative volatility."""
        feat = pd.DataFrame(index=prices_a.index)
        feat["rsi_a"] = self._rsi(prices_a)
        feat["rsi_b"] = self._rsi(prices_b)

        macd_a = self._macd(prices_a)
        macd_b = self._macd(prices_b)
        if macd_a is not None and not macd_a.empty:
            feat["macd_a"] = macd_a["MACD_12_26_9"]
            feat["macd_signal_a"] = macd_a["MACDs_12_26_9"]
            feat["macd_hist_a"] = macd_a["MACDh_12_26_9"]
        else:
            feat["macd_a"] = np.nan
            feat["macd_signal_a"] = np.nan
            feat["macd_hist_a"] = np.nan

        if macd_b is not None and not macd_b.empty:
            feat["macd_b"] = macd_b["MACD_12_26_9"]
            feat["macd_signal_b"] = macd_b["MACDs_12_26_9"]
            feat["macd_hist_b"] = macd_b["MACDh_12_26_9"]
        else:
            feat["macd_b"] = np.nan
            feat["macd_signal_b"] = np.nan
            feat["macd_hist_b"] = np.nan

        vol_a = prices_a.pct_change().rolling(20).std()
        vol_b = prices_b.pct_change().rolling(20).std()
        feat["vol_ratio_20"] = vol_a / vol_b.replace(0.0, np.nan)

        return feat.ffill().bfill().fillna(0.0)

    def _rsi(self, prices: pd.Series, length: int = 14) -> pd.Series:
        if ta is not None:
            return ta.rsi(prices, length=length)

        delta = prices.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))

    def _macd(self, prices: pd.Series) -> pd.DataFrame | None:
        if ta is not None:
            return ta.macd(prices)

        ema_fast = prices.ewm(span=12, adjust=False).mean()
        ema_slow = prices.ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        hist_line = macd_line - signal_line
        return pd.DataFrame(
            {
                "MACD_12_26_9": macd_line,
                "MACDs_12_26_9": signal_line,
                "MACDh_12_26_9": hist_line,
            },
            index=prices.index,
        )

    def predict_success_probability(
        self, prices_a: pd.Series, prices_b: pd.Series
    ) -> float | None:
        """Return Class-1 success probability; None when model is unavailable."""
        model = self._load_model()
        if model is None:
            return None

        feature_frame = self.generate_features(prices_a, prices_b)
        latest = feature_frame.tail(1)
        if latest.empty:
            return None

        probabilities = model.predict_proba(latest)
        return float(probabilities[0][1])

    def should_trade(self, prices_a: pd.Series, prices_b: pd.Series) -> bool:
        """Return True when ML confidence exceeds configured threshold."""
        success_probability = self.predict_success_probability(prices_a, prices_b)
        if success_probability is None:
            return True
        return success_probability >= config.ML_PROBABILITY_THRESHOLD

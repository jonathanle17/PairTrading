"""Train the ML gatekeeper classifier for pairs-trade entry filtering."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

import config
from src.data_loader import load_price_data
from src.ml_filter import TradeGatekeeper
from src.stats_tools import (
    calculate_cointegration_p_value,
    calculate_half_life,
    calculate_hedge_ratio,
    calculate_spread,
    calculate_z_score,
)


def _label_signal(
    log_a: pd.Series,
    log_b: pd.Series,
    entry_idx: int,
    trade_half_life: float,
) -> int | None:
    """Label entry by forward outcome: 1 for Z_EXIT, 0 for stop/time-stop."""
    trade_days = 0
    n = len(log_a)
    for j in range(entry_idx + 1, n):
        if j < config.LOOKBACK_WINDOW:
            continue
        window_slice = slice(j - config.LOOKBACK_WINDOW, j + 1)
        win_log_a = log_a.iloc[window_slice]
        win_log_b = log_b.iloc[window_slice]
        beta = calculate_hedge_ratio(win_log_a, win_log_b)
        spread_window = calculate_spread(win_log_a, win_log_b, beta)
        z_score = calculate_z_score(spread_window)

        trade_days += 1
        if abs(z_score) < config.Z_EXIT:
            return 1
        if abs(z_score) > config.Z_STOP_LOSS:
            return 0
        if trade_days > 2 * trade_half_life:
            return 0

    return None


def _build_training_set(price_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Collect ML features and labels from historical statistical entries."""
    gatekeeper = TradeGatekeeper()
    all_features: list[pd.DataFrame] = []
    all_labels: list[int] = []

    for asset_a, asset_b in config.PAIRS:
        prices_a = price_data[asset_a]
        prices_b = price_data[asset_b]
        log_a = prices_a.apply(np.log)
        log_b = prices_b.apply(np.log)

        n = len(price_data)
        for i in range(config.LOOKBACK_WINDOW, n):
            window_slice = slice(i - config.LOOKBACK_WINDOW, i + 1)
            win_log_a = log_a.iloc[window_slice]
            win_log_b = log_b.iloc[window_slice]

            beta = calculate_hedge_ratio(win_log_a, win_log_b)
            spread_window = calculate_spread(win_log_a, win_log_b, beta)
            p_value = calculate_cointegration_p_value(win_log_a, win_log_b)
            half_life = calculate_half_life(spread_window)
            z_score = calculate_z_score(spread_window)
            current_deviation = abs(spread_window.iloc[-1] - spread_window.mean())

            can_enter = (
                p_value < config.P_VALUE_THRESHOLD
                and half_life < config.MAX_HALF_LIFE
                and abs(z_score) > config.Z_ENTRY
                and current_deviation >= config.MIN_SPREAD_PCT
            )
            if not can_enter:
                continue

            ml_window = 30
            start_idx = max(0, i - ml_window + 1)
            price_window_a = prices_a.iloc[start_idx : i + 1]
            price_window_b = prices_b.iloc[start_idx : i + 1]
            feature_row = gatekeeper.generate_features(price_window_a, price_window_b).tail(1)
            if feature_row.empty:
                continue

            label = _label_signal(log_a, log_b, i, max(1.0, half_life))
            if label is None:
                continue

            all_features.append(feature_row)
            all_labels.append(label)

    if not all_features:
        raise ValueError("No training samples generated from historical signals.")

    features = pd.concat(all_features, axis=0).reset_index(drop=True)
    labels = pd.Series(all_labels, dtype=int)
    return features, labels


def main() -> None:
    """Train and persist the trade gatekeeper model."""
    all_pair_tickers = [ticker for pair in config.PAIRS for ticker in pair]
    price_data = load_price_data(
        tickers=all_pair_tickers,
        benchmark=config.BENCHMARK,
        start_date=config.START_DATE,
        end_date=config.END_DATE,
    )

    X, y = _build_training_set(price_data)
    if len(X) < 10:
        raise ValueError(f"Not enough samples to train robustly (got {len(X)}).")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    model_path = Path(config.MODEL_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    print(f"Samples: {len(X)} | Success ratio: {float(y.mean()):.2%}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()

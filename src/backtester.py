"""Backtesting logic for pairs trading and benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

import config
from src.ml_filter import TradeGatekeeper
from src.stats_tools import (
    calculate_cointegration_p_value,
    calculate_half_life,
    calculate_hedge_ratio,
    calculate_spread,
    calculate_z_score,
)


@dataclass
class Trade:
    """Represents a completed trade."""

    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    side: str
    entry_z: float
    exit_z: float
    stop_reason: str | None


def run_pairs_backtest(price_data: pd.DataFrame, tickers: tuple[str, str]) -> dict[str, Any]:
    """Run pairs strategy with cointegration and mean-reversion controls."""
    asset_a, asset_b = tickers
    prices_a = price_data[asset_a]
    prices_b = price_data[asset_b]
    returns_a = prices_a.pct_change().fillna(0.0)
    returns_b = prices_b.pct_change().fillna(0.0)

    rolling_vol_a = returns_a.rolling(config.VOL_WINDOW).std()
    rolling_vol_b = returns_b.rolling(config.VOL_WINDOW).std()

    log_a = prices_a.apply(np.log)
    log_b = prices_b.apply(np.log)

    dates = price_data.index
    n = len(price_data)

    spread_series = pd.Series(index=dates, dtype=float)
    zscore_series = pd.Series(index=dates, dtype=float)
    pvalue_series = pd.Series(index=dates, dtype=float)
    half_life_series = pd.Series(index=dates, dtype=float)
    beta_series = pd.Series(index=dates, dtype=float)
    position_series = pd.Series(0, index=dates, dtype=int)

    equity = pd.Series(np.nan, index=dates, dtype=float)
    equity.iloc[0] = config.INITIAL_CAPITAL

    position = 0
    trade_days = 0
    trade_half_life = float("inf")
    trade_entry_date: pd.Timestamp | None = None
    trade_entry_z = 0.0
    trades: list[Trade] = []

    round_trip_cost_rate = (config.COMMISSION_RATE + config.SLIPPAGE_RATE) * 2.0
    gatekeeper = TradeGatekeeper()

    for i in range(1, n):
        date = dates[i]
        prev_equity = equity.iloc[i - 1]
        if np.isnan(prev_equity):
            prev_equity = config.INITIAL_CAPITAL

        vol_a = rolling_vol_a.iloc[i - 1]
        vol_b = rolling_vol_b.iloc[i - 1]
        if (
            pd.isna(vol_a)
            or pd.isna(vol_b)
            or vol_a <= 0
            or vol_b <= 0
        ):
            weight_a = 0.5
            weight_b = 0.5
        else:
            inv_vol_a = 1.0 / vol_a
            inv_vol_b = 1.0 / vol_b
            weight_a = inv_vol_a / (inv_vol_a + inv_vol_b)
            weight_b = inv_vol_b / (inv_vol_a + inv_vol_b)

        pair_return = position * (
            weight_a * returns_a.iloc[i] - weight_b * returns_b.iloc[i]
        )
        curr_equity = prev_equity * (1.0 + pair_return)

        if i >= config.LOOKBACK_WINDOW:
            window_slice = slice(i - config.LOOKBACK_WINDOW, i + 1)
            win_log_a = log_a.iloc[window_slice]
            win_log_b = log_b.iloc[window_slice]

            beta = calculate_hedge_ratio(win_log_a, win_log_b)
            spread_window = calculate_spread(win_log_a, win_log_b, beta)
            p_value = calculate_cointegration_p_value(win_log_a, win_log_b)
            half_life = calculate_half_life(spread_window)
            z_score = calculate_z_score(spread_window)

            beta_series.iloc[i] = beta
            spread_series.iloc[i] = spread_window.iloc[-1]
            pvalue_series.iloc[i] = p_value
            half_life_series.iloc[i] = half_life
            zscore_series.iloc[i] = z_score

            if position != 0:
                trade_days += 1
                stop_reason: str | None = None
                should_exit = False

                if abs(z_score) > config.Z_STOP_LOSS:
                    stop_reason = "divergence_stop"
                    should_exit = True
                elif trade_days > 2 * trade_half_life:
                    stop_reason = "time_stop"
                    should_exit = True
                elif p_value > 0.10:
                    stop_reason = "statistical_stop"
                    should_exit = True
                elif abs(z_score) < config.Z_EXIT:
                    should_exit = True

                if should_exit:
                    curr_equity -= prev_equity * round_trip_cost_rate
                    trades.append(
                        Trade(
                            entry_date=trade_entry_date if trade_entry_date else date,
                            exit_date=date,
                            side="long_spread" if position == 1 else "short_spread",
                            entry_z=trade_entry_z,
                            exit_z=z_score,
                            stop_reason=stop_reason,
                        )
                    )
                    position = 0
                    trade_days = 0
                    trade_half_life = float("inf")
                    trade_entry_date = None
                    trade_entry_z = 0.0

            if position == 0:
                # 1. Calculate the current deviation from the equilibrium (mean)
                current_deviation = abs(spread_window.iloc[-1] - spread_window.mean())

                # 2. Use the deviation for the filter instead of the raw spread
                can_enter = (
                    p_value < config.P_VALUE_THRESHOLD
                    and half_life < config.MAX_HALF_LIFE
                    and abs(z_score) > config.Z_ENTRY
                    and current_deviation >= config.MIN_SPREAD_PCT
                )
                ml_confirmed = True
                ml_probability: float | None = None
                ml_failure_probability: float | None = None
                if can_enter and config.USE_ML_FILTER:
                    # Align with training context: compute indicators over all history up to i.
                    price_history_a = prices_a.iloc[: i + 1]
                    price_history_b = prices_b.iloc[: i + 1]
                    ml_confirmed, ml_probability, ml_failure_probability = gatekeeper.should_trade(
                        price_history_a, price_history_b
                    )
                    status = "CONFIRMED" if ml_confirmed else "REJECTED"
                    if ml_probability is None or ml_failure_probability is None:
                        print(f"[{date}] ML Probabilities - Success: N/A, Failure: N/A | Result: {status}")
                    else:
                        print(
                            f"[{date}] ML Probabilities - Success: {ml_probability:.2%}, Failure: {ml_failure_probability:.2%} | Result: {status}"
                        )

                if can_enter and ml_confirmed:
                    curr_equity -= prev_equity * round_trip_cost_rate
                    position = -1 if z_score > 0 else 1
                    trade_days = 0
                    trade_half_life = max(1.0, half_life)
                    trade_entry_date = date
                    trade_entry_z = z_score

        position_series.iloc[i] = position
        equity.iloc[i] = max(curr_equity, 0.0)

    returns = equity.pct_change().fillna(0.0)
    cumulative_returns = equity / config.INITIAL_CAPITAL

    return {
        "equity": equity,
        "returns": returns,
        "cumulative_returns": cumulative_returns,
        "spread": spread_series,
        "zscore": zscore_series,
        "pvalue": pvalue_series,
        "half_life": half_life_series,
        "beta": beta_series,
        "position": position_series,
        "trades": trades,
    }


def run_benchmark_backtest(price_data: pd.DataFrame) -> dict[str, pd.Series]:
    """Run buy-and-hold benchmark simulation."""
    bench_prices = price_data[config.BENCHMARK]
    bench_returns = bench_prices.pct_change().fillna(0.0)
    equity = config.INITIAL_CAPITAL * (1.0 + bench_returns).cumprod()
    cumulative_returns = equity / config.INITIAL_CAPITAL
    return {
        "equity": equity,
        "returns": bench_returns,
        "cumulative_returns": cumulative_returns,
    }

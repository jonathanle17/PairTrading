"""Backtesting logic for pairs trading and benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

import config
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


def run_pairs_backtest(price_data: pd.DataFrame) -> dict[str, Any]:
    """Run pairs strategy with cointegration and mean-reversion controls."""
    asset_a, asset_b = config.TICKERS
    prices_a = price_data[asset_a]
    prices_b = price_data[asset_b]
    returns_a = prices_a.pct_change().fillna(0.0)
    returns_b = prices_b.pct_change().fillna(0.0)

    log_a = np.log(prices_a)
    log_b = np.log(prices_b)

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

    for i in range(1, n):
        date = dates[i]
        prev_equity = equity.iloc[i - 1]
        if np.isnan(prev_equity):
            prev_equity = config.INITIAL_CAPITAL

        pair_return = position * 0.5 * (returns_a.iloc[i] - returns_b.iloc[i])
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
                can_enter = (
                    p_value < config.P_VALUE_THRESHOLD
                    and half_life < config.MAX_HALF_LIFE
                    and abs(z_score) > config.Z_ENTRY
                )
                if can_enter:
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

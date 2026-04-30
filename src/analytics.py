"""Performance analytics and visualization."""

from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def calculate_sharpe_ratio(returns: pd.Series) -> float:
    """Calculate annualized Sharpe ratio."""
    excess_daily = returns - (config.RISK_FREE_RATE_ANNUAL / config.TRADING_DAYS_PER_YEAR)
    vol = excess_daily.std(ddof=0)
    if vol == 0 or np.isnan(vol):
        return 0.0
    return float(math.sqrt(config.TRADING_DAYS_PER_YEAR) * excess_daily.mean() / vol)


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum drawdown from an equity curve."""
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1.0
    return float(drawdown.min())


def build_summary_row(name: str, cumulative_returns: pd.Series, returns: pd.Series) -> dict[str, float | str]:
    """Create summary metrics for a strategy."""
    total_return = float(cumulative_returns.iloc[-1] - 1.0)
    sharpe = calculate_sharpe_ratio(returns)
    max_dd = calculate_max_drawdown(cumulative_returns)
    return {
        "Strategy": name,
        "Total Return": total_return,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
    }


def print_summary_table(strategy_summary: dict[str, Any], benchmark_summary: dict[str, Any]) -> None:
    """Print key performance metrics for strategy and benchmark."""
    rows = [strategy_summary, benchmark_summary]
    frame = pd.DataFrame(rows)

    display_frame = frame.copy()
    display_frame["Total Return"] = display_frame["Total Return"].map(lambda x: f"{x:.2%}")
    display_frame["Sharpe Ratio"] = display_frame["Sharpe Ratio"].map(lambda x: f"{x:.2f}")
    display_frame["Max Drawdown"] = display_frame["Max Drawdown"].map(lambda x: f"{x:.2%}")

    print("\n=== Performance Summary ===")
    print(display_frame.to_string(index=False))


def plot_strategy_dashboard(
    price_data: pd.DataFrame,
    spread: pd.Series,
    zscore: pd.Series,
    tickers: tuple[str, str],
) -> None:
    """Plot normalized prices, z-score signals, and spread."""
    asset_a, asset_b = tickers
    norm_prices = price_data[[asset_a, asset_b]] / price_data[[asset_a, asset_b]].iloc[0]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(f"Pairs Strategy Dashboard: {asset_a}/{asset_b}", fontsize=14)

    axes[0].plot(norm_prices.index, norm_prices[asset_a], label=asset_a)
    axes[0].plot(norm_prices.index, norm_prices[asset_b], label=asset_b)
    axes[0].set_ylabel("Normalized Price")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)

    axes[1].plot(zscore.index, zscore, color="purple", label="Z-Score")
    axes[1].axhline(config.Z_ENTRY, color="red", linestyle="--", label="Entry")
    axes[1].axhline(-config.Z_ENTRY, color="red", linestyle="--")
    axes[1].axhline(config.Z_EXIT, color="green", linestyle=":", label="Exit")
    axes[1].axhline(-config.Z_EXIT, color="green", linestyle=":")
    axes[1].axhline(config.Z_STOP_LOSS, color="black", linestyle="-.", label="Stop")
    axes[1].axhline(-config.Z_STOP_LOSS, color="black", linestyle="-.")
    axes[1].set_ylabel("Z-Score")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.3)

    axes[2].plot(spread.index, spread, color="steelblue", label="Spread")
    axes[2].set_ylabel("Spread")
    axes[2].set_xlabel("Date")
    axes[2].legend(loc="upper left")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_comparison_dashboard(
    strategy_cumulative: pd.Series,
    benchmark_cumulative: pd.Series,
    strategy_label: str = "Pairs Strategy",
) -> None:
    """Plot cumulative returns for strategy versus benchmark."""
    plt.figure(figsize=(14, 5))
    plt.plot(strategy_cumulative.index, strategy_cumulative, label=strategy_label, linewidth=2)
    plt.plot(benchmark_cumulative.index, benchmark_cumulative, label=f"Buy & Hold {config.BENCHMARK}", linewidth=2)
    plt.title("Cumulative Return Comparison")
    plt.ylabel("Growth of $1")
    plt.xlabel("Date")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

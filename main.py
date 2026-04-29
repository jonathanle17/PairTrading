"""Entry point for the pairs trading backtesting engine."""

from __future__ import annotations

from src.analytics import (
    build_summary_row,
    plot_comparison_dashboard,
    plot_strategy_dashboard,
    print_summary_table,
)
from src.backtester import run_benchmark_backtest, run_pairs_backtest
from src.data_loader import load_price_data

import config


def main() -> None:
    """Run backtest, analytics, and visual outputs."""
    price_data = load_price_data(
        tickers=config.TICKERS,
        benchmark=config.BENCHMARK,
        start_date=config.START_DATE,
        end_date=config.END_DATE,
    )

    strategy_results = run_pairs_backtest(price_data)
    benchmark_results = run_benchmark_backtest(price_data)

    strategy_summary = build_summary_row(
        name="Pairs Strategy",
        cumulative_returns=strategy_results["cumulative_returns"],
        returns=strategy_results["returns"],
    )
    benchmark_summary = build_summary_row(
        name=f"Buy & Hold {config.BENCHMARK}",
        cumulative_returns=benchmark_results["cumulative_returns"],
        returns=benchmark_results["returns"],
    )

    print_summary_table(strategy_summary, benchmark_summary)
    plot_strategy_dashboard(
        price_data=price_data,
        spread=strategy_results["spread"],
        zscore=strategy_results["zscore"],
    )
    plot_comparison_dashboard(
        strategy_cumulative=strategy_results["cumulative_returns"],
        benchmark_cumulative=benchmark_results["cumulative_returns"],
    )


if __name__ == "__main__":
    main()

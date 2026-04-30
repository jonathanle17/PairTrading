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
import pandas as pd


def main() -> None:
    """Run backtests for configured pairs and show analytics."""
    all_pair_tickers = [ticker for pair in config.PAIRS for ticker in pair]
    price_data = load_price_data(
        tickers=all_pair_tickers,
        benchmark=config.BENCHMARK,
        start_date=config.START_DATE,
        end_date=config.END_DATE,
    )

    benchmark_results = run_benchmark_backtest(price_data)
    benchmark_summary = build_summary_row(
        name=f"Buy & Hold {config.BENCHMARK}",
        cumulative_returns=benchmark_results["cumulative_returns"],
        returns=benchmark_results["returns"],
    )

    pair_rows: list[dict[str, float | str]] = []
    pair_runs: list[tuple[tuple[str, str], dict[str, pd.Series]]] = []
    for pair in config.PAIRS:
        strategy_results = run_pairs_backtest(price_data, pair)
        pair_name = f"Pairs {pair[0]}/{pair[1]}"
        strategy_summary = build_summary_row(
            name=pair_name,
            cumulative_returns=strategy_results["cumulative_returns"],
            returns=strategy_results["returns"],
        )
        pair_rows.append(strategy_summary)
        pair_runs.append((pair, strategy_results))

    summary_frame = pd.DataFrame(pair_rows).sort_values("Total Return", ascending=False)
    display_frame = summary_frame.copy()
    display_frame["Total Return"] = display_frame["Total Return"].map(lambda x: f"{x:.2%}")
    display_frame["Sharpe Ratio"] = display_frame["Sharpe Ratio"].map(lambda x: f"{x:.2f}")
    display_frame["Max Drawdown"] = display_frame["Max Drawdown"].map(lambda x: f"{x:.2%}")
    print("\n=== Pairs Performance (Ranked) ===")
    print(display_frame.to_string(index=False))
    print_summary_table(summary_frame.iloc[0].to_dict(), benchmark_summary)

    best_pair_name = str(summary_frame.iloc[0]["Strategy"]).replace("Pairs ", "")
    best_pair = tuple(best_pair_name.split("/"))
    best_pair_results = next(result for pair, result in pair_runs if pair == best_pair)
    plot_strategy_dashboard(
        price_data=price_data,
        spread=best_pair_results["spread"],
        zscore=best_pair_results["zscore"],
        tickers=best_pair,
    )
    plot_comparison_dashboard(
        strategy_cumulative=best_pair_results["cumulative_returns"],
        benchmark_cumulative=benchmark_results["cumulative_returns"],
        strategy_label=f"Pairs {best_pair[0]}/{best_pair[1]}",
    )


if __name__ == "__main__":
    main()

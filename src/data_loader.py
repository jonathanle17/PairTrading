"""Data loading utilities for market data."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def load_price_data(
    tickers: list[str],
    benchmark: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch adjusted close prices for strategy assets and benchmark.

    Args:
        tickers: Pair assets used in the strategy.
        benchmark: Benchmark ticker for buy-and-hold comparison.
        start_date: Inclusive start date in ISO format.
        end_date: Inclusive end date in ISO format.

    Returns:
        DataFrame indexed by date with one column per ticker.

    Raises:
        ValueError: If no data is fetched or required tickers are missing.
    """

    all_tickers = list(dict.fromkeys([*tickers, benchmark]))
    raw_data = yf.download(
        tickers=all_tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    if raw_data.empty:
        raise ValueError("No data returned from yfinance.")

    if isinstance(raw_data.columns, pd.MultiIndex):
        price_data = raw_data["Close"].copy()
    else:
        # Single ticker fallback.
        price_data = raw_data.to_frame(name=all_tickers[0])

    price_data = price_data.dropna(how="all")
    missing = [symbol for symbol in all_tickers if symbol not in price_data.columns]
    if missing:
        raise ValueError(f"Missing tickers in downloaded data: {missing}")

    return price_data[all_tickers].dropna()

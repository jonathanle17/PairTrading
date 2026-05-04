"""Global configuration for the pairs trading backtest."""

from __future__ import annotations

from datetime import date, timedelta


PAIRS: list[tuple[str, str]] = [
    ("MSFT", "AAPL"),
    ("KO", "PEP"),
    ("NVDA", "AMD"),
]
BENCHMARK: str = "QQQ"

# Five-year window ending in April 2026.
END_DATE: str = "2026-04-30"
START_DATE: str = (date(2026, 4, 30) - timedelta(days=365 * 5)).isoformat()

INITIAL_CAPITAL: float = 100_000.0

LOOKBACK_WINDOW: int = 90
P_VALUE_THRESHOLD: float = 0.05
Z_ENTRY: float = 2
MIN_SPREAD_PCT: float = 0.005
Z_EXIT: float = 0.5
Z_STOP_LOSS: float = 4.0
MAX_HALF_LIFE: int = 30

COMMISSION_RATE: float = 0.001
SLIPPAGE_RATE: float = 0.0005

RISK_FREE_RATE_ANNUAL: float = 0.02
TRADING_DAYS_PER_YEAR: int = 252

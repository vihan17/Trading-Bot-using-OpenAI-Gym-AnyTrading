"""Data loading and optional Binance fetching.

Supports three sources:

* Local pickle (``.pickle``) — the original repo's ``measurement.pickle``
* Local CSV (``.csv``)
* Fresh Binance klines via public API (``fetch_binance_data``)
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a DataFrame from a ``.pickle`` or ``.csv`` file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix.lower() in {".pkl", ".pickle"}:
        df = pd.read_pickle(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported data format '{path.suffix}'. Use .pickle or .csv."
        )

    if "Close" not in df.columns:
        raise ValueError("Dataframe must contain a 'Close' price column.")
    return df.reset_index(drop=True)


def fetch_binance_data(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    days: int = 100,
    output: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch klines from Binance's public API and return as a DataFrame.

    Note: this returns raw OHLCV only. It does **not** reproduce the
    technical-indicator columns present in ``measurement.pickle``.
    """
    import requests

    limit = min(days * 24, 1000)  # Binance caps at 1000 candles per call.
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    rows = resp.json()
    df = pd.DataFrame(rows, columns=[
        "open_time", "Open", "High", "Low", "Close", "Volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
    ])
    df = df[["open_time", "Open", "High", "Low", "Close", "Volume"]].copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)

    if output is not None:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(out)
    return df


def train_test_split(
    df: pd.DataFrame, train_fraction: float = 0.7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data chronologically into train/test frames."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1).")
    split = int(len(df) * train_fraction)
    return df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)
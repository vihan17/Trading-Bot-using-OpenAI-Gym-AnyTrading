"""Fetch fresh Binance klines into data/raw.

Usage::

    python scripts/fetch_data.py --symbol BTCUSDT --interval 1h --days 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data import fetch_binance_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Binance klines.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--days", type=int, default=100)
    parser.add_argument("--output", default="data/raw/binance_klines.pickle")
    args = parser.parse_args()

    out = Path(args.output)
    df = fetch_binance_data(args.symbol, args.interval, args.days, out)
    print(f"[fetch] {len(df)} rows saved to {out}")


if __name__ == "__main__":
    main()
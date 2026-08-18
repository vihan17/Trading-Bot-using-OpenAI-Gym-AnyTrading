"""Portfolio / backtest metrics used during evaluation."""

from __future__ import annotations

import numpy as np


def total_return(values: list[float]) -> float:
    """Total percentage return of a portfolio value series."""
    if not values or values[0] == 0:
        return 0.0
    return values[-1] / values[0] - 1.0


def max_drawdown(values: list[float]) -> float:
    """Maximum drawdown (positive fraction lost from a peak)."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(values)
    dd = (running_max - values) / np.maximum(running_max, 1e-12)
    return float(np.max(dd)) if dd.size else 0.0


def sharpe_ratio(
    values: list[float], periods_per_year: int = 8760
) -> float:
    """Annualized Sharpe ratio of a portfolio value series.

    ``periods_per_year`` defaults to 8760 (hourly candles: 24*365).
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return 0.0
    returns = np.diff(values) / np.maximum(values[:-1], 1e-12)
    if returns.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / returns.std())


def win_rate(values: list[float]) -> float:
    """Fraction of steps where portfolio value increased."""
    if len(values) < 2:
        return 0.0
    diffs = np.diff(values)
    return float(np.mean(diffs > 0))


def summary(values: list[float], periods_per_year: int = 8760) -> dict:
    """Compute a compact metrics dict for a portfolio value series."""
    return {
        "total_return": total_return(values),
        "max_drawdown": max_drawdown(values),
        "sharpe": sharpe_ratio(values, periods_per_year),
        "win_rate": win_rate(values),
        "final_value": float(values[-1]) if values else 0.0,
    }
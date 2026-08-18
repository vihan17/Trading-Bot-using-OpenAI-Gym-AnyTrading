"""Unit tests for the TradingEnv.

Run with::

    python -m pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.trading_env import TradingEnv


def make_df(n: int = 200) -> pd.DataFrame:
    """Synthetic OHLCV + feature dataframe."""
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": rng.uniform(100, 500, n),
            "rsi": rng.uniform(0, 100, n),
        }
    )
    return df


@pytest.fixture
def df() -> pd.DataFrame:
    return make_df()


@pytest.fixture
def env(df: pd.DataFrame) -> TradingEnv:
    return TradingEnv(df, window_size=5, max_steps=30)


def test_observation_shape(env: TradingEnv) -> None:
    obs, info = env.reset()
    assert obs.shape == (env.window_size, env.n_features)
    assert env.observation_space.contains(obs)


def test_observation_is_scaled_to_unit_box(env: TradingEnv) -> None:
    obs, _ = env.reset()
    assert obs.min() >= 0.0 and obs.max() <= 1.0


def test_step_returns_gymnasium_tuple(env: TradingEnv) -> None:
    obs, _ = env.reset()
    out = env.step(0)
    assert len(out) == 5  # obs, reward, terminated, truncated, info


def test_buy_spends_cash_and_increases_shares(env: TradingEnv) -> None:
    env.reset()
    before_balance = env.balance
    before_shares = env.shares
    env.step(0)
    assert env.shares > before_shares
    assert env.balance < before_balance


def test_sell_decreases_shares_and_increases_cash(env: TradingEnv) -> None:
    env.reset()
    env.step(0)  # buy first
    after_buy_shares = env.shares
    after_buy_balance = env.balance
    env.step(1)  # sell
    assert env.shares < after_buy_shares
    assert env.balance > after_buy_balance


def test_balance_never_goes_negative(env: TradingEnv) -> None:
    env.reset()
    for _ in range(200):
        obs, _, terminated, truncated, info = env.step(0)
        assert env.balance >= -1e-9
        if terminated or truncated:
            env.reset()


def test_sell_with_no_shares_is_noop(env: TradingEnv) -> None:
    env.reset()
    env.balance = env.initial_value
    env.shares = 0.0
    before_balance = env.balance
    env.step(1)
    assert env.balance == pytest.approx(before_balance)


def test_hold_action_only_when_enabled() -> None:
    env_no_hold = TradingEnv(make_df(), hold=False)
    assert env_no_hold.action_space.n == 2
    env_hold = TradingEnv(make_df(), hold=True)
    assert env_hold.action_space.n == 3


def test_truncation_respects_max_steps(env: TradingEnv) -> None:
    env.reset()
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(0)
        steps += 1
        assert steps <= env.max_steps + 1
    assert truncated
    assert steps == env.max_steps


def test_termination_at_data_end() -> None:
    env = TradingEnv(make_df(n=60), window_size=5, max_steps=1000)
    env.reset(options={"start_index": 50})
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(0)
    assert terminated


def test_reset_with_random_split_bounds(env: TradingEnv) -> None:
    env.reset()
    assert env.balance >= 0
    assert env.shares >= 0
    total = env.balance + env.shares * env.prices[env._current_step]
    assert total == pytest.approx(env.initial_value, abs=1e-6)


def test_invalid_action_raises(env: TradingEnv) -> None:
    env.reset()
    with pytest.raises(ValueError):
        env.step(99)


def test_out_of_range_start_index_raises(env: TradingEnv) -> None:
    with pytest.raises(ValueError):
        env.reset(options={"start_index": 10_000})


def test_short_data_raises() -> None:
    with pytest.raises(ValueError):
        TradingEnv(make_df(n=5), window_size=10)


def test_custom_reward_function() -> None:
    env = TradingEnv(make_df(), reward_function=lambda action: 42.0)
    env.reset()
    _, reward, _, _, _ = env.step(0)
    assert reward == pytest.approx(42.0)


def test_reward_is_finite(env: TradingEnv) -> None:
    env.reset()
    for _ in range(10):
        _, reward, _, _, _ = env.step(0)
        assert np.isfinite(reward)


def test_deterministic_reset_with_seed() -> None:
    df = make_df()
    e1 = TradingEnv(df, window_size=5)
    e2 = TradingEnv(df, window_size=5)
    o1, _ = e1.reset(seed=7)
    o2, _ = e2.reset(seed=7)
    assert np.array_equal(o1, o2)
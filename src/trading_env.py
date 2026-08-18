"""Crypto trading environment for reinforcement learning.

A Gymnasium environment that models trading a single asset (e.g. BTC)
with realistic market frictions: trade fees, slippage, and fractional
order sizing. Designed to be used with Stable-Baselines3.

The agent observes a window of scaled technical-indicator features and
picks an action every step:

* ``0`` — BUY  : spend ``order_fraction`` of available cash on the asset
* ``1`` — SELL : sell ``order_fraction`` of held shares
* ``2`` — HOLD : do nothing (only when ``hold=True``)

Reward is the relative change in total portfolio value per step, minus a
diversity penalty that discourages the agent from collapsing onto a
single action.

This is a clean-room rewrite of ``tradinggym.py`` from the original
repo. The bugs fixed vs. the original:

* ``done`` compared ``current_step`` (an absolute index) to ``max_steps``
  (a relative count) — termination is now a proper step counter plus a
  data-end guard.
* ``step`` did not increment the step counter, so ``max_steps`` was never
  respected.
* ``_get_observation`` sliced a ``2 * window`` chunk then re-sliced the
  last ``window`` rows — now it uses exactly one window.
* ``_get_reward`` relied on a ``time_shift`` hack with off-by-one price
  indexing — reward is now computed from the same prices used by ``step``.
* ``reset`` accepted an inconsistent ``start_index`` range check.
* Observation bounds were ``[0, inf)`` even though features are min-max
  scaled — now ``[0, 1]``.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

Action = int
Observation = np.ndarray


class TradingEnv(gym.Env):
    """Single-asset trading environment with transaction frictions."""

    metadata = {"render_modes": ["human", "none"]}

    def __init__(
        self,
        df: pd.DataFrame,
        price_column: str = "Close",
        window_size: int = 5,
        initial_value: float = 10_000.0,
        order_fraction: float = 0.2,
        trade_fee: float = 0.0045,
        slippage: float = 0.005,
        hold: bool = False,
        diversity_penalty: float = 0.001,
        random_split: bool = True,
        max_steps: Optional[int] = None,
        reward_function: Optional[Callable[[Action], float]] = None,
        render_mode: str = "none",
    ) -> None:
        """Initialize the environment.

        Args:
            df: OHLCV + feature dataframe. Must contain ``price_column``.
            price_column: Column used for pricing/valuation.
            window_size: Number of past rows in each observation.
            initial_value: Starting portfolio value in cash.
            order_fraction: Fraction of available cash/shares per order.
            trade_fee: Proportional fee paid on every fill.
            slippage: Max random slippage applied to fills.
            hold: Add a HOLD action (action space becomes Discrete(3)).
            diversity_penalty: Penalty applied when the agent's recent
                actions have low entropy (action collapse).
            random_split: Start each episode with a random cash/shares
                split instead of all-cash.
            max_steps: Max steps per episode; ``None`` runs until data ends.
            reward_function: Optional custom reward taking the action and
                returning a float.
            render_mode: ``"none"`` or ``"human"``.
        """
        if df is None or len(df) <= 2 * window_size:
            raise ValueError(
                "Observation data must be longer than 2 * window_size."
            )
        if not 0 < order_fraction <= 1:
            raise ValueError("order_fraction must be in (0, 1].")
        if trade_fee < 0 or slippage < 0:
            raise ValueError("trade_fee and slippage must be non-negative.")

        self.df = df.reset_index(drop=True)
        self.price_column = price_column
        self.window_size = window_size
        self.initial_value = float(initial_value)
        self.order_fraction = float(order_fraction)
        self.trade_fee = float(trade_fee)
        self.slippage = float(slippage)
        self.hold = hold
        self.diversity_penalty = float(diversity_penalty)
        self.random_split = random_split
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.prices = self.df[self.price_column].to_numpy(dtype=np.float64)
        self.feature_columns = [
            c for c in self.df.columns if c != self.price_column
        ]
        self.n_features = len(self.feature_columns)

        # Action space: BUY=0, SELL=1, (HOLD=2)
        self.action_space = spaces.Discrete(3 if hold else 2)

        # Observation: a (window_size, n_features) block of scaled features.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.window_size, self.n_features),
            dtype=np.float64,
        )

        # Episode state
        self.balance = self.initial_value
        self.shares = 0.0
        self._current_step = 0
        self._steps_taken = 0
        self._action_history: list[Action] = []
        self._max_step_index = len(self.df) - 1

        self._custom_reward = reward_function

        # Matplotlib figure for "human" rendering.
        self._fig = None
        self._ax_price = None
        self._ax_value = None
        self._ax_reward = None
        self._render_data: list[dict] = []

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[Observation, dict]:
        """Reset the episode and return the initial observation."""
        super().reset(seed=seed)
        rng = self.np_random

        if options and options.get("start_index") is not None:
            start = int(options["start_index"])
            low, high = self._valid_start_bounds()
            if not (low <= start <= high):
                raise ValueError(
                    f"start_index must be in [{low}, {high}], got {start}."
                )
            self._current_step = start
        else:
            low, high = self._valid_start_bounds()
            self._current_step = int(rng.integers(low, high + 1))

        if self.random_split:
            split = float(rng.random())
            price = self.prices[self._current_step]
            self.balance = 0.0
            self.shares = (self.initial_value * split) / price
            self.balance = self.initial_value - self.shares * price
        else:
            self.balance = self.initial_value
            self.shares = 0.0

        self._steps_taken = 0
        self._action_history = []
        self._render_data = []

        return self._get_observation(), {"step": self._current_step}

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]:
        """Apply an action, advance one step, return the transition."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}.")

        self._action_history.append(int(action))
        if len(self._action_history) > self.window_size:
            self._action_history.pop(0)

        price = self.prices[self._current_step]
        fees = self.trade_fee + float(
            self.np_random.uniform(-self.slippage, self.slippage)
        )

        if action == 0:  # BUY
            if self.balance > 0:
                spend = self.order_fraction * self.balance
                price_with_fee = price * (1.0 + fees)
                shares_bought = spend / price_with_fee
                self.shares += shares_bought
                self.balance -= spend
        elif action == 1:  # SELL
            if self.shares > 0:
                sell_shares = self.order_fraction * self.shares
                proceeds = sell_shares * price * (1.0 - fees)
                self.shares -= sell_shares
                self.balance += proceeds
        # action == 2: HOLD, no-op.

        prev_value = self._portfolio_value(price)
        self._current_step += 1
        self._steps_taken += 1

        # Compute reward on the *new* price.
        if self._current_step <= self._max_step_index:
            new_price = self.prices[self._current_step]
        else:
            new_price = price  # terminal step: no forward price available.
        new_value = self._portfolio_value(new_price)

        if self._custom_reward is not None:
            reward = float(self._custom_reward(action))
        else:
            reward = self._default_reward(action, prev_value, new_value)

        terminated = self._current_step >= self._max_step_index
        truncated = (
            self.max_steps is not None and self._steps_taken >= self.max_steps
        )

        info = {
            "step": self._current_step,
            "balance": self.balance,
            "shares": self.shares,
            "portfolio_value": new_value,
            "price": new_price,
            "action": int(action),
            "reward_raw": reward,
        }

        if self.render_mode == "human":
            self._render_data.append(info)

        return self._get_observation(), reward, terminated, truncated, info

    def render(self) -> None:
        """Render portfolio value and rewards vs. price over the episode."""
        if not self._render_data:
            return
        import matplotlib.pyplot as plt

        prices = [d["price"] for d in self._render_data]
        values = [d["portfolio_value"] for d in self._render_data]
        rewards = [d["reward_raw"] for d in self._render_data]
        steps = list(range(len(prices)))

        if self._fig is None:
            self._fig, (self._ax_price, self._ax_value, self._ax_reward) = (
                plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            )
        self._ax_price.clear()
        self._ax_value.clear()
        self._ax_reward.clear()

        self._ax_price.plot(steps, prices, label="Price")
        self._ax_price.set_ylabel("Price")
        self._ax_price.grid(alpha=0.4)
        self._ax_price.legend()

        self._ax_value.plot(steps, values, color="#15ab5b")
        self._ax_value.set_ylabel("Portfolio value")
        self._ax_value.grid(alpha=0.4)

        self._ax_reward.plot(steps, rewards, color="#e89f0c")
        self._ax_reward.set_ylabel("Reward")
        self._ax_reward.grid(alpha=0.4)

        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

    def close(self) -> None:
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _valid_start_bounds(self) -> tuple[int, int]:
        """Return (low, high) valid episode start indices.

        The episode must leave at least one step of data after the start
        so it can terminate naturally. If ``max_steps`` is set and the
        data is long enough to support that many steps, the start is also
        capped so truncation is reachable. When ``max_steps`` exceeds the
        remaining data, termination dominates and the cap is skipped.
        """
        low = self.window_size
        high = self._max_step_index - 1
        if self.max_steps is not None:
            candidate = self._max_step_index - self.max_steps
            if candidate >= low:
                high = min(high, candidate)
        return low, max(low, high)

    def _get_observation(self) -> Observation:
        """Return the min-max scaled feature window at the current step."""
        start = self._current_step - self.window_size
        end = self._current_step
        window = self.df[self.feature_columns].iloc[start:end].to_numpy(
            dtype=np.float64
        )
        if window.shape[0] < self.window_size:
            # Warm-up padding when starting near the data boundary.
            pad = np.zeros((self.window_size - window.shape[0], self.n_features))
            window = np.vstack([pad, window])
        mins = window.min(axis=0)
        maxs = window.max(axis=0)
        denom = maxs - mins
        denom[denom == 0] = 1.0  # avoid divide-by-zero for constant features
        return ((window - mins) / denom).astype(np.float64)

    def _default_reward(
        self, action: Action, prev_value: float, new_value: float
    ) -> float:
        """Return portfolio-relative reward minus diversity penalty."""
        if prev_value > 0:
            reward = (new_value - prev_value) / prev_value
        else:
            reward = 0.0

        if self.diversity_penalty > 0 and len(self._action_history) >= 2:
            history = self._action_history[-self.window_size :]
            counts = np.bincount(history, minlength=int(self.action_space.n))
            probs = counts / len(history)
            probs = probs[probs > 0]
            entropy = -float(np.sum(probs * np.log(probs)))
            max_entropy = np.log(int(self.action_space.n))
            reward -= self.diversity_penalty * (1.0 - entropy / max_entropy)

        return float(reward)

    def _portfolio_value(self, price: float) -> float:
        """Total portfolio value = cash + shares valued at ``price``."""
        return self.balance + self.shares * price
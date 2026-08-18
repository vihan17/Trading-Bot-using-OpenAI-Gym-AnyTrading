# RL Trading Bot (Crypto)

A reinforcement-learning agent that trades a single cryptocurrency using
technical-indicator features, built with a custom **Gymnasium** environment and
**Stable-Baselines3**. Transaction frictions (fees, slippage, fractional
orders) make the environment realistic, and an entropy-based diversity penalty
stops the agent from collapsing onto one action.

This repo is a clean rebuild of the original `Trading-Bot-using-OpenAI-Gym-AnyTrading`
project: the environment is preserved (with bugs fixed), and the scattered
notebooks + 11 MB zip are replaced by a structured, testable package.

---

## Quick start

```bash
# 1. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Train (smoke test — use configs/default.yaml for a real run)
python -m src.train --config configs/default.yaml --timesteps 100000

# 3. Evaluate on held-out data
python -m src.evaluate --config configs/default.yaml --model models/ppo_100000

# 4. Run the unit tests
python -m pytest tests/ -v
```

Outputs:
- trained models → `models/`
- evaluation metrics → `results/eval_report.json`
- equity curve → `results/equity_curve.png`

---

## Project structure

```
.
├── configs/
│   └── default.yaml        # data, env, and training/eval configuration
├── data/
│   └── raw/
│       └── measurement.pickle   # 144k hourly BTCUSDT rows + 10 indicators
├── scripts/
│   └── fetch_data.py       # fetch fresh Binance klines
├── src/
│   ├── trading_env.py      # Gymnasium TradingEnv (the core)
│   ├── data.py             # data loading + train/test split
│   ├── train.py            # SB3 training CLI
│   ├── evaluate.py         # held-out evaluation CLI
│   └── utils.py            # portfolio metrics (return, Sharpe, max DD)
├── tests/
│   └── test_trading_env.py # 17 unit tests for the environment
├── models/                 # saved agents (gitignored)
└── results/                # eval reports + figures (gitignored)
```

---

## The environment

The agent observes a **window of the last `window_size` rows** of
technical-indicator features (14 features, min-max scaled to `[0, 1]`), and
each step picks one of:

| Action | Meaning |
|--------|---------|
| `0` — BUY  | Spend `order_fraction` of available cash on the asset |
| `1` — SELL | Sell `order_fraction` of held shares |
| `2` — HOLD | Do nothing (only when `hold: true`) |

Frictions modeled per trade: a proportional `trade_fee`, random `slippage`
uniform on `[-slippage, +slippage]`, and fractional order sizing via
`order_fraction`.

**Reward** = relative change in total portfolio value per step, minus a
*diversity penalty* that scales with `1 - entropy(action_history)`. This
discourages degenerate policies that repeatedly pick one action.

**Episode flow** — `reset()` optionally starts each episode with a *random
cash/shares split* (curriculum-style diversity), then steps until either the
data ends (`terminated`) or `max_steps` is reached (`truncated`).

### Bugs fixed vs. the original `tradinggym.py`

- Termination compared an absolute index to `max_steps` (a count) — now a
  proper step counter plus a data-end guard.
- `max_steps` was never respected because the counter never incremented.
- Observation slicing grabbed `2*window` rows then re-sliced — now exactly one
  window, with warm-up padding at the data boundary.
- Reward used an off-by-one `time_shift` hack — now computed from the same
  prices used by `step`.
- Observation space bounds were `[0, inf)` for min-max-scaled features — now
  `[0, 1]`.
- Migration from unmaintained `gym` to `gymnasium` (5-tuple step API).

---

## Data

`data/raw/measurement.pickle` contains **144,634 hourly BTCUSDT candles**
(starting Jan 2022) with OHLCV plus 10 technical indicators: AROON, Stochastic
%D, Keltner Channels (KCU/KCL), Force Index, RSI diff, Stochastic RSI, TRIX
histogram, and Awesome Oscillator.

Fetch fresh data from Binance:

```bash
python scripts/fetch_data.py --symbol BTCUSDT --interval 1h --days 100
```

---

## Results & honest expectations

RL trading is hard, and this project says so plainly. On the held-out test
window (a strong bull run), a **buy-and-hold baseline returned +59.8%** with an
annualized Sharpe of ~0.75. An undertrained 2,000-step PPO agent lost money —
RL will not beat buy-and-hold in a trending market without careful tuning.

Treat this as a **research scaffold**, not a trading system:

- Use proper walk-forward splits and many seeds before drawing conclusions.
- The `diversity_penalty`, reward shaping, and `order_fraction` are the levers
  most likely to move results.
- Nothing here is financial advice.

---

## License

MIT — see `LICENSE`.
"""Train an RL trading agent on the crypto environment.

Usage::

    python -m src.train --config configs/default.yaml
    python -m src.train --config configs/default.yaml --timesteps 50000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.data import load_dataframe, train_test_split
from src.trading_env import TradingEnv

ALGORITHMS = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN,
}


def make_env(df, env_cfg: dict, price_column: str):
    """Create a single TradingEnv from a config dict."""
    return TradingEnv(
        df=df,
        price_column=price_column,
        window_size=env_cfg["window_size"],
        initial_value=env_cfg["initial_value"],
        order_fraction=env_cfg["order_fraction"],
        trade_fee=env_cfg["trade_fee"],
        slippage=env_cfg["slippage"],
        hold=env_cfg["hold"],
        diversity_penalty=env_cfg["diversity_penalty"],
        random_split=env_cfg["random_split"],
        max_steps=env_cfg["max_steps"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an RL trading agent.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--algorithm", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_cfg, env_cfg, train_cfg = cfg["data"], cfg["env"], cfg["train"]

    timesteps = args.timesteps or train_cfg["total_timesteps"]
    algorithm = args.algorithm or train_cfg["algorithm"]
    if algorithm not in ALGORITHMS:
        raise SystemExit(f"Unknown algorithm: {algorithm}. Choose from {list(ALGORITHMS)}.")

    df = load_dataframe(Path(data_cfg["path"]))
    train_df, _ = train_test_split(df, data_cfg["train_fraction"])

    env = DummyVecEnv([lambda: make_env(train_df, env_cfg, data_cfg["price_column"])])
    model = ALGORITHMS[algorithm](
        train_cfg["policy"],
        env,
        learning_rate=train_cfg["learning_rate"],
        gamma=train_cfg["gamma"],
        seed=train_cfg["seed"],
        device=train_cfg["device"],
        verbose=1,
    )

    print(f"[train] {algorithm} on {len(train_df):,} rows, {timesteps:,} timesteps")
    model.learn(total_timesteps=timesteps)

    model_dir = Path(train_cfg.get("model_dir", "models"))
    model_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or str(model_dir / f"{algorithm.lower()}_{timesteps}")
    model.save(output)
    print(f"[train] model saved to {output}")


if __name__ == "__main__":
    main()
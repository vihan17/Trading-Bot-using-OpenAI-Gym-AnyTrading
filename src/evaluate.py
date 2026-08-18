"""Evaluate a trained RL agent on held-out data.

Runs ``n_episodes`` evaluation episodes from spread-out start indices on
the test split, records portfolio-value series, computes metrics, and
saves a JSON report plus an equity-curve figure.

Usage::

    python -m src.evaluate --config configs/default.yaml --model models/ppo_100000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import yaml

from stable_baselines3 import A2C, DQN, PPO

from src.data import load_dataframe, train_test_split
from src.trading_env import TradingEnv
from src.utils import summary

ALGORITHMS = {"PPO": PPO, "A2C": A2C, "DQN": DQN}


def make_env(df, env_cfg: dict, price_column: str):
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


def infer_algorithm(model_path: str, cfg: dict) -> str:
    """Infer the algorithm from --algorithm, config, or the model path."""
    if cfg.get("algorithm"):
        return cfg["algorithm"]
    for name in ALGORITHMS:
        if name in model_path:
            return name
    return "PPO"


def run_episode(env, model, start_index: int) -> tuple[list[float], dict]:
    """Run one deterministic episode, returning (values, metrics)."""
    obs, _ = env.reset(options={"start_index": start_index})
    values = [env._portfolio_value(env.prices[env._current_step])]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        values.append(info["portfolio_value"])
        done = terminated or truncated
    return values, summary(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained RL agent.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", required=True, help="Path to saved SB3 model.")
    parser.add_argument("--algorithm", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_cfg, env_cfg, eval_cfg = cfg["data"], cfg["env"], cfg["eval"]

    df = load_dataframe(Path(data_cfg["path"]))
    _, test_df = train_test_split(df, data_cfg["train_fraction"])

    algorithm = args.algorithm or infer_algorithm(args.model, eval_cfg)
    model = ALGORITHMS[algorithm].load(args.model)

    n_episodes = eval_cfg["n_episodes"]
    periods_per_year = eval_cfg.get("periods_per_year", 8760)

    env = make_env(test_df, env_cfg, data_cfg["price_column"])
    low, high = env._valid_start_bounds()

    # Spread episode starts evenly across the test window.
    step = max(1, (high - low) // n_episodes)
    starts = [low + i * step for i in range(n_episodes)]

    values_by_episode = []
    results = []
    for start in starts:
        values, metrics = run_episode(env, model, start)
        values_by_episode.append(values)
        results.append(metrics)

    # Chart: one equity curve per episode.
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, values in enumerate(values_by_episode):
        ax.plot(values, label=f"episode {i + 1} (start {starts[i]})")
    ax.axhline(env_cfg["initial_value"], color="gray", ls="--", label="Initial capital")
    ax.set_xlabel("Step")
    ax.set_ylabel("Portfolio value")
    ax.set_title(f"RL agent ({algorithm}) on held-out data")
    ax.legend()
    ax.grid(alpha=0.4)

    results_dir = Path(eval_cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(results_dir / "equity_curve.png", dpi=120)

    mean_metrics = {
        k: round(sum(r[k] for r in results) / len(results), 4) for k in results[0]
    }
    report = {
        "model": args.model,
        "algorithm": algorithm,
        "episodes": n_episodes,
        "periods_per_year": periods_per_year,
        "episode_starts": starts,
        "episode_metrics": results,
        "mean": mean_metrics,
    }
    out_path = results_dir / "eval_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[eval] report saved to {out_path}")
    print(f"[eval] equity curve saved to {results_dir / 'equity_curve.png'}")


if __name__ == "__main__":
    main()
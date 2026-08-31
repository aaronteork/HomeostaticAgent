"""Evaluate trained PPO policies using survival-oriented rollouts.

This mirrors :mod:`utils.evaluate_dreamer_survival`, but loads the plain
``HomeostaticPPO.state_dict`` checkpoints produced by ``train_ppo.py``.
Those checkpoints do not contain their training configuration, so evaluation
uses the current ``PPOConfig`` defaults (apart from device and worker count).

Example:
    python utils/evaluate_ppo_survival.py --checkpoint-dir models
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from configs.config_ppo import PPOConfig
from utils.utils import set_seed
from utils.utils_env import create_env
from utils.utils_ppo import HomeostaticPPO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure PPO policy survival.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Folder containing PPO .pt state-dictionary checkpoints.")
    parser.add_argument("--episodes", type=int, default=16, help="Independent evaluation episodes per checkpoint.")
    parser.add_argument("--num-workers", type=int, default=16, help="Concurrent AsyncVectorEnv workers.")
    parser.add_argument("--max-steps", type=int, default=None, help="Administrative per-episode cutoff (default: PPOConfig.eval_max_steps).")
    parser.add_argument("--seed", type=int, default=0, help="First reset seed; subsequent episodes increment it by one.")
    parser.add_argument("--device", default=None, help="Torch device, such as cuda or cpu (default: available accelerator).")
    parser.add_argument("--stochastic", action="store_true", help="Sample the Beta policy instead of using its deterministic mode.")
    parser.add_argument("--output-dir", type=Path, default=Path("survival_evaluation_ppo"), help="Directory for survival_by_checkpoint.csv and survival_by_checkpoint.png.")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1.")
    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1.")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be at least 1.")
    if not args.checkpoint_dir.is_dir():
        parser.error(f"Checkpoint folder not found: {args.checkpoint_dir}")
    return args


def find_checkpoints(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    """Return PPO checkpoints in a stable order.

    ``train_ppo.py`` records completed PPO iterations as ``_chkpt<N>`` and
    marks its final policy ``_final``.  Legacy files without either suffix use
    their sorted evaluation order.
    """
    paths = sorted(
        path
        for path in checkpoint_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".pt" and "ppo" in path.stem.lower()
    )
    if not paths:
        raise FileNotFoundError(f"No PPO .pt checkpoints found in {checkpoint_dir}.")
    pattern = re.compile(r"_chkpt(?P<step>\d+)", re.IGNORECASE)
    final_iteration = PPOConfig().total_updates
    checkpoints = []
    for index, path in enumerate(paths, start=1):
        match = pattern.search(path.stem)
        if match:
            step = int(match.group("step"))
        elif path.stem.lower().endswith("_final"):
            step = final_iteration
        else:
            step = index
        checkpoints.append((step, path))
    return checkpoints


def torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch before weights_only was added.
        return torch.load(path, map_location=device)


def load_policy(checkpoint: Path, requested_device: str | None, num_workers: int) -> tuple[PPOConfig, HomeostaticPPO]:
    device = torch.device(requested_device) if requested_device else (
        torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")
    )
    payload = torch_load(checkpoint, device)
    if not isinstance(payload, dict):
        raise TypeError("PPO checkpoint must be a HomeostaticPPO state dictionary.")
    state_dict = {key.replace("._orig_mod.", ".").removeprefix("_orig_mod."): value for key, value in payload.items()}
    config = PPOConfig(device=device, is_training=False, num_workers=num_workers)
    agent = HomeostaticPPO(config).to(device)
    agent.load_state_dict(state_dict, strict=True)
    return config, agent.eval()


def info_scalar(info: dict[str, Any], name: str, worker: int, default: float = 0.0) -> float:
    """Read one worker's scalar from a vector-environment info dictionary."""
    if name not in info:
        return default
    return float(np.asarray(info[name])[worker])


@torch.inference_mode()
def run_parallel_episodes(env: Any, config: PPOConfig, agent: HomeostaticPPO, seeds: list[int], active_count: int, max_steps: int, stochastic: bool) -> list[dict[str, Any]]:
    """Run up to one independent evaluation episode on each async worker."""
    worker_count = config.num_workers
    obs, _ = env.reset(seed=seeds)
    active = np.arange(worker_count) < active_count
    flips_observed = np.zeros(worker_count, dtype=np.int64)
    food_consumed = np.zeros(worker_count, dtype=np.int64)
    water_consumed = np.zeros(worker_count, dtype=np.int64)
    results: list[dict[str, Any]] = []

    for steps in range(1, max_steps + 1):
        action, _, _, _ = agent(obs["vision"], obs["proprioception"], obs["internal_state"], deterministic=not stochastic)
        obs, _, terminated, truncated, info = env.step(action.cpu().numpy())
        for worker in np.flatnonzero(active):
            flips_observed[worker] += int(bool(info_scalar(info, "is_flipped", worker)))
            food_consumed[worker] = int(info_scalar(info, "food_consumed", worker))
            water_consumed[worker] = int(info_scalar(info, "water_consumed", worker))
            if terminated[worker] or truncated[worker] or steps == max_steps:
                outcome = "homeostatic_termination" if terminated[worker] else ("environment_truncation" if truncated[worker] else "evaluation_cutoff")
                results.append({
                    "seed": seeds[worker], "steps": steps, "outcome": outcome,
                    "final_state_label": int(info_scalar(info, "termination_reason", worker)),
                    "food_consumed": int(food_consumed[worker]), "water_consumed": int(water_consumed[worker]),
                    "flipped_steps": int(flips_observed[worker]),
                    "final_hunger": info_scalar(info, "hunger", worker),
                    "final_thirst": info_scalar(info, "thirst", worker),
                    "final_posture": info_scalar(info, "posture", worker),
                    "final_height": info_scalar(info, "z_pos", worker),
                })
                active[worker] = False
        if not active.any():
            return results
    raise RuntimeError("Parallel evaluation exited without completing all active episodes.")


def evaluate_checkpoint(args: argparse.Namespace, step: int, checkpoint: Path) -> dict[str, Any]:
    config, agent = load_policy(checkpoint, args.device, args.num_workers)
    max_steps = args.max_steps or config.eval_max_steps
    set_seed(args.seed)
    env = create_env(config, multiple_env=True)
    try:
        episodes = []
        for offset in range(0, args.episodes, args.num_workers):
            active_count = min(args.num_workers, args.episodes - offset)
            seeds = [args.seed + offset + worker for worker in range(args.num_workers)]
            episodes.extend(run_parallel_episodes(env, config, agent, seeds, active_count, max_steps, args.stochastic))
    finally:
        env.close()

    lengths = np.asarray([episode["steps"] for episode in episodes], dtype=np.float64)
    outcomes = ("homeostatic_termination", "environment_truncation", "evaluation_cutoff")
    return {
        "checkpoint_step": step, "checkpoint": str(checkpoint), "policy_mode": "stochastic" if args.stochastic else "deterministic",
        "episodes": args.episodes, "num_workers": args.num_workers, "max_steps": max_steps, "seed": args.seed,
        "survival_steps": {"mean": float(lengths.mean()), "median": float(np.median(lengths)), "min": int(lengths.min()), "max": int(lengths.max())},
        "survival_rate_to_cutoff": sum(episode["outcome"] == "evaluation_cutoff" for episode in episodes) / args.episodes,
        "outcomes": {name: sum(episode["outcome"] == name for episode in episodes) for name in outcomes},
        "episode_details": episodes,
    }


def save_survival_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Creating the survival graph requires matplotlib in the evaluation environment.") from error
    ordered = sorted(rows, key=lambda row: int(row["checkpoint_step"]))
    steps = [int(row["checkpoint_step"]) for row in ordered]
    minimums = [float(row["min_survival_steps"]) for row in ordered]
    maximums = [float(row["max_survival_steps"]) for row in ordered]
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for axis, (column, ylabel) in zip(axes, (("mean_survival_steps", "Mean survival steps"), ("median_survival_steps", "Median survival steps"))):
        values = [float(row[column]) for row in ordered]
        axis.fill_between(steps, minimums, maximums, color="tab:blue", alpha=0.2, label="Episode range (min--max)")
        axis.plot(steps, values, marker="o", linewidth=2, color="tab:blue", label=ylabel)
        axis.set_ylabel(ylabel)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("Completed PPO iterations (or sorted checkpoint order)")
    axes[-1].set_xticks(steps)
    axes[-1].ticklabel_format(style="plain", axis="x")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    results = []
    for step, checkpoint in find_checkpoints(args.checkpoint_dir.resolve()):
        checkpoint = checkpoint.resolve()
        print(f"Evaluating checkpoint {step}: {checkpoint.name}", flush=True)
        results.append(evaluate_checkpoint(args, step, checkpoint))
    rows = []
    for result in results:
        survival, outcomes = result["survival_steps"], result["outcomes"]
        rows.append({
            "checkpoint_step": result["checkpoint_step"], "checkpoint": result["checkpoint"], "policy_mode": result["policy_mode"],
            "episodes": result["episodes"], "num_workers": result["num_workers"], "max_steps": result["max_steps"], "seed": result["seed"],
            "mean_survival_steps": survival["mean"], "median_survival_steps": survival["median"], "min_survival_steps": survival["min"], "max_survival_steps": survival["max"],
            "survival_rate_to_cutoff": result["survival_rate_to_cutoff"], "homeostatic_terminations": outcomes["homeostatic_termination"],
            "environment_truncations": outcomes["environment_truncation"], "evaluation_cutoffs": outcomes["evaluation_cutoff"],
        })
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path, plot_path = output_dir / "survival_by_checkpoint.csv", output_dir / "survival_by_checkpoint.png"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_survival_plot(rows, plot_path)
    print(f"Saved survival CSV to {csv_path}", flush=True)
    print(f"Saved survival plot to {plot_path}", flush=True)


if __name__ == "__main__":
    main()

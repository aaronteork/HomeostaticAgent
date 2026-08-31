"""Evaluate a trained Dreamer policy using survival-oriented rollouts.

This is intentionally separate from training and world-model diagnostics.  It
reports how long the policy remains alive, resource collection, and stability
indicators.  A rollout ends only when the environment reports a homeostatic
termination or when ``--max-steps`` is reached; the latter is an administrative
evaluation cutoff, not a death.

Example:
    python evaluate_dreamer_survival.py --checkpoint-dir models
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from configs.config_dreamer import DreamerConfig
from utils.utils import set_seed
from utils.utils_dreamer import obs_to_tensor_dict
from utils.utils_env import create_env
from utils.world_model import BetaActorNetwork, GaussianActorNetwork, WorldModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure Dreamer policy survival.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Folder containing full train_dreamer.py checkpoints.")
    parser.add_argument("--episodes", type=int, default=16, help="Number of independent evaluation episodes per checkpoint.")
    parser.add_argument("--num-workers", type=int, default=16, help="Concurrent AsyncVectorEnv workers.")
    parser.add_argument("--max-steps", type=int, default=None, help="Administrative per-episode cutoff (default: checkpoint eval_max_steps).")
    parser.add_argument("--seed", type=int, default=0, help="First reset seed; subsequent episodes increment it by one.")
    parser.add_argument("--device", default=None, help="Torch device, such as cuda or cpu (default: checkpoint configuration).")
    parser.add_argument("--stochastic", action="store_true", help="Sample the policy instead of using its deterministic mode.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("survival_evaluation"),
        help="Directory for survival_by_checkpoint.csv and survival_by_checkpoint.png (mean and median panels).",
    )
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
    """Return step-labelled checkpoints, excluding any explicitly final file."""
    pattern = re.compile(r"_chkpt(?P<step>\d+)\.pt$", re.IGNORECASE)
    checkpoints = []
    for path in checkpoint_dir.iterdir():
        match = pattern.search(path.name)
        if path.is_file() and "_final" not in path.stem.lower() and match:
            checkpoints.append((int(match.group("step")), path))
    checkpoints.sort(key=lambda item: (item[0], item[1].name))
    if not checkpoints:
        raise FileNotFoundError(
            f"No step-labelled checkpoints found in {checkpoint_dir}. "
            "Expected names ending in _chkpt<step>.pt."
        )
    return checkpoints


def torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch before weights_only was added.
        return torch.load(path, map_location=device)


def cleaned_state(payload: Any, name: str) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict) or name not in payload:
        raise ValueError(f"Checkpoint has no '{name}' state dictionary; use a full train_dreamer.py checkpoint.")
    component = payload[name]
    if not isinstance(component, dict):
        raise TypeError(f"Checkpoint component '{name}' is not a state dictionary.")
    return {
        key.replace("._orig_mod.", ".").removeprefix("_orig_mod.").removeprefix(f"{name}."): value
        for key, value in component.items()
    }


def load_policy(
    checkpoint: Path, requested_device: str | None, num_workers: int,
) -> tuple[DreamerConfig, WorldModel, torch.nn.Module]:
    device = torch.device(requested_device) if requested_device else (
        torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")
    )
    payload = torch_load(checkpoint, device)
    saved_config = payload.get("config") if isinstance(payload, dict) else None
    if not isinstance(saved_config, dict):
        raise ValueError("Checkpoint has no saved configuration; survival evaluation requires a full current Dreamer checkpoint.")

    valid_fields = {field.name for field in fields(DreamerConfig)}
    values = {key: value for key, value in saved_config.items() if key in valid_fields and key not in {"device", "use_amp", "is_training", "num_workers"}}
    config = replace(DreamerConfig(**values), device=device, use_amp=False, is_training=False, num_workers=num_workers)
    world_model = WorldModel(config).to(device)
    world_model.load_state_dict(cleaned_state(payload, "world_model"), strict=True)

    if config.actor_policy == "beta":
        actor: torch.nn.Module = BetaActorNetwork(config).to(device)
    elif config.actor_policy == "gaussian":
        actor = GaussianActorNetwork(config).to(device)
    else:
        raise ValueError(f"Unsupported actor_policy in checkpoint: {config.actor_policy!r}")
    actor.load_state_dict(cleaned_state(payload, "actor"), strict=True)
    return config, world_model.eval(), actor.eval()


def info_scalar(info: dict[str, Any], name: str, worker: int, default: float = 0.0) -> float:
    """Read one worker's scalar from a vector-environment info dictionary."""
    if name not in info:
        return default
    return float(np.asarray(info[name])[worker])


@torch.inference_mode()
def run_parallel_episodes(
    env: Any,
    config: DreamerConfig,
    world_model: WorldModel,
    actor: torch.nn.Module,
    seeds: list[int],
    active_count: int,
    max_steps: int,
    stochastic: bool,
) -> list[dict[str, Any]]:
    """Run up to one independent evaluation episode on each async worker."""
    worker_count = config.num_workers
    obs, _ = env.reset(seed=seeds)
    previous_action = torch.zeros(worker_count, config.action_space_dim, device=config.device)
    recurrent_state = None
    previous_stochastic = None
    is_first = torch.ones(worker_count, dtype=torch.bool, device=config.device)
    active = np.arange(worker_count) < active_count
    flips_observed = np.zeros(worker_count, dtype=np.int64)
    food_consumed = np.zeros(worker_count, dtype=np.int64)
    water_consumed = np.zeros(worker_count, dtype=np.int64)
    results: list[dict[str, Any]] = []

    for steps in range(1, max_steps + 1):
        embedding = world_model.encode(obs_to_tensor_dict(obs, config))
        latent, recurrent_state, _, _ = world_model.observe(
            previous_action, embedding, is_first, recurrent_state=recurrent_state,
            previous_stochastic=previous_stochastic, deterministic=not stochastic,
        )
        previous_stochastic, recurrent_state = world_model.rssm.split_feature(latent)
        action, _, _ = actor(latent, deterministic=not stochastic)
        action_np = action.cpu().numpy()
        # Beta actions are already in the RescaleAction wrapper's [0, 1] coordinates.
        if config.actor_policy == "beta":
            env_action = action_np
        else:
            env_action = np.clip(action_np, -1.0, 1.0)
        obs, _, terminated, truncated, info = env.step(env_action)
        for worker in np.flatnonzero(active):
            flips_observed[worker] += int(bool(info_scalar(info, "is_flipped", worker)))
            food_consumed[worker] = int(info_scalar(info, "food_consumed", worker))
            water_consumed[worker] = int(info_scalar(info, "water_consumed", worker))
            if terminated[worker] or truncated[worker] or steps == max_steps:
                # Only a homeostatic limit terminates; flip and height remain
                # state labels, so they are never reported as death causes.
                outcome = "homeostatic_termination" if terminated[worker] else (
                    "environment_truncation" if truncated[worker] else "evaluation_cutoff"
                )
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
        previous_action = action.detach()
        is_first = torch.zeros(worker_count, dtype=torch.bool, device=config.device)

    raise RuntimeError("Parallel evaluation exited without completing all active episodes.")


def evaluate_checkpoint(args: argparse.Namespace, step: int, checkpoint: Path) -> dict[str, Any]:
    config, world_model, actor = load_policy(checkpoint, args.device, args.num_workers)
    max_steps = args.max_steps or config.eval_max_steps
    set_seed(args.seed)
    env = create_env(config, multiple_env=True, rescale_action=config.actor_policy == "beta")
    try:
        episodes = []
        for offset in range(0, args.episodes, args.num_workers):
            active_count = min(args.num_workers, args.episodes - offset)
            seeds = [args.seed + offset + worker for worker in range(args.num_workers)]
            episodes.extend(
                run_parallel_episodes(
                    env, config, world_model, actor, seeds, active_count,
                    max_steps, args.stochastic,
                )
            )
    finally:
        env.close()

    lengths = np.asarray([episode["steps"] for episode in episodes], dtype=np.float64)
    return {
        "checkpoint_step": step, "checkpoint": str(checkpoint),
        "policy_mode": "stochastic" if args.stochastic else "deterministic",
        "episodes": args.episodes, "num_workers": args.num_workers,
        "max_steps": max_steps, "seed": args.seed,
        "survival_steps": {"mean": float(lengths.mean()), "median": float(np.median(lengths)), "min": int(lengths.min()), "max": int(lengths.max())},
        "survival_rate_to_cutoff": sum(episode["outcome"] == "evaluation_cutoff" for episode in episodes) / args.episodes,
        "outcomes": {name: sum(episode["outcome"] == name for episode in episodes) for name in ("homeostatic_termination", "environment_truncation", "evaluation_cutoff")},
        "episode_details": episodes,
        # Keep enough provenance for comparing runs while remaining JSON-safe.
        "config": {
            "actor_policy": config.actor_policy,
            "num_food": config.num_food,
            "num_water": config.num_water,
            "num_heat": config.num_heat,
            "hunger_decay": config.hunger_decay,
            "thirst_decay": config.thirst_decay,
            "replenish_rate": config.replenish_rate,
            "device": str(config.device),
        },
    }


def save_survival_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Plot mean and median survival, with min--max episode bounds, by checkpoint."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Creating the survival graph requires matplotlib in the evaluation environment."
        ) from error

    ordered_rows = sorted(rows, key=lambda row: int(row["checkpoint_step"]))
    steps = [int(row["checkpoint_step"]) for row in ordered_rows]
    minimums = [float(row["min_survival_steps"]) for row in ordered_rows]
    maximums = [float(row["max_survival_steps"]) for row in ordered_rows]
    summaries = (
        ("mean_survival_steps", "Mean survival steps", "Mean survival by checkpoint"),
        ("median_survival_steps", "Median survival steps", "Median survival by checkpoint"),
    )
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for axis, (column, ylabel, title) in zip(axes, summaries):
        survival = [float(row[column]) for row in ordered_rows]
        axis.fill_between(
            steps, minimums, maximums, color="tab:blue", alpha=0.2,
            label="Episode range (min--max)",
        )
        axis.plot(steps, survival, marker="o", linewidth=2, color="tab:blue", label=ylabel)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("Training steps")
    axes[-1].set_xticks(steps)
    axes[-1].ticklabel_format(style="plain", axis="x")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    checkpoint_results = []
    for step, checkpoint in find_checkpoints(checkpoint_dir):
        checkpoint = checkpoint.resolve()
        print(f"Evaluating checkpoint step {step}: {checkpoint.name}", flush=True)
        checkpoint_results.append(evaluate_checkpoint(args, step, checkpoint))

    rows = []
    for result in checkpoint_results:
        outcomes = result["outcomes"]
        survival = result["survival_steps"]
        rows.append({
            "checkpoint_step": result["checkpoint_step"],
            "checkpoint": result["checkpoint"],
            "policy_mode": result["policy_mode"],
            "episodes": result["episodes"],
            "num_workers": result["num_workers"],
            "max_steps": result["max_steps"],
            "seed": result["seed"],
            "mean_survival_steps": survival["mean"],
            "median_survival_steps": survival["median"],
            "min_survival_steps": survival["min"],
            "max_survival_steps": survival["max"],
            "survival_rate_to_cutoff": result["survival_rate_to_cutoff"],
            "homeostatic_terminations": outcomes["homeostatic_termination"],
            "environment_truncations": outcomes["environment_truncation"],
            "evaluation_cutoffs": outcomes["evaluation_cutoff"],
        })
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "survival_by_checkpoint.csv"
    plot_path = output_dir / "survival_by_checkpoint.png"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_survival_plot(rows, plot_path)
    print(f"Saved survival CSV to {csv_path}", flush=True)
    print(f"Saved survival plot to {plot_path}", flush=True)


if __name__ == "__main__":
    main()

"""Evaluate Homeostatic Ant Dreamer checkpoints without MLflow.

The script has two independent data sources:

* deterministic online episodes from fixed evaluation seeds, used for the
  behavioural curves; and
* a frozen validation set collected once with a chosen trained reference
  checkpoint, used for comparable world-model losses.

Examples (run in the MuJoCo-enabled project environment)::

    # Create the frozen validation trajectories from a capable reference model.
    python checks/evaluate_dreamer_checkpoints.py --build-validation \
      --reference-checkpoint models/dreamer_step_2000000.pt \
      --validation-path evaluation/validation.pt

    python utils/evaluate_dreamer.py \
        --checkpoints path/to/model_run_folder \
        --output-dir evaluation/my_run_eval \
        --validation-path evaluation/dreamer_validation.pt


Checkpoint payloads saved by current ``train_dreamer.py`` are supported. For
old weight-only checkpoints, supply ``--checkpoint-steps`` in the same order
as ``--checkpoints`` because a timestamp-only filename has no recoverable
training-step value.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import glob
import hashlib
import json
import math
import re
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from configs.config_dreamer import DreamerConfig
from utils.utils_dreamer import compute_world_model_loss, obs_to_tensor_dict, to_tensor
from utils.utils_env import create_env
from utils.world_model import ActorNetwork, CriticNetwork, WorldModel


STATE_HISTOGRAM_BINS = np.linspace(-1.0, 1.0, 101)


def parse_seed_spec(value: str) -> list[int]:
    """Parse ``1000:1016`` or ``1000,1003,1008`` seed specifications."""
    if ":" in value:
        start, stop = (int(part) for part in value.split(":", 1))
        if stop <= start:
            raise argparse.ArgumentTypeError("seed range stop must exceed start")
        return list(range(start, stop))
    return [int(part) for part in value.split(",") if part]


def resolve_checkpoints(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        p = Path(value)
        if p.is_dir():
            matches = sorted(p.glob("*.pt"))
            paths.extend(matches)
        else:
            matches = [Path(item) for item in sorted(glob.glob(value))]
            paths.extend(matches if matches else [Path(value)])
    # Exclude convenience duplicates such as *_final.pt
    paths = [p for p in paths if not (p.stem.endswith("_final") or p.stem == "final")]
    unique = list(dict.fromkeys(path.resolve() for path in paths))
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError("checkpoint(s) not found: " + ", ".join(missing))
    return unique


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def config_to_jsonable(cfg: DreamerConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(cfg), default=json_default))


def config_from_payload(payload: dict[str, Any], fallback: DreamerConfig) -> DreamerConfig:
    saved = payload.get("config")
    if not isinstance(saved, dict):
        return fallback
    fields = {field.name for field in dataclasses.fields(DreamerConfig)}
    values = {key: value for key, value in saved.items() if key in fields}
    if "device" in values:
        values["device"] = torch.device(str(values["device"]))
    return DreamerConfig(**values)


def unwrap_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    """Accept ordinary and torch.compile-prefixed state dictionaries."""
    if all(key.startswith("_orig_mod.") for key in state):
        return {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    return state


def load_models(path: Path, fallback_cfg: DreamerConfig) -> tuple[dict[str, Any], DreamerConfig, WorldModel, ActorNetwork, CriticNetwork | None]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "world_model" not in payload or "actor" not in payload:
        raise ValueError(f"{path} is not a full Dreamer checkpoint")
    cfg = config_from_payload(payload, fallback_cfg)
    world_model = WorldModel(cfg).to(cfg.device)
    actor = ActorNetwork(cfg).to(cfg.device)
    world_model.load_state_dict(unwrap_state_dict(payload["world_model"]))
    actor.load_state_dict(unwrap_state_dict(payload["actor"]))
    critic = None
    if "critic" in payload:
        critic = CriticNetwork(cfg).to(cfg.device)
        critic.load_state_dict(unwrap_state_dict(payload["critic"]))
        critic.eval()
    world_model.eval()
    actor.eval()
    return payload, cfg, world_model, actor, critic


def checkpoint_step(path: Path, payload: dict[str, Any], supplied: int | None) -> int:
    for key in ("env_transition_step", "global_step", "training_step"):
        if key in payload:
            return int(payload[key])
    if supplied is not None:
        return supplied
    match = re.search(r"(?:step|steps|envsteps|chkpt)[_-]?(\d+)", path.stem, flags=re.I)
    if match:
        return int(match.group(1))
    raise ValueError(
        f"Cannot determine training step for {path.name}. Supply --checkpoint-steps "
        "or save future checkpoints with env_transition_step metadata."
    )


def array_stats(values: list[float] | np.ndarray, prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {f"{prefix}_{name}": float("nan") for name in ("min", "p10", "median", "mean", "p95", "max")}
    return {
        f"{prefix}_min": float(array.min()),
        f"{prefix}_p10": float(np.percentile(array, 10)),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_p95": float(np.percentile(array, 95)),
        f"{prefix}_max": float(array.max()),
    }


# def sha256_file(path: Path) -> str:
#     digest = hashlib.sha256()
#     with path.open("rb") as handle:
#         while block := handle.read(1024 * 1024):
#             digest.update(block)
#     return digest.hexdigest()


def summary_row(episodes: list[dict[str, Any]], checkpoint_step_value: int) -> dict[str, Any]:
    row: dict[str, Any] = {"checkpoint_step": checkpoint_step_value, "episodes": len(episodes)}
    numeric_keys = [
        "survival_length", "food_consumed", "water_consumed", "food_per_step",
        "water_per_step", "reward_per_step", "homeostatic_reward_per_step",
        "movement_penalty_per_step", "posture_penalty_per_step", "hunger_time_mean",
        "thirst_time_mean", "final_hunger", "final_thirst", "hunger_abs_p95", "thirst_abs_p95", "hunger_abs_max",
        "thirst_abs_max", "hunger_danger_080_fraction", "thirst_danger_080_fraction",
        "hunger_danger_095_fraction", "thirst_danger_095_fraction", "posture_mean",
        "posture_p95", "posture_max", "action_magnitude_mean", "action_magnitude_p95",
        "flipped_step_fraction", "first_flip_step",
    ]
    for key in numeric_keys:
        values = [episode[key] for episode in episodes if np.isfinite(episode[key])]
        row.update(array_stats(values, key))
    for key in ("homeostatic_ended", "eval_capped", "ever_flipped", "food_collected", "water_collected"):
        row[f"{key}_fraction"] = float(np.mean([episode[key] for episode in episodes]))
    return row


def evaluate_policy(
    world_model: WorldModel,
    actor: ActorNetwork,
    critic: CriticNetwork | None = None,
    cfg: DreamerConfig | None = None,
    seeds: list[int] = None,
    max_steps: int = 60_000,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    if cfg is None:
        cfg = DreamerConfig()
    if seeds is None:
        seeds = [1000]
    if len(seeds) != cfg.num_workers:
        cfg = replace(cfg, num_workers=len(seeds))
    eval_cfg = replace(cfg, is_training=True, max_steps=max_steps)
    env = create_env(eval_cfg, multiple_env=True)
    try:
        obs, _ = env.reset(seed=seeds)
        workers = len(seeds)
        previous_action = torch.zeros(workers, eval_cfg.action_space_dim, device=eval_cfg.device)
        is_first = torch.ones(workers, device=eval_cfg.device)
        recurrent_state = None
        previous_stochastic = None
        active = np.ones(workers, dtype=bool)
        # terminal_bootstraps: list[dict[str, Any]] = []
        data = [
            {"seed": seed, "hunger": [], "thirst": [], "posture": [], "action_magnitude": [],
             "reward": [], "homeostatic_reward": [], "movement_penalty": [], "posture_penalty": [],
             "flipped": [], "first_flip_step": -1, "ended": False,
             "homeostatic_ended": False, "eval_capped": False,
             "termination_reason": 0, "food_consumed": 0.0, "water_consumed": 0.0}
            for seed in seeds
        ]
        neutral = np.full((workers, eval_cfg.action_space_dim), 0.5, dtype=np.float32)
        while active.any():
            with torch.inference_mode():
                embed = world_model.encode(obs_to_tensor_dict(obs, eval_cfg))
                latent, recurrent_state, _, _ = world_model.observe(previous_action, embed, is_first, recurrent_state, previous_stochastic, deterministic=True)
                previous_stochastic, recurrent_state = world_model.rssm.split_feature(latent)
                action, _, _ = actor(latent, deterministic=True)
                action = action.cpu().numpy()
            action[~active] = neutral[~active]
            next_obs, rewards, terminations, truncations, infos = env.step(action)
            for index in np.flatnonzero(active):
                item = data[index]
                item["hunger"].append(float(infos["hunger"][index]))
                item["thirst"].append(float(infos["thirst"][index]))
                item["posture"].append(float(infos["posture"][index]))
                item["action_magnitude"].append(float(infos["action_magnitude"][index]))
                item["reward"].append(float(rewards[index]))
                item["homeostatic_reward"].append(float(infos["reward_homeostatic"][index]))
                item["movement_penalty"].append(float(infos["reward_movement_penalty"][index]))
                item["posture_penalty"].append(float(infos["reward_posture_penalty"][index]))
                flipped = bool(infos["is_flipped"][index])
                item["flipped"].append(flipped)
                if flipped and item["first_flip_step"] < 0:
                    item["first_flip_step"] = len(item["reward"])
            ended = active & (np.asarray(terminations, dtype=bool) | np.asarray(truncations, dtype=bool))
            # Terminal bootstrap diagnostic commented out:
            # if ended.any() and critic is not None:
            #     with torch.inference_mode():
            #         final_embed = world_model.encode(obs_to_tensor_dict(next_obs, eval_cfg))
            #         final_latent, _, _, _ = world_model.observe(
            #             to_tensor(action, eval_cfg.device), final_embed,
            #             torch.zeros(workers, device=eval_cfg.device),
            #             recurrent_state, previous_stochastic, deterministic=True,
            #         )
            #         final_values = critic(final_latent).mode.squeeze(-1).cpu().numpy()
            #     for index in np.flatnonzero(ended):
            #         terminal_reward = float(rewards[index])
            #         value = float(final_values[index])
            #         terminal_bootstraps.append({
            #             "seed": seeds[index],
            #             "survival_length": len(data[index]["reward"]),
            #             "homeostatic_ended": float(bool(terminations[index])),
            #             "eval_capped": float(bool(truncations[index])),
            #             "termination_reason": int(infos["termination_reason"][index]),
            #             "final_hunger": float(infos["hunger"][index]),
            #             "final_thirst": float(infos["thirst"][index]),
            #             "terminal_reward": terminal_reward,
            #             "critic_final_value": value,
            #             "discounted_bootstrap": float(eval_cfg.discount * value),
            #             "one_step_bootstrapped_target": float(terminal_reward + eval_cfg.discount * value),
            #             "one_step_no_bootstrap_target": terminal_reward,
            #         })
            for index in np.flatnonzero(ended):
                item = data[index]
                item["ended"] = True
                item["homeostatic_ended"] = bool(terminations[index])
                item["eval_capped"] = bool(truncations[index])
                item["termination_reason"] = int(infos["termination_reason"][index])
                item["food_consumed"] = float(infos["food_consumed"][index])
                item["water_consumed"] = float(infos["water_consumed"][index])
            active[ended] = False
            previous = action.copy()
            previous[~active] = neutral[~active]
            previous_action = to_tensor(previous, eval_cfg.device)
            is_first = torch.zeros(workers, device=eval_cfg.device)
            obs = next_obs

        episodes: list[dict[str, Any]] = []
        histograms: dict[str, Any] = {}
        all_hunger = np.concatenate([np.asarray(item["hunger"]) for item in data])
        all_thirst = np.concatenate([np.asarray(item["thirst"]) for item in data])
        all_posture = np.concatenate([np.asarray(item["posture"]) for item in data])
        all_actions = np.concatenate([np.asarray(item["action_magnitude"]) for item in data])
        all_rewards = np.concatenate([np.asarray(item["reward"]) for item in data])
        all_homeostatic_rewards = np.concatenate([np.asarray(item["homeostatic_reward"]) for item in data])
        all_movement_penalties = np.concatenate([np.asarray(item["movement_penalty"]) for item in data])
        all_posture_penalties = np.concatenate([np.asarray(item["posture_penalty"]) for item in data])
        histograms["state_histogram_edges"] = STATE_HISTOGRAM_BINS.tolist()
        histograms["hunger_counts"] = np.histogram(all_hunger, bins=STATE_HISTOGRAM_BINS)[0].tolist()
        histograms["thirst_counts"] = np.histogram(all_thirst, bins=STATE_HISTOGRAM_BINS)[0].tolist()
        histograms["timestep_metrics"] = {
            **array_stats(all_hunger, "timestep_hunger"),
            **array_stats(all_thirst, "timestep_thirst"),
            **array_stats(np.abs(all_hunger), "timestep_abs_hunger"),
            **array_stats(np.abs(all_thirst), "timestep_abs_thirst"),
            **array_stats(all_posture, "timestep_posture"),
            **array_stats(all_actions, "timestep_action_magnitude"),
            **array_stats(all_rewards, "timestep_reward"),
            **array_stats(all_homeostatic_rewards, "timestep_homeostatic_reward"),
            **array_stats(all_movement_penalties, "timestep_movement_penalty"),
            **array_stats(all_posture_penalties, "timestep_posture_penalty"),
        }
        for index, item in enumerate(data):
            hunger, thirst = np.asarray(item["hunger"]), np.asarray(item["thirst"])
            posture, action_mag = np.asarray(item["posture"]), np.asarray(item["action_magnitude"])
            rewards = np.asarray(item["reward"])
            length = len(rewards)
            # The per-worker info fields remain available at the terminal step.
            # Cumulative food/water are captured on the last recorded transition.
            episodes.append({
                "seed": item["seed"], "survival_length": length,
                "homeostatic_ended": float(item["homeostatic_ended"]), "eval_capped": float(item["eval_capped"]),
                "termination_reason": item["termination_reason"],
                "food_consumed": item["food_consumed"], "water_consumed": item["water_consumed"],
                "food_collected": float(item["food_consumed"] > 0), "water_collected": float(item["water_consumed"] > 0),
                "food_per_step": float(item["food_consumed"] / length), "water_per_step": float(item["water_consumed"] / length),
                "final_hunger": float(hunger[-1]), "final_thirst": float(thirst[-1]),
                "hunger_time_mean": float(hunger.mean()), "thirst_time_mean": float(thirst.mean()),
                "hunger_abs_p95": float(np.percentile(np.abs(hunger), 95)), "thirst_abs_p95": float(np.percentile(np.abs(thirst), 95)),
                "hunger_abs_max": float(np.abs(hunger).max()), "thirst_abs_max": float(np.abs(thirst).max()),
                "hunger_danger_080_fraction": float((np.abs(hunger) >= .8).mean()), "thirst_danger_080_fraction": float((np.abs(thirst) >= .8).mean()),
                "hunger_danger_095_fraction": float((np.abs(hunger) >= .95).mean()), "thirst_danger_095_fraction": float((np.abs(thirst) >= .95).mean()),
                "reward_per_step": float(rewards.mean()), "homeostatic_reward_per_step": float(np.mean(item["homeostatic_reward"])),
                "movement_penalty_per_step": float(np.mean(item["movement_penalty"])), "posture_penalty_per_step": float(np.mean(item["posture_penalty"])),
                "posture_mean": float(posture.mean()), "posture_p95": float(np.percentile(posture, 95)), "posture_max": float(posture.max()),
                "action_magnitude_mean": float(action_mag.mean()), "action_magnitude_p95": float(np.percentile(action_mag, 95)),
                "ever_flipped": float(any(item["flipped"])), "flipped_step_fraction": float(np.mean(item["flipped"])),
                "first_flip_step": float(item["first_flip_step"]) if item["first_flip_step"] >= 0 else float("nan"),
            })
        return episodes, histograms  #, terminal_bootstraps
    finally:
        env.close()


def build_validation(reference_path: Path, cfg: DreamerConfig, seeds: list[int], steps: int, output: Path) -> None:
    _, reference_cfg, world_model, actor, _ = load_models(reference_path, cfg)
    if len(seeds) != reference_cfg.num_workers:
        reference_cfg = replace(reference_cfg, num_workers=len(seeds))
    env = create_env(replace(reference_cfg, is_training=True, max_steps=steps), multiple_env=True)
    try:
        obs, _ = env.reset(seed=seeds)
        workers = len(seeds)
        previous_action = torch.zeros(workers, reference_cfg.action_space_dim, device=reference_cfg.device)
        incoming_reward = np.zeros(workers, dtype=np.float32)
        is_first = torch.ones(workers, device=reference_cfg.device)
        recurrent_state = None
        previous_stochastic = None
        active = np.ones(workers, dtype=bool)
        trajectories = [{key: [] for key in ("vision", "proprioception", "internal_state", "prev_action", "action", "reward", "terminal", "is_first")} for _ in seeds]
        neutral = np.full((workers, reference_cfg.action_space_dim), .5, dtype=np.float32)
        while active.any():
            with torch.inference_mode():
                embed = world_model.encode(obs_to_tensor_dict(obs, reference_cfg))
                latent, recurrent_state, _, _ = world_model.observe(previous_action, embed, is_first, recurrent_state, previous_stochastic, deterministic=True)
                previous_stochastic, recurrent_state = world_model.rssm.split_feature(latent)
                action, _, _ = actor(latent, deterministic=True)
                action = action.cpu().numpy()
            action[~active] = neutral[~active]
            for index in np.flatnonzero(active):
                row = trajectories[index]
                for key in ("vision", "proprioception", "internal_state"):
                    row[key].append(np.array(obs[key][index], copy=True))
                row["prev_action"].append(previous_action[index].detach().cpu().numpy())
                row["action"].append(action[index].copy())
                row["reward"].append(float(incoming_reward[index]))
                row["terminal"].append(False)
                row["is_first"].append(bool(is_first[index].item()))
            next_obs, rewards, terminations, truncations, _ = env.step(action)
            ended = active & (np.asarray(terminations, dtype=bool) | np.asarray(truncations, dtype=bool))
            active[ended] = False
            incoming_reward = np.asarray(rewards, dtype=np.float32)
            previous = action.copy(); previous[~active] = neutral[~active]
            previous_action = to_tensor(previous, reference_cfg.device)
            is_first = torch.zeros(workers, device=reference_cfg.device)
            obs = next_obs
        packed = []
        for seed, row in zip(seeds, trajectories):
            # A final row's incoming reward is unknown because we deliberately
            # stop before crossing an autoreset boundary; discard that row.
            for key in row:
                row[key] = np.asarray(row[key][:-1])
            if len(row["reward"]) >= reference_cfg.batch_length:
                packed.append({"seed": seed, **row})
        if not packed:
            raise RuntimeError("validation trajectories are shorter than batch_length")
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": 1, "reference_checkpoint": str(reference_path), "seeds": seeds,
                    "max_steps": steps, "config": config_to_jsonable(reference_cfg), "trajectories": packed}, output)
    finally:
        env.close()


def validation_losses(world_model: WorldModel, cfg: DreamerConfig, validation_path: Path, seed: int) -> dict[str, float]:
    saved = torch.load(validation_path, map_location="cpu", weights_only=False)
    trajectories = saved.get("trajectories", [])
    chunks = []
    for trajectory in trajectories:
        length = len(trajectory["reward"])
        for start in range(0, length - cfg.batch_length + 1, cfg.batch_length):
            chunks.append({key: trajectory[key][start:start + cfg.batch_length] for key in trajectory if key != "seed"})
    if not chunks:
        raise ValueError("validation set contains no complete batch_length chunks")
    totals: dict[str, float] = {}
    torch.manual_seed(seed)
    with torch.inference_mode():
        for offset in range(0, len(chunks), cfg.batch_size):
            batch = chunks[offset:offset + cfg.batch_size]
            obs = {key: to_tensor(np.stack([row[key] for row in batch]), cfg.device) for key in ("vision", "proprioception", "internal_state")}
            prev_action = to_tensor(np.stack([row["prev_action"] for row in batch]), cfg.device)
            action = to_tensor(np.stack([row["action"] for row in batch]), cfg.device)
            reward = to_tensor(np.stack([row["reward"] for row in batch]), cfg.device)
            terminal = to_tensor(np.stack([row["terminal"] for row in batch]), cfg.device)
            is_first = to_tensor(np.stack([row["is_first"] for row in batch]), cfg.device)
            embed = world_model.encode({key: value.reshape(-1, *value.shape[2:]) for key, value in obs.items()}).reshape(len(batch), cfg.batch_length, -1)
            _, metrics = compute_world_model_loss(world_model.rssm, world_model.decoder, world_model.reward_predictor, world_model.continue_predictor, prev_action, action, embed, obs, is_first, reward, terminal, cfg)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * len(batch)
    return {f"validation/{key.removeprefix('world_model/')}": value / len(chunks) for key, value in totals.items()}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def plot_seed_curves(episodes: list[dict[str, Any]], summaries: list[dict[str, Any]], output: Path) -> None:
    panels = [
        ("survival_length", "Survival length"), ("food_per_step", "Food per step"),
        ("water_per_step", "Water per step"), ("hunger_time_mean", "Mean hunger"),
        ("thirst_time_mean", "Mean thirst"), ("hunger_abs_p95", "Hunger |state| p95"),
        ("thirst_abs_p95", "Thirst |state| p95"), ("posture_mean", "Mean posture error"),
        ("flipped_step_fraction", "Flipped-step fraction"), ("reward_per_step", "Reward per step"),
        ("homeostatic_reward_per_step", "Homeostatic reward / step"), ("action_magnitude_mean", "Mean action magnitude"),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(16, 15), constrained_layout=True)
    steps = sorted({int(row["checkpoint_step"]) for row in episodes})
    for axis, (metric, title) in zip(axes.flat, panels):
        for seed in sorted({int(row["seed"]) for row in episodes}):
            rows = sorted((row for row in episodes if int(row["seed"]) == seed), key=lambda row: row["checkpoint_step"])
            axis.plot([row["checkpoint_step"] for row in rows], [row[metric] for row in rows], color="C0", alpha=.18, linewidth=.8)
        means, lows, highs = [], [], []
        for step in steps:
            values = np.asarray([row[metric] for row in episodes if row["checkpoint_step"] == step], dtype=float)
            means.append(values.mean()); lows.append(np.percentile(values, 10)); highs.append(np.percentile(values, 90))
        axis.fill_between(steps, lows, highs, color="C0", alpha=.22, label="p10–p90")
        axis.plot(steps, means, color="C0", linewidth=2.4, label="mean")
        axis.set_title(title); axis.set_xlabel("training environment step"); axis.grid(alpha=.25)
    fig.suptitle("Dreamer checkpoint behavioural evaluation: fixed seeds", fontsize=15)
    fig.savefig(output, dpi=160); plt.close(fig)


def plot_validation(summaries: list[dict[str, Any]], output: Path) -> None:
    keys = [key for key in summaries[0] if key.startswith("validation/")]
    if not keys:
        return
    columns = 3
    rows_count = math.ceil(len(keys) / columns)
    fig, axes = plt.subplots(rows_count, columns, figsize=(15, 4 * rows_count), constrained_layout=True, squeeze=False)
    for axis, key in zip(axes.flat, keys):
        rows = sorted(summaries, key=lambda row: row["checkpoint_step"])
        axis.plot([row["checkpoint_step"] for row in rows], [row.get(key, np.nan) for row in rows], linewidth=2)
        axis.set_title(key.removeprefix("validation/")); axis.set_xlabel("training environment step"); axis.grid(alpha=.25)
    for axis in axes.flat[len(keys):]: axis.set_visible(False)
    fig.suptitle("Frozen trained-policy validation losses", fontsize=15)
    fig.savefig(output, dpi=160); plt.close(fig)


# def plot_terminal_bootstraps(rows: list[dict[str, Any]], output: Path) -> None:
#     """Plot critic bootstraps only for actual homeostatic ends, not 60k caps."""
#     rows = [row for row in rows if row["homeostatic_ended"]]
#     if not rows:
#         return
#     metrics = [
#         ("critic_final_value", "Critic value at final state"),
#         ("discounted_bootstrap", "Discounted bootstrap"),
#         ("one_step_bootstrapped_target", "One-step bootstrapped target"),
#     ]
#     fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4.5), constrained_layout=True)
#     for axis, (metric, title) in zip(axes, metrics):
#         for seed in sorted({int(row["seed"]) for row in rows}):
#             values = sorted((row for row in rows if int(row["seed"]) == seed), key=lambda row: row["checkpoint_step"])
#             axis.plot([row["checkpoint_step"] for row in values], [row[metric] for row in values], color="C3", alpha=.22, linewidth=.8)
#         for step in sorted({int(row["checkpoint_step"]) for row in rows}):
#             values = np.asarray([row[metric] for row in rows if row["checkpoint_step"] == step], dtype=float)
#             axis.scatter([step], [values.mean()], color="C3", zorder=3)
#             axis.vlines(step, np.percentile(values, 10), np.percentile(values, 90), color="C3", linewidth=2)
#         axis.axhline(0.0, color="black", linewidth=.8); axis.set_title(title)
#         axis.set_xlabel("training environment step"); axis.grid(alpha=.25)
#     fig.suptitle("Homeostatic-death bootstrap diagnostic", fontsize=15)
#     fig.savefig(output, dpi=160); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", nargs="+", help="Checkpoint paths or glob patterns to score.")
    parser.add_argument("--checkpoint-steps", nargs="+", type=int, help="Steps for weight-only checkpoints, matching --checkpoints after glob expansion.")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/dreamer_checkpoints"))
    parser.add_argument("--evaluation-seeds", type=parse_seed_spec, default=parse_seed_spec("1000:1016"))
    parser.add_argument("--eval-max-steps", type=int, default=60_000)
    parser.add_argument("--validation-path", type=Path, default=Path("evaluation/dreamer_validation.pt"))
    parser.add_argument("--build-validation", action="store_true")
    parser.add_argument("--reference-checkpoint", type=Path, help="Trained model used only to collect frozen validation trajectories.")
    parser.add_argument("--validation-seeds", type=parse_seed_spec, default=parse_seed_spec("2000:2016"))
    parser.add_argument("--validation-max-steps", type=int, default=512)
    parser.add_argument("--validation-rng-seed", type=int, default=9876)
    args = parser.parse_args()
    cfg = DreamerConfig()
    if args.build_validation:
        if args.reference_checkpoint is None:
            parser.error("--build-validation requires --reference-checkpoint")
        build_validation(args.reference_checkpoint.resolve(), cfg, args.validation_seeds, args.validation_max_steps, args.validation_path.resolve())
        print(f"Saved frozen validation set to {args.validation_path.resolve()}")
    if not args.checkpoints:
        return
    paths = resolve_checkpoints(args.checkpoints)
    if args.checkpoint_steps and len(args.checkpoint_steps) != len(paths):
        parser.error("--checkpoint-steps must contain one value per expanded checkpoint")
    if not args.validation_path.is_file():
        parser.error("validation set not found; first run with --build-validation")
    episode_rows: list[dict[str, Any]] = []
    # terminal_bootstrap_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    histograms: dict[str, Any] = {}
    checkpoint_configs: dict[str, dict[str, Any]] = {}
    for index, path in enumerate(paths):
        payload, checkpoint_cfg, world_model, actor, critic = load_models(path, cfg)
        # critic check not strictly required if terminal bootstraps are omitted
        # if critic is None:
        #     raise ValueError(f"{path.name} has no critic, required for the terminal-bootstrap diagnostic")
        supplied = args.checkpoint_steps[index] if args.checkpoint_steps else None
        step = checkpoint_step(path, payload, supplied)
        print(f"Evaluating {path.name} at step {step}", flush=True)
        episodes, histogram, bootstraps = evaluate_policy(world_model, actor, critic, checkpoint_cfg, args.evaluation_seeds, args.eval_max_steps)
        for row in episodes:
            row["checkpoint_step"] = step; row["checkpoint_path"] = str(path)
        # for row in bootstraps:
        #     row["checkpoint_step"] = step; row["checkpoint_path"] = str(path)
        summary = summary_row(episodes, step)
        summary.update(histogram["timestep_metrics"])
        summary["checkpoint_path"] = str(path)
        checkpoint_config = config_to_jsonable(checkpoint_cfg)
        # summary["config_sha256"] = hashlib.sha256(json.dumps(checkpoint_config, sort_keys=True).encode()).hexdigest()
        checkpoint_configs[str(step)] = checkpoint_config
        summary.update(validation_losses(world_model, checkpoint_cfg, args.validation_path, args.validation_rng_seed))
        episode_rows.extend(episodes); summaries.append(summary); histograms[str(step)] = histogram
        # terminal_bootstrap_rows.extend(bootstraps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "episodes.csv", episode_rows)
    # write_csv(args.output_dir / "terminal_bootstraps.csv", terminal_bootstrap_rows)
    write_csv(args.output_dir / "checkpoint_summary.csv", summaries)
    (args.output_dir / "state_histograms.json").write_text(json.dumps(histograms, indent=2), encoding="utf-8")
    provenance = {"evaluation_seeds": args.evaluation_seeds, "eval_max_steps": args.eval_max_steps,
                  "validation_path": str(args.validation_path.resolve()), "validation_rng_seed": args.validation_rng_seed,
                  # "checkpoint_sha256": {str(path): sha256_file(path) for path in paths},
                  "checkpoint_config": checkpoint_configs,
                  # "source_sha256": {str(path): sha256_file(path) for path in (Path(__file__), Path("train_dreamer.py"), Path("utils/world_model.py"), Path("utils/utils_dreamer.py"))}
                  }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    plot_seed_curves(episode_rows, summaries, args.output_dir / "behavioural_curves.png")
    plot_validation(summaries, args.output_dir / "validation_losses.png")
    # plot_terminal_bootstraps(terminal_bootstrap_rows, args.output_dir / "terminal_bootstraps.png")
    print(f"Saved evaluation results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

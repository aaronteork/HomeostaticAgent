"""Unified Dreamer world-model reconstruction and prediction audit.

This diagnostic deliberately separates six claims that a single reconstruction
MSE cannot distinguish:

1. fixed-frame capacity under the production and reconstruction-only objectives;
2. held-out posterior reconstruction of natural RGB-D trajectories;
3. matched empty/food/water colour and depth preservation;
4. paired categorical-sample stability rather than sample-mean images alone;
5. one-step action-conditioned prior prediction without observing the target;
6. posterior and prior reward/continuation calibration on controlled events and
   matched near-resource hard negatives.

It writes one JSON report and a small number of RGB/depth panels.  It is a
controlled diagnostic, not an online actor-critic evaluation.

Example:

    python checks/audit_world_model_reconstruction.py \
        --output-dir checks/world_model_reconstruction_audit

To audit an existing compatible world-model checkpoint instead of training the
trajectory model from scratch, pass ``--checkpoint``.  The two tiny capacity
tests still train fresh models because they answer a different question.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config_dreamer import DreamerConfig
from utils.utils_dreamer import compute_world_model_loss, symlog
from utils.utils_env import create_env
from utils.world_model import ActorNetwork, CriticNetwork, WorldModel


OBS_KEYS = ("vision", "proprioception", "internal_state")
RESOURCES = ("empty", "food", "water")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("checks/world_model_reconstruction_audit"))
    parser.add_argument("--checkpoint", type=Path, default=None, help="Skip trajectory training and audit this compatible checkpoint.")
    parser.add_argument(
        "--event-policy-only", action="store_true",
        help="With --checkpoint, run only controlled reward/prior/critic/actor probes; skip capacity training and visual audits.",
    )
    parser.add_argument(
        "--use-sru", choices=("auto", "true", "false"), default="auto",
        help="RSSM recurrence for checkpoints without saved config metadata (default: infer from checkpoint keys).",
    )
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available, otherwise CPU.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trajectory-lengths", type=int, nargs="+", default=[8, 32, 64])
    parser.add_argument("--train-trajectories-per-case", type=int, default=4)
    parser.add_argument("--validation-trajectories-per-case", type=int, default=2)
    parser.add_argument("--train-updates", type=int, default=5000)
    parser.add_argument("--capacity-updates", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--posterior-samples", type=int, default=32)
    parser.add_argument("--matched-seeds", type=int, default=3)
    parser.add_argument("--matched-distances", type=float, nargs="+", default=[1.5, 2.5, 4.0])
    parser.add_argument(
        "--depth-encoding", choices=("raw_opengl", "normalized_metric"), default="raw_opengl",
        help="Meaning of the fourth observation channel. The current live renderer path is raw_opengl.",
    )
    parser.add_argument("--depth-clip-min-m", type=float, default=0.15)
    parser.add_argument("--depth-clip-max-m", type=float, default=8.0)
    parser.add_argument("--frames-per-panel", type=int, default=3)
    parser.add_argument(
        "--policy-rollout-steps", type=int, default=64,
        help="Maximum real-environment steps for each controlled actor probe.",
    )
    parser.add_argument(
        "--policy-rollout-samples", type=int, default=16,
        help="Number of stochastic actor rollouts per unsafe resource case.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.event_policy_only and args.checkpoint is None:
        raise ValueError("--event-policy-only requires --checkpoint.")
    positive = {
        "train trajectories": args.train_trajectories_per_case,
        "validation trajectories": args.validation_trajectories_per_case,
        "train updates": args.train_updates,
        "capacity updates": args.capacity_updates,
        "learning rate": args.learning_rate,
        "posterior samples": args.posterior_samples,
        "matched seeds": args.matched_seeds,
        "frames per panel": args.frames_per_panel,
        "policy rollout steps": args.policy_rollout_steps,
        "policy rollout samples": args.policy_rollout_samples,
    }
    if any(value <= 0 for value in positive.values()):
        raise ValueError(f"All count/rate arguments must be positive: {positive}")
    if any(length < 2 for length in args.trajectory_lengths):
        raise ValueError("Every trajectory length must be at least two for one-step prior evaluation.")
    if any(distance <= 0 for distance in args.matched_distances):
        raise ValueError("Matched distances must be positive.")
    if args.depth_clip_max_m <= args.depth_clip_min_m:
        raise ValueError("--depth-clip-max-m must exceed --depth-clip-min-m.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def base_environment(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def copy_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(obs[key], dtype=np.float32).copy() for key in OBS_KEYS}


def camera_forward_xy(raw) -> np.ndarray:
    axes = raw.data.cam_xmat[raw.pov_camera_id].reshape(3, 3)
    direction = -axes[:, 2][:2].astype(np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        raise RuntimeError("POV camera has no horizontal forward direction.")
    return direction / norm


def projection_metadata(raw) -> dict[str, float]:
    extent = float(raw.model.stat.extent)
    znear = float(raw.model.vis.map.znear)
    zfar = float(raw.model.vis.map.zfar)
    return {
        "model_extent": extent,
        "xml_znear": znear,
        "xml_zfar": zfar,
        "projection_near_m": extent * znear,
        "projection_far_m": extent * zfar,
    }


def place_resource(raw, resource: str, distance: float) -> None:
    if resource == "empty":
        raw.object = []
        return
    ant_xy = raw.data.xpos[raw.ant_body_id][:2].astype(np.float64)
    position = ant_xy + distance * camera_forward_xy(raw)
    raw.object = [(resource, float(position[0]), float(position[1]))]


def capture_matched_triplet(config: DreamerConfig, distance: float, seed: int) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, float]]:
    """Render empty, food, and water without changing pose, vectors, or lighting."""
    env = create_env(config, multiple_env=False)
    try:
        env.reset(seed=seed)
        raw = base_environment(env)
        frames = {}
        for resource in RESOURCES:
            place_resource(raw, resource, distance)
            frames[resource] = copy_obs(raw._get_obs())
        return frames, projection_metadata(raw)
    finally:
        env.close()


def collect_trajectory(config: DreamerConfig, resource: str, length: int, seed: int) -> dict[str, Any]:
    """Collect a seeded native-action trajectory with an initially visible resource."""
    env = create_env(config, multiple_env=False)
    rng = np.random.default_rng(seed)
    try:
        obs, _ = env.reset(seed=seed)
        env.action_space.seed(seed)
        raw = base_environment(env)
        place_resource(raw, resource, 1.75 + 0.25 * (seed % 7))
        obs = raw._get_obs()
        rows, actions, prev_actions, rewards, terminals, firsts = [], [], [], [], [], []
        previous_action = np.zeros(config.action_space_dim, dtype=np.float32)
        incoming_reward, incoming_terminal, is_first = 0.0, False, True
        for step in range(length):
            rows.append(copy_obs(obs))
            prev_actions.append(previous_action.copy())
            rewards.append(incoming_reward)
            terminals.append(incoming_terminal)
            firsts.append(is_first)
            action = rng.uniform(-1.0, 1.0, config.action_space_dim).astype(np.float32)
            actions.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            previous_action = action
            # Production replay treats homeostatic death as an episode/RSSM
            # boundary but not as an absorbing value terminal.  Preserve that
            # distinction instead of feeding the environment's reset signal to
            # the continuation target.
            incoming_reward, incoming_terminal = float(reward), False
            is_first = False
            if terminated or truncated:
                obs, _ = env.reset(seed=seed + step + 1)
                raw = base_environment(env)
                place_resource(raw, resource, 1.75 + 0.25 * (seed % 7))
                obs = raw._get_obs()
                previous_action.fill(0.0)
                incoming_reward, incoming_terminal, is_first = 0.0, False, True
        return {
            "resource": resource, "length": length, "seed": seed, "obs": rows,
            "action": np.stack(actions), "prev_action": np.stack(prev_actions),
            "reward": np.asarray(rewards, np.float32), "terminal": np.asarray(terminals, np.float32),
            "is_first": np.asarray(firsts, np.float32),
        }
    finally:
        env.close()


def tensor_trajectory(trajectory: dict[str, Any], device: torch.device):
    obs = {key: torch.from_numpy(np.stack([row[key] for row in trajectory["obs"]])).to(device) for key in OBS_KEYS}
    return (
        obs,
        torch.from_numpy(trajectory["action"]).unsqueeze(0).to(device),
        torch.from_numpy(trajectory["prev_action"]).unsqueeze(0).to(device),
        torch.from_numpy(trajectory["reward"]).unsqueeze(0).to(device),
        torch.from_numpy(trajectory["terminal"]).unsqueeze(0).to(device),
        torch.from_numpy(trajectory["is_first"]).unsqueeze(0).to(device),
    )


def world_model_loss(model: WorldModel, trajectory: dict[str, Any], config: DreamerConfig):
    obs, action, prev_action, reward, terminal, is_first = tensor_trajectory(trajectory, config.device)
    embed = model.encode(obs).view(1, prev_action.shape[1], -1)
    return compute_world_model_loss(
        model.rssm, model.decoder, model.reward_predictor, model.continue_predictor,
        prev_action, action, embed, {key: value.unsqueeze(0) for key, value in obs.items()},
        is_first, reward, terminal, config,
    )


def train_model(
    model: WorldModel,
    cases: list[dict[str, Any]],
    config: DreamerConfig,
    updates: int,
    learning_rate: float,
    label: str,
    *,
    balanced: bool = False,
) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    model.train()
    for update in range(1, updates + 1):
        selected_cases = cases if balanced else [cases[np.random.randint(len(cases))]]
        optimizer.zero_grad(set_to_none=True)
        evaluations = [world_model_loss(model, case, config) for case in selected_cases]
        loss = torch.stack([item[0] for item in evaluations]).mean()
        metrics = {
            key: float(np.mean([item[1][key] for item in evaluations]))
            for key in evaluations[0][1]
        }
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()
        if update == 1 or update % max(1, updates // 20) == 0:
            row = {"update": update, "total_loss": float(loss.item()), "grad_norm": float(grad_norm), **metrics}
            history.append(row)
            print(f"{label}: {update}/{updates} loss={loss.item():.6g}", flush=True)
    return history


def checkpoint_state(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("world_model", "world_model_state_dict"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint does not contain a world-model state dictionary.")
    state = {}
    for key, value in payload.items():
        key = key.replace("._orig_mod.", ".").removeprefix("_orig_mod.").removeprefix("world_model.")
        state[key] = value
    return state


def component_state(payload: Any, name: str) -> dict[str, torch.Tensor]:
    """Extract one compiled-or-uncompiled checkpoint component."""
    if not isinstance(payload, dict) or name not in payload:
        raise ValueError(
            f"Checkpoint has no '{name}' state dictionary. "
            "Use a full Dreamer checkpoint saved by train_dreamer.py."
        )
    component = payload[name]
    if not isinstance(component, dict):
        raise TypeError(f"Checkpoint component '{name}' is not a state dictionary.")
    return {
        key.replace("._orig_mod.", ".").removeprefix("_orig_mod.").removeprefix(f"{name}."): value
        for key, value in component.items()
    }


def load_model(device: torch.device, path: Path, use_sru: str = "auto") -> tuple[WorldModel, DreamerConfig]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    saved = payload.get("config") if isinstance(payload, dict) else None
    values = {}
    if isinstance(saved, dict):
        valid = {field.name for field in fields(DreamerConfig)}
        values = {key: value for key, value in saved.items() if key in valid and key not in ("device", "use_amp", "is_training")}
    state = checkpoint_state(payload)
    inferred_sru = any("transform_gate" in key for key in state)
    if use_sru != "auto":
        inferred_sru = use_sru == "true"
    values["use_sru"] = inferred_sru
    config = replace(DreamerConfig(**values), device=device, use_amp=False, is_training=False)
    model = WorldModel(config).to(config.device)
    model.load_state_dict(state, strict=True)
    return model.eval(), config


def load_actor_critic(device: torch.device, path: Path, config: DreamerConfig) -> tuple[ActorNetwork, CriticNetwork]:
    """Load the policy and critic needed for controlled checkpoint probes."""
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    actor = ActorNetwork(config).to(device)
    critic = CriticNetwork(config).to(device)
    actor.load_state_dict(component_state(payload, "actor"), strict=True)
    critic.load_state_dict(component_state(payload, "critic"), strict=True)
    return actor.eval(), critic.eval()


@torch.no_grad()
def posterior_sequence(model: WorldModel, trajectory: dict[str, Any], deterministic: bool = True):
    obs, _, prev_action, _, _, is_first = tensor_trajectory(trajectory, model.config.device)
    embed = model.encode(obs).view(1, prev_action.shape[1], -1)
    latent, _, priors, posteriors = model.observe(prev_action, embed, is_first, deterministic=deterministic)
    return obs, latent, model.decode(latent), priors, posteriors


def squared_metrics(prediction: np.ndarray, target: np.ndarray, baseline_mse: float | None = None) -> dict[str, float | None]:
    error = prediction.astype(np.float64) - target.astype(np.float64)
    mse = float(np.mean(error ** 2))
    variance = float(np.var(target.astype(np.float64)))
    return {
        "mse": mse,
        "mae": float(np.mean(np.abs(error))),
        "target_mean": float(np.mean(target)),
        "target_std": float(np.std(target)),
        "explained_variance": None if variance <= 1e-12 else 1.0 - mse / variance,
        "relative_to_baseline": None if baseline_mse is None or baseline_mse <= 1e-12 else mse / baseline_mse,
    }


def training_baselines(trajectories: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    vision = np.concatenate([np.stack([row["vision"] for row in trajectory["obs"]]) for trajectory in trajectories])
    result = {
        "vision_channel_mean": vision.mean(axis=(0, 2, 3), keepdims=True),
        "vision_pixel_mean": vision.mean(axis=0, keepdims=True),
    }
    for key in ("proprioception", "internal_state"):
        values = np.concatenate([np.stack([row[key] for row in trajectory["obs"]]) for trajectory in trajectories])
        transformed = np.sign(values) * np.log1p(np.abs(values))
        result[f"{key}_symlog_mean"] = transformed.mean(axis=0, keepdims=True)
    return result


def depth_to_metres(depth: np.ndarray, args: argparse.Namespace, projection: dict[str, float]) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float64)
    if args.depth_encoding == "normalized_metric":
        return args.depth_clip_min_m + np.clip(depth, 0.0, 1.0) * (args.depth_clip_max_m - args.depth_clip_min_m)
    near, far = projection["projection_near_m"], projection["projection_far_m"]
    value = np.clip(depth, 0.0, 1.0)
    return near * far / np.maximum(far - value * (far - near), 1e-12)


@torch.no_grad()
def held_out_posterior_audit(model: WorldModel, trajectories: list[dict[str, Any]], baselines: dict[str, np.ndarray], args, projection) -> dict[str, Any]:
    targets, predictions = [], []
    vector_targets = {key: [] for key in ("proprioception", "internal_state")}
    vector_predictions = {key: [] for key in vector_targets}
    per_case = []
    for trajectory in trajectories:
        obs, _, decoded, _, _ = posterior_sequence(model, trajectory, deterministic=True)
        target = obs["vision"].cpu().numpy()
        prediction = decoded["vision"][0].float().cpu().numpy()
        channel_base = np.broadcast_to(baselines["vision_channel_mean"], target.shape)
        pixel_base = np.broadcast_to(baselines["vision_pixel_mean"], target.shape)
        case = {"resource": trajectory["resource"], "length": trajectory["length"], "seed": trajectory["seed"], "heads": {}}
        for name, channels in (("rgb", slice(0, 3)), ("depth", slice(3, 4)), ("rgbd", slice(0, 4))):
            mean_mse = float(np.mean((channel_base[:, channels] - target[:, channels]) ** 2))
            pixel_mse = float(np.mean((pixel_base[:, channels] - target[:, channels]) ** 2))
            metrics = squared_metrics(prediction[:, channels], target[:, channels], min(mean_mse, pixel_mse))
            metrics.update({"channel_mean_baseline_mse": mean_mse, "pixel_mean_baseline_mse": pixel_mse})
            if len(target) > 1:
                metrics["previous_frame_baseline_mse"] = float(np.mean((target[:-1, channels] - target[1:, channels]) ** 2))
            case["heads"][name] = metrics
        for key in vector_targets:
            vector_target = symlog(obs[key]).cpu().numpy()
            vector_prediction = decoded[key][0].float().cpu().numpy()
            mean_base = np.broadcast_to(baselines[f"{key}_symlog_mean"], vector_target.shape)
            mean_mse = float(np.mean((mean_base - vector_target) ** 2))
            vector_metrics = squared_metrics(vector_prediction, vector_target, mean_mse)
            vector_metrics["training_mean_baseline_mse"] = mean_mse
            if len(vector_target) > 1:
                vector_metrics["previous_frame_baseline_mse"] = float(np.mean((vector_target[:-1] - vector_target[1:]) ** 2))
            case["heads"][key] = vector_metrics
            vector_targets[key].append(vector_target)
            vector_predictions[key].append(vector_prediction)
        target_m = depth_to_metres(target[:, 3], args, projection)
        prediction_m = depth_to_metres(prediction[:, 3], args, projection)
        case["heads"]["depth_metres"] = squared_metrics(prediction_m, target_m)
        per_case.append(case)
        targets.append(target); predictions.append(prediction)
    target = np.concatenate(targets); prediction = np.concatenate(predictions)
    aggregate = {}
    for name, channels in (("rgb", slice(0, 3)), ("depth", slice(3, 4)), ("rgbd", slice(0, 4))):
        channel_base = np.broadcast_to(baselines["vision_channel_mean"], target.shape)[:, channels]
        pixel_base = np.broadcast_to(baselines["vision_pixel_mean"], target.shape)[:, channels]
        baseline_mse = min(float(np.mean((channel_base - target[:, channels]) ** 2)), float(np.mean((pixel_base - target[:, channels]) ** 2)))
        aggregate[name] = squared_metrics(prediction[:, channels], target[:, channels], baseline_mse)
    aggregate["depth_background_fraction_at_0_999"] = float(np.mean(target[:, 3] >= 0.999))
    for key in vector_targets:
        vector_target = np.concatenate(vector_targets[key])
        vector_prediction = np.concatenate(vector_predictions[key])
        mean_base = np.broadcast_to(baselines[f"{key}_symlog_mean"], vector_target.shape)
        mean_mse = float(np.mean((mean_base - vector_target) ** 2))
        aggregate[key] = squared_metrics(vector_prediction, vector_target, mean_mse)
        aggregate[key]["training_mean_baseline_mse"] = mean_mse
    return {"aggregate": aggregate, "cases": per_case}


@torch.no_grad()
def one_step_prior_audit(model: WorldModel, trajectories: list[dict[str, Any]], baselines: dict[str, np.ndarray], args, projection) -> dict[str, Any]:
    targets, predictions, copies, rewards_target, rewards_pred, cont_target, cont_pred = [], [], [], [], [], [], []
    vector_targets = {key: [] for key in ("proprioception", "internal_state")}
    vector_predictions = {key: [] for key in vector_targets}
    vector_copies = {key: [] for key in vector_targets}
    for trajectory in trajectories:
        obs, latent, _, _, _ = posterior_sequence(model, trajectory, deterministic=True)
        actions = trajectory["action"]
        for index in range(len(actions) - 1):
            if trajectory["is_first"][index + 1]:
                continue
            current = latent[:, index]
            _, recurrent = model.rssm.split_feature(current)
            action = torch.from_numpy(actions[index:index + 1]).to(model.config.device)
            prior_latent, _, _ = model.imagine_step(current, recurrent, action, deterministic=True)
            decoded_heads = model.decode(prior_latent)
            decoded = decoded_heads["vision"][0].float().cpu().numpy()
            target = obs["vision"][index + 1].cpu().numpy()
            targets.append(target); predictions.append(decoded); copies.append(obs["vision"][index].cpu().numpy())
            for key in vector_targets:
                vector_targets[key].append(symlog(obs[key][index + 1]).cpu().numpy())
                vector_copies[key].append(symlog(obs[key][index]).cpu().numpy())
                vector_predictions[key].append(decoded_heads[key][0].float().cpu().numpy())
            rewards_target.append(float(trajectory["reward"][index + 1]))
            rewards_pred.append(float(model.predict_reward(prior_latent).mode.detach().cpu().reshape(-1)[0]))
            terminal = float(trajectory["terminal"][index + 1])
            cont_target.append((1.0 - terminal) * model.config.discount if model.config.contdisc else 1.0 - terminal)
            cont_pred.append(float(model.predict_continue(prior_latent).cpu().reshape(-1)[0]))
    target = np.stack(targets); prediction = np.stack(predictions); copy_base = np.stack(copies)
    report = {"count": len(target), "heads": {}}
    for name, channels in (("rgb", slice(0, 3)), ("depth", slice(3, 4)), ("rgbd", slice(0, 4))):
        pixel_base = np.broadcast_to(baselines["vision_pixel_mean"][0], target.shape)
        base_mse = min(float(np.mean((copy_base[:, channels] - target[:, channels]) ** 2)), float(np.mean((pixel_base[:, channels] - target[:, channels]) ** 2)))
        metrics = squared_metrics(prediction[:, channels], target[:, channels], base_mse)
        metrics["previous_frame_baseline_mse"] = float(np.mean((copy_base[:, channels] - target[:, channels]) ** 2))
        metrics["pixel_mean_baseline_mse"] = float(np.mean((pixel_base[:, channels] - target[:, channels]) ** 2))
        report["heads"][name] = metrics
    report["heads"]["depth_metres"] = squared_metrics(depth_to_metres(prediction[:, 3], args, projection), depth_to_metres(target[:, 3], args, projection))
    for key in vector_targets:
        vector_target = np.stack(vector_targets[key])
        vector_prediction = np.stack(vector_predictions[key])
        vector_copy = np.stack(vector_copies[key])
        mean_base = np.broadcast_to(baselines[f"{key}_symlog_mean"], vector_target.shape)
        copy_mse = float(np.mean((vector_copy - vector_target) ** 2))
        mean_mse = float(np.mean((mean_base - vector_target) ** 2))
        metrics = squared_metrics(vector_prediction, vector_target, min(copy_mse, mean_mse))
        metrics["previous_frame_baseline_mse"] = copy_mse
        metrics["training_mean_baseline_mse"] = mean_mse
        report["heads"][key] = metrics
    report["reward"] = squared_metrics(np.asarray(rewards_pred), np.asarray(rewards_target))
    report["continuation"] = squared_metrics(np.asarray(cont_pred), np.asarray(cont_target))
    return report


def resource_mask(empty_rgb: np.ndarray, resource_rgb: np.ndarray) -> np.ndarray | None:
    difference = np.max(np.abs(resource_rgb - empty_rgb), axis=0)
    threshold = max(0.02, float(difference.max()) * 0.15)
    mask = difference >= threshold
    if not mask.any():
        return None
    return mask


@torch.no_grad()
def paired_posterior(model: WorldModel, observation_np: dict[str, np.ndarray], base_seed: int, samples: int) -> dict[str, np.ndarray]:
    observation = {key: torch.from_numpy(observation_np[key][None]).to(model.config.device) for key in OBS_KEYS}
    embed = model.encode(observation)
    action = torch.zeros(1, model.config.action_space_dim, device=model.config.device)
    first = torch.ones(1, device=model.config.device)
    _, _, _, posteriors = model.observe(action, embed, first, deterministic=True)
    decoded = []
    for index in range(samples):
        torch.manual_seed(base_seed + index)
        if model.config.device.type == "cuda":
            torch.cuda.manual_seed_all(base_seed + index)
        latent, _, _, _ = model.observe(action, embed, first, deterministic=False)
        decoded.append(model.decode(latent)["vision"][0].float().cpu().numpy())
    return {
        "embedding": embed[0].float().cpu().numpy(),
        "posterior_logits": posteriors[0].base_dist.logits[0].float().cpu().numpy(),
        "samples": np.stack(decoded),
    }


def masked_values(images: np.ndarray, mask: np.ndarray, channels: slice | int) -> np.ndarray:
    if images.ndim == 3:
        return images[channels][:, mask] if isinstance(channels, slice) else images[channels][mask]
    return images[:, channels, :, :][..., mask] if isinstance(channels, slice) else images[:, channels, :, :][..., mask]


def dominant_correct(samples: np.ndarray, mask: np.ndarray, resource: str) -> float:
    means = samples[:, :3, :, :][:, :, mask].mean(axis=2)
    expected_channel = {"food": 0, "water": 2}.get(resource)
    if expected_channel is None:
        raise ValueError(f"No expected dominant colour for resource {resource!r}.")
    return float(np.mean(np.argmax(means, axis=1) == expected_channel))


def matched_probe_metrics(model: WorldModel, frames, args, projection, sample_seed: int) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    recon = {name: paired_posterior(model, frames[name], sample_seed, args.posterior_samples) for name in RESOURCES}
    masks = {resource: resource_mask(frames["empty"]["vision"][:3], frames[resource]["vision"][:3]) for resource in ("food", "water")}
    invisible = [resource for resource, mask in masks.items() if mask is None]
    if invisible:
        return {
            "available": False,
            "reason": "resource_not_visible",
            "invisible_resources": invisible,
        }, recon
    masks = {resource: mask for resource, mask in masks.items() if mask is not None}
    union = masks["food"] | masks["water"]
    report: dict[str, Any] = {"available": True, "resource_pixels": {key: int(value.sum()) for key, value in masks.items()}, "per_resource": {}}
    for resource in ("food", "water"):
        mask = masks[resource]
        target = frames[resource]["vision"]
        target_empty = frames["empty"]["vision"]
        decoded = recon[resource]["samples"]
        decoded_empty = recon["empty"]["samples"]
        rgb_target_contrast = float(np.mean(np.abs(masked_values(target, mask, slice(0, 3)) - masked_values(target_empty, mask, slice(0, 3)))))
        rgb_sample_contrast = np.mean(np.abs(masked_values(decoded, mask, slice(0, 3)) - masked_values(decoded_empty, mask, slice(0, 3))), axis=(1, 2))
        rgb_sample_error = np.mean(np.abs(masked_values(decoded, mask, slice(0, 3)) - masked_values(target, mask, slice(0, 3))[None]), axis=(1, 2))
        background = ~mask
        rgb_background_error = np.mean(np.abs(masked_values(decoded, background, slice(0, 3)) - masked_values(target, background, slice(0, 3))[None]), axis=(1, 2))
        depth_target_m = depth_to_metres(masked_values(target, mask, 3), args, projection)
        depth_decoded_m = depth_to_metres(masked_values(decoded, mask, 3), args, projection)
        report["per_resource"][resource] = {
            "target_vs_empty_rgb_mae": rgb_target_contrast,
            "decoded_vs_decoded_empty_rgb_mae_mean": float(rgb_sample_contrast.mean()),
            "decoded_vs_decoded_empty_rgb_mae_std": float(rgb_sample_contrast.std()),
            "decoded_to_target_rgb_mae_mean": float(rgb_sample_error.mean()),
            "decoded_to_target_rgb_mae_std": float(rgb_sample_error.std()),
            "decoded_to_target_background_rgb_mae_mean": float(rgb_background_error.mean()),
            "correct_dominant_colour_fraction": dominant_correct(decoded, mask, resource),
            "depth_mae_m_mean": float(np.mean(np.abs(depth_decoded_m - depth_target_m[None]))),
            "encoder_l2_from_empty": float(np.linalg.norm(recon[resource]["embedding"] - recon["empty"]["embedding"])),
            "posterior_logit_l2_from_empty": float(np.linalg.norm(recon[resource]["posterior_logits"] - recon["empty"]["posterior_logits"])),
        }
    target_pair = np.abs(masked_values(frames["food"]["vision"], union, slice(0, 3)) - masked_values(frames["water"]["vision"], union, slice(0, 3))).mean()
    decoded_pair = np.mean(np.abs(masked_values(recon["food"]["samples"], union, slice(0, 3)) - masked_values(recon["water"]["samples"], union, slice(0, 3))), axis=(1, 2))
    report["food_vs_water"] = {
        "target_rgb_mae": float(target_pair),
        "paired_sample_rgb_mae_mean": float(decoded_pair.mean()),
        "paired_sample_rgb_mae_std": float(decoded_pair.std()),
        "paired_sample_rgb_mae_values": [float(value) for value in decoded_pair],
        "colour_separation_ratio": None if target_pair <= 0 else float(decoded_pair.mean() / target_pair),
    }
    return report, recon


def save_matched_panel(cases: list[tuple[float, dict[str, Any], dict[str, Any]]], path: Path) -> None:
    """Save matched target/posterior rows for every configured distance."""
    entries = []
    for distance, frames, recon in cases:
        for resource in RESOURCES:
            prefix = f"d={distance:g}m {resource}"
            entries.extend([
                (f"{prefix} RGB target", frames[resource]["vision"][:3], False),
                (f"{prefix} RGB posterior mean", recon[resource]["samples"].mean(axis=0)[:3], False),
                (f"{prefix} depth target", frames[resource]["vision"][3], True),
                (f"{prefix} depth posterior mean", recon[resource]["samples"].mean(axis=0)[3], True),
            ])
    panel = Image.new("RGB", (256 * 4, 286 * len(RESOURCES) * len(cases)), "white")
    draw = ImageDraw.Draw(panel)
    for index, (label, array, depth) in enumerate(entries):
        row, column = divmod(index, 4)
        if depth:
            image = Image.fromarray((np.clip(array, 0, 1) * 255).round().astype(np.uint8)).convert("RGB")
        else:
            image = Image.fromarray((np.clip(array.transpose(1, 2, 0), 0, 1) * 255).round().astype(np.uint8))
        panel.paste(image.resize((256, 256)), (column * 256, row * 286 + 30))
        draw.text((column * 256 + 5, row * 286 + 6), label, fill="black")
    panel.save(path)


@torch.no_grad()
def save_held_out_panel(model: WorldModel, trajectory: dict[str, Any], path: Path, frame_count: int) -> None:
    """Save target, posterior, and genuinely predictive one-step-prior views."""
    obs, latent, posterior, _, _ = posterior_sequence(model, trajectory, deterministic=True)
    valid = [index for index in range(1, trajectory["length"]) if not trajectory["is_first"][index]]
    if not valid:
        return
    selected = np.unique(np.linspace(0, len(valid) - 1, min(frame_count, len(valid)), dtype=int))
    frame_indices = [valid[index] for index in selected]
    rows = []
    for frame in frame_indices:
        current = latent[:, frame - 1]
        _, recurrent = model.rssm.split_feature(current)
        action = torch.from_numpy(trajectory["action"][frame - 1:frame]).to(model.config.device)
        prior_latent, _, _ = model.imagine_step(current, recurrent, action, deterministic=True)
        prior = model.decode(prior_latent)["vision"][0].float().cpu().numpy()
        rows.append((
            frame,
            obs["vision"][frame].float().cpu().numpy(),
            posterior["vision"][0, frame].float().cpu().numpy(),
            prior,
        ))
    labels = ("RGB target", "RGB posterior", "RGB one-step prior", "depth target", "depth posterior", "depth one-step prior")
    panel = Image.new("RGB", (256 * len(labels), 286 * len(rows)), "white")
    draw = ImageDraw.Draw(panel)
    for row_index, (frame, target, post, prior) in enumerate(rows):
        arrays = (target[:3], post[:3], prior[:3], target[3], post[3], prior[3])
        for column, (label, array) in enumerate(zip(labels, arrays)):
            if array.ndim == 2:
                image = Image.fromarray((np.clip(array, 0, 1) * 255).round().astype(np.uint8)).convert("RGB")
            else:
                image = Image.fromarray((np.clip(array.transpose(1, 2, 0), 0, 1) * 255).round().astype(np.uint8))
            x, y = column * 256, row_index * 286
            panel.paste(image.resize((256, 256)), (x, y + 30))
            draw.text((x + 5, y + 6), f"t={frame} {label}", fill="black")
    panel.save(path)


def make_one_step_case(frames: dict[str, dict[str, np.ndarray]], name: str, action_dim: int) -> dict[str, Any]:
    observation = [copy_obs(frames[name])]
    return {
        "resource": name, "length": 1, "seed": 0, "obs": observation,
        "action": np.zeros((1, action_dim), np.float32), "prev_action": np.zeros((1, action_dim), np.float32),
        "reward": np.zeros(1, np.float32), "terminal": np.zeros(1, np.float32), "is_first": np.ones(1, np.float32),
    }


def capacity_audit(config: DreamerConfig, frames, args, projection) -> dict[str, Any]:
    cases = [make_one_step_case(frames, name, config.action_space_dim) for name in RESOURCES]
    results = {}
    objectives = {
        "production": config,
        "reconstruction_only": replace(config, dyn_loss_weight=0.0, rep_loss_weight=0.0, reward_weight=0.0, continue_weight=0.0),
    }
    for offset, (name, objective_config) in enumerate(objectives.items()):
        set_seed(args.seed + 1000 + offset)
        model = WorldModel(objective_config).to(objective_config.device)
        # This is a fixed three-frame capacity test, so every optimizer update
        # must see empty, food, and water equally. Random single-case updates
        # add avoidable exposure imbalance and make the comparison noisy.
        history = train_model(
            model,
            cases,
            objective_config,
            args.capacity_updates,
            args.learning_rate,
            f"capacity/{name}",
            balanced=True,
        )
        metrics, _ = matched_probe_metrics(model.eval(), frames, args, projection, args.seed + 2000)
        results[name] = {"history": history, "matched": metrics}
    return results


def set_orientation(raw, roll: float) -> None:
    raw.data.qpos[3:7] = np.array([math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0])
    mujoco.mj_forward(raw.model, raw.data)


def capture_event(config: DreamerConfig, name: str, seed: int) -> dict[str, Any]:
    """Create a controlled transition for reward/continuation calibration."""
    env = create_env(config, multiple_env=False)
    try:
        env.reset(seed=seed)
        raw = base_environment(env)
        raw.object = []
        distance = None
        if name.startswith("beneficial_food"):
            raw.hunger, distance = -0.5, 0.75
        elif name.startswith("harmful_food"):
            raw.hunger, distance = 0.85, 0.75
        elif name.startswith("lethal_food"):
            raw.hunger, distance = 0.95, 0.75
        elif name.startswith("beneficial_water"):
            raw.thirst, distance = -0.5, 0.75
        elif name.startswith("harmful_water"):
            raw.thirst, distance = 0.85, 0.75
        elif name.startswith("lethal_water"):
            raw.thirst, distance = 0.95, 0.75
        elif name == "food_near_hard_negative":
            raw.hunger, distance = -0.5, float(config.object_interaction_dist) + 0.15
        elif name == "water_near_hard_negative":
            raw.thirst, distance = -0.5, float(config.object_interaction_dist) + 0.15
        elif name == "deprivation_death":
            raw.hunger = -0.99999
        elif name == "high_posture":
            set_orientation(raw, 1.2)
        elif name == "flipped":
            set_orientation(raw, math.pi)
        else:
            raise ValueError(name)
        if "food" in name and distance is not None:
            place_resource(raw, "food", distance)
        elif "water" in name and distance is not None:
            place_resource(raw, "water", distance)
        raw.prev_drive = raw._calculate_drive()
        before = copy_obs(raw._get_obs())
        # ``create_env()`` applies ``RescaleAction(0, 1)``.  A wrapper action
        # of 0.5 maps to neutral native torque, whereas 0.0 would command
        # maximum negative torque at every joint and confound this
        # resource-consumption probe with locomotion/posture effects.
        action = np.full(config.action_space_dim, 0.5, dtype=np.float32)
        after, reward, terminated, truncated, info = env.step(action)
        return {
            "name": name, "before": before, "after": copy_obs(after), "action": action,
            "reward": float(reward), "terminal": bool(terminated), "truncated": bool(truncated),
            "food_consumed": int(info["food_consumed"]), "water_consumed": int(info["water_consumed"]),
            "is_flipped": bool(info["is_flipped"]), "posture": float(info["posture"]),
            "termination_reason": int(info["termination_reason"]),
        }
    finally:
        env.close()


@torch.no_grad()
def event_prediction(model: WorldModel, event: dict[str, Any], critic: CriticNetwork | None = None) -> dict[str, Any]:
    device = model.config.device
    before = {key: torch.from_numpy(event["before"][key][None]).to(device) for key in OBS_KEYS}
    after = {key: torch.from_numpy(event["after"][key][None]).to(device) for key in OBS_KEYS}
    zero = torch.zeros(1, model.config.action_space_dim, device=device)
    first = torch.ones(1, device=device)
    before_embed = model.encode(before)
    current, _, _, _ = model.observe(zero, before_embed, first, deterministic=True)
    _, recurrent = model.rssm.split_feature(current)
    action = torch.from_numpy(event["action"][None]).to(device)
    prior_next, _, _ = model.imagine_step(current, recurrent, action, deterministic=True)
    after_embed = model.encode(after)
    posterior_next, _, _, _ = model.observe(action, after_embed, torch.zeros_like(first), recurrent_state=recurrent, previous_stochastic=model.rssm.split_feature(current)[0], deterministic=True)
    # Match production replay: a homeostatic reset is not a value terminal.
    value_terminal = 0.0
    target_continue = (1.0 - value_terminal) * model.config.discount if model.config.contdisc else 1.0 - value_terminal
    result = {
        key: event[key] for key in ("name", "reward", "terminal", "truncated", "food_consumed", "water_consumed", "is_flipped", "posture", "termination_reason")
    }
    if critic is not None:
        result["critic_value_before_action"] = float(critic(current).mode.cpu().reshape(-1)[0])
    result["target_continue"] = target_continue
    result["value_terminal_target"] = bool(value_terminal)
    for label, latent in (("prior", prior_next), ("posterior", posterior_next)):
        result[label] = {
            "predicted_reward": float(model.predict_reward(latent).mode.cpu().reshape(-1)[0]),
            "predicted_continue": float(model.predict_continue(latent).cpu().reshape(-1)[0]),
        }
    return result


def configure_resource_event(raw, name: str, *, distance: float) -> None:
    """Set a single resource and the internal state for an unsafe-contact case."""
    raw.object = []
    if name.endswith("food"):
        raw.hunger = -0.5 if name.startswith("beneficial") else (0.85 if name.startswith("harmful") else 0.95)
        place_resource(raw, "food", distance)
    elif name.endswith("water"):
        raw.thirst = -0.5 if name.startswith("beneficial") else (0.85 if name.startswith("harmful") else 0.95)
        place_resource(raw, "water", distance)
    else:
        raise ValueError(f"Expected a food or water event, got {name!r}")
    raw.prev_drive = raw._calculate_drive()


def actor_summary(actor: ActorNetwork, latent: torch.Tensor) -> dict[str, Any]:
    """Expose the policy distribution rather than only a sampled motor command."""
    mode, _, dist = actor(latent, deterministic=True)
    base = dist.base_dist
    return {
        "mode": mode.detach().cpu().reshape(-1).tolist(),
        "mean": base.mean.detach().cpu().reshape(-1).tolist(),
        "stddev": base.stddev.detach().cpu().reshape(-1).tolist(),
        "mean_abs": float(base.mean.abs().mean().cpu()),
        "stddev_mean": float(base.stddev.mean().cpu()),
    }


@torch.no_grad()
def policy_rollout(
    model: WorldModel,
    actor: ActorNetwork,
    critic: CriticNetwork,
    config: DreamerConfig,
    name: str,
    seed: int,
    steps: int,
    deterministic: bool,
) -> dict[str, Any]:
    """Run one actor rollout from just outside automatic-consumption range.

    Unlike ``capture_event``, this leaves room for the actor to turn away. It
    records whether the policy enters the resource zone and whether that causes
    the positive-bound homeostatic death under investigation.
    """
    env = create_env(config, multiple_env=False)
    try:
        obs, _ = env.reset(seed=seed)
        raw = base_environment(env)
        configure_resource_event(raw, name, distance=config.object_interaction_dist + 0.25)
        obs = copy_obs(raw._get_obs())
        device = config.device
        prev_action = torch.zeros(1, config.action_space_dim, device=device)
        is_first = torch.ones(1, device=device)
        recurrent = None
        stochastic = None
        reward_sum = 0.0
        first_actor = None
        first_value = None
        consumed = False
        terminated = False
        info: dict[str, Any] = {}
        for step in range(steps):
            tensor_obs = {key: torch.from_numpy(obs[key][None]).to(device) for key in OBS_KEYS}
            embed = model.encode(tensor_obs)
            latent, recurrent, _, _ = model.observe(
                prev_action, embed, is_first, recurrent_state=recurrent,
                previous_stochastic=stochastic, deterministic=True,
            )
            if first_actor is None:
                first_actor = actor_summary(actor, latent)
                first_value = float(critic(latent).mode.cpu().reshape(-1)[0])
            action, _, _ = actor(latent, deterministic=deterministic)
            action_np = action.detach().cpu().numpy()[0].astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action_np)
            reward_sum += float(reward)
            consumed = consumed or bool(int(info["food_consumed"]) + int(info["water_consumed"]))
            if terminated or truncated:
                break
            stochastic, _ = model.rssm.split_feature(latent)
            prev_action = action
            is_first = torch.zeros(1, device=device)
        hunger = float(info["hunger"]) if info else float(raw.hunger)
        thirst = float(info["thirst"]) if info else float(raw.thirst)
        return {
            "deterministic": deterministic,
            "steps": step + 1,
            "consumed_resource": consumed,
            "terminated": bool(terminated),
            "positive_bound_death": bool(terminated and (hunger >= 0.99 or thirst >= 0.99)),
            "reward_sum": reward_sum,
            "final_hunger": hunger,
            "final_thirst": thirst,
            "first_critic_value": first_value,
            "first_actor": first_actor,
        }
    finally:
        env.close()


def policy_calibration(
    model: WorldModel, actor: ActorNetwork, critic: CriticNetwork,
    config: DreamerConfig, seed: int, steps: int, samples: int,
) -> dict[str, Any]:
    """Measure whether policy samples approach dangerous resources from matched states."""
    cases = ("harmful_food", "lethal_food", "harmful_water", "lethal_water")
    report: dict[str, Any] = {}
    for offset, name in enumerate(cases):
        deterministic = policy_rollout(model, actor, critic, config, name, seed + offset * 10_000, steps, True)
        stochastic = [
            policy_rollout(model, actor, critic, config, name, seed + offset * 10_000 + sample + 1, steps, False)
            for sample in range(samples)
        ]
        report[name] = {
            "deterministic": deterministic,
            "stochastic_summary": {
                "trials": samples,
                "consumption_fraction": float(np.mean([row["consumed_resource"] for row in stochastic])),
                "termination_fraction": float(np.mean([row["terminated"] for row in stochastic])),
                "positive_bound_death_fraction": float(np.mean([row["positive_bound_death"] for row in stochastic])),
                "reward_sum_mean": float(np.mean([row["reward_sum"] for row in stochastic])),
            },
            "stochastic_rollouts": stochastic,
        }
    return report


def event_calibration(model: WorldModel, config: DreamerConfig, seed: int, critic: CriticNetwork | None = None) -> dict[str, Any]:
    names = (
        "beneficial_food", "beneficial_water", "harmful_food", "harmful_water",
        "lethal_food", "lethal_water", "food_near_hard_negative", "water_near_hard_negative",
        "deprivation_death", "high_posture", "flipped",
    )
    events = [capture_event(config, name, seed + index) for index, name in enumerate(names)]
    rows = [event_prediction(model, event, critic) for event in events]
    reward_target = np.asarray([row["reward"] for row in rows])
    continue_target = np.asarray([row["target_continue"] for row in rows])
    aggregate = {}
    for label in ("prior", "posterior"):
        aggregate[label] = {
            "reward": squared_metrics(np.asarray([row[label]["predicted_reward"] for row in rows]), reward_target),
            "continuation": squared_metrics(np.asarray([row[label]["predicted_continue"] for row in rows]), continue_target),
        }
    return {"aggregate": aggregate, "cases": rows}


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def source_hashes() -> dict[str, str]:
    """Record the exact local sources that define this diagnostic and model."""
    relative_paths = (
        "checks/audit_world_model_reconstruction.py",
        "configs/config_dreamer.py",
        "envs/ant_env.py",
        "envs/ant_env.xml",
        "utils/utils_dreamer.py",
        "utils/world_model.py",
    )
    return {
        relative_path: hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        for relative_path in relative_paths
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = replace(DreamerConfig(device=device, shift=True, is_training=False, num_food=0, num_water=0), use_amp=False)
    actor = None
    critic = None
    if args.checkpoint is not None:
        model, config = load_model(device, args.checkpoint.resolve(), args.use_sru)
        try:
            actor, critic = load_actor_critic(device, args.checkpoint.resolve(), config)
        except (TypeError, ValueError, RuntimeError) as error:
            print(f"Actor/critic audit unavailable: {error}", flush=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.event_policy_only:
        events = event_calibration(model, config, args.seed + 50_000, critic)
        policy = None
        if actor is not None and critic is not None:
            policy = policy_calibration(
                model, actor, critic, config, args.seed + 60_000,
                args.policy_rollout_steps, args.policy_rollout_samples,
            )
        report = {
            "purpose": "Checkpoint-only controlled homeostatic consumption audit.",
            "limitations": [
                "This intentionally skips capacity training, trajectory collection, reconstruction, and RGB-D matched-resource panels.",
                "Policy rollouts start just outside automatic-consumption range and test this fixed geometry, not every natural approach trajectory.",
                "Actor/critic results are null when the checkpoint contains only a world model.",
            ],
            "config": asdict(config),
            "arguments": vars(args),
            "checkpoint": args.checkpoint.resolve(),
            "source_sha256": source_hashes(),
            "event_calibration": events,
            "policy_and_critic": policy,
            "interpretation_order": [
                "Events: at harmful/lethal consumption, the prior reward should follow the true negative reward.",
                "Critic: harmful and lethal states should receive lower values than beneficial states.",
                "Policy: harmful/lethal stochastic resource-entry and positive-bound-death fractions should be near zero.",
            ],
        }
        report_path = output_dir / "report.json"
        report_path.write_text(json.dumps(json_ready(report), indent=2), encoding="utf-8")
        print(f"Saved checkpoint-only report to {report_path}")
        return

    matched_frames, projection = capture_matched_triplet(config, args.matched_distances[0], args.seed)
    capacity = capacity_audit(config, matched_frames, args, projection)

    train, validation = [], []
    seed = args.seed + 10_000
    for resource in RESOURCES:
        for length in args.trajectory_lengths:
            for _ in range(args.train_trajectories_per_case):
                train.append(collect_trajectory(config, resource, length, seed)); seed += 1
            for _ in range(args.validation_trajectories_per_case):
                validation.append(collect_trajectory(config, resource, length, seed)); seed += 1
    baselines = training_baselines(train)

    if args.checkpoint is None:
        set_seed(args.seed + 20_000)
        model = WorldModel(config).to(device)
        history = train_model(model, train, config, args.train_updates, args.learning_rate, "trajectory")
        checkpoint_path = output_dir / "world_model_only.pt"
        torch.save({"world_model": model.state_dict(), "world_model_only": True, "config": asdict(config)}, checkpoint_path)
    else:
        checkpoint_path = args.checkpoint.resolve()
        history = []
    model.eval()

    posterior = held_out_posterior_audit(model, validation, baselines, args, projection)
    prior = one_step_prior_audit(model, validation, baselines, args, projection)
    save_held_out_panel(model, validation[0], output_dir / "held_out_posterior_prior_panel.png", args.frames_per_panel)

    matched_reports = []
    matched_panel_cases = []
    for seed_offset in range(args.matched_seeds):
        for distance in args.matched_distances:
            frames, probe_projection = capture_matched_triplet(config, distance, args.seed + 30_000 + seed_offset)
            metrics, recon = matched_probe_metrics(model, frames, args, probe_projection, args.seed + 40_000 + seed_offset)
            matched_reports.append({"seed": args.seed + 30_000 + seed_offset, "distance": distance, **metrics})
            if seed_offset == 0 and metrics["available"]:
                matched_panel_cases.append((distance, frames, recon))
    if matched_panel_cases:
        save_matched_panel(matched_panel_cases, output_dir / "matched_rgb_depth_panel.png")

    events = event_calibration(model, config, args.seed + 50_000, critic)
    policy = None
    if actor is not None and critic is not None:
        policy = policy_calibration(
            model, actor, critic, config, args.seed + 60_000,
            args.policy_rollout_steps, args.policy_rollout_samples,
        )
    report = {
        "purpose": "Unified capacity, posterior reconstruction, paired RGB-D contrast, one-step prior, and event-calibration audit.",
        "limitations": [
            "The reconstruction/prior sections exclude actor, critic, imagination rollouts, replay sampling, and the production joint optimizer.",
            "Posterior reconstruction is compression evidence; the one-step prior section is the predictive evidence.",
            "Event cases are controlled calibration probes and are not estimates of natural replay frequency.",
            "Policy rollouts start just outside automatic-consumption range. They test this fixed geometry, not every natural approach trajectory.",
        ],
        "config": asdict(config),
        "arguments": vars(args),
        "checkpoint": checkpoint_path,
        "source_sha256": source_hashes(),
        "projection": projection,
        "depth_interpretation": {
            "encoding": args.depth_encoding,
            "clip_min_m": args.depth_clip_min_m if args.depth_encoding == "normalized_metric" else None,
            "clip_max_m": args.depth_clip_max_m if args.depth_encoding == "normalized_metric" else None,
        },
        "dataset": {"train_cases": len(train), "validation_cases": len(validation)},
        "capacity": capacity,
        "trajectory_training_history": history,
        "held_out_posterior": posterior,
        "held_out_one_step_prior": prior,
        "matched_colour_depth": matched_reports,
        "event_calibration": events,
        "policy_and_critic": policy,
        "interpretation_order": [
            "Capacity: production should fit; reconstruction-only localises objective competition if production does not.",
            "Matched: inspect decoded-vs-decoded-empty contrast, paired-sample distributions, dominant-colour fraction, and depth metres.",
            "Posterior: require improvement over channel/pixel mean baselines, not the zero predictor.",
            "Prior: require improvement over previous-frame and pixel-mean baselines without observing the target frame.",
            "Events: compare beneficial, harmful, lethal, posture/flip, and hard-negative predictions case by case; aggregate MSE alone is insufficient.",
            "Policy/critic: at harmful and lethal states, require low stochastic resource-entry/death fractions and lower critic values than matched safe states.",
        ],
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(json_ready(report), indent=2), encoding="utf-8")
    print(f"Saved unified report to {report_path}")


if __name__ == "__main__":
    main()

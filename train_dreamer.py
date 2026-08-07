import datetime as dt
from dataclasses import asdict, replace
from zoneinfo import ZoneInfo

import mlflow
import numpy as np
import torch
from torch.optim.lr_scheduler import LinearLR

from configs.config_dreamer import DreamerConfig
from utils.utils import set_seed
from utils.agc import agc
from utils.episode_telemetry import EpisodeTelemetry, RolloutTelemetryWindow
from utils.laprop import LaProp
from utils.replay_buffer_dreamer import SequenceReplayBuffer
from utils.utils_dreamer import (
    PercentileEMANormalizer,
    Ratio,
    compute_actor_critic_loss,
    compute_world_model_loss,
    imagination_rollout,
    obs_to_tensor_dict,
    prepare_sequence_batch,
    select_nonboundary_imagination_starts,
    set_requires_grad,
    to_tensor,
)
from utils.utils_env import create_env
from utils.utils_logger import create_logger
from utils.world_model import ActorNetwork, CriticNetwork, WorldModel


def run_deterministic_stability_probe(world_model, actor, cfg):
    """Run a fixed-length deterministic probe without modifying training state."""
    probe_cfg = replace(cfg, is_training=False, max_steps=float("inf"))
    probe_env = create_env(probe_cfg, multiple_env=True, rescale_action=False)
    try:
        obs, _ = probe_env.reset(seed=cfg.seed)
        previous_action = torch.zeros(
            cfg.num_workers, cfg.action_space_dim, device=cfg.device
        )
        is_first = torch.ones(cfg.num_workers, device=cfg.device)
        recurrent_state = None
        previous_stochastic = None
        current_is_last = np.zeros(cfg.num_workers, dtype=bool)
        episode_age = np.zeros(cfg.num_workers, dtype=np.int64)
        first_flip_step = np.full(cfg.num_workers, -1, dtype=np.int64)
        reward_sum = 0.0
        valid_count = 0
        episode_count = 0
        postures = []
        action_magnitudes = []
        flipped_rows = []

        for _ in range(cfg.deterministic_probe_steps):
            with torch.inference_mode():
                obs_embed = world_model.encode(obs_to_tensor_dict(obs, cfg))
                latent, recurrent_state, _, _ = world_model.observe(
                    previous_action,
                    obs_embed,
                    is_first,
                    recurrent_state=recurrent_state,
                    previous_stochastic=previous_stochastic,
                    deterministic=True,
                )
                previous_stochastic, recurrent_state = world_model.rssm.split_feature(
                    latent
                )
                action, _, _ = actor(latent, deterministic=True)
                action = action.detach().cpu().numpy()

            next_obs, rewards, terminations, truncations, infos = probe_env.step(action)
            valid = ~current_is_last
            indices = np.flatnonzero(valid)
            episode_age[current_is_last] = 0
            episode_age[indices] += 1
            reward_sum += float(np.asarray(rewards)[indices].sum())
            valid_count += int(indices.size)

            if indices.size:
                posture = np.asarray(infos["posture"], dtype=np.float64)
                flipped = np.asarray(infos["is_flipped"], dtype=bool) & valid
            else:
                posture = np.zeros(cfg.num_workers, dtype=np.float64)
                flipped = np.zeros(cfg.num_workers, dtype=bool)
            postures.append(posture[indices])
            action_magnitudes.append(np.linalg.norm(action[indices], axis=1))
            flipped_rows.append(flipped[indices])
            newly_flipped = flipped & (first_flip_step < 0)
            first_flip_step[newly_flipped] = episode_age[newly_flipped]

            episode_end = np.asarray(terminations) | np.asarray(truncations)
            episode_count += int((episode_end & valid).sum())
            next_is_first = current_is_last
            previous_action_np = action.copy()
            previous_action_np[next_is_first] = 0.0
            previous_action = to_tensor(previous_action_np, cfg.device)
            is_first = to_tensor(next_is_first, cfg.device)
            recurrent_state = recurrent_state * (1.0 - is_first.unsqueeze(-1))
            previous_stochastic = previous_stochastic * (
                1.0 - is_first.view(-1, 1, 1)
            )
            current_is_last = episode_end.astype(bool, copy=True)
            obs = next_obs

        posture_values = np.concatenate(postures)
        magnitude_values = np.concatenate(action_magnitudes)
        flipped_values = np.concatenate(flipped_rows)
        observed_first_flips = first_flip_step[first_flip_step >= 0]
        return {
            "deterministic_probe/reward_per_step": (
                reward_sum / valid_count if valid_count else 0.0
            ),
            "deterministic_probe/posture_mean": float(posture_values.mean()),
            "deterministic_probe/posture_p95": float(
                np.percentile(posture_values, 95)
            ),
            "deterministic_probe/posture_max": float(posture_values.max()),
            "deterministic_probe/action_magnitude_mean": float(
                magnitude_values.mean()
            ),
            "deterministic_probe/action_magnitude_p95": float(
                np.percentile(magnitude_values, 95)
            ),
            "deterministic_probe/flipped_transition_fraction": float(
                flipped_values.mean()
            ),
            "deterministic_probe/workers_ever_flipped_fraction": float(
                (first_flip_step >= 0).mean()
            ),
            "deterministic_probe/first_flip_step_mean": float(
                observed_first_flips.mean() if observed_first_flips.size else -1.0
            ),
            "deterministic_probe/completed_episodes": float(episode_count),
            "deterministic_probe/valid_transitions": float(valid_count),
        }
    finally:
        probe_env.close()


def train_dreamer():

    # Create logger
    logger = create_logger(name="Dreamer", log_file="./logs/logs_dreamer.log")
    logger.info("Starting Dreamer V3 training...")

    # Create mlflow
    mlflow.set_tracking_uri("sqlite:///runs.db?timeout=50000")
    mlflow.set_experiment("HomoeostaticAgent")

    # Get config
    cfg = DreamerConfig()
    logger.info(f"Config: {cfg}")

    # Set seed
    set_seed(cfg.seed)

    # Check that the config does not have frame stack key
    assert not hasattr(cfg, "frame_stack_key"), (
        "DreamerConfig should not have frame_stack_key attribute"
    )

    # Create environment
    env = create_env(cfg, multiple_env=True, rescale_action=False)
    # frame_skip = env.envs[0].unwrapped.frame_skip
    logger.info(f"Created parallel environment with {cfg.num_workers} workers")

    # Create world model
    torch.set_float32_matmul_precision("high")
    world_model = WorldModel(cfg).to(cfg.device)
    world_model.encoder = torch.compile(world_model.encoder, dynamic=False)
    world_model.rssm = torch.compile(world_model.rssm, dynamic=False)
    world_model.decoder = torch.compile(world_model.decoder, dynamic=False)
    world_model.reward_predictor = torch.compile(
        world_model.reward_predictor, dynamic=False
    )
    world_model.continue_predictor = torch.compile(
        world_model.continue_predictor, dynamic=False
    )

    logger.info("Created world model")

    # Create actor-critic
    actor = ActorNetwork(cfg)
    actor.to(cfg.device)
    critic = CriticNetwork(cfg)
    critic.to(cfg.device)
    ema_critic = CriticNetwork(cfg)
    ema_critic.to(cfg.device)
    ema_critic.load_state_dict(critic.state_dict())
    ema_critic.eval()
    set_requires_grad([ema_critic], False)
    return_normalizer = PercentileEMANormalizer(
        rate=cfg.return_norm_rate,
        limit=cfg.return_norm_limit,
        percentile_low=cfg.return_norm_percentile_low,
        percentile_high=cfg.return_norm_percentile_high,
        device=cfg.device,
    )
    actor = torch.compile(actor, dynamic=False)
    critic = torch.compile(critic, dynamic=True)
    logger.info("Created actor, critic, and EMA critic networks")

    # Create model save timestamp
    timestamp = dt.datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y-%m-%d_%H-%M-%S")
    model_path = f"./models/dreamer_{timestamp}.pt"

    # Create replay buffer
    replay_buffer = SequenceReplayBuffer(cfg, device=cfg.device)
    logger.info("Created replay buffer")

    # Use one optimizer for the combined Dreamer objective. Replay-value
    # gradients can then update both the critic and posterior representation.
    optimizer = LaProp(
        list(world_model.parameters())
        + list(actor.parameters())
        + list(critic.parameters()),
        lr=cfg.laprop_lr,
        eps=cfg.laprop_eps,
        betas=(cfg.laprop_beta1, cfg.laprop_beta2),
    )
    logger.info("Created joint world-model, actor, and critic optimizer")
    scheduler = LinearLR(
        optimizer, start_factor=0.001, end_factor=1.0, total_iters=cfg.warmup_steps
    )

    # AMP only accelerates CUDA training. Keeping it disabled on CPU preserves
    # the existing numerical path for development and test runs.
    amp_enabled = cfg.use_amp and cfg.device.type == "cuda"
    amp_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    def amp_autocast():
        return torch.autocast(
            device_type=cfg.device.type,
            dtype=torch.bfloat16,
            enabled=amp_enabled,
        )

    logger.info("CUDA AMP is %s", "enabled (bfloat16)" if amp_enabled else "disabled")

    # Training
    logger.info("Starting training loop")
    global_step = 0
    env_transition_step = 0
    episodes_finished = 0
    cumulative_episode_ends = 0
    cumulative_homeostatic_terminations = 0
    cumulative_truncations = 0
    comparison_window_episode_ends = 0
    comparison_window_transitions = 0
    next_comparison_metrics_step = cfg.comparison_metrics_interval
    next_training_metrics_step = 0
    next_rollout_metrics_step = cfg.rollout_metrics_interval
    next_per_joint_metrics_step = cfg.per_joint_metrics_interval
    next_deterministic_probe_step = cfg.deterministic_probe_interval
    obs, info = env.reset(seed=cfg.seed)
    prev_action = torch.zeros(cfg.num_workers, cfg.action_space_dim, device=cfg.device)
    is_first = torch.ones(cfg.num_workers, device=cfg.device)
    replay_reward = np.zeros(cfg.num_workers, dtype=np.float32)
    replay_terminal = np.zeros(cfg.num_workers, dtype=bool)
    replay_is_last = np.zeros(cfg.num_workers, dtype=bool)
    recurrent_state = None
    previous_stochastic = None
    iteration = 0
    batch_steps = cfg.batch_size * cfg.batch_length
    should_train = Ratio(
        cfg.replay_ratio / (batch_steps)
    )  # Removed frame skip from the denominator for now
    replay_ratio_started = False
    episode_telemetry = EpisodeTelemetry(cfg.num_workers)
    rollout_telemetry = RolloutTelemetryWindow(cfg.action_space_dim)
    per_joint_telemetry = RolloutTelemetryWindow(cfg.action_space_dim)

    with mlflow.start_run(
        run_name="Dreamer V3 Training - "
        + dt.datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y-%m-%d_%H-%M-%S")
    ):
        mlflow.log_params(asdict(cfg))
        mlflow.log_metrics(
            {
                "comparison/env_transition_step": 0.0,
                "comparison/cumulative_episode_ends": 0.0,
                "comparison/cumulative_homeostatic_terminations": 0.0,
                "comparison/cumulative_truncations": 0.0,
                "comparison/episode_end_rate_per_million": 0.0,
            },
            step=0,
        )

        while global_step < cfg.total_env_steps:
            # ===== PHASE 1: Real World Interaction =====
            logger.debug(f"Iteration {iteration}: Starting environment interaction")
            actor.eval()
            world_model.eval()

            list_iterations_episode_length = []
            # random_action_fraction = 0.0
            replay_is_first = is_first.detach().cpu().numpy().astype(bool)
            current_is_last = replay_is_last.astype(bool, copy=True)
            with torch.no_grad():
                # Embed observations
                obs_tensor = obs_to_tensor_dict(obs, cfg)
                obs_embed = world_model.encode(obs_tensor)

                # Get s_t (with z_t concatenated) and h_t from the world model
                latent, recurrent_state, _, _ = world_model.observe(
                    prev_action,
                    obs_embed,
                    is_first,
                    recurrent_state=recurrent_state,
                    previous_stochastic=previous_stochastic,
                    deterministic=False,
                )
                previous_stochastic, recurrent_state = world_model.rssm.split_feature(
                    latent
                )
                # Official replay entries are posterior states *after* observing
                # this row. They form the one-row context prefix for later chunks.
                replay_context_stochastic = previous_stochastic.detach().cpu().numpy()
                replay_context_recurrent = recurrent_state.detach().cpu().numpy()

                # Get action from actor
                action, log_prob, dist = actor(latent, deterministic=False)
                actor_loc = dist.base_dist.loc.detach().cpu().numpy()
                actor_std = dist.base_dist.scale.detach().cpu().numpy()
                action = action.detach().cpu().numpy()

                # Check for NaN
                if np.any(np.isnan(action)) or np.any(np.isinf(action)):
                    logger.error(
                        f"NaN/Inf detected in actions at iteration {iteration}!"
                    )
                    raise RuntimeError("NaN/Inf detected in actions")

            # Store the current replay row before stepping, matching the official
            # Dreamer convention: row t contains obs_t, action_t, and the
            # reward/done that led into obs_t from action_{t-1}. The replay
            # buffer reconstructs prev_action_t by shifting stored actions.
            # Stores the following:
            #   Current observation
            #   Action taken because of the current observation
            #   The reward from the previous action that led to this state
            #   Whether the incoming transition is an absorbing value terminal.
            #   Homeostatic episode endings are reset boundaries, not value
            #   terminals, so this remains false.
            #   The is_first flag for the current observation
            #   Posterior RSSM entry after observing the current observation.
            for i in range(cfg.num_workers):
                obs_single = {
                    "vision": obs["vision"][i : i + 1],
                    "proprioception": obs["proprioception"][i : i + 1],
                    "internal_state": obs["internal_state"][i : i + 1],
                }
                if cfg.num_heat > 0:
                    obs_single["heat_sensor"] = obs["heat_sensor"][i : i + 1]

                replay_buffer.add(
                    obs_dict=obs_single,
                    action=action[i : i + 1],
                    reward=replay_reward[i : i + 1],
                    terminal=replay_terminal[i : i + 1],
                    is_last=replay_is_last[i : i + 1],
                    is_first=bool(replay_is_first[i]),
                    context_stochastic=replay_context_stochastic[i : i + 1],
                    context_recurrent=replay_context_recurrent[i : i + 1],
                )

            # Match official DreamerV3's ClipAction wrapper: keep the raw
            # Normal sample for replay, policy likelihoods, and prev_action,
            # while sending only native action-space values to MuJoCo.
            env_action = np.clip(action, -1.0, 1.0)
            next_obs, rewards, terminations, truncations, infos = env.step(env_action)
            episode_end = terminations | truncations
            valid_transitions = ~current_is_last
            newly_flipped_workers = episode_telemetry.add_transition(
                valid=valid_transitions,
                rewards=rewards,
                infos=infos,
                native_actions=action,
                executed_actions=env_action,
                actor_loc=actor_loc,
                actor_std=actor_std,
            )
            if valid_transitions.any():
                is_flipped = np.asarray(infos["is_flipped"], dtype=bool)
            else:
                is_flipped = np.zeros(cfg.num_workers, dtype=bool)
            for telemetry_window in (rollout_telemetry, per_joint_telemetry):
                telemetry_window.add_transition(
                    valid=valid_transitions,
                    raw_actions=action,
                    executed_actions=env_action,
                    actor_loc=actor_loc,
                    actor_std=actor_std,
                    is_flipped=is_flipped,
                )

            obs = next_obs
            next_is_first = current_is_last
            replay_reward = rewards.astype(np.float32, copy=True)
            # Match Appendix A of the homeostatic reference paper: reset the
            # environment at a homeostatic limit without assigning zero value
            # to the final observation. ``is_last`` below still cuts replay
            # traces at the reset boundary.
            replay_terminal = np.zeros_like(terminations, dtype=bool)
            replay_is_last = episode_end.astype(bool, copy=True)
            replay_reward[next_is_first] = 0.0
            replay_terminal[next_is_first] = False
            replay_is_last[next_is_first] = False

            prev_action_np = action.copy()
            prev_action_np[next_is_first] = 0.0
            prev_action = to_tensor(prev_action_np, cfg.device)
            is_first = to_tensor(next_is_first, cfg.device)
            # Reset the live RSSM state only for rows that are actually reset
            # observations. Final observations are consumed on the next loop
            # with is_first=False so reward/continue can learn from their
            # posterior latent.
            recurrent_state = recurrent_state * (1.0 - is_first.unsqueeze(-1))
            previous_stochastic = previous_stochastic * (1.0 - is_first.view(-1, 1, 1))
            global_step += cfg.num_workers
            valid_transition_count = int(valid_transitions.sum())
            env_transition_step += valid_transition_count
            comparison_window_transitions += valid_transition_count

            episode_finished_mask = np.asarray(
                infos.get("_episode", np.zeros(cfg.num_workers, dtype=bool)),
                dtype=bool,
            )
            completed_this_step = int(episode_finished_mask.sum())
            cumulative_episode_ends += completed_this_step
            comparison_window_episode_ends += completed_this_step
            cumulative_homeostatic_terminations += int(
                (episode_finished_mask & np.asarray(terminations, dtype=bool)).sum()
            )
            cumulative_truncations += int(
                (episode_finished_mask & np.asarray(truncations, dtype=bool)).sum()
            )
            comparison_heartbeat_due = (
                env_transition_step >= next_comparison_metrics_step
            )
            if completed_this_step or comparison_heartbeat_due:
                comparison_metrics = {
                    "comparison/env_transition_step": float(env_transition_step),
                    "comparison/cumulative_episode_ends": float(
                        cumulative_episode_ends
                    ),
                    "comparison/cumulative_homeostatic_terminations": float(
                        cumulative_homeostatic_terminations
                    ),
                    "comparison/cumulative_truncations": float(
                        cumulative_truncations
                    ),
                }
                if comparison_heartbeat_due:
                    comparison_metrics[
                        "comparison/episode_end_rate_per_million"
                    ] = (
                        1_000_000.0
                        * comparison_window_episode_ends
                        / comparison_window_transitions
                        if comparison_window_transitions
                        else 0.0
                    )
                    comparison_window_episode_ends = 0
                    comparison_window_transitions = 0
                    while env_transition_step >= next_comparison_metrics_step:
                        next_comparison_metrics_step += (
                            cfg.comparison_metrics_interval
                        )
                mlflow.log_metrics(comparison_metrics, step=env_transition_step)

            if newly_flipped_workers.size:
                flip_rows = newly_flipped_workers
                mlflow.log_metrics(
                    {
                        "flip_event/new_first_flip_count": float(flip_rows.size),
                        "flip_event/episode_age_mean": float(
                            episode_telemetry.episode_age[flip_rows].mean()
                        ),
                        "flip_event/posture_mean": float(
                            np.asarray(infos["posture"])[flip_rows].mean()
                        ),
                        "flip_event/actor_std_mean": float(
                            actor_std[flip_rows].mean()
                        ),
                        "flip_event/raw_action_clip_coordinate_fraction": float(
                            (np.abs(action[flip_rows]) > 1.0).mean()
                        ),
                    },
                    step=env_transition_step,
                )

            if env_transition_step >= next_rollout_metrics_step:
                rollout_metrics = rollout_telemetry.summary()
                rollout_metrics.update(episode_telemetry.active_survival_metrics())
                rollout_metrics.update(
                    {
                        "rollout/env_transition_step": float(env_transition_step),
                        "rollout/dreamer_global_step": float(global_step),
                    }
                )
                mlflow.log_metrics(rollout_metrics, step=env_transition_step)
                while env_transition_step >= next_rollout_metrics_step:
                    next_rollout_metrics_step += cfg.rollout_metrics_interval

            if env_transition_step >= next_per_joint_metrics_step:
                mlflow.log_metrics(
                    per_joint_telemetry.per_joint_summary(),
                    step=env_transition_step,
                )
                while env_transition_step >= next_per_joint_metrics_step:
                    next_per_joint_metrics_step += cfg.per_joint_metrics_interval

            if env_transition_step >= next_deterministic_probe_step:
                logger.info(
                    "Running deterministic stability probe at transition step %d",
                    env_transition_step,
                )
                mlflow.log_metrics(
                    run_deterministic_stability_probe(world_model, actor, cfg),
                    step=env_transition_step,
                )
                while env_transition_step >= next_deterministic_probe_step:
                    next_deterministic_probe_step += cfg.deterministic_probe_interval

            # avg_episode_length = sum(list_iterations_episode_length) / len(list_iterations_episode_length) if list_iterations_episode_length else 0
            # avg_wm_loss = 0.0
            # avg_reconstruction_loss = 0.0
            actor_loss = None
            critic_loss = None
            actor_loss_value = None
            critic_loss_value = None
            ac_metrics = {}
            num_wm_updates = 0
            num_ac_updates = 0
            train_batches_due = 0
            wm_sample_failures = 0
            wm_metrics = {}
            # last_actor_grad_norm = 0.0
            # avg_reward_loss = 0.0
            # avg_continue_loss = 0.0
            # replay_terminal_count = 0.0
            # replay_terminal_items = 0
            if len(replay_buffer) >= cfg.min_buffer_size_before_training:
                # if not replay_ratio_started:
                #     # Do not retrospectively schedule updates for the replay
                #     # collection phase before training became eligible.
                #     should_train.start(global_step)
                #     replay_ratio_started = True
                # else:
                train_batches_due = should_train(global_step)

            # ===== PHASE 2: World Model Training =====
            if (
                len(replay_buffer) >= cfg.min_buffer_size_before_training
                and train_batches_due > 0
            ):
                logger.debug(f"Iteration {iteration}: Starting world model training")
                world_model.train()

                total_wm_loss = 0
                total_wm_metrics = {}
                total_replay_terminal_count = 0.0
                total_replay_terminal_items = 0
                total_actor_loss = 0.0
                total_critic_loss = 0.0
                total_ac_metrics = {}

                for wm_step in range(train_batches_due):
                    # Sample fresh online sequences first, then fill with uniform replay.
                    batch_data = replay_buffer.sample_mixed()
                    if batch_data is None:
                        wm_sample_failures += 1
                        break

                    (
                        batch_size,
                        seq_len,
                        obs_batch_flat,
                        obs_target,
                        actions_batch,
                        prev_actions,
                        rewards_batch,
                        terminals_batch,
                        is_last_batch,
                        is_first_batch,
                        initial_stochastic,
                        initial_recurrent,
                    ) = prepare_sequence_batch(batch_data, cfg)
                    total_replay_terminal_count += float(terminals_batch.sum().item())
                    total_replay_terminal_items += int(terminals_batch.numel())

                    with amp_autocast():
                        embed_batch = world_model.encode(obs_batch_flat).view(
                            batch_size, seq_len, -1
                        )

                        wm_loss, wm_metrics, wm_latents = compute_world_model_loss(
                            world_model.rssm,
                            world_model.decoder,
                            world_model.reward_predictor,
                            world_model.continue_predictor,
                            prev_actions,
                            actions_batch,
                            embed_batch,
                            obs_target,
                            is_first_batch,
                            rewards_batch,
                            terminals_batch,
                            cfg,
                            initial_stochastic=initial_stochastic,
                            initial_recurrent=initial_recurrent,
                            return_latents=True,
                        )

                    total_wm_loss += wm_loss.item()
                    for key, value in wm_metrics.items():
                        total_wm_metrics[key] = total_wm_metrics.get(key, 0.0) + value
                    num_wm_updates += 1

                    # Train actor-critic from the same replay batch used for
                    # this world-model update, matching DreamerV3's batch flow
                    # and avoiding a second replay sample that would drain a
                    # different online queue item.
                    actor.train()
                    critic.train()

                    ac_batch_size = batch_size
                    replay_source_latents = wm_latents
                    ac_rewards = rewards_batch.detach()
                    ac_terminals = terminals_batch.detach()
                    ac_is_last = is_last_batch.detach()
                    if ac_batch_size > cfg.imagine_batch_size:
                        take = slice(0, cfg.imagine_batch_size)
                        ac_batch_size = cfg.imagine_batch_size
                        replay_source_latents = replay_source_latents[take]
                        ac_rewards = ac_rewards[take]
                        ac_terminals = ac_terminals[take]
                        ac_is_last = ac_is_last[take]

                    start_count = (
                        seq_len
                        if cfg.imagine_last == 0
                        else min(cfg.imagine_last, seq_len)
                    )
                    # Replay value learning remains connected to the posterior
                    # representation. Imagination starts stay detached, which
                    # preserves the official ac_grads=False behavior.
                    replay_latents = replay_source_latents[:, -start_count:]
                    replay_rewards = ac_rewards[:, -start_count:]
                    replay_terminals = ac_terminals[:, -start_count:]
                    replay_last_rows = ac_is_last[:, -start_count:]
                    init_latent, imagination_start_mask = (
                        select_nonboundary_imagination_starts(
                            replay_latents, replay_last_rows
                        )
                    )
                    if init_latent.shape[0] == 0:
                        raise RuntimeError(
                            "Replay batch contains no non-boundary states for imagination"
                        )
                    _, init_recurrent_state = world_model.rssm.split_feature(
                        init_latent
                    )

                    with amp_autocast():
                        imagined_trajectories = imagination_rollout(
                            world_model.rssm,
                            actor,
                            world_model.reward_predictor,
                            init_latent,
                            init_recurrent_state,
                            cfg,
                        )
                        imagined_trajectories["start_batch_size"] = ac_batch_size
                        imagined_trajectories["start_count"] = start_count
                        imagined_trajectories["imagination_start_mask"] = (
                            imagination_start_mask
                        )
                        replay_trajectories = {
                            "latents": replay_latents,
                            "rewards": replay_rewards,
                            "terminals": replay_terminals,
                            "is_last": replay_last_rows,
                        }

                        actor_loss, critic_loss, ac_metrics = compute_actor_critic_loss(
                            critic,
                            ema_critic,
                            imagined_trajectories,
                            cfg,
                            return_normalizer,
                            replay_trajectories,
                        )

                    optimizer.zero_grad()
                    total_loss = wm_loss + actor_loss + critic_loss
                    amp_scaler.scale(total_loss).backward()
                    amp_scaler.unscale_(optimizer)

                    # Retain component-wise AGC statistics with the joint
                    # optimizer.
                    agc(world_model.parameters())
                    agc(actor.parameters())
                    agc(critic.parameters())

                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                    scheduler.step()

                    # Persist contexts in float32 so replay does not gradually
                    # accumulate fp16 rounding error across updates.
                    replay_buffer.update_contexts(
                        batch_data, wm_latents.detach().float()
                    )

                    for param, ema_param in zip(
                        critic.parameters(), ema_critic.parameters()
                    ):
                        ema_param.data.copy_(
                            cfg.ema_critic_tau * param.data
                            + (1.0 - cfg.ema_critic_tau) * ema_param.data
                        )

                    total_actor_loss += actor_loss.item()
                    total_critic_loss += critic_loss.item()
                    for key, value in ac_metrics.items():
                        total_ac_metrics[key] = total_ac_metrics.get(key, 0.0) + value
                    num_ac_updates += 1

                if num_ac_updates > 0:
                    actor_loss_value = total_actor_loss / num_ac_updates
                    critic_loss_value = total_critic_loss / num_ac_updates
                    ac_metrics = {
                        key: value / num_ac_updates
                        for key, value in total_ac_metrics.items()
                    }
                if num_wm_updates > 0:
                    wm_metrics = {
                        key: value / num_wm_updates
                        for key, value in total_wm_metrics.items()
                    }
                    # These are update-averaged and use global environment
                    # steps, unlike episode metrics below. They provide a
                    # stable training diagnostic even when episodes are long.
                    if global_step >= next_training_metrics_step:
                        mlflow.log_metrics(
                            {
                                f"train_update/{key.removeprefix('world_model/')}": value
                                for key, value in wm_metrics.items()
                            },
                            step=global_step,
                        )
                        while global_step >= next_training_metrics_step:
                            next_training_metrics_step += cfg.train_metrics_interval
                # if total_replay_terminal_items > 0:
                #     replay_terminal_rate = (
                #         total_replay_terminal_count / total_replay_terminal_items
                #     )
                #     logger.info(
                #         "ReplayTerminals=%d/%d (%.6f)",
                #         int(total_replay_terminal_count),
                #         total_replay_terminal_items,
                #         replay_terminal_rate,
                #     )

            # Log finished episodes and carry the step outcome to the next
            # replay row, where it is aligned with the resulting observation.
            # When the current replay row was terminal, this vector step returns
            # the reset observation; that next row should be first with zero
            # reward/done, not another terminal transition.
            # episode_global_step = global_step + cfg.num_workers
            for i in range(cfg.num_workers):
                if "_episode" in infos and infos["_episode"][i]:
                    episodes_finished += 1
                    list_iterations_episode_length.append(infos["episode"]["l"][i])
                    avg_episode_length = (
                        sum(list_iterations_episode_length)
                        / len(list_iterations_episode_length)
                        if list_iterations_episode_length
                        else 0.0
                    )
                    episode_metrics = {
                        "episode/return": infos["episode"]["r"][i],
                        "episode/length": infos["episode"]["l"][i],
                        "episode/food_consumed": infos["food_consumed"][i],
                        "episode/water_consumed": infos["water_consumed"][i],
                        "episode/posture": infos["posture"][i],
                        "episode/termination_reason": infos["termination_reason"][i],
                        "episode/final_hunger": infos["hunger"][i],
                        "episode/final_thirst": infos["thirst"][i],
                        "average_episode_length": avg_episode_length,
                        "global_step": global_step,
                        "episode/env_transition_step": env_transition_step,
                        "episode/start_global_step_estimate": max(
                            0,
                            global_step
                            - int(infos["episode"]["l"][i]) * cfg.num_workers,
                        ),
                    }
                    episode_metrics.update(
                        episode_telemetry.finish_episode(
                            i,
                            episode_return=float(infos["episode"]["r"][i]),
                            episode_length=int(infos["episode"]["l"][i]),
                            food_consumed=float(infos["food_consumed"][i]),
                            water_consumed=float(infos["water_consumed"][i]),
                        )
                    )
                    # Episodes can finish while replay is still warming up.
                    # In that phase actor_loss_value and critic_loss_value are
                    # intentionally None because no training update has run;
                    # MLflow does not accept None as a metric value.
                    if num_ac_updates > 0:
                        episode_metrics.update(
                            {
                                "train/policy_loss": actor_loss_value,
                                "train/value_loss": critic_loss_value,
                                "train/critic_imagined_return_loss": ac_metrics.get(
                                    "actor_critic/critic_return_loss", 0.0
                                ),
                                "train/critic_imagined_slow_loss": ac_metrics.get(
                                    "actor_critic/critic_slow_loss", 0.0
                                ),
                                "train/critic_imagined_loss": ac_metrics.get(
                                    "actor_critic/critic_imagined_loss", 0.0
                                ),
                                "train/critic_imagined_weighted_loss": ac_metrics.get(
                                    "actor_critic/critic_imagined_weighted_loss", 0.0
                                ),
                                "train/critic_replay_return_loss": ac_metrics.get(
                                    "actor_critic/critic_replay_return_loss", 0.0
                                ),
                                "train/critic_replay_slow_loss": ac_metrics.get(
                                    "actor_critic/critic_replay_slow_loss", 0.0
                                ),
                                "train/critic_replay_loss": ac_metrics.get(
                                    "actor_critic/critic_replay_loss", 0.0
                                ),
                                "train/critic_replay_weighted_loss": ac_metrics.get(
                                    "actor_critic/critic_replay_weighted_loss", 0.0
                                ),
                                "train/entropy": ac_metrics.get(
                                    "actor_critic/entropy", 0.0
                                ),
                                "train/learning_rate": optimizer.param_groups[0]["lr"],
                                # "train/kl_divergence": 0.0,
                                "train/dyn_loss": wm_metrics.get(
                                    "world_model/dyn_loss", 0.0
                                ),
                                "train/rep_loss": wm_metrics.get(
                                    "world_model/rep_loss", 0.0
                                ),
                                "train/explained_variance": ac_metrics.get(
                                    "actor_critic/explained_variance", 0.0
                                ),
                            }
                        )
                    mlflow.log_metrics(episode_metrics, step=episodes_finished)

                    logger.info(
                        f"Episode {episodes_finished} finished at global step {global_step}: return={infos['episode']['r'][i]}, length={infos['episode']['l'][i]}, food_consumed={infos['food_consumed'][i]}, water_consumed={infos['water_consumed'][i]}, posture={infos['posture'][i]}, termination_reason={infos['termination_reason'][i]}, final_hunger={infos['hunger'][i]}, final_thirst={infos['thirst'][i]}"
                    )
            iteration += 1

            if global_step % 100_000 == 0 and global_step > 0:
                logger.info(f"Global step {global_step}: Saving models...")
                torch.save(
                    {
                        "world_model": world_model.state_dict(),
                        "actor": actor.state_dict(),
                        "critic": critic.state_dict(),
                    },
                    model_path
                )

        if episodes_finished != cumulative_episode_ends:
            raise RuntimeError(
                "Dreamer episode counter mismatch: "
                f"episode logs={episodes_finished}, comparison={cumulative_episode_ends}"
            )
        final_comparison_metrics = {
            "comparison/env_transition_step": float(env_transition_step),
            "comparison/cumulative_episode_ends": float(cumulative_episode_ends),
            "comparison/cumulative_homeostatic_terminations": float(
                cumulative_homeostatic_terminations
            ),
            "comparison/cumulative_truncations": float(cumulative_truncations),
        }
        if comparison_window_transitions:
            final_comparison_metrics[
                "comparison/episode_end_rate_per_million"
            ] = (
                1_000_000.0
                * comparison_window_episode_ends
                / comparison_window_transitions
            )
        mlflow.log_metrics(final_comparison_metrics, step=env_transition_step)

    # Save models
    torch.save(
        {
            "world_model": world_model.state_dict(),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
        },
        model_path
    )
    logger.info(
        f"Models saved to {model_path}"
    )
    print(
        f"Models saved to {model_path}"
    )


if __name__ == "__main__":
    train_dreamer()

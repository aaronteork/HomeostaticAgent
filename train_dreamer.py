import datetime as dt
from dataclasses import asdict

import numpy as np
import mlflow
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from tqdm.auto import tqdm

from configs.config_dreamer import DreamerConfig
from utils.utils_env import create_env
from utils.utils_logger import create_logger
from utils.utils_dreamer import (
    WorldModel,
    ActorNetwork,
    CriticNetwork,
    SequenceReplayBuffer,
    PercentileEMANormalizer,
    compute_world_model_loss,
    compute_actor_critic_loss,
    imagination_rollout,
    to_tensor,
    Ratio,
    prepare_sequence_batch,
    set_requires_grad,
    obs_to_tensor_dict,
)


def train_dreamer():

    # Create logger
    logger = create_logger(name="Dreamer", log_file="./logs/logs_dreamer.log")
    logger.info("Starting Dreamer V3 training...")

    # Create mlflow
    mlflow.set_tracking_uri("sqlite:///runs.db")
    mlflow.set_experiment("HomoeostaticAgent")

    # Get config
    cfg = DreamerConfig()
    logger.info(f"Config: {cfg}")

    # Create environment
    env = create_env(cfg, multiple_env=True)
    frame_skip = env.envs[0].unwrapped.frame_skip
    logger.info(f"Created parallel environment with {cfg.num_workers} workers")

    # Create world model
    world_model = WorldModel(cfg).to(cfg.device)
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
    logger.info("Created actor, critic, and EMA critic networks")

    # Create replay buffer
    replay_buffer = SequenceReplayBuffer(cfg, device=cfg.device)
    logger.info("Created replay buffer")

    # Create optimizers
    # Stick to Adam for simplicity, the paper mentioned that they used some LaProp and adaptive global gradient clipping to stabilise training
    world_model_params = list(world_model.parameters())
    world_model_optimizer = Adam(world_model_params, lr=cfg.world_model_lr, eps=cfg.adam_eps)
    actor_optimizer = Adam(actor.parameters(), lr=cfg.actor_lr, eps=cfg.adam_eps)
    critic_optimizer = Adam(critic.parameters(), lr=cfg.critic_lr, eps=cfg.adam_eps)
    logger.info("Created optimizers")

    # # Learning rate schedulers
    # No need learning rate schedulers
    # total_train_updates = max(1, int(cfg.total_env_steps * cfg.replay_ratio / (cfg.batch_size * cfg.batch_length)))
    # world_model_scheduler = LinearLR(world_model_optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_train_updates)
    # actor_scheduler = LinearLR(actor_optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_train_updates)
    # critic_scheduler = LinearLR(critic_optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_train_updates)

    # Training
    logger.info("Starting training loop")
    global_step = 0
    episodes_finished = 0
    obs, info = env.reset()
    prev_action = torch.zeros(cfg.num_workers, cfg.action_space_dim, device=cfg.device)
    is_first = torch.ones(cfg.num_workers, device=cfg.device)
    recurrent_state = None
    previous_stochastic = None
    iteration = 0
    batch_steps = cfg.batch_size * cfg.batch_length
    should_train = Ratio(cfg.replay_ratio / (batch_steps * frame_skip))  # 0.5

    with mlflow.start_run(run_name="Dreamer V3 Training - " + dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")):
        mlflow.log_params(asdict(cfg))

        while global_step < cfg.total_env_steps:
            # ===== PHASE 1: Real World Interaction =====
            logger.debug(f"Iteration {iteration}: Starting environment interaction")
            actor.eval()
            world_model.eval()

            list_iterations_episode_length = []
            replay_is_first = is_first.detach().cpu().numpy().astype(bool)
            with torch.no_grad():
                if recurrent_state is None or previous_stochastic is None:
                    context_stochastic, context_recurrent = world_model.rssm.initial_state(
                        cfg.num_workers,
                        cfg.device,
                        deterministic=False,
                    )
                else:
                    context_stochastic = previous_stochastic
                    context_recurrent = recurrent_state
                replay_context_stochastic = context_stochastic.detach().cpu().numpy()
                replay_context_recurrent = context_recurrent.detach().cpu().numpy()

                # Embed observations
                obs_tensor = obs_to_tensor_dict(obs, cfg)
                obs_embed = world_model.encode(obs_tensor)

                # Infer current latent state from the latest real observation.
                latent, recurrent_state, _, _ = world_model.observe(
                    prev_action,
                    obs_embed,
                    is_first,
                    recurrent_state=recurrent_state,
                    previous_stochastic=previous_stochastic,
                    deterministic=False,
                )
                previous_stochastic, recurrent_state = world_model.rssm.split_feature(latent)

                # Get action from actor
                action, log_prob, dist = actor(latent, deterministic=False)
                action = action.detach().cpu().numpy()

                # # Clamp action to valid range [0, 1]
                # # Not required since we are using Beta Distribution
                # action = np.clip(action, 0.0, 1.0)

                # Check for NaN
                if np.any(np.isnan(action)) or np.any(np.isinf(action)):
                    logger.error(f"NaN/Inf detected in actions at iteration {iteration}!")
                    raise RuntimeError("NaN/Inf detected in actions")

            # Step environment
            next_obs, rewards, terminations, truncations, infos = env.step(action)
            done = terminations | truncations

            # Store in replay buffer (one transition per worker)
            for i in range(cfg.num_workers):
                if "_episode" in infos and infos["_episode"][i]:
                    episodes_finished += 1
                    list_iterations_episode_length.append(infos["episode"]["l"][i])
                    mlflow.log_metrics(
                        {
                            "episode/return": infos["episode"]["r"][i],
                            "episode/length": infos["episode"]["l"][i],
                            "episode/food_consumed": infos["food_consumed"][i],
                            "episode/water_consumed": infos["water_consumed"][i],
                            "episode/posture": infos["posture"][i],
                            "episode/termination_reason": infos["termination_reason"][i],
                            "episode/final_hunger": infos["hunger"][i],
                            "episode/final_thirst": infos["thirst"][i],
                        },
                        step=episodes_finished,
                    )

                obs_single = {
                    "vision": obs["vision"][i:i+1],
                    "proprioception": obs["proprioception"][i:i+1],
                    "internal_state": obs["internal_state"][i:i+1],
                }
                if cfg.num_heat > 0:
                    obs_single["heat_sensor"] = obs["heat_sensor"][i:i+1]

                replay_buffer.add(
                    obs_dict=obs_single,
                    action=action[i:i+1],
                    reward=rewards[i:i+1],
                    done=done[i:i+1],
                    is_first=bool(replay_is_first[i]),
                    context_stochastic=replay_context_stochastic[i:i+1],
                    context_recurrent=replay_context_recurrent[i:i+1],
                )

            obs = next_obs
            prev_action = to_tensor(action, cfg.device)
            is_first = to_tensor(done, cfg.device)
            recurrent_state = recurrent_state * (1.0 - is_first.unsqueeze(-1))
            previous_stochastic = previous_stochastic * (1.0 - is_first.view(-1, 1, 1))
            global_step += cfg.num_workers

            avg_episode_length = sum(list_iterations_episode_length) / len(list_iterations_episode_length) if list_iterations_episode_length else 0
            avg_wm_loss = 0.0
            avg_reconstruction_loss = 0.0
            avg_kl_loss = 0.0
            actor_loss = None
            critic_loss = None
            actor_loss_value = None
            critic_loss_value = None
            ac_metrics = {}
            num_wm_updates = 0
            num_ac_updates = 0
            train_batches_due = 0
            if len(replay_buffer) >= cfg.min_buffer_size_before_training:
                train_batches_due = should_train(global_step)

            # ===== PHASE 2: World Model Training =====
            if len(replay_buffer) >= cfg.min_buffer_size_before_training and train_batches_due > 0:
                logger.debug(f"Iteration {iteration}: Starting world model training")
                world_model.train()

                total_wm_loss = 0
                total_reconstruction_loss = 0
                total_kl_loss = 0

                for wm_step in range(train_batches_due):
                    # Sample fresh online sequences first, then fill with uniform replay.
                    batch_data = replay_buffer.sample_mixed()
                    if batch_data is None:
                        break

                    (
                        batch_size,
                        seq_len,
                        obs_batch_flat,
                        obs_target,
                        prev_actions,
                        rewards_batch,
                        dones_batch,
                        is_first_batch,
                        initial_stochastic,
                        initial_recurrent,
                    ) = prepare_sequence_batch(batch_data, cfg)

                    embed_batch = world_model.encode(obs_batch_flat).view(batch_size, seq_len, -1)

                    wm_loss, wm_metrics, wm_latents = compute_world_model_loss(
                        world_model.rssm,
                        world_model.decoder,
                        world_model.reward_predictor,
                        world_model.terminal_predictor,
                        prev_actions, embed_batch, obs_target, is_first_batch,
                        rewards_batch, dones_batch, cfg,
                        initial_stochastic=initial_stochastic,
                        initial_recurrent=initial_recurrent,
                        return_latents=True,
                    )

                    world_model_optimizer.zero_grad()
                    wm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(world_model_params, cfg.world_model_grad_norm_clip)
                    world_model_optimizer.step()
                    # world_model_scheduler.step()
                    replay_buffer.update_contexts(batch_data, wm_latents.detach())

                    total_wm_loss += wm_loss.item()
                    total_reconstruction_loss += wm_metrics['world_model/reconstruction_loss']
                    total_kl_loss += wm_metrics['world_model/kl_loss']
                    num_wm_updates += 1

                avg_wm_loss = total_wm_loss / max(num_wm_updates, 1)
                avg_reconstruction_loss = total_reconstruction_loss / max(num_wm_updates, 1)
                avg_kl_loss = total_kl_loss / max(num_wm_updates, 1)

                # ===== PHASE 3: Imagination Rollout & Actor-Critic Training =====
                if num_wm_updates > 0:
                    logger.debug(f"Iteration {iteration}: Starting imagination rollout")
                    actor.train()
                    critic.train()
                    world_model.eval()

                    total_actor_loss = 0.0
                    total_critic_loss = 0.0
                    total_ac_metrics = {}

                    set_requires_grad([world_model], False)
                    for ac_step in range(num_wm_updates):
                        batch_data = replay_buffer.sample()
                        if batch_data is None:
                            break

                        (
                            batch_size,
                            seq_len,
                            obs_batch_flat,
                            obs_target,
                            prev_actions,
                            rewards_batch,
                            dones_batch,
                            is_first_batch,
                            initial_stochastic,
                            initial_recurrent,
                        ) = prepare_sequence_batch(batch_data, cfg)

                        if batch_size > cfg.imagine_batch_size:
                            take = slice(0, cfg.imagine_batch_size)
                            batch_size = cfg.imagine_batch_size
                            prev_actions = prev_actions[take]
                            rewards_batch = rewards_batch[take]
                            dones_batch = dones_batch[take]
                            is_first_batch = is_first_batch[take]
                            obs_target = {
                                key: value[take]
                                for key, value in obs_target.items()
                            }
                            # Flatten the batch size and sequence length but still keep the shape of the remaining data so that it is okay for CNN encoders
                            obs_batch_flat = {
                                key: value.view(-1, seq_len, *value.shape[1:])[take].reshape(
                                    batch_size * seq_len, *value.shape[1:]
                                )
                                for key, value in obs_batch_flat.items()
                            }

                        with torch.no_grad():
                            embed_batch = world_model.encode(obs_batch_flat).view(batch_size, seq_len, -1)
                            posterior_latents, init_recurrent_state, _, _ = world_model.observe(
                                prev_actions,
                                embed_batch,
                                is_first_batch,
                                recurrent_state=initial_recurrent,
                                previous_stochastic=initial_stochastic,
                                deterministic=False,
                            )
                            start_count = seq_len if cfg.imagine_last == 0 else min(cfg.imagine_last, seq_len)
                            replay_latents = posterior_latents[:, -start_count:].detach()
                            replay_rewards = rewards_batch[:, -start_count:].detach()
                            replay_terminals = dones_batch[:, -start_count:].detach()
                            init_latent = replay_latents.reshape(batch_size * start_count, -1)
                            _, init_recurrent_state = world_model.rssm.split_feature(init_latent)

                        imagined_trajectories = imagination_rollout(
                            world_model.rssm,
                            actor,
                            world_model.reward_predictor,
                            world_model.terminal_predictor,
                            init_latent, init_recurrent_state, cfg
                        )
                        imagined_trajectories['start_batch_size'] = batch_size
                        imagined_trajectories['start_count'] = start_count
                        replay_trajectories = {
                            'latents': replay_latents,
                            'rewards': replay_rewards,
                            'terminals': replay_terminals,
                        }

                        actor_loss, critic_loss, ac_metrics = compute_actor_critic_loss(
                            critic, ema_critic, imagined_trajectories, cfg, return_normalizer, replay_trajectories
                        )

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.actor_grad_norm_clip)
                        actor_optimizer.step()
                        # actor_scheduler.step()

                        critic_optimizer.zero_grad()
                        critic_loss.backward()
                        torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.critic_grad_norm_clip)
                        critic_optimizer.step()
                        # critic_scheduler.step()

                        for param, ema_param in zip(critic.parameters(), ema_critic.parameters()):
                            ema_param.data.copy_(cfg.ema_critic_tau * param.data + (1.0 - cfg.ema_critic_tau) * ema_param.data)

                        total_actor_loss += actor_loss.item()
                        total_critic_loss += critic_loss.item()
                        for key, value in ac_metrics.items():
                            total_ac_metrics[key] = total_ac_metrics.get(key, 0.0) + value
                        num_ac_updates += 1

                    set_requires_grad([world_model], True)

                    if num_ac_updates > 0:
                        actor_loss_value = total_actor_loss / num_ac_updates
                        critic_loss_value = total_critic_loss / num_ac_updates
                        ac_metrics = {
                            key: value / num_ac_updates
                            for key, value in total_ac_metrics.items()
                        }
                    else:
                        actor_loss_value = None
                        critic_loss_value = None

            # Log metrics
            metrics_to_log = {
                'global_step': global_step,
                'train/average_episode_length': avg_episode_length,
                'train/episodes_finished': episodes_finished,
                'train/buffer_size': len(replay_buffer),
                'train/world_model_learning_rate': world_model_optimizer.param_groups[0]['lr'],
                'train/actor_learning_rate': actor_optimizer.param_groups[0]['lr'],
                'train/critic_learning_rate': critic_optimizer.param_groups[0]['lr'],
            }

            if len(replay_buffer) >= cfg.min_buffer_size_before_training:
                metrics_to_log.update({
                    'world_model/reconstruction_loss': avg_reconstruction_loss,
                    'world_model/kl_loss': avg_kl_loss,
                    'world_model/total_loss': avg_wm_loss,
                    'train/kl_divergence': avg_kl_loss,
                })
                if ac_metrics:
                    metrics_to_log.update(ac_metrics)
                    metrics_to_log.update({
                        'train/policy_loss': ac_metrics['actor_critic/actor_loss'],
                        'train/value_loss': ac_metrics['actor_critic/critic_loss'],
                        'train/entropy': ac_metrics['actor_critic/entropy'],
                        'train/learning_rate': actor_optimizer.param_groups[0]['lr'],
                        'train/explained_variance': ac_metrics['actor_critic/explained_variance'],
                    })

            mlflow.log_metrics(metrics_to_log, step=iteration)

            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: Steps={global_step}/{cfg.total_env_steps}, Buffer size={len(replay_buffer)}, Episodes finished={episodes_finished}")
                if len(replay_buffer) >= cfg.min_buffer_size_before_training:
                    logger.info(f"  WM Loss: {avg_wm_loss:.4f} (Recon: {avg_reconstruction_loss:.4f}, KL: {avg_kl_loss:.4f})")
                    if actor_loss_value is not None:
                        logger.info(f"  Actor Loss: {actor_loss_value:.4f}, Critic Loss: {critic_loss_value:.4f}")
            iteration += 1

    # Save models
    timestamp = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_path_world_model = f"./models/dreamer_world_model_{timestamp}.pt"
    model_path_actor = f"./models/dreamer_actor_{timestamp}.pt"
    model_path_critic = f"./models/dreamer_critic_{timestamp}.pt"

    torch.save(world_model.state_dict(), model_path_world_model)
    torch.save(actor.state_dict(), model_path_actor)
    torch.save(critic.state_dict(), model_path_critic)

    logger.info(f"Models saved to {model_path_world_model}, {model_path_actor}, {model_path_critic}")
    print(f"Models saved to {model_path_world_model}, {model_path_actor}, {model_path_critic}")


if __name__ == "__main__":
    train_dreamer()

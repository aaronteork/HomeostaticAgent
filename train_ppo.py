import datetime as dt
from dataclasses import asdict

import numpy as np
import mlflow
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from tqdm.auto import tqdm

from configs.config_ppo import PPOConfig
from utils.utils_env import create_env
from utils.utils_logger import create_logger
from utils.utils_ppo import (
    HomeostaticPPO,
    add_to_replay_buffer,
    compute_gae_from_buffer,
    create_replay_buffer,
    reshape_trajectory,
    compute_explained_variance,
)


def train_ppo():

    # Create logger
    logger = create_logger(name="PPO", log_file="./logs/logs_ppo.log")
    logger.info("Starting PPO training...")

    # Create mlflow
    mlflow.set_tracking_uri("sqlite:///runs.db?timeout=500")
    mlflow.set_experiment("HomoeostaticAgent")

    # Get config
    cfg = PPOConfig()
    logger.info(f"Config: {cfg}")

    # Create environment
    env = create_env(cfg, multiple_env=True)
    logger.info(f"Created parallel environment with {cfg.num_workers} workers")

    # Create replay buffer
    replay_buffer = create_replay_buffer(cfg)
    logger.info("Created replay buffer")

    # Create agent
    torch.set_float32_matmul_precision('high') 
    agent = HomeostaticPPO(cfg)
    agent.to(cfg.device)
    agent = torch.compile(agent)
    optimizer = Adam(agent.parameters(), lr=cfg.lr_start, eps=cfg.adam_eps)
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=cfg.total_updates)
    logger.info("Created PPO agent and optimizer")

    # Test out agent and environment
    logger.info("Running agent and environment")
    global_step = 0
    episodes_finished = 0
    obs, info = env.reset()
    # SyncVectorEnv uses next-step autoreset in this project. After an episode
    # boundary, the next env.step() ignores that worker's action and only
    # returns its reset observation. Keep the row for temporal alignment and
    # truncation bootstrapping, but exclude it from PPO optimization.
    reset_only = np.zeros(cfg.num_workers, dtype=bool)
    # PPO iterations
    with mlflow.start_run(run_name="PPO Training - " + dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")):
        mlflow.log_params(asdict(cfg))
        for iterations in range(cfg.total_updates):
            # Rollout phase
            agent.eval()
            list_iterations_episode_length = []
            for _ in range(cfg.rollout_steps):
                # Get next state and store it buffer
                with torch.no_grad():
                    actions, log_prob, entropy, value = agent(obs["vision"], obs["proprioception"], obs["internal_state"])
                actions = actions.detach().cpu().numpy()
                log_prob = log_prob.detach().cpu().numpy()
                entropy = entropy.detach().cpu().numpy()
                value = value.detach().cpu().numpy()

                # Check for NaN values in actions
                if np.any(np.isnan(actions)) or np.any(np.isinf(actions)):
                    logger.error(f"NaN/Inf detected in actions at iteration {iterations}, step {_}!")
                    logger.error(f"  actions sample: {actions[0]}")
                    raise RuntimeError("NaN/Inf detected in actions")

                next_obs, rewards, terminations, truncations, infos = env.step(actions)
                valid = ~reset_only
                add_to_replay_buffer(
                    replay_buffer,
                    obs,
                    actions,
                    log_prob,
                    rewards,
                    terminations,
                    truncations,
                    valid,
                    value,
                )

                # Debug check: log sample values on first step
                if _ == 0:
                    logger.debug(f"[ROLLOUT] Sample log_prob: {log_prob[0]}, shape: {log_prob.shape}")
                    logger.debug(f"[ROLLOUT] Sample entropy: {entropy[0]}, shape: {entropy.shape}")
                    logger.debug(f"[ROLLOUT] Sample value: {value[0]}, shape: {value.shape}")

                # Next step and global step update
                obs = next_obs
                global_step += int(valid.sum())
                reset_only = (terminations | truncations).astype(bool, copy=True)

                # Track metrics
                if "_episode" in infos:
                    for i in range(cfg.num_workers):
                        if infos["_episode"][i]:
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
                                }, step=episodes_finished
                            )
            # Get the value of the very last observation in your rollout
            trajectory = replay_buffer.sample(cfg.rollout_steps * cfg.num_workers)

            # Keep observations on CPU for indexing, move scalar data to GPU for GAE
            trajectory_for_gae = {
                "rewards": trajectory["rewards"].to(cfg.device),
                "values": trajectory["values"].to(cfg.device),
                "terminations": trajectory["terminated"].to(cfg.device),
                "truncations": trajectory["truncated"].to(cfg.device),
                "valid": trajectory["valid"].to(cfg.device),
            }

            # Reshape for GAE computation, then flatten back
            trajectory_reshaped = reshape_trajectory(trajectory_for_gae, cfg.rollout_steps, cfg.num_workers)
            advantages, returns = compute_gae_from_buffer(
                rewards=trajectory_reshaped["rewards"],
                values=trajectory_reshaped["values"],
                next_obs=obs,
                terminations=trajectory_reshaped["terminations"],
                truncations=trajectory_reshaped["truncations"],
                agent=agent,
                gamma=cfg.gamma,
                lam=cfg.gae_lambda
            )

            # Flatten advantages and returns back to [batch_size] for training
            valid = trajectory_reshaped["valid"].reshape(-1).bool()
            advantages = advantages.reshape(-1)[valid]
            returns = returns.reshape(-1)[valid]

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

            # Use original flat trajectory for unpacking (keep on CPU to avoid indexing issues on CUDA)
            valid_cpu = valid.cpu()
            obs_trajectory = {
                key: value[valid_cpu] for key, value in trajectory["obs"].items()
            }
            actions = trajectory["actions"][valid_cpu]
            old_log_probs = trajectory["log_probs"][valid_cpu]
            old_values = trajectory["values"].squeeze(-1)[valid_cpu]
            advantages = advantages.cpu() if advantages.is_cuda else advantages
            returns = returns.cpu() if returns.is_cuda else returns

            # Update phase - multiple epochs
            agent.train()
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy_loss = 0
            total_explained_var = 0
            num_updates = 0

            for epoch in range(cfg.epochs):
                # Create random permutation of all data
                indices = torch.randperm(len(advantages))

                # Create minibatches
                for start_idx in range(0, len(advantages), cfg.minibatch_size):
                    end_idx = min(start_idx + cfg.minibatch_size, len(advantages))
                    batch_indices = indices[start_idx:end_idx]
                    batch_indices_cpu = batch_indices.cpu() if batch_indices.is_cuda else batch_indices

                    # Get minibatch data
                    batch_obs = {
                        "vision": obs_trajectory["vision"][batch_indices_cpu].to(cfg.device),
                        "proprioception": obs_trajectory["proprioception"][batch_indices_cpu].to(cfg.device),
                        "internal_state": obs_trajectory["internal_state"][batch_indices_cpu].to(cfg.device),
                    }
                    batch_actions = actions[batch_indices_cpu].to(cfg.device)
                    batch_old_log_probs = old_log_probs[batch_indices_cpu].to(cfg.device)
                    batch_advantages = advantages[batch_indices_cpu].to(cfg.device)
                    batch_returns = returns[batch_indices_cpu].to(cfg.device)
                    batch_old_values = old_values[batch_indices_cpu].to(cfg.device)

                    # Forward pass - evaluate log_prob of old actions under new policy
                    new_log_probs, entropy, new_values = agent(
                        batch_obs["vision"],
                        batch_obs["proprioception"],
                        batch_obs["internal_state"],
                        evaluate_actions=batch_actions
                    )
                    new_values = new_values.squeeze(-1)
                    total_explained_var += compute_explained_variance(new_values, batch_returns)

                    # Policy loss (PPO clipped surrogate objective)
                    log_ratio = new_log_probs - batch_old_log_probs
                    ratio = torch.exp(log_ratio)
                    # Compute KL divergence
                    kl_div = -log_ratio.mean()
                    if kl_div > cfg.target_kl:
                        break

                    # Sanity check: ratio should be 1.0 in first epoch/minibatch (no updates yet)
                    if iterations == 0 and epoch == 0 and start_idx == 0:
                        ratio_mean = ratio.mean().item()
                        ratio_std = ratio.std().item()
                        logger.debug(f"[SANITY CHECK] First minibatch ratio - mean: {ratio_mean:.6f}, std: {ratio_std:.6f}")
                        if abs(ratio_mean - 1.0) > 0.01:
                            logger.warning("WARNING: Ratio not close to 1.0! Check if log_probs were saved correctly.")

                        # Additional diagnostic: check log_probs consistency
                        with torch.no_grad():
                            det_log_probs, _, _ = agent(
                                batch_obs["vision"], batch_obs["proprioception"],
                                batch_obs["internal_state"], deterministic=True,
                                evaluate_actions=batch_actions
                            )
                        logger.debug(f"[DEBUG] Stochastic log_probs sample: {new_log_probs[0:3]}")
                        logger.debug(f"[DEBUG] Deterministic log_probs sample: {det_log_probs[0:3]}")
                        logger.debug(f"[DEBUG] Old log_probs sample: {batch_old_log_probs[0:3]}")

                    pg_loss1 = -batch_advantages * ratio
                    pg_loss2 = -batch_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                    policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss (clipped)
                    value_pred_clipped = batch_old_values + (new_values - batch_old_values).clamp(-cfg.clip_coef, cfg.clip_coef)
                    value_loss_unclipped = (new_values - batch_returns) ** 2
                    value_loss_clipped = (value_pred_clipped - batch_returns) ** 2
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                    # Entropy bonus
                    entropy_loss = entropy.mean()

                    # Total loss
                    loss = policy_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * value_loss

                    # Check for NaN before backprop
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.error(f"NaN/Inf detected in loss at iteration {iterations}, epoch {epoch}!")
                        logger.error(f"  policy_loss: {policy_loss}, entropy_loss: {entropy_loss}, value_loss: {value_loss}")
                        logger.error(f"  new_log_probs sample: {new_log_probs[:3]}")
                        logger.error(f"  new_values sample: {new_values[:3]}")
                        logger.error(f"  batch_advantages sample: {batch_advantages[:3]}")
                        raise RuntimeError("NaN/Inf detected in loss")

                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                    optimizer.step()

                    # Track metrics
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy_loss += entropy_loss.item()
                    num_updates += 1

            # Update learning rate
            scheduler.step()

            # Log training metrics
            avg_policy_loss = total_policy_loss / num_updates if num_updates > 0 else 0
            avg_value_loss = total_value_loss / num_updates if num_updates > 0 else 0
            avg_entropy = total_entropy_loss / num_updates if num_updates > 0 else 0
            avg_episode_length = sum(list_iterations_episode_length) / len(list_iterations_episode_length) if list_iterations_episode_length else 0
            avg_explained_var = total_explained_var / num_updates if num_updates > 0 else 0

            mlflow.log_metrics(
                {
                    "train/policy_loss": avg_policy_loss,
                    "train/value_loss": avg_value_loss,
                    "train/entropy": avg_entropy,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/kl_divergence": kl_div.item(),
                    "global_step": global_step,
                    "train/average_episode_length": avg_episode_length,
                    "train/episodes_finished": episodes_finished,
                    "train/explained_variance": avg_explained_var,  # Add this line
                },
                step=iterations
            )

            logger.info(f"Iteration {iterations}: Policy Loss={avg_policy_loss:.4f}, Value Loss={avg_value_loss:.4f}, Entropy={avg_entropy:.4f}")

            replay_buffer.empty()

            # Clear GPU cache to prevent memory accumulation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    model_path = f"./models/ppo_agent_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt"
    torch.save(agent.state_dict(), model_path)
    print(f"model saved to {model_path}")

if __name__ == "__main__":
    train_ppo()

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import (
    kl_divergence,
)

from configs.config_dreamer import DreamerConfig


# ---------------- Helper functions ---------------------------
def to_tensor(value, device, dtype=torch.float32):
    """Convert numpy arrays or tensors to float tensors on the training device."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


# From https://github.com/NM512/r2dreamer/blob/546e4fab8146ea4b14e1d7726bbc1a8a1d50322f/distributions.py
def to_f32(x):
    return x.to(dtype=torch.float32)


# From https://github.com/NM512/r2dreamer/blob/546e4fab8146ea4b14e1d7726bbc1a8a1d50322f/distributions.py
def to_i32(x):
    return x.to(dtype=torch.int32)


# From https://github.com/NM512/r2dreamer/blob/546e4fab8146ea4b14e1d7726bbc1a8a1d50322f/distributions.py
def symexp_twohot(logits, bin_num, **kwargs):
    if bin_num % 2 == 1:
        half = torch.linspace(-20, 0, (bin_num - 1) // 2 + 1, dtype=torch.float32, device=logits.device)
        half = symexp(half)
        bins = torch.concatenate([half, -half[:-1].flip(dims=(0,))], 0)
    else:
        half = torch.linspace(-20, 0, bin_num // 2, dtype=torch.float32, device=logits.device)
        half = symexp(half)
        bins = torch.concatenate([half, -half.flip(dims=(0,))], 0)
    return TwoHot(to_f32(logits), bins)


# From https://github.com/Eclectic-Sheep/sheeprl/blob/main/sheeprl/utils/utils.py
# From https://github.com/danijar/dreamerv3/blob/8fa35f83eee1ce7e10f3dee0b766587d0a713a60/dreamerv3/jaxutils.py
def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x: Tensor) -> Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def obs_to_tensor_dict(obs, cfg):
    keys = ["vision", "proprioception", "internal_state"]
    if cfg.num_heat > 0:
        keys.append("heat_sensor")
    return {key: to_tensor(obs[key], cfg.device) for key in keys}


def set_requires_grad(modules, requires_grad):
    for module in modules:
        for param in module.parameters():
            param.requires_grad_(requires_grad)


# ------------------ Data and replay preparation ------------------------
def stack_sequence_observations(obs_sequences, cfg):
    keys = ["vision", "proprioception", "internal_state"]
    if cfg.num_heat > 0:
        keys.append("heat_sensor")

    stacked = {}
    for key in keys:
        flat_values = []
        for obs_seq in obs_sequences:
            for obs_dict in obs_seq:
                flat_values.append(to_tensor(obs_dict[key], cfg.device).squeeze(0))
        stacked[key] = torch.stack(flat_values, dim=0)
    return stacked


def reshape_sequence_observations(obs_flat, batch_size, seq_len):
    return {
        key: value.view(batch_size, seq_len, *value.shape[1:])
        for key, value in obs_flat.items()
    }


def prepare_sequence_batch(batch_data, cfg):
    batch_size = len(batch_data["obs"])
    seq_len = cfg.batch_length
    context_len = cfg.replay_context
    stored_seq_len = seq_len + context_len
    if len(batch_data["obs"][0]) != stored_seq_len:
        raise ValueError(
            f"expected replay sequence length {stored_seq_len}, got {len(batch_data['obs'][0])}"
        )
    obs_batch_full_flat = stack_sequence_observations(batch_data["obs"], cfg)
    obs_target_full = reshape_sequence_observations(
        obs_batch_full_flat, batch_size, stored_seq_len
    )
    obs_target = {
        key: value[:, context_len:] for key, value in obs_target_full.items()
    }
    # The replay context only restores the RSSM state and supplies the first
    # previous action. It is not part of the world-model loss sequence. Keep
    # the encoder input aligned with the post-context targets; otherwise
    # flattening B * (T + context) rows and reshaping them as B * T silently
    # inflates the embedding dimension (for example 1536 -> 1560 at T=64).
    obs_batch_flat = {
        key: value.reshape(batch_size * seq_len, *value.shape[2:])
        for key, value in obs_target.items()
    }

    actions = torch.stack(
        [
            to_tensor(action, cfg.device).squeeze(0)
            for seq in batch_data["actions"]
            for action in seq
        ]
    ).view(batch_size, stored_seq_len, -1)
    rewards_full = torch.stack(
        [
            to_tensor(reward, cfg.device).reshape(())
            for seq in batch_data["rewards"]
            for reward in seq
        ]
    ).view(batch_size, stored_seq_len)
    terminals_full = to_tensor(
        np.asarray(batch_data["terminals"]).squeeze(-1), cfg.device
    )
    is_last_full = to_tensor(
        np.asarray(batch_data["is_last"]).squeeze(-1), cfg.device
    )
    is_first_full = to_tensor(np.asarray(batch_data["is_first"]), cfg.device)

    if context_len:
        prev_actions = actions[:, context_len - 1 : -1]
    else:
        prev_actions = torch.zeros_like(actions)
        prev_actions[:, 1:] = actions[:, :-1]
    actions = actions[:, context_len:]
    rewards = rewards_full[:, context_len:]
    terminals = terminals_full[:, context_len:]
    is_last = is_last_full[:, context_len:]
    is_first = is_first_full[:, context_len:]

    reset_mask = 1.0 - is_first
    prev_actions = prev_actions * reset_mask.unsqueeze(-1)

    initial_stochastic = None
    initial_recurrent = None
    if context_len and "context_stochastic" in batch_data and "context_recurrent" in batch_data:
        initial_stochastic = torch.stack(
            [
                to_tensor(seq[context_len - 1], cfg.device).squeeze(0)
                for seq in batch_data["context_stochastic"]
            ]
        )
        initial_recurrent = torch.stack(
            [
                to_tensor(seq[context_len - 1], cfg.device).squeeze(0)
                for seq in batch_data["context_recurrent"]
            ]
        )

    return (
        batch_size,
        seq_len,
        obs_batch_flat,
        obs_target,
        actions,
        prev_actions,
        rewards,
        terminals,
        is_last,
        is_first,
        initial_stochastic,
        initial_recurrent,
    )


# ------------------ World Model Training functions ---------------------------
def compute_world_model_loss(
    rssm,
    decoder,
    reward_predictor,
    continue_predictor,
    prev_action,
    action,
    embed,
    obs,
    is_first,
    reward,
    terminal,
    config: DreamerConfig,
    initial_stochastic=None,
    initial_recurrent=None,
    return_latents=False,
):
    """
    Compute complete world model loss: reconstruction + KL + reward + continue.

    Args:
        rssm: RSSM model
        decoder: CNN decoder
        reward_predictor: Reward prediction network
        continue_predictor: Continue prediction network
        prev_action: (batch, seq_len, action_dim)
        action: (batch, seq_len, action_dim)
        embed: (batch, seq_len, encoder_dim)
        obs: dict of observation targets
        is_first: (batch, seq_len)
        reward: (batch, seq_len) - target rewards
        terminal: (batch, seq_len) - true environmental terminal targets;
            time-limit truncations remain nonterminal
        config: DreamerConfig

    Returns:
        loss: scalar loss
        metrics: dict with loss components
    """
    batch_size, seq_len = prev_action.shape[0], prev_action.shape[1]

    # Forward through RSSM
    # Compute RSSM outputs here to get the grad
    latent, recurrent_state, prior_dists, posterior_dists = rssm(
        prev_action,
        embed,
        is_first,
        recurrent_state=initial_recurrent,
        previous_stochastic=initial_stochastic,
        deterministic=False,
    )

    # ===== Reconstruction Loss (decoder recovers Ant observation dict) =====
    # Use normal MSE for images
    # Use symlog + MSE for proprioception and internal state to handle large dynamic range since they are vector inputs
    # This is a simpler way of doing compared to SheepRL's implementation which has a MSEDistribution
    reconstructed_obs = decoder(latent)
    recon_losses = {
        key: F.mse_loss(reconstructed_obs[key], obs[key].detach(), reduction="none").sum(dim=(-3, -2, -1)).mean()
        if key == "vision"
        else F.mse_loss(reconstructed_obs[key], symlog(obs[key].detach()), reduction="none").sum(dim=-1).mean()
        for key in reconstructed_obs.keys()
    }
    recon_loss = (
        config.vision_reconstruction_weight * recon_losses["vision"]
        + config.proprioception_reconstruction_weight * recon_losses["proprioception"]
        + config.internal_state_reconstruction_weight * recon_losses["internal_state"]
    )
    if "heat_sensor" in recon_losses:
        recon_loss = (
            recon_loss
            + config.heat_sensor_reconstruction_weight * recon_losses["heat_sensor"]
        )

    # ===== DreamerV3 KL balancing with free nats =====
    free_nats = torch.tensor(config.free_nats, device=prev_action.device)
    dyn_loss = torch.tensor(0.0, device=prev_action.device)
    rep_loss = torch.tensor(0.0, device=prev_action.device)

    for prior_dist, posterior_dist in zip(prior_dists, posterior_dists):
        prior_logits = prior_dist.base_dist.logits
        posterior_logits = posterior_dist.base_dist.logits
        prior_sg = rssm._dist(prior_logits.detach())
        posterior_sg = rssm._dist(posterior_logits.detach())

        dyn_kl = kl_divergence(posterior_sg, prior_dist)
        rep_kl = kl_divergence(posterior_dist, prior_sg)
        dyn_loss = dyn_loss + torch.maximum(dyn_kl, free_nats).mean()
        rep_loss = rep_loss + torch.maximum(rep_kl, free_nats).mean()

    dyn_loss = dyn_loss / len(prior_dists)
    rep_loss = rep_loss / len(prior_dists)

    # ===== Reward Prediction Loss =====
    # Replay rows follow the Dreamer convention: latent_t is built from
    # (obs_t, action_{t-1}), and reward_t/terminal_t are the transition outcome that
    # led into obs_t. Train reward and continuation from the posterior feature,
    # matching the official DreamerV3 world-model loss.
    reward_target = reward.contiguous().view(batch_size, seq_len, 1)
    reward_dist = reward_predictor(latent)  # two-hot distribution over rewards
    reward_loss = -reward_dist.log_prob(reward_target).mean()

    # ===== Continue Prediction Loss =====
    # DreamerV3 predicts continuation, and with contdisc enabled it predicts the
    # discounted continuation used directly in lambda returns.
    # BCE on sigmoid probabilities is not autocast-safe. Train directly from
    # logits, while the predictor's normal forward path continues to expose
    # probabilities for imagination and diagnostics.
    predicted_continue_logits = continue_predictor.forward_logits(latent)
    terminal_target = terminal.contiguous().view(batch_size, seq_len, 1).float()
    continue_target = 1.0 - terminal_target
    if config.contdisc:
        continue_target = continue_target * config.discount
    continue_loss = F.binary_cross_entropy_with_logits(
        predicted_continue_logits, continue_target
    )
    # ===== Combined Loss with Weighting =====
    loss = (
        recon_loss
        + config.reward_weight * reward_loss
        + config.continue_weight * continue_loss
        + config.dyn_loss_weight * dyn_loss
        + config.rep_loss_weight * rep_loss
    )

    metrics = {
        # "world_model/reconstruction_loss": recon_loss.item(),
        # "world_model/reconstruction_vision_loss": recon_losses["vision"].item(),
        # "world_model/reconstruction_proprioception_loss": recon_losses[
        #     "proprioception"
        # ].item(),
        # "world_model/reconstruction_internal_state_loss": recon_losses[
        #     "internal_state"
        # ].item(),
        "world_model/dyn_loss": dyn_loss.item(),
        "world_model/rep_loss": rep_loss.item(),
        # "world_model/reward_loss": reward_loss.item(),
        # "world_model/continue_loss": continue_loss.item(),
        # "world_model/predicted_continue": predicted_continue.mean().item(),
        # "world_model/total_loss": loss.item(),
    }
    if "heat_sensor" in recon_losses:
        metrics["world_model/reconstruction_heat_sensor_loss"] = recon_losses[
            "heat_sensor"
        ].item()

    if return_latents:
        return loss, metrics, latent
    return loss, metrics


def imagination_rollout(
    rssm,
    actor,
    reward_predictor,
    continue_predictor,
    init_latent,
    init_recurrent_state,
    config: DreamerConfig,
):
    """
    Rollout imagination trajectories for actor-critic training.

    The rollout starts from detached posterior states sampled from replay. For
    DreamerV3, actor learning uses a REINFORCE-style surrogate with stopped
    imagined returns, so gradients should flow through the actor log-probs and
    entropy terms, not through predicted dynamics, rewards, or terminals.

    Imagined rewards and continues is not for the latent on the same index, but the one after

    Args:
        rssm: RSSM model
        actor: Actor network
        reward_predictor: Reward predictor network
        continue_predictor: Continue predictor network
        init_latent: (batch, latent_dim) - posterior latent from real obs
        init_recurrent_state: (batch, gru_units)
        config: DreamerConfig

    Returns:
        imagined_trajectories: dict with a single latent state sequence,
        actions, rewards, continues, entropies, and log-probs.
    """
    horizon = config.imagine_horizon

    # Initialize from replay states without backpropagating into representation learning.
    latent = init_latent.detach()
    recurrent_state = init_recurrent_state.detach()

    imagined_latents = [latent]
    imagined_actions = []
    imagined_rewards = []
    imagined_continues = []
    imagined_entropies = []
    imagined_log_probs = []

    # Official DreamerV3 includes continuation at the replay start when
    # constructing cumulative actor/value loss weights. This masks imagined
    # losses that start from terminal posterior states. Keep the factor in the
    # same discounted convention as successor continuation factors.
    with torch.no_grad():
        start_continue = continue_predictor(latent).squeeze(-1)
        if not config.contdisc:
            start_continue = start_continue * config.gamma

    for step in range(horizon):
        action, _, dist = actor(latent.detach(), deterministic=False)
        imagined_actions.append(action)
        imagined_log_probs.append(dist.log_prob(action.detach()).sum(dim=-1))
        imagined_entropies.append(dist.entropy().sum(dim=-1))

        with torch.no_grad():
            latent, recurrent_state, _ = rssm.imagine_step(
                latent,
                recurrent_state,
                action,
                deterministic=False,
            )

            imagined_latents.append(latent)
            reward_dist = reward_predictor(latent)
            imagined_rewards.append(reward_dist.mode.squeeze(-1))
            continue_head_output = continue_predictor(latent)
            if config.contdisc:
                continue_factor = continue_head_output
            else:
                continue_factor = continue_head_output * config.gamma
            imagined_continues.append(continue_factor.squeeze(-1))

    imagined_latents = torch.stack(
        imagined_latents, dim=1
    )  # (batch, horizon + 1, latent_dim), s_0 ... s_H
    imagined_actions = torch.stack(
        imagined_actions, dim=1
    )  # (batch, horizon, action_dim)
    imagined_rewards = torch.stack(imagined_rewards, dim=1)  # (batch, horizon)
    imagined_continues = torch.stack(imagined_continues, dim=1)  # (batch, horizon)
    imagined_entropies = torch.stack(imagined_entropies, dim=1)  # (batch, horizon)
    imagined_log_probs = torch.stack(imagined_log_probs, dim=1)  # (batch, horizon)

    return {
        "latents": imagined_latents,
        "actions": imagined_actions,
        "rewards": imagined_rewards,
        "continues": imagined_continues,
        "start_continue": start_continue,
        "entropies": imagined_entropies,
        "log_probs": imagined_log_probs,
    }


def compute_lambda_returns_from_continues(
    rewards, values, continues, bootstrap, lam=0.95
):
    """Compute Dreamer-style lambda returns from continuation factors."""
    _, horizon = rewards.shape
    returns = torch.zeros_like(rewards)
    next_return = bootstrap

    for t in reversed(range(horizon)):
        next_value = bootstrap if t == horizon - 1 else values[:, t + 1]
        target = rewards[:, t] + continues[:, t] * (
            (1.0 - lam) * next_value + lam * next_return
        )
        returns[:, t] = target
        next_return = target

    return returns


def compute_discount_weights_from_continues(
    continues, start_continue=None, discount_factor=1.0
):
    _, horizon = continues.shape
    discount_weights = torch.ones_like(continues)
    if start_continue is not None:
        if discount_factor <= 0:
            raise ValueError(f"discount_factor must be positive, got {discount_factor}")
        discount_weights[:, 0] = start_continue / discount_factor
    for t in range(1, horizon):
        discount_weights[:, t] = discount_weights[:, t - 1] * continues[:, t - 1]
    return discount_weights


def compute_replay_lambda_returns(
    rewards, terminals, is_last, imagination_bootstraps, config
):
    """Return imagination-annotated replay targets without crossing boundaries.

    True terminals suppress bootstrapping. Truncated ``is_last`` rows stop the
    lambda trace but retain the one-step imagination bootstrap from their final
    observation. Each replay state has its own imagination return annotation,
    matching DreamerV3/R2Dreamer replay value learning.
    """
    if rewards.shape[1] < 2:
        raise ValueError("Replay value loss needs at least two timesteps")
    if terminals.shape != rewards.shape or is_last.shape != rewards.shape:
        raise ValueError(
            "rewards, terminals, and is_last must have matching (batch, time) shapes"
        )
    if imagination_bootstraps.shape != rewards.shape:
        raise ValueError(
            "imagination_bootstraps must provide one annotation per replay state"
        )

    rewards_next = rewards[:, 1:]
    terminals_next = terminals[:, 1:]
    is_last_next = is_last[:, 1:]
    current_is_last = is_last[:, :-1]
    live_next = (1.0 - terminals_next.float()) * config.discount
    trace_next = (1.0 - is_last_next.float()) * config.gae_lambda

    replay_returns = torch.zeros_like(rewards_next)
    detached_bootstraps = imagination_bootstraps.detach()
    next_return = detached_bootstraps[:, -1]
    for t in reversed(range(rewards_next.shape[1])):
        next_bootstrap = detached_bootstraps[:, t + 1]
        target = rewards_next[:, t] + live_next[:, t] * (
            (1.0 - trace_next[:, t]) * next_bootstrap
            + trace_next[:, t] * next_return
        )
        replay_returns[:, t] = target
        next_return = target

    # Replay sequences can cross reset boundaries, so cumulative imagination
    # weights are inappropriate here. Match official replay value learning by
    # masking only boundary rows; reset rows start a fresh episode at weight 1.
    discount_weights = 1.0 - current_is_last.detach().float()
    return replay_returns, discount_weights


def compute_replay_value_loss(
    critic,
    ema_critic,
    replay_trajectories,
    imagination_bootstraps,
    config: DreamerConfig,
):
    # Let replay-value prediction shape the posterior encoder/RSSM features.
    # Replay targets and environment labels remain detached below.
    latents = replay_trajectories["latents"]
    rewards = replay_trajectories["rewards"].detach()
    terminals = replay_trajectories["terminals"].detach()
    is_last = replay_trajectories["is_last"].detach()

    with torch.no_grad():
        replay_returns, discount_weights = compute_replay_lambda_returns(
            rewards,
            terminals,
            is_last,
            imagination_bootstraps=imagination_bootstraps,
            config=config,
        )

    value_dist = critic(latents[:, :-1])
    replay_return_loss = -(
        discount_weights * value_dist.log_prob(replay_returns.unsqueeze(-1))
    ).mean()
    with torch.no_grad():
        ema_values = ema_critic(latents[:, :-1]).mode
    replay_slow_loss = -(
        discount_weights * value_dist.log_prob(ema_values.detach())
    ).mean()
    replay_loss = (
        replay_return_loss + config.critic_slow_regularization * replay_slow_loss
    )
    return replay_loss, replay_return_loss, replay_slow_loss, replay_returns


def compute_actor_critic_loss(
    critic,
    ema_critic,
    imagined_trajectories,
    config: DreamerConfig,
    return_normalizer=None,
    replay_trajectories=None,
):
    """
    Compute actor and critic losses on imagined trajectories.

    Args:
        critic: Critic network
        ema_critic: Exponential moving average of the critic for value regularization
        imagined_trajectories: dict with imagined data from imagination_rollout
        config: DreamerConfig

    Returns:
        actor_loss: scalar
        critic_loss: scalar
        metrics: dict with loss components
    """
    # Imagination stores a single state sequence s_0 ... s_H. Policy log-probs
    # were produced from s_t, while rewards/continues were predicted from
    # s_{t+1}; use the pre-action slice for actor and value losses.
    latent_sequence = imagined_trajectories["latents"]
    rewards = imagined_trajectories["rewards"]
    continues = imagined_trajectories["continues"]
    start_continue = imagined_trajectories["start_continue"]
    entropies = imagined_trajectories["entropies"]
    log_probs = imagined_trajectories["log_probs"]

    batch_size, horizon = rewards.shape
    if latent_sequence.shape[1] != horizon + 1:
        raise ValueError(
            "imagined_trajectories['latents'] must contain horizon + 1 states"
        )
    latents = latent_sequence[:, :-1]
    last_latent = latent_sequence[:, -1]

    with torch.no_grad():
        current_values = critic(latents.detach()).mode.squeeze(-1)
        bootstrap = critic(last_latent.detach()).mode.squeeze(-1)
        critic_returns = compute_lambda_returns_from_continues(
            rewards=rewards.detach(),
            values=current_values,
            continues=continues.detach(),
            bootstrap=bootstrap,
            lam=config.gae_lambda,
        )
        if return_normalizer is None:
            return_offset = critic_returns.detach().quantile(
                config.return_norm_percentile_low / 100.0
            )
            return_high = critic_returns.detach().quantile(
                config.return_norm_percentile_high / 100.0
            )
            return_scale = (return_high - return_offset).clamp_min(
                config.return_norm_limit
            )
        else:
            return_offset, return_scale = return_normalizer(critic_returns, update=True)

        advantages = (critic_returns - current_values) / return_scale

    with torch.no_grad():
        discount_weights = compute_discount_weights_from_continues(
            continues.detach(),
            start_continue=start_continue.detach(),
            discount_factor=config.discount,
        )

    entropy = entropies.mean()
    actor_objective = (
        discount_weights
        * (log_probs * advantages.detach() + config.ent_coef * entropies)
    ).mean()
    actor_loss = -actor_objective

    value_dist = critic(latents.detach())
    critic_return_loss = -(
        discount_weights * value_dist.log_prob(critic_returns.unsqueeze(-1))
    ).mean()
    predicted_values = value_dist.mode.squeeze(-1)
    returns_var = torch.var(critic_returns)
    if returns_var.item() < 1e-10:
        explained_variance = 0.0
    else:
        explained_variance = (
            1.0 - torch.var(critic_returns - predicted_values) / returns_var
        ).item()
    with torch.no_grad():
        ema_values = ema_critic(latents.detach()).mode
    critic_slow_loss = -(
        discount_weights * value_dist.log_prob(ema_values.detach())
    ).mean()
    critic_imagined_loss = (
        critic_return_loss + config.critic_slow_regularization * critic_slow_loss
    )
    critic_loss = config.critic_imagined_loss * critic_imagined_loss

    replay_loss = torch.zeros((), device=latents.device)
    replay_return_loss = torch.zeros((), device=latents.device)
    replay_slow_loss = torch.zeros((), device=latents.device)
    replay_returns = None
    if replay_trajectories is not None and config.critic_replay_loss > 0.0:
        start_batch_size = imagined_trajectories.get("start_batch_size")
        start_count = imagined_trajectories.get("start_count")
        if start_batch_size is None or start_count is None:
            raise ValueError(
                "imagined_trajectories must include start_batch_size and start_count for replay value loss"
            )
        replay_bootstraps = (
            critic_returns[:, 0]
            .detach()
            .view(start_batch_size, start_count)
        )
        replay_loss, replay_return_loss, replay_slow_loss, replay_returns = (
            compute_replay_value_loss(
                critic,
                ema_critic,
                replay_trajectories,
                replay_bootstraps,
                config,
            )
        )
        critic_loss = critic_loss + config.critic_replay_loss * replay_loss

    metrics = {
        "actor_critic/actor_loss": actor_loss.item(),
        "actor_critic/critic_loss": critic_loss.item(),
        # "actor_critic/critic_return_loss": critic_return_loss.item(),
        # "actor_critic/critic_slow_loss": critic_slow_loss.item(),
        # "actor_critic/critic_imagined_loss": critic_imagined_loss.item(),
        # "actor_critic/critic_replay_loss": replay_loss.item(),
        # "actor_critic/critic_replay_return_loss": replay_return_loss.item(),
        # "actor_critic/critic_replay_slow_loss": replay_slow_loss.item(),
        "actor_critic/entropy": entropy.item(),
        # "actor_critic/actor_objective": actor_objective.item(),
        # "actor_critic/average_return": critic_returns.mean().item(),
        # "actor_critic/return_norm_offset": return_offset.item(),
        # "actor_critic/return_norm_scale": return_scale.item(),
        # "actor_critic/average_advantage": advantages.mean().item(),
        # "actor_critic/advantage_std": advantages.std().item(),
        # "actor_critic/log_prob": log_probs.mean().item(),
        "actor_critic/explained_variance": explained_variance,
    }
    if replay_returns is not None:
        metrics["actor_critic/replay_return"] = replay_returns.mean().item()

    return actor_loss, critic_loss, metrics


# ------------------ Helper Classes ---------------------------
# From https://github.com/Eclectic-Sheep/sheeprl/blob/main/sheeprl/utils/distribution.py
class TwoHotEncodingDistribution:
    def __init__(
        self,
        logits: Tensor,
        dims: int = 0,
        low: int = -20,
        high: int = 20,
        transfwd: Callable[[Tensor], Tensor] = symlog,
        transbwd: Callable[[Tensor], Tensor] = symexp,
    ):
        self.logits = logits
        self.probs = F.softmax(logits, dim=-1)
        self.dims = tuple([-x for x in range(1, dims + 1)])
        self.bins = torch.linspace(low, high, logits.shape[-1], device=logits.device)
        self.low = low
        self.high = high
        self.transfwd = transfwd
        self.transbwd = transbwd
        self._batch_shape = logits.shape[: len(logits.shape) - dims]
        self._event_shape = logits.shape[len(logits.shape) - dims : -1] + (1,)

    @property
    def mean(self) -> Tensor:
        return self.transbwd((self.probs * self.bins).sum(dim=self.dims, keepdim=True))

    @property
    def mode(self) -> Tensor:
        return self.transbwd((self.probs * self.bins).sum(dim=self.dims, keepdim=True))

    def log_prob(self, x: Tensor) -> Tensor:
        x = self.transfwd(x)
        # below in [-1, len(self.bins) - 1]
        below = (self.bins <= x).type(torch.int32).sum(dim=-1, keepdim=True) - 1
        # above in [0, len(self.bins)]
        above = below + 1

        # above in [0, len(self.bins) - 1]
        above = torch.minimum(above, torch.full_like(above, len(self.bins) - 1))
        # below in [0, len(self.bins) - 1]
        below = torch.maximum(below, torch.zeros_like(below))

        equal = below == above
        dist_to_below = torch.where(equal, 1, torch.abs(self.bins[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.bins[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, len(self.bins)) * weight_below[..., None]
            + F.one_hot(above, len(self.bins)) * weight_above[..., None]
        ).squeeze(-2)
        log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdims=True)
        return (target * log_pred).sum(dim=self.dims)


# From https://github.com/NM512/r2dreamer/blob/546e4fab8146ea4b14e1d7726bbc1a8a1d50322f/distributions.py
class TwoHot:
    def __init__(self, logits, bins, squash=None, unsquash=None):
        # (..., N_bins), (N_bins,)
        self.logits = to_f32(logits)
        assert self.logits.shape[-1] == len(bins), (self.logits.shape, len(bins))

        self.bins = bins
        self.probs = F.softmax(self.logits, dim=-1)  # (..., N_bins)
        self.squash = squash if squash is not None else (lambda x: x)
        self.unsquash = unsquash if unsquash is not None else (lambda x: x)

    @property
    def mode(self):
        # (..., N_bins), (N_bins,) -> (..., 1)
        n = self.logits.shape[-1]
        if n % 2 == 1:
            m = (n - 1) // 2
            p1 = self.probs[..., :m]
            p2 = self.probs[..., m : m + 1]
            p3 = self.probs[..., m + 1 :]
            b1 = self.bins[..., :m]
            b2 = self.bins[..., m : m + 1]
            b3 = self.bins[..., m + 1 :]
            wavg = (p2 * b2).sum(dim=-1, keepdim=True) + ((p1 * b1).flip(dims=(-1,)) + (p3 * b3)).sum(
                dim=-1, keepdim=True
            )
            return self.unsquash(wavg)
        p1 = self.probs[..., : n // 2]
        p2 = self.probs[..., n // 2 :]
        b1 = self.bins[..., : n // 2]
        b2 = self.bins[..., n // 2 :]
        wavg = ((p1 * b1).flip(dims=(-1,)) + (p2 * b2)).sum(dim=-1, keepdim=True)
        return self.unsquash(wavg)

    def log_prob(self, target):
        # (..., 1)
        assert target.dtype == self.probs.dtype
        target = target.squeeze(-1)  # (...,)
        target_squashed = self.squash(target).detach()  # (...,)
        # below/above: (...,)
        below = to_i32(self.bins <= target_squashed.unsqueeze(-1)).sum(dim=-1) - 1
        above = len(self.bins) - to_i32(self.bins > target_squashed.unsqueeze(-1)).sum(dim=-1)
        below = torch.clamp(below, 0, len(self.bins) - 1)
        above = torch.clamp(above, 0, len(self.bins) - 1)
        equal = below == above
        dist_to_below = torch.where(
            equal,
            torch.tensor(1.0, device=target.device, dtype=torch.float32),
            (self.bins[below] - target_squashed).abs(),
        )
        dist_to_above = torch.where(
            equal,
            torch.tensor(1.0, device=target.device, dtype=torch.float32),
            (self.bins[above] - target_squashed).abs(),
        )
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        oh_below = to_f32(F.one_hot(below, num_classes=len(self.bins)))
        oh_above = to_f32(F.one_hot(above, num_classes=len(self.bins)))
        # (..., N_bins)
        mixed_target = oh_below * weight_below.unsqueeze(-1) + oh_above * weight_above.unsqueeze(-1)
        log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdim=True)  # (..., N_bins)
        return (mixed_target * log_pred).sum(dim=-1)  # (...)

class PercentileEMANormalizer:
    """DreamerV3-style percentile EMA normalizer for imagined returns."""

    def __init__(
        self,
        rate: float = 0.01,
        limit: float = 1.0,
        percentile_low: float = 5.0,
        percentile_high: float = 95.0,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.rate = rate
        self.limit = limit
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.device = torch.device(device)
        self.initialized = False
        self.low = torch.tensor(0.0, device=self.device)
        self.high = torch.tensor(1.0, device=self.device)

    def _ensure_device(self, values):
        if self.low.device != values.device:
            self.device = values.device
            self.low = self.low.to(values.device)
            self.high = self.high.to(values.device)

    @torch.no_grad()
    def __call__(self, values, update=True):
        self._ensure_device(values)
        flat = values.detach().reshape(-1).float()
        batch_low = torch.quantile(flat, self.percentile_low / 100.0)
        batch_high = torch.quantile(flat, self.percentile_high / 100.0)

        if update:
            if not self.initialized:
                self.low.copy_(batch_low)
                self.high.copy_(batch_high)
                self.initialized = True
            else:
                self.low.lerp_(batch_low, self.rate)
                self.high.lerp_(batch_high, self.rate)

        scale = (self.high - self.low).clamp_min(self.limit)
        return self.low.detach(), scale.detach()


class Ratio:
    """Reference-style scheduler for update-to-environment-step ratios.
    The conversion to integers helps to determine if we need to train any batches.

    Call ``start(step)`` when a delayed training phase becomes eligible. This
    establishes the scheduler baseline without retrospectively charging the
    pre-training collection phase to the update budget.
    """

    def __init__(self, ratio: float):
        if ratio < 0:
            raise ValueError(f"ratio must be non-negative, got {ratio}")
        self.ratio = ratio
        self.prev = None

    def start(self, step: int) -> None:
        """Establish the update-ratio baseline at ``step`` without updates."""
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}")
        self.prev = step

    def __call__(self, step: int) -> int:
        if self.ratio == 0:
            return 0
        if self.prev is None:
            self.prev = step
            return int(step * self.ratio)
        repeats = int((step - self.prev) * self.ratio)
        self.prev += repeats / self.ratio
        return repeats

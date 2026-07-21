from typing import Callable

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
    obs_batch_flat = stack_sequence_observations(batch_data["obs"], cfg)
    obs_target = reshape_sequence_observations(obs_batch_flat, batch_size, seq_len)

    actions = torch.stack(
        [
            to_tensor(action, cfg.device).squeeze(0)
            for seq in batch_data["actions"]
            for action in seq
        ]
    ).view(batch_size, seq_len, -1)
    rewards = torch.stack(
        [
            to_tensor(reward, cfg.device).reshape(())
            for seq in batch_data["rewards"]
            for reward in seq
        ]
    ).view(batch_size, seq_len)
    terminals = to_tensor(
        np.asarray(batch_data["terminals"]).squeeze(-1), cfg.device
    )
    is_last = to_tensor(np.asarray(batch_data["is_last"]).squeeze(-1), cfg.device)
    is_first = to_tensor(np.asarray(batch_data["is_first"]), cfg.device)

    if "prev_actions" in batch_data:
        prev_actions = torch.stack(
            [
                to_tensor(action, cfg.device).squeeze(0)
                for seq in batch_data["prev_actions"]
                for action in seq
            ]
        ).view(batch_size, seq_len, -1)
    else:
        prev_actions = torch.zeros_like(actions)
        prev_actions[:, 1:] = actions[:, :-1]

    reset_mask = 1.0 - is_first
    prev_actions = prev_actions * reset_mask.unsqueeze(-1)

    initial_stochastic = None
    initial_recurrent = None
    if "context_stochastic" in batch_data and "context_recurrent" in batch_data:
        initial_stochastic = torch.stack(
            [
                to_tensor(seq[0], cfg.device).squeeze(0)
                for seq in batch_data["context_stochastic"]
            ]
        )
        initial_recurrent = torch.stack(
            [
                to_tensor(seq[0], cfg.device).squeeze(0)
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
        key: F.mse_loss(reconstructed_obs[key], obs[key])
        if key == "vision"
        else F.mse_loss(reconstructed_obs[key], symlog(obs[key]))
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
    predicted_continue = continue_predictor(latent)  # (batch, seq_len, 1)
    terminal_target = terminal.contiguous().view(batch_size, seq_len, 1).float()
    continue_target = 1.0 - terminal_target
    if config.contdisc:
        continue_target = continue_target * config.discount
    continue_loss = F.binary_cross_entropy(predicted_continue, continue_target)
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
            imagined_rewards.append(reward_dist.mean.squeeze(-1))
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
    rewards, values, terminals, is_last, bootstrap, config
):
    """Return replay targets without crossing reset boundaries.

    True terminals suppress bootstrapping. Truncated ``is_last`` rows stop the
    lambda trace but retain the one-step value bootstrap from their final
    observation.
    """
    if rewards.shape[1] < 2:
        raise ValueError("Replay value loss needs at least two timesteps")

    rewards_next = rewards[:, 1:]
    terminals_next = terminals[:, 1:]
    is_last_next = is_last[:, 1:]
    current_is_last = is_last[:, :-1]
    values_for_loss = values[:, :-1]
    live_next = (1.0 - terminals_next.float()) * config.discount
    trace_next = (1.0 - is_last_next.float()) * config.gae_lambda

    replay_returns = torch.zeros_like(rewards_next)
    next_return = bootstrap.detach()
    for t in reversed(range(rewards_next.shape[1])):
        next_value = (
            bootstrap.detach()
            if t == rewards_next.shape[1] - 1
            else values_for_loss[:, t + 1]
        )
        target = rewards_next[:, t] + live_next[:, t] * (
            (1.0 - trace_next[:, t]) * next_value
            + trace_next[:, t] * next_return
        )
        replay_returns[:, t] = target
        next_return = target

    # Replay sequences can cross reset boundaries, so cumulative imagination
    # weights are inappropriate here. Match official replay value learning by
    # masking only boundary rows; reset rows start a fresh episode at weight 1.
    discount_weights = 1.0 - current_is_last.detach()
    return replay_returns, discount_weights


def compute_replay_value_loss(
    critic, ema_critic, replay_trajectories, bootstrap, config: DreamerConfig
):
    latents = replay_trajectories["latents"].detach()
    rewards = replay_trajectories["rewards"].detach()
    terminals = replay_trajectories["terminals"].detach()
    is_last = replay_trajectories["is_last"].detach()

    with torch.no_grad():
        values = ema_critic(latents).mean.squeeze(-1)
        replay_returns, discount_weights = compute_replay_lambda_returns(
            rewards,
            values,
            terminals,
            is_last,
            bootstrap=bootstrap.detach(),
            config=config,
        )

    value_dist = critic(latents[:, :-1])
    replay_return_loss = -(
        discount_weights * value_dist.log_prob(replay_returns.unsqueeze(-1))
    ).mean()
    with torch.no_grad():
        ema_values = ema_critic(latents[:, :-1]).mean
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
        current_values = ema_critic(latents.detach()).mean.squeeze(-1)
        bootstrap = ema_critic(last_latent.detach()).mean.squeeze(-1)
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
    predicted_values = value_dist.mean.squeeze(-1)
    returns_var = torch.var(critic_returns)
    if returns_var.item() < 1e-10:
        explained_variance = 0.0
    else:
        explained_variance = (
            1.0 - torch.var(critic_returns - predicted_values) / returns_var
        ).item()
    with torch.no_grad():
        ema_values = ema_critic(latents.detach()).mean
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
        replay_bootstrap = (
            critic_returns[:, 0].detach().view(start_batch_size, start_count)[:, -1]
        )
        replay_loss, replay_return_loss, replay_slow_loss, replay_returns = (
            compute_replay_value_loss(
                critic, ema_critic, replay_trajectories, replay_bootstrap, config
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

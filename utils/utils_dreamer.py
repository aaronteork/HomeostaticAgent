from collections import deque
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import (
    Beta,
    Independent,
    OneHotCategoricalStraightThrough,
    kl_divergence,
)

from configs.config_dreamer import DreamerConfig
from utils.vision import VisionDecoder, VisionEncoder


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
    dones = to_tensor(np.asarray(batch_data["dones"]).squeeze(-1), cfg.device)
    is_first = to_tensor(np.asarray(batch_data["is_first"]), cfg.device)

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
        prev_actions,
        rewards,
        dones,
        is_first,
        initial_stochastic,
        initial_recurrent,
    )


# ------------------ World Model Training functions ---------------------------


def compute_world_model_loss(
    rssm,
    decoder,
    reward_predictor,
    terminal_predictor,
    action,
    embed,
    obs,
    is_first,
    reward,
    done,
    config: DreamerConfig,
    initial_stochastic=None,
    initial_recurrent=None,
    return_latents=False,
):
    """
    Compute complete world model loss: reconstruction + KL + reward + terminal.

    Args:
        rssm: RSSM model
        decoder: CNN decoder
        reward_predictor: Reward prediction network
        terminal_predictor: Terminal prediction network
        action: (batch, seq_len, action_dim)
        embed: (batch, seq_len, encoder_dim)
        obs: dict of observation targets
        is_first: (batch, seq_len)
        reward: (batch, seq_len) - target rewards
        done: (batch, seq_len) - target terminals
        config: DreamerConfig

    Returns:
        loss: scalar loss
        metrics: dict with loss components
    """
    batch_size, seq_len = action.shape[0], action.shape[1]

    # Forward through RSSM
    # Compute RSSM outputs here to get the grad
    latent, recurrent_state, prior_dists, posterior_dists = rssm(
        action,
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
        else F.mse_loss(symlog(reconstructed_obs[key]), symlog(obs[key]))
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
    free_nats = torch.tensor(config.free_nats, device=action.device)
    dyn_loss = torch.tensor(0.0, device=action.device)
    rep_loss = torch.tensor(0.0, device=action.device)

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
    kl_loss = config.dyn_loss_weight * dyn_loss + config.rep_loss_weight * rep_loss

    # ===== Reward Prediction Loss =====
    reward_dist = reward_predictor(latent)  # two-hot distribution over rewards
    reward_target = reward.view(batch_size, seq_len, 1)
    reward_loss = -reward_dist.log_prob(reward_target).mean()

    # ===== Terminal/Done Prediction Loss =====
    predicted_terminal = terminal_predictor(latent)  # (batch, seq_len, 1)
    done_target = done.view(batch_size, seq_len, 1).float()
    terminal_loss = F.binary_cross_entropy(predicted_terminal, done_target)

    # ===== Combined Loss with Weighting =====
    loss = (
        recon_loss
        + config.kl_weight * kl_loss
        + config.reward_weight * reward_loss
        + config.terminal_weight * terminal_loss
    )

    metrics = {
        "world_model/reconstruction_loss": recon_loss.item(),
        "world_model/reconstruction_vision_loss": recon_losses["vision"].item(),
        "world_model/reconstruction_proprioception_loss": recon_losses[
            "proprioception"
        ].item(),
        "world_model/reconstruction_internal_state_loss": recon_losses[
            "internal_state"
        ].item(),
        "world_model/kl_loss": kl_loss.item(),
        "world_model/dyn_loss": dyn_loss.item(),
        "world_model/rep_loss": rep_loss.item(),
        "world_model/reward_loss": reward_loss.item(),
        "world_model/terminal_loss": terminal_loss.item(),
        "world_model/total_loss": loss.item(),
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
    terminal_predictor,
    init_latent,
    init_recurrent_state,
    config: DreamerConfig,
):
    """
    Rollout imagination trajectories for actor-critic training.

    The rollout starts from detached posterior states sampled from replay. The
    policy sees stop-gradient imagined states, matching DreamerV3's default
    ac_grads=False behavior, while sampled actions are still used to advance the
    frozen world model.

    Args:
        rssm: RSSM model
        actor: Actor network
        reward_predictor: Reward predictor network
        terminal_predictor: Terminal predictor network
        init_latent: (batch, latent_dim) - posterior latent from real obs
        init_recurrent_state: (batch, gru_units)
        config: DreamerConfig

    Returns:
        imagined_trajectories: dict with latents, actions, rewards, terminals,
        entropies, and final latent.
    """
    horizon = config.imagine_horizon

    # Initialize from replay states without backpropagating into representation learning.
    latent = init_latent.detach()
    recurrent_state = init_recurrent_state.detach()

    imagined_latents = []
    imagined_actions = []
    imagined_rewards = []
    imagined_terminals = []
    imagined_entropies = []
    imagined_log_probs = []

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
            terminal_prob = terminal_predictor(latent)
            imagined_terminals.append(terminal_prob.squeeze(-1))

    imagined_latents = torch.stack(
        imagined_latents, dim=1
    )  # (batch, horizon, latent_dim)
    imagined_actions = torch.stack(
        imagined_actions, dim=1
    )  # (batch, horizon, action_dim)
    imagined_rewards = torch.stack(imagined_rewards, dim=1)  # (batch, horizon)
    imagined_terminals = torch.stack(imagined_terminals, dim=1)  # (batch, horizon)
    imagined_entropies = torch.stack(imagined_entropies, dim=1)  # (batch, horizon)
    imagined_log_probs = torch.stack(imagined_log_probs, dim=1)  # (batch, horizon)

    return {
        "latents": imagined_latents,
        "actions": imagined_actions,
        "rewards": imagined_rewards,
        "terminals": imagined_terminals,
        "entropies": imagined_entropies,
        "log_probs": imagined_log_probs,
        "last_latent": latent,
    }


def compute_lambda_returns(rewards, values, terminals, bootstrap, gamma=0.99, lam=0.95):
    """Compute Dreamer-style lambda returns for imagined trajectories."""
    _, horizon = rewards.shape
    returns = torch.zeros_like(rewards)
    next_return = bootstrap
    continuation = 1.0 - terminals

    for t in reversed(range(horizon)):
        next_value = bootstrap if t == horizon - 1 else values[:, t + 1]
        target = rewards[:, t] + gamma * continuation[:, t] * (
            (1.0 - lam) * next_value + lam * next_return
        )
        returns[:, t] = target
        next_return = target

    return returns


def compute_discount_weights(terminals, gamma):
    _, horizon = terminals.shape
    continuation = 1.0 - terminals.detach()
    discount_weights = torch.ones_like(terminals)
    for t in range(1, horizon):
        discount_weights[:, t] = (
            discount_weights[:, t - 1] * gamma * continuation[:, t - 1]
        )
    return discount_weights


def compute_replay_value_loss(
    critic, ema_critic, replay_trajectories, bootstrap, config: DreamerConfig
):
    latents = replay_trajectories["latents"].detach()
    rewards = replay_trajectories["rewards"].detach()
    terminals = replay_trajectories["terminals"].detach()

    with torch.no_grad():
        values = critic(latents).mean.squeeze(-1)
        replay_returns = compute_lambda_returns(
            rewards=rewards,
            values=values,
            terminals=terminals,
            bootstrap=bootstrap.detach(),
            gamma=config.gamma,
            lam=config.gae_lambda,
        )
        discount_weights = compute_discount_weights(terminals, config.gamma)

    value_dist = critic(latents)
    replay_return_loss = -(
        discount_weights * value_dist.log_prob(replay_returns.unsqueeze(-1))
    ).mean()
    with torch.no_grad():
        ema_values = ema_critic(latents).mean
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
        imagined_trajectories: dict with DETACHED imagined data (from imagination_rollout)
        config: DreamerConfig

    Returns:
        actor_loss: scalar
        critic_loss: scalar
        metrics: dict with loss components
    """
    latents = imagined_trajectories["latents"]
    rewards = imagined_trajectories["rewards"]
    terminals = imagined_trajectories["terminals"]
    entropies = imagined_trajectories["entropies"]
    log_probs = imagined_trajectories["log_probs"]
    last_latent = imagined_trajectories["last_latent"]

    batch_size, horizon = rewards.shape

    with torch.no_grad():
        current_values = critic(latents.detach()).mean.squeeze(-1)
        bootstrap = critic(last_latent.detach()).mean.squeeze(-1)
        critic_returns = compute_lambda_returns(
            rewards=rewards.detach(),
            values=current_values,
            terminals=terminals.detach(),
            bootstrap=bootstrap,
            gamma=config.gamma,
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
        discount_weights = compute_discount_weights(terminals, config.gamma)

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
        "actor_critic/critic_return_loss": critic_return_loss.item(),
        "actor_critic/critic_slow_loss": critic_slow_loss.item(),
        "actor_critic/critic_imagined_loss": critic_imagined_loss.item(),
        "actor_critic/critic_replay_loss": replay_loss.item(),
        "actor_critic/critic_replay_return_loss": replay_return_loss.item(),
        "actor_critic/critic_replay_slow_loss": replay_slow_loss.item(),
        "actor_critic/entropy": entropy.item(),
        "actor_critic/actor_objective": actor_objective.item(),
        "actor_critic/average_return": critic_returns.mean().item(),
        "actor_critic/return_norm_offset": return_offset.item(),
        "actor_critic/return_norm_scale": return_scale.item(),
        "actor_critic/average_advantage": advantages.mean().item(),
        "actor_critic/advantage_std": advantages.std().item(),
        "actor_critic/log_prob": log_probs.mean().item(),
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
    """Reference-style scheduler for update-to-environment-step ratios."""

    def __init__(self, ratio: float):
        if ratio < 0:
            raise ValueError(f"ratio must be non-negative, got {ratio}")
        self.ratio = ratio
        self.prev = None

    def __call__(self, step: int) -> int:
        if self.ratio == 0:
            return 0
        if self.prev is None:
            self.prev = step
            return int(step * self.ratio)
        repeats = int((step - self.prev) * self.ratio)
        self.prev += repeats / self.ratio
        return repeats


# ------------------ World Model Components ---------------------------
class ObservationEncoder(nn.Module):
    """Encode Ant image and vector observations into a single RSSM embedding."""

    def __init__(self, config: DreamerConfig):
        super().__init__()
        self.config = config

        # Vision encoder
        self.vision_encoder = VisionEncoder(
            input_channels=4, depth=config.base_cnn_channels
        )
        # Get output dimension of vision encoder by passing a dummy input
        with torch.no_grad():
            dummy_input = torch.zeros(1, 4, 64, 64)
            vision_output_dim = self.vision_encoder(dummy_input).shape[-1]

        # Vector encoder for proprioception
        self.proprioception_encoder = MLP(
            config,
            input_dim=config.obs_space_dim,
            output_dim=config.hidden_dim,
            num_layers=config.mlp_n_layers,
        )

        # Vector encoder for internal state
        self.internal_state_encoder = MLP(
            config,
            input_dim=2 if config.num_heat == 0 else 3,
            output_dim=config.hidden_dim,
            num_layers=config.mlp_n_layers,
        )

        self.output_dim = vision_output_dim + config.hidden_dim + config.hidden_dim
        if self.output_dim != config.encoder_dim:
            raise ValueError(
                f"Encoder output dim {self.output_dim} does not match config.encoder_dim {config.encoder_dim}"
            )

        self._init_weights()

    def forward(self, vision, proprioception=None, internal_state=None):
        if isinstance(vision, dict):
            obs = vision
            vision = obs["vision"]
            proprioception = obs["proprioception"]
            internal_state = obs["internal_state"]

        vision = to_tensor(vision, self.config.device)
        proprioception = to_tensor(proprioception, self.config.device)
        proprioception = symlog(proprioception)
        internal_state = to_tensor(internal_state, self.config.device)
        internal_state = symlog(internal_state)

        vision_embed = self.vision_encoder(vision)
        proprioception_embed = self.proprioception_encoder(proprioception)
        internal_state_embed = self.internal_state_encoder(internal_state)
        return torch.cat(
            [vision_embed, proprioception_embed, internal_state_embed], dim=-1
        )

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class ObservationDecoder(nn.Module):
    """Multi-head observation decoder, mirroring SheepRL's per-key decoders."""

    def __init__(self, config: DreamerConfig):
        super().__init__()
        self.config = config
        # Vision decoder
        self.vision_decoder = VisionDecoder(
            feature_dim=config.latent_dim,
            output_channels=4,
            depth=config.base_cnn_channels,
            output_shape=(64, 64),
        )

        # Proprioception decoder
        self.proprioception_decoder = MLP(
            config,
            input_dim=config.latent_dim,
            output_dim=config.hidden_dim,
            num_layers=config.mlp_n_layers,
        )
        self.proprioception_decoder_projection = nn.Linear(
            config.hidden_dim, config.obs_space_dim
        )  # Projection layer for proprioception

        # Internal state decoder
        self.internal_state_decoder = MLP(
            config,
            input_dim=config.latent_dim,
            output_dim=config.hidden_dim,
            num_layers=config.mlp_n_layers,
        )
        self.internal_state_decoder_projection = nn.Linear(
            config.hidden_dim, 2 if config.num_heat == 0 else 3
        )  # Projection layer for internal state

        self._init_weights()

    def forward(self, latent):
        latent = to_tensor(latent, self.config.device)
        sequence_shape = latent.shape[:-1]
        latent_flat = latent.reshape(-1, latent.shape[-1])

        # Vision
        vision = self.vision_decoder(latent_flat)
        vision = vision.view(*sequence_shape, *vision.shape[1:])
        # Proprioception
        proprioception = self.proprioception_decoder(latent)
        proprioception = self.proprioception_decoder_projection(proprioception)
        proprioception = symexp(
            proprioception
        )  # Apply symexp to map back to original scale
        # Internal state
        internal_state = self.internal_state_decoder(latent)
        internal_state = self.internal_state_decoder_projection(internal_state)
        internal_state = symexp(
            internal_state
        )  # Apply symexp to map back to original scale
        return {
            "vision": vision,
            "proprioception": proprioception,
            "internal_state": internal_state,
        }

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class MLP(nn.Module):
    def __init__(
        self, config: DreamerConfig, input_dim: int, output_dim: int, num_layers: int
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                dim_in = input_dim
                dim_out = config.hidden_dim
            elif i == num_layers - 1:
                dim_in = config.hidden_dim
                dim_out = output_dim
            else:
                dim_in = config.hidden_dim
                dim_out = config.hidden_dim
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(nn.RMSNorm(dim_out))
            layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# From https://github.com/eclectic-sheep/sheeprl/blob/33b636681fd8b5340b284f2528db8821ab8dcd0b/sheeprl/models/models.py
class LayerNormGRUCell(nn.Module):
    """A GRU cell with a LayerNorm, taken
    from https://github.com/danijar/dreamerv2/blob/main/dreamerv2/common/nets.py#L317.

    This particular GRU cell accepts 3-D inputs, with a sequence of length 1, and applies
    a LayerNorm after the projection of the inputs.

    Args:
        input_size (int): the input size.
        hidden_size (int): the hidden state size
        bias (bool, optional): whether to apply a bias to the input projection.
            Defaults to True.
        batch_first (bool, optional): whether the first dimension represent the batch dimension or not.
            Defaults to False.
        layer_norm_cls (Callable[..., nn.Module]): the layer norm to apply after the input projection.
            Defaults to nn.Identiy.
        layer_norm_kw (Dict[str, Any]): the kwargs of the layer norm.
            Default to {}.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        batch_first: bool = False,
        layer_norm_cls: Callable[..., nn.Module] = nn.LayerNorm,
        layer_norm_kw: Dict[str, Any] = {},
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first
        self.linear = nn.Linear(
            input_size + hidden_size, 3 * hidden_size, bias=self.bias
        )
        # Avoid multiple values for the `normalized_shape` argument
        layer_norm_kw.pop("normalized_shape", None)
        self.layer_norm = layer_norm_cls(3 * hidden_size, **layer_norm_kw)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        is_3d = input.dim() == 3
        if is_3d:
            if input.shape[int(self.batch_first)] == 1:
                input = input.squeeze(int(self.batch_first))
            else:
                raise AssertionError(
                    "LayerNormGRUCell: Expected input to be 3-D with sequence length equal to 1 but received "
                    f"a sequence of length {input.shape[int(self.batch_first)]}"
                )
        if hx.dim() == 3:
            hx = hx.squeeze(0)
        assert input.dim() in (
            1,
            2,
        ), (
            f"LayerNormGRUCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"
        )

        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx = torch.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx

        input = torch.cat((hx, input), -1)
        x = self.linear(input)
        x = self.layer_norm(x)
        reset, cand, update = torch.chunk(x, 3, -1)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        hx = update * cand + (1 - update) * hx

        if not is_batched:
            hx = hx.squeeze(0)
        elif is_3d:
            hx = hx.unsqueeze(0)

        return hx


class RSSM(nn.Module):
    """DreamerV3-style recurrent state-space model.

    The latent feature exposed to the rest of this codebase is the concatenation
    of a flattened categorical stochastic state and the deterministic recurrent
    state, matching the feature used by DreamerV3 actor, critic, and heads.
    """

    def __init__(self, config: DreamerConfig):
        super().__init__()
        self.config = config
        self.recurrent_units = config.recurrent_units
        self.stochastic_units = config.stochastic_units
        self.discrete_classes = config.discrete_classes
        self.stochastic_size = config.stochastic_units * config.discrete_classes
        self.feature_dim = config.latent_dim
        self.unimix = config.rssm_unimix
        self.initial_recurrent_state = nn.Parameter(torch.zeros(self.recurrent_units))

        recurrent_input_dim = config.action_space_dim + self.stochastic_size
        self.recurrent_input = nn.Sequential(
            nn.Linear(recurrent_input_dim, config.hidden_dim),
            nn.RMSNorm(config.hidden_dim),
            nn.SiLU(),
        )
        self.gru = LayerNormGRUCell(config.hidden_dim, self.recurrent_units)

        # Both prior and posterior networks are MLPs with one hidden layer, outputting logits for the categorical distribution
        self.prior_network = nn.Sequential(
            MLP(
                config,
                input_dim=self.recurrent_units,
                output_dim=config.hidden_dim,
                num_layers=1,
            ),
            nn.Linear(config.hidden_dim, self.stochastic_units * self.discrete_classes),
        )
        self.posterior_network = nn.Sequential(
            MLP(
                config,
                input_dim=self.recurrent_units + config.encoder_dim,
                output_dim=config.hidden_dim,
                num_layers=1,
            ),
            nn.Linear(config.hidden_dim, self.stochastic_units * self.discrete_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                if len(param.shape) > 1:
                    nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def _uniform_mix(self, logits: Tensor) -> Tensor:
        logits = logits.view(
            *logits.shape[:-1], self.stochastic_units, self.discrete_classes
        )
        if self.unimix > 0.0:
            probs = logits.softmax(dim=-1)
            uniform = torch.ones_like(probs) / self.discrete_classes
            probs = (1.0 - self.unimix) * probs + self.unimix * uniform
            logits = torch.log(probs.clamp_min(1e-8))
        return logits

    def _dist(self, logits: Tensor) -> Independent:
        return Independent(OneHotCategoricalStraightThrough(logits=logits), 1)

    def _sample_stochastic(self, logits: Tensor, deterministic: bool = False) -> Tensor:
        """Returns a stochastic state sample from the categorical distribution defined by logits. If deterministic is True, returns the mode of the distribution."""
        if deterministic:
            index = logits.argmax(dim=-1)
            return F.one_hot(index, self.discrete_classes).to(dtype=logits.dtype)
        return self._dist(logits).rsample()

    def initial_state(
        self, batch_size: int, device, dtype=torch.float32, deterministic: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """Get the initial stochastic and recurrent states for a batch of sequences.
        Initial stochastic state is derived from the recurrent state."""
        recurrent_state = torch.tanh(self.initial_recurrent_state).to(
            device=device, dtype=dtype
        )
        recurrent_state = recurrent_state.unsqueeze(0).expand(batch_size, -1)
        logits = self.prior_logits(recurrent_state)
        stochastic_state = self._sample_stochastic(logits, deterministic=deterministic)
        return stochastic_state, recurrent_state

    def _feature(self, stochastic_state: Tensor, recurrent_state: Tensor) -> Tensor:
        """Combine stochastic and recurrent states into a single latent feature vector."""
        stochastic_flat = stochastic_state.reshape(
            *stochastic_state.shape[:-2], self.stochastic_size
        )
        return torch.cat([stochastic_flat, recurrent_state], dim=-1)

    def split_feature(self, latent: Tensor) -> Tuple[Tensor, Tensor]:
        stochastic_flat, recurrent_state = torch.split(
            latent, [self.stochastic_size, self.recurrent_units], dim=-1
        )
        stochastic_state = stochastic_flat.view(
            *stochastic_flat.shape[:-1], self.stochastic_units, self.discrete_classes
        )
        return stochastic_state, recurrent_state

    def prior_logits(self, recurrent_state: Tensor) -> Tensor:
        logits = self.prior_network(recurrent_state)
        return self._uniform_mix(logits)

    def posterior_logits(self, recurrent_state: Tensor, embed: Tensor) -> Tensor:
        x = torch.cat([recurrent_state, embed], dim=-1)
        logits = self.posterior_network(x)
        return self._uniform_mix(logits)

    def prior(self, recurrent_state: Tensor) -> Independent:
        """Return the categorical prior distribution p(z_t | h_t)."""
        return self._dist(self.prior_logits(recurrent_state))

    def posterior(self, recurrent_state: Tensor, embed: Tensor) -> Independent:
        """Return the categorical posterior distribution q(z_t | h_t, o_t)."""
        return self._dist(self.posterior_logits(recurrent_state, embed))

    def recurrent_step(
        self, stochastic_state: Tensor, action: Tensor, recurrent_state: Tensor
    ) -> Tensor:
        # Gets h_t from z_{t-1} and a_{t-1}
        # Included the GRU here
        stochastic_flat = stochastic_state.reshape(
            stochastic_state.shape[0], self.stochastic_size
        )
        action = action / torch.maximum(torch.ones_like(action), action.abs())
        recurrent_input = torch.cat([stochastic_flat, action], dim=-1)
        return self.gru(self.recurrent_input(recurrent_input), recurrent_state)

    def imagine_step(
        self,
        latent: Tensor,
        recurrent_state: Tensor,
        action: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor, Independent]:
        stochastic_state, _ = self.split_feature(latent)
        recurrent_state = self.recurrent_step(stochastic_state, action, recurrent_state)
        logits = self.prior_logits(recurrent_state)
        next_stochastic = self._sample_stochastic(logits, deterministic=deterministic)
        next_latent = self._feature(next_stochastic, recurrent_state)
        return next_latent, recurrent_state, self._dist(logits)

    def forward(
        self,
        action,
        embed,
        is_first,
        recurrent_state=None,
        previous_stochastic=None,
        deterministic=False,
    ):
        """
        Forward pass through RSSM.
        One entire step in the RSSM

        Args:
            action: (batch, seq_len, action_dim) or (batch, action_dim)
            embed: (batch, seq_len, encoder_dim) or (batch, encoder_dim)
            is_first: (batch, seq_len) or (batch,)
            recurrent_state: (batch, recurrent_units) or None
            previous_stochastic: (batch, stochastic_units, discrete_classes) or None
            deterministic: if True, use categorical modes instead of samples

        Returns:
            latent: (batch, seq_len, latent_dim) or (batch, latent_dim)
            recurrent_state: (batch, recurrent_units)
            prior_dists: list of prior distributions
            posterior_dists: list of posterior distributions
        """
        if len(action.shape) == 2:
            action = action.unsqueeze(1)
            embed = embed.unsqueeze(1)
            is_first = is_first.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, seq_len, _ = action.shape

        initial_stochastic, initial_recurrent = self.initial_state(
            batch_size,
            action.device,
            dtype=action.dtype,
            deterministic=deterministic,
        )
        if recurrent_state is None:
            recurrent_state = initial_recurrent
        if previous_stochastic is None:
            previous_stochastic = initial_stochastic

        prior_dists = []
        posterior_dists = []
        latents = []

        for t in range(seq_len):
            # Reset recurrent state on episode start
            reset_mask = 1.0 - is_first[:, t : t + 1].to(dtype=action.dtype)
            recurrent_state = recurrent_state * reset_mask + initial_recurrent * (
                1.0 - reset_mask
            )
            previous_stochastic = previous_stochastic * reset_mask[
                ..., None
            ] + initial_stochastic * (1.0 - reset_mask[..., None])
            step_action = action[:, t] * reset_mask

            # Compute h_t from z_{t-1} and a_{t-1}, then predict prior/posterior.
            recurrent_state = self.recurrent_step(
                previous_stochastic, step_action, recurrent_state
            )
            prior_logits = self.prior_logits(recurrent_state)
            posterior_logits = self.posterior_logits(recurrent_state, embed[:, t])
            prior_dist = self._dist(prior_logits)
            posterior_dist = self._dist(posterior_logits)
            prior_dists.append(prior_dist)
            posterior_dists.append(posterior_dist)

            stochastic_state = self._sample_stochastic(
                posterior_logits, deterministic=deterministic
            )
            previous_stochastic = stochastic_state
            latents.append(self._feature(stochastic_state, recurrent_state))

        latent_stack = torch.stack(latents, dim=1)
        if squeeze_output:
            latent_stack = latent_stack.squeeze(1)

        return latent_stack, recurrent_state, prior_dists, posterior_dists


class ActorNetwork(nn.Module):
    """Actor network for policy in latent space."""

    def __init__(self, config: DreamerConfig):
        super().__init__()
        self.config = config
        input_dim = config.latent_dim
        action_dim = config.action_space_dim

        self.net = MLP(
            config, input_dim=input_dim, output_dim=config.hidden_dim, num_layers=3
        )

        # Beta distribution parameters for continuous actions
        self.alpha = nn.Linear(config.hidden_dim, action_dim)
        self.beta = nn.Linear(config.hidden_dim, action_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, latent, deterministic=False):
        """
        Forward pass to get action distribution.

        Args:
            latent: (batch, latent_dim) or (batch, seq_len, latent_dim)
            deterministic: if True, return mean action

        Returns:
            action: (batch, action_dim) or (batch, seq_len, action_dim)
            log_prob: (batch,) or (batch, seq_len)
        """
        if len(latent.shape) == 3:
            batch_size, seq_len, latent_dim = latent.shape
            latent_flat = latent.reshape(batch_size * seq_len, latent_dim)
            squeeze_output = True
        else:
            batch_size = latent.shape[0]
            latent_flat = latent
            squeeze_output = False

        x = self.net(latent_flat)
        alpha = F.softplus(self.alpha(x)) + 1.0
        beta = F.softplus(self.beta(x)) + 1.0

        dist = Beta(alpha, beta)
        if deterministic:
            action = dist.mode  # TODO: Use mode for deterministic action, if it doesnt work, use mean instead
        else:
            action = dist.rsample()

        log_prob = dist.log_prob(action).sum(dim=-1)

        if squeeze_output:
            action = action.view(batch_size, seq_len, -1)
            log_prob = log_prob.view(batch_size, seq_len)

        return action, log_prob, dist


class CriticNetwork(nn.Module):
    """Critic network for value function in latent space."""

    def __init__(self, config: DreamerConfig):
        super().__init__()
        self.config = config
        input_dim = config.latent_dim

        self.net = nn.Sequential(
            MLP(
                config, input_dim=input_dim, output_dim=config.hidden_dim, num_layers=3
            ),
            nn.Linear(config.hidden_dim, config.two_hot_bins),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.net[1].weight)
        nn.init.zeros_(self.net[1].bias)

    def forward(self, latent):
        """
        Compute a two-hot value distribution.

        Args:
            latent: (batch, latent_dim) or (batch, seq_len, latent_dim)

        Returns:
            TwoHotEncodingDistribution over scalar values.
        """
        if len(latent.shape) == 3:
            batch_size, seq_len, latent_dim = latent.shape
            latent_flat = latent.reshape(batch_size * seq_len, latent_dim)
            squeeze_output = True
        else:
            batch_size = latent.shape[0]
            latent_flat = latent
            squeeze_output = False

        logits = self.net(latent_flat)

        if squeeze_output:
            logits = logits.view(batch_size, seq_len, -1)

        return TwoHotEncodingDistribution(
            logits,
            dims=1,
            low=self.config.two_hot_low,
            high=self.config.two_hot_high,
        )


class RewardPredictor(nn.Module):
    """Predict a two-hot reward distribution from latent state."""

    def __init__(self, config: DreamerConfig):
        super().__init__()
        self.config = config
        input_dim = config.latent_dim
        output_dim = config.hidden_dim
        self.net = MLP(config, input_dim=input_dim, output_dim=output_dim, num_layers=1)
        self.reward_projection = nn.Linear(output_dim, config.two_hot_bins)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.reward_projection.weight)
        nn.init.zeros_(self.reward_projection.bias)

    def forward(self, latent):
        """Predict a two-hot reward distribution from latent state."""
        latent = to_tensor(latent, self.config.device)
        latent = self.net(latent)
        logits = self.reward_projection(latent)
        return TwoHotEncodingDistribution(
            logits,
            dims=1,
            low=self.config.two_hot_low,
            high=self.config.two_hot_high,
        )


class ContinuePredictor(nn.Module):
    """Predict terminal probability from latent state."""

    def __init__(self, config: DreamerConfig):
        super().__init__()
        self.config = config
        input_dim = config.latent_dim
        output_dim = config.hidden_dim
        self.net = MLP(config, input_dim=input_dim, output_dim=output_dim, num_layers=1)
        self.terminal_projection = nn.Sequential(nn.Linear(output_dim, 1), nn.Sigmoid())
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, latent):
        """Predict terminal probability from latent state."""
        latent = to_tensor(latent, self.config.device)
        return self.terminal_projection(self.net(latent))


class WorldModel(nn.Module):
    """Thin orchestration module for Dreamer world-model components."""

    def __init__(self, config: DreamerConfig):
        super().__init__()
        self.config = config
        self.encoder = ObservationEncoder(config)
        self.rssm = RSSM(config)
        self.decoder = ObservationDecoder(config)
        self.reward_predictor = RewardPredictor(config)
        self.terminal_predictor = ContinuePredictor(config)

    def encode(self, obs):
        return self.encoder(obs)

    def observe(
        self,
        prev_action,
        embed,
        is_first,
        recurrent_state=None,
        previous_stochastic=None,
        deterministic=False,
    ):
        return self.rssm(
            prev_action,
            embed,
            is_first,
            recurrent_state=recurrent_state,
            previous_stochastic=previous_stochastic,
            deterministic=deterministic,
        )

    def imagine_step(self, latent, recurrent_state, action, deterministic=False):
        return self.rssm.imagine_step(
            latent,
            recurrent_state,
            action,
            deterministic=deterministic,
        )

    def decode(self, latent):
        return self.decoder(latent)

    def predict_reward(self, latent):
        return self.reward_predictor(latent)

    def predict_terminal(self, latent):
        return self.terminal_predictor(latent)


class SequenceReplayBuffer:
    """Replay buffer that stores same-environment sequences for RSSM training.

    The training loop adds one transition per worker at each environment step. This
    buffer groups those calls into one time-major row, so sampling can return
    contiguous chunks from a single worker trajectory instead of accidentally
    mixing workers within one RSSM sequence.
    """

    def __init__(self, config: DreamerConfig, device="cpu"):
        self.config = config
        self.device = device
        self.num_workers = config.num_workers
        if self.num_workers <= 0:
            raise ValueError(f"num_workers must be positive, got {self.num_workers}")
        # Maximum time steps per worker to store, ensuring total capacity is as specified in config
        self.replay_capacity = max(1, config.replay_capacity // self.num_workers)
        self.batch_length = config.batch_length
        self.batch_size = config.batch_size
        self._rng = np.random.default_rng()

        # Each item in the list is a time step of all environments, so the shape of each item is [num_workers, ...].
        # The deques ensure we only keep the most recent replay_capacity time steps per worker.
        self.observations = deque(maxlen=self.replay_capacity)
        self.actions = deque(maxlen=self.replay_capacity)
        self.rewards = deque(maxlen=self.replay_capacity)
        self.dones = deque(maxlen=self.replay_capacity)
        self.episode_starts = deque(maxlen=self.replay_capacity)
        self.context_stochastics = deque(maxlen=self.replay_capacity)
        self.context_recurrents = deque(maxlen=self.replay_capacity)

        self._pending_observations = []
        self._pending_actions = []
        self._pending_rewards = []
        self._pending_dones = []
        self._pending_episode_starts = []
        self._pending_context_stochastics = []
        self._pending_context_recurrents = []

    def add(
        self,
        obs_dict,
        action,
        reward,
        done,
        is_first=False,
        context_stochastic=None,
        context_recurrent=None,
    ):
        """Add one worker transition.

        Calls are expected in worker order. Once all workers for a timestep have
        been added, the row is committed as [env0, env1, ...].
        """
        if len(self._pending_observations) >= self.num_workers:
            raise RuntimeError(
                "Pending replay row already has num_workers entries; this indicates an add() ordering bug."
            )

        self._pending_observations.append(obs_dict)
        self._pending_actions.append(action)
        self._pending_rewards.append(reward)
        self._pending_dones.append(done)
        self._pending_episode_starts.append(is_first)
        self._pending_context_stochastics.append(context_stochastic)
        self._pending_context_recurrents.append(context_recurrent)

        if len(self._pending_observations) == self.num_workers:
            self.observations.append(self._pending_observations)
            self.actions.append(self._pending_actions)
            self.rewards.append(self._pending_rewards)
            self.dones.append(self._pending_dones)
            self.episode_starts.append(self._pending_episode_starts)
            self.context_stochastics.append(self._pending_context_stochastics)
            self.context_recurrents.append(self._pending_context_recurrents)

            self._pending_observations = []
            self._pending_actions = []
            self._pending_rewards = []
            self._pending_dones = []
            self._pending_episode_starts = []
            self._pending_context_stochastics = []
            self._pending_context_recurrents = []

    def _build_sequence_batch(self, start_indices, env_indices, force_first_reset):
        """Build same-worker contiguous sequences and keep replay coordinates."""
        obs_sequences = []
        action_sequences = []
        reward_sequences = []
        done_sequences = []
        is_first_sequences = []
        context_stochastic_sequences = []
        context_recurrent_sequences = []
        replay_indices = []

        for start_idx, env_idx in zip(start_indices, env_indices):
            obs_seq = [
                self.observations[start_idx + i][env_idx]
                for i in range(self.batch_length)
            ]
            action_seq = [
                self.actions[start_idx + i][env_idx] for i in range(self.batch_length)
            ]
            reward_seq = [
                self.rewards[start_idx + i][env_idx] for i in range(self.batch_length)
            ]
            done_seq = [
                self.dones[start_idx + i][env_idx] for i in range(self.batch_length)
            ]
            is_first_seq = [
                self.episode_starts[start_idx + i][env_idx]
                for i in range(self.batch_length)
            ]
            context_stochastic_seq = [
                self.context_stochastics[start_idx + i][env_idx]
                for i in range(self.batch_length)
            ]
            context_recurrent_seq = [
                self.context_recurrents[start_idx + i][env_idx]
                for i in range(self.batch_length)
            ]
            if force_first_reset:
                is_first_seq[0] = True

            obs_sequences.append(obs_seq)
            action_sequences.append(action_seq)
            reward_sequences.append(reward_seq)
            done_sequences.append(done_seq)
            is_first_sequences.append(is_first_seq)
            context_stochastic_sequences.append(context_stochastic_seq)
            context_recurrent_sequences.append(context_recurrent_seq)
            replay_indices.append((int(start_idx), int(env_idx)))

        return {
            "obs": obs_sequences,
            "actions": action_sequences,
            "rewards": reward_sequences,
            "dones": done_sequences,
            "is_first": is_first_sequences,
            "context_stochastic": context_stochastic_sequences,
            "context_recurrent": context_recurrent_sequences,
            "replay_indices": replay_indices,
        }

    def _sample_uniform(self, batch_size):
        """Sample same-worker contiguous sequences from the whole buffer."""
        time_len = len(self.observations)
        if (
            len(self) < self.config.min_buffer_size_before_training
            or time_len < self.batch_length
        ):
            return None

        max_start = time_len - self.batch_length + 1
        start_indices = self._rng.integers(0, max_start, size=batch_size)
        env_indices = self._rng.integers(0, self.num_workers, size=batch_size)
        return self._build_sequence_batch(
            start_indices, env_indices, force_first_reset=False
        )

    def _sample_online(self, batch_size):
        """Take newest non-overlapping chunks before falling back to uniform replay."""
        time_len = len(self.observations)
        if batch_size <= 0 or time_len < self.batch_length:
            return None

        starts = []
        envs = []
        for end_idx in range(time_len, self.batch_length - 1, -self.batch_length):
            start_idx = end_idx - self.batch_length
            for env_idx in range(self.num_workers):
                starts.append(start_idx)
                envs.append(env_idx)
                if len(starts) == batch_size:
                    return self._build_sequence_batch(
                        starts, envs, force_first_reset=False
                    )

        if not starts:
            return None
        return self._build_sequence_batch(starts, envs, force_first_reset=False)

    def _merge_batches(self, batches):
        batches = [batch for batch in batches if batch is not None and batch["obs"]]
        if not batches:
            return None
        merged = {}
        for key in batches[0].keys():
            merged[key] = []
            for batch in batches:
                merged[key].extend(batch[key])
        return merged

    def sample(self, batch_size=None):
        """Sample a uniform batch for callers that do not need online mixing."""
        return self._sample_uniform(batch_size or self.batch_size)

    def sample_mixed(self):
        """Sample fresh online chunks first, then fill the batch with uniform replay."""
        online_size = int(round(self.batch_size * self.config.online_batch_fraction))
        online_size = max(0, min(self.batch_size, online_size))
        replay_size = self.batch_size - online_size

        online_batch = self._sample_online(online_size)
        online_count = len(online_batch["obs"]) if online_batch is not None else 0
        replay_batch = self._sample_uniform(replay_size + (online_size - online_count))
        return self._merge_batches([online_batch, replay_batch])

    def update_contexts(self, batch_data, latents):
        """Write refreshed posterior RSSM states back as context for later items."""
        if batch_data is None or "replay_indices" not in batch_data:
            return
        stochastic_states, recurrent_states = (
            latents.detach()
            .cpu()
            .split(
                [self.config.stochastic_size, self.config.recurrent_units],
                dim=-1,
            )
        )
        stochastic_states = stochastic_states.view(
            *stochastic_states.shape[:-1],
            self.config.stochastic_units,
            self.config.discrete_classes,
        ).numpy()
        recurrent_states = recurrent_states.numpy()

        time_len = len(self.observations)
        for batch_idx, (start_idx, env_idx) in enumerate(batch_data["replay_indices"]):
            for offset in range(1, self.batch_length):
                target_idx = start_idx + offset
                if target_idx >= time_len:
                    break
                if self.episode_starts[target_idx][env_idx]:
                    continue
                self.context_stochastics[target_idx][env_idx] = stochastic_states[
                    batch_idx, offset - 1
                ].copy()
                self.context_recurrents[target_idx][env_idx] = recurrent_states[
                    batch_idx, offset - 1
                ].copy()
            next_idx = start_idx + self.batch_length
            if next_idx < time_len and not self.episode_starts[next_idx][env_idx]:
                self.context_stochastics[next_idx][env_idx] = stochastic_states[
                    batch_idx, -1
                ].copy()
                self.context_recurrents[next_idx][env_idx] = recurrent_states[
                    batch_idx, -1
                ].copy()

    def __len__(self):
        return len(self.observations) * self.num_workers + len(
            self._pending_observations
        )

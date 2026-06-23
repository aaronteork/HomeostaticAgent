from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Beta, Independent, OneHotCategoricalStraightThrough
import numpy as np
from collections import deque

from configs.config_dreamer import DreamerConfig
from utils.vision import VisionEncoder, VisionDecoder


def to_tensor(value, device, dtype=torch.float32):
    """Convert numpy arrays or tensors to float tensors on the training device."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


class ObservationEncoder(nn.Module):
    """Encode Ant image and vector observations into a single RSSM embedding."""

    def __init__(self, config: DreamerConfig):
        super().__init__()
        self.config = config

        # Vision encoder
        self.vision_encoder = VisionEncoder(input_channels=4, depth=config.base_cnn_channels)
        # Get output dimension of vision encoder by passing a dummy input
        with torch.no_grad():
            dummy_input = torch.zeros(1, 4, 64, 64)
            vision_output_dim = self.vision_encoder(dummy_input).shape[-1]

        # Vector encoder for proprioception
        self.proprioception_encoder = MLP(config, input_dim=config.obs_space_dim, output_dim=config.hidden_dim, num_layers=config.mlp_n_layers)

        # Vector encoder for internal state
        self.internal_state_encoder = MLP(config, input_dim=2 if config.num_heat == 0 else 3, output_dim=config.hidden_dim, num_layers=config.mlp_n_layers)

        self.output_dim = vision_output_dim + config.hidden_dim + config.hidden_dim
        if self.output_dim != config.encoder_dim:
            raise ValueError(f"Encoder output dim {self.output_dim} does not match config.encoder_dim {config.encoder_dim}")

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
        return torch.cat([vision_embed, proprioception_embed, internal_state_embed], dim=-1)

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
        self.vision_decoder = VisionDecoder(feature_dim=config.latent_dim, output_channels=4, depth=config.base_cnn_channels, output_shape=(64, 64))
        
        # Proprioception decoder
        self.proprioception_decoder = MLP(config, input_dim=config.latent_dim, output_dim=config.hidden_dim, num_layers=config.mlp_n_layers)
        self.proprioception_decoder_projection = nn.Linear(config.hidden_dim, config.obs_space_dim)  # Projection layer for proprioception

        # Internal state decoder
        self.internal_state_decoder = MLP(config, input_dim=config.latent_dim, output_dim=config.hidden_dim, num_layers=config.mlp_n_layers)
        self.internal_state_decoder_projection = nn.Linear(config.hidden_dim, 2 if config.num_heat == 0 else 3)  # Projection layer for internal state

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
        proprioception = symexp(proprioception)  # Apply symexp to map back to original scale
        # Internal state
        internal_state = self.internal_state_decoder(latent)
        internal_state = self.internal_state_decoder_projection(internal_state)
        internal_state = symexp(internal_state)  # Apply symexp to map back to original scale
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
    def __init__(self, config: DreamerConfig, input_dim: int, output_dim: int, num_layers: int):
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
        self.linear = nn.Linear(input_size + hidden_size, 3 * hidden_size, bias=self.bias)
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
        ), f"LayerNormGRUCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"

        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
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

        recurrent_input_dim = config.action_space_dim + self.stochastic_size
        self.recurrent_input = nn.Sequential(
            nn.Linear(recurrent_input_dim, config.hidden_dim),
            nn.RMSNorm(config.hidden_dim),
            nn.SiLU(),
        )
        self.gru = LayerNormGRUCell(config.hidden_dim, self.recurrent_units)

        # Both prior and posterior networks are MLPs with one hidden layer, outputting logits for the categorical distribution
        self.prior_network = nn.Sequential(
            MLP(config, input_dim=self.recurrent_units, output_dim=config.hidden_dim, num_layers=1),
            nn.Linear(config.hidden_dim, self.stochastic_units * self.discrete_classes),
        )
        self.posterior_network = nn.Sequential(
            MLP(config, input_dim=self.recurrent_units + config.encoder_dim, output_dim=config.hidden_dim, num_layers=1),
            nn.Linear(config.hidden_dim, self.stochastic_units * self.discrete_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) > 1:
                    nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def _uniform_mix(self, logits: Tensor) -> Tensor:
        logits = logits.view(*logits.shape[:-1], self.stochastic_units, self.discrete_classes)
        if self.unimix > 0.0:
            probs = logits.softmax(dim=-1)
            uniform = torch.ones_like(probs) / self.discrete_classes
            probs = (1.0 - self.unimix) * probs + self.unimix * uniform
            logits = torch.log(probs.clamp_min(1e-8))
        return logits

    def _dist(self, logits: Tensor) -> Independent:
        return Independent(OneHotCategoricalStraightThrough(logits=logits), 1)

    def _sample_stochastic(self, logits: Tensor, deterministic: bool = False) -> Tensor:
        if deterministic:
            index = logits.argmax(dim=-1)
            return F.one_hot(index, self.discrete_classes).to(dtype=logits.dtype)
        return self._dist(logits).rsample()

    def _feature(self, stochastic_state: Tensor, recurrent_state: Tensor) -> Tensor:
        stochastic_flat = stochastic_state.reshape(*stochastic_state.shape[:-2], self.stochastic_size)
        return torch.cat([stochastic_flat, recurrent_state], dim=-1)

    def split_feature(self, latent: Tensor) -> Tuple[Tensor, Tensor]:
        stochastic_flat, recurrent_state = torch.split(latent, [self.stochastic_size, self.recurrent_units], dim=-1)
        stochastic_state = stochastic_flat.view(*stochastic_flat.shape[:-1], self.stochastic_units, self.discrete_classes)
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

    def recurrent_step(self, stochastic_state: Tensor, action: Tensor, recurrent_state: Tensor) -> Tensor:
        # Gets h_t from z_{t-1} and a_{t-1}
        # Included the GRU here
        stochastic_flat = stochastic_state.reshape(stochastic_state.shape[0], self.stochastic_size)
        action = action / torch.maximum(torch.ones_like(action), action.abs())
        recurrent_input = torch.cat([stochastic_flat, action], dim=-1)
        return self.gru(self.recurrent_input(recurrent_input), recurrent_state)

    def imagine_step(self, latent: Tensor, recurrent_state: Tensor, action: Tensor, deterministic: bool = False) -> Tuple[Tensor, Tensor, Independent]:
        stochastic_state, _ = self.split_feature(latent)
        recurrent_state = self.recurrent_step(stochastic_state, action, recurrent_state)
        logits = self.prior_logits(recurrent_state)
        next_stochastic = self._sample_stochastic(logits, deterministic=deterministic)
        next_latent = self._feature(next_stochastic, recurrent_state)
        return next_latent, recurrent_state, self._dist(logits)

    def forward(self, action, embed, is_first, recurrent_state=None, deterministic=False):
        """
        Forward pass through RSSM.

        Args:
            action: (batch, seq_len, action_dim) or (batch, action_dim)
            embed: (batch, seq_len, encoder_dim) or (batch, encoder_dim)
            is_first: (batch, seq_len) or (batch,)
            recurrent_state: (batch, recurrent_units) or None
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

        # Initialise for the start of an episode
        if recurrent_state is None:
            recurrent_state = torch.zeros(batch_size, self.recurrent_units, device=action.device)

        previous_stochastic = torch.zeros(
            batch_size,
            self.stochastic_units,
            self.discrete_classes,
            device=action.device,
            dtype=action.dtype,
        )

        prior_dists = []
        posterior_dists = []
        latents = []

        for t in range(seq_len):
            # Reset recurrent state on episode start
            reset_mask = 1.0 - is_first[:, t:t+1]
            recurrent_state = recurrent_state * reset_mask
            previous_stochastic = previous_stochastic * reset_mask[..., None]
            step_action = action[:, t] * reset_mask

            # Compute h_t from z_{t-1} and a_{t-1}, then predict prior/posterior.
            recurrent_state = self.recurrent_step(previous_stochastic, step_action, recurrent_state)
            prior_logits = self.prior_logits(recurrent_state)
            posterior_logits = self.posterior_logits(recurrent_state, embed[:, t])
            prior_dist = self._dist(prior_logits)
            posterior_dist = self._dist(posterior_logits)
            prior_dists.append(prior_dist)
            posterior_dists.append(posterior_dist)

            stochastic_state = self._sample_stochastic(posterior_logits, deterministic=deterministic)
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

        self.net = MLP(config, input_dim=input_dim, output_dim=config.hidden_dim, num_layers=3)

        # Beta distribution parameters for continuous actions
        self.alpha = nn.Linear(config.hidden_dim, action_dim)
        self.beta = nn.Linear(config.hidden_dim, action_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
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
            MLP(config, input_dim=input_dim, output_dim=config.hidden_dim, num_layers=3),
            nn.Linear(config.hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, latent):
        """
        Compute value estimate.

        Args:
            latent: (batch, latent_dim) or (batch, seq_len, latent_dim)

        Returns:
            value: (batch, 1) or (batch, seq_len, 1)
        """
        if len(latent.shape) == 3:
            batch_size, seq_len, latent_dim = latent.shape
            latent_flat = latent.reshape(batch_size * seq_len, latent_dim)
            squeeze_output = True
        else:
            batch_size = latent.shape[0]
            latent_flat = latent
            squeeze_output = False

        value = self.net(latent_flat)

        if squeeze_output:
            value = value.view(batch_size, seq_len, 1)

        return value


class RewardPredictor(nn.Module):
    """Predict rewards from latent state.
    Use TwoHotEncodingDistribution after this"""

    def __init__(self, config: DreamerConfig):
        super().__init__()
        self.config = config
        input_dim = config.latent_dim
        output_dim = config.hidden_dim
        self.net = MLP(config, input_dim=input_dim, output_dim=output_dim, num_layers=1)
        self.reward_projection = nn.Linear(output_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, latent):
        """Predict reward from latent state."""
        latent = to_tensor(latent, self.config.device)
        latent = self.net(latent)
        return self.reward_projection(latent)


class ContinuePredictor(nn.Module):
    """Predict terminal probability from latent state."""

    def __init__(self, config: DreamerConfig):
        super().__init__()
        self.config = config
        input_dim = config.latent_dim
        output_dim = config.hidden_dim
        self.net = MLP(config, input_dim=input_dim, output_dim=output_dim, num_layers=1)
        self.terminal_projection = nn.Sequential(
            nn.Linear(output_dim, 1),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, latent):
        """Predict terminal probability from latent state."""
        latent = to_tensor(latent, self.config.device)
        return self.terminal_projection(self.net(latent))


class SequenceReplayBuffer:
    """Replay buffer that stores same-environment sequences for RSSM training.

    The training loop adds one transition per worker at each environment step. This
    buffer groups those calls into one time-major row, so sampling can return
    contiguous chunks from a single worker trajectory instead of accidentally
    mixing workers within one RSSM sequence.
    """

    def __init__(self, config: DreamerConfig, device='cpu'):
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

        self._pending_observations = []
        self._pending_actions = []
        self._pending_rewards = []
        self._pending_dones = []
        self._pending_episode_starts = []

    def add(self, obs_dict, action, reward, done, is_first=False):
        """Add one worker transition.

        Calls are expected in worker order. Once all workers for a timestep have
        been added, the row is committed as [env0, env1, ...].
        """
        if len(self._pending_observations) >= self.num_workers:
            raise RuntimeError("Pending replay row already has num_workers entries; this indicates an add() ordering bug.")

        self._pending_observations.append(obs_dict)
        self._pending_actions.append(action)
        self._pending_rewards.append(reward)
        self._pending_dones.append(done)
        self._pending_episode_starts.append(is_first)

        if len(self._pending_observations) == self.num_workers:
            self.observations.append(self._pending_observations)
            self.actions.append(self._pending_actions)
            self.rewards.append(self._pending_rewards)
            self.dones.append(self._pending_dones)
            self.episode_starts.append(self._pending_episode_starts)

            self._pending_observations = []
            self._pending_actions = []
            self._pending_rewards = []
            self._pending_dones = []
            self._pending_episode_starts = []

    def sample(self):
        """Sample same-worker contiguous sequences from the buffer."""
        time_len = len(self.observations)
        if len(self) < self.config.min_buffer_size_before_training or time_len < self.batch_length:
            return None

        max_start = time_len - self.batch_length + 1
        start_indices = self._rng.integers(0, max_start, size=self.batch_size)
        env_indices = self._rng.integers(0, self.num_workers, size=self.batch_size)

        obs_sequences = []
        action_sequences = []
        reward_sequences = []
        done_sequences = []
        is_first_sequences = []

        for start_idx, env_idx in zip(start_indices, env_indices):
            obs_seq = [self.observations[start_idx + i][env_idx] for i in range(self.batch_length)]
            action_seq = [self.actions[start_idx + i][env_idx] for i in range(self.batch_length)]
            reward_seq = [self.rewards[start_idx + i][env_idx] for i in range(self.batch_length)]
            done_seq = [self.dones[start_idx + i][env_idx] for i in range(self.batch_length)]
            is_first_seq = [self.episode_starts[start_idx + i][env_idx] for i in range(self.batch_length)]
            is_first_seq[0] = True

            obs_sequences.append(obs_seq)
            action_sequences.append(action_seq)
            reward_sequences.append(reward_seq)
            done_sequences.append(done_seq)
            is_first_sequences.append(is_first_seq)

        return {
            'obs': obs_sequences,
            'actions': action_sequences,
            'rewards': reward_sequences,
            'dones': done_sequences,
            'is_first': is_first_sequences,
        }

    def __len__(self):
        return len(self.observations) * self.num_workers + len(self._pending_observations)


# From https://github.com/Eclectic-Sheep/sheeprl/blob/main/sheeprl/utils/utils.py
# From https://github.com/danijar/dreamerv3/blob/8fa35f83eee1ce7e10f3dee0b766587d0a713a60/dreamerv3/jaxutils.py
def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x: Tensor) -> Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


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

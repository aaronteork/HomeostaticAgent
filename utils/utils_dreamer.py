from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal, Beta, Independent
from torchrl.data import ListStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from tensordict import TensorDict
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
        self.vision_encoder = VisionEncoder(channels=4, depth=config.base_cnn_channels)
        # Get output dimension of vision encoder by passing a dummy input
        with torch.no_grad():
            dummy_input = torch.zeros(1, 4, 64, 64)
            vision_output_dim = self.vision_encoder(dummy_input).shape[-1]

        # Vector encoder for proprioception
        self.proprioception_encoder = MLP(config, input_dims=config.obs_space_dim, output_dims=config.hidden_dim, num_layers=config.mlp_n_layers)

        # Vector encoder for internal state
        self.internal_state_encoder = MLP(config, input_dims=2 if config.num_heat == 0 else 3, output_dims=config.hidden_dim, num_layers=config.mlp_n_layers)

        # Fusion layer
        self.fusion = nn.Linear(vision_output_dim + config.hidden_dim + config.hidden_dim, config.encoder_dim)

        self._init_weights()

    def forward(self, vision, proprioception, internal_state):
        vision = torch.from_numpy(vision).to(self.config.device)
        proprioception = torch.from_numpy(proprioception).to(self.config.device)
        proprioception = symlog(proprioception)
        internal_state = torch.from_numpy(internal_state).to(self.config.device)
        internal_state = symlog(internal_state)

        vision_embed = self.vision_encoder(vision)
        proprioception_embed = self.proprioception_encoder(proprioception)
        internal_state_embed = self.internal_state_encoder(internal_state)
        return self.fusion(torch.cat([vision_embed, proprioception_embed, internal_state_embed], dim=-1))

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
        self.vision_decoder = VisionDecoder(feature_dim=config.stochastic_units + config.recurrent_units, output_channels=4, depth=config.base_cnn_channels, output_shape=(64, 64))
        
        # Proprioception decoder
        self.proprioception_decoder = MLP(config, input_dims=config.stochastic_units + config.recurrent_units, output_dims=config.hidden_dim, num_layers=config.mlp_n_layers)
        self.proprioception_decoder_projection = nn.Linear(config.hidden_dim, config.obs_space_dim)  # Projection layer for proprioception

        # Internal state decoder
        self.internal_state_decoder = MLP(config, input_dims=config.stochastic_units + config.recurrent_units, output_dims=config.hidden_dim, num_layers=config.mlp_n_layers)
        self.internal_state_decoder_projection = nn.Linear(config.hidden_dim, 2 if config.num_heat == 0 else 3)  # Projection layer for internal state

        self._init_weights()

    def forward(self, latent):
        latent = torch.from_numpy(latent).to(self.config.device)
        # Vision
        vision = self.vision_decoder(latent)
        # Proprioception
        proprioception = self.proprioception_decoder(latent)
        proprioception = self.proprioception_decoder_projection(proprioception)
        proprioception = symexp(proprioception)  # Apply symexp to map back to original scale
        # Internal state
        internal_state = self.internal_state_decoder(latent)
        internal_state = self.internal_state_decoder_projection(internal_state)
        internal_state = symexp(internal_state)  # Apply symexp to map back to original scale
        return vision, proprioception, internal_state

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class MLP(nn.Module):
    def __init__(self, cfg: DreamerConfig, input_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                dim_in = input_dim
                dim_out = cfg.hidden_dim
            elif i == num_layers - 1:
                dim_in = cfg.hidden_dim
                dim_out = output_dim
            else:
                dim_in = cfg.hidden_dim
                dim_out = cfg.hidden_dim
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(nn.RMSNorm(dim_out))
            layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class RSSM(nn.Module):
    """Recurrent State-Space Model for learning world dynamics."""

    def __init__(self, config: DreamerConfig):
        super().__init__()
        self.config = config

        # GRU for recurrent state
        input_size = config.action_space_dim + config.latent_dim + 1  # action + z + is_first
        self.gru = nn.GRU(input_size, config.gru_units, num_layers=config.gru_layers, batch_first=True)

        # Prior distribution (from recurrent state)
        self.prior_mean = nn.Linear(config.gru_units, config.latent_dim)
        self.prior_std = nn.Linear(config.gru_units, config.latent_dim)

        # Posterior distribution (from recurrent state + embedding)
        self.posterior_mean = nn.Linear(config.gru_units + config.encoder_dim, config.latent_dim)
        self.posterior_std = nn.Linear(config.gru_units + config.encoder_dim, config.latent_dim)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) > 1:
                    nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def _compute_dist(self, mean, std):
        """Compute Normal distribution with learned parameters."""
        std = F.softplus(std) + 0.1
        return Normal(mean, std)

    def prior(self, recurrent_state):
        """Compute prior distribution p(z_t | h_t)."""
        mean = self.prior_mean(recurrent_state)
        std = self.prior_std(recurrent_state)
        return self._compute_dist(mean, std)

    def posterior(self, recurrent_state, embed):
        """Compute posterior distribution q(z_t | h_t, o_t)."""
        x = torch.cat([recurrent_state, embed], dim=-1)
        mean = self.posterior_mean(x)
        std = self.posterior_std(x)
        return self._compute_dist(mean, std)

    def forward(self, action, embed, is_first, recurrent_state=None, deterministic=False):
        """
        Forward pass through RSSM.

        Args:
            action: (batch, seq_len, action_dim) or (batch, action_dim)
            embed: (batch, seq_len, encoder_dim) or (batch, encoder_dim)
            is_first: (batch, seq_len) or (batch,)
            recurrent_state: (batch, gru_units) or None
            deterministic: if True, use prior mean (for imagination); else use posterior sample

        Returns:
            latent: (batch, seq_len, latent_dim) or (batch, latent_dim)
            recurrent_state: (batch, gru_units)
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

        if recurrent_state is None:
            recurrent_state = torch.zeros(batch_size, self.gru_units, device=action.device)

        prior_dists = []
        posterior_dists = []
        latents = []

        for t in range(seq_len):
            # Reset recurrent state on episode start
            recurrent_state = recurrent_state * (1.0 - is_first[:, t:t+1])

            # Compute prior from current recurrent state
            prior_dist = self.prior(recurrent_state)
            prior_dists.append(prior_dist)

            # Compute posterior from recurrent state + observation
            posterior_dist = self.posterior(recurrent_state, embed[:, t])
            posterior_dists.append(posterior_dist)

            # Sample latent: use posterior during training, prior during imagination
            if deterministic:
                latent = prior_dist.mean
            else:
                latent = posterior_dist.rsample()
            latents.append(latent)

            # Update recurrent state via GRU for next timestep
            gru_input = torch.cat([action[:, t], latent, is_first[:, t:t+1]], dim=-1)
            _, recurrent_state_new = self.gru(gru_input.unsqueeze(1), recurrent_state.unsqueeze(0))
            recurrent_state = recurrent_state_new.squeeze(0)

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

        layers = []
        prev_dim = input_dim
        for hidden_dim in config.actor_hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim

        self.net = nn.Sequential(*layers)

        # Beta distribution parameters for continuous actions
        self.alpha = nn.Linear(prev_dim, action_dim)
        self.beta = nn.Linear(prev_dim, action_dim)

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
            action = dist.mean
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

        layers = []
        prev_dim = input_dim
        for hidden_dim in config.critic_hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

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
    """Predict rewards from latent state."""

    def __init__(self, config: DreamerConfig):
        super().__init__()
        input_dim = config.latent_dim
        layers = [
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        ]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, latent):
        """Predict reward from latent state."""
        return self.net(latent)


class TerminalPredictor(nn.Module):
    """Predict terminal state from latent state."""

    def __init__(self, config: DreamerConfig):
        super().__init__()
        input_dim = config.latent_dim
        layers = [
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        ]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, latent):
        """Predict terminal probability from latent state."""
        return torch.sigmoid(self.net(latent))


class SequenceReplayBuffer:
    """Replay buffer that stores sequences for RSSM training."""

    def __init__(self, config: DreamerConfig, device='cpu'):
        self.config = config
        self.device = device
        self.buffer_size = config.buffer_size
        self.seq_length = config.replay_buffer_seq_length
        self.batch_size = config.replay_buffer_batch_size

        self.observations = deque(maxlen=config.buffer_size)
        self.actions = deque(maxlen=config.buffer_size)
        self.rewards = deque(maxlen=config.buffer_size)
        self.dones = deque(maxlen=config.buffer_size)
        self.episode_starts = deque(maxlen=config.buffer_size)

    def add(self, obs_dict, action, reward, done, is_first=False):
        """Add transition to buffer."""
        self.observations.append(obs_dict)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.episode_starts.append(is_first)

    def sample(self):
        """Sample sequences from buffer."""
        buffer_len = len(self.observations)
        if buffer_len < self.config.min_buffer_size_before_training:
            return None

        # Sample random starting indices
        indices = np.random.randint(0, buffer_len - self.seq_length, size=self.batch_size)

        obs_sequences = []
        action_sequences = []
        reward_sequences = []
        done_sequences = []
        is_first_sequences = []

        for idx in indices:
            obs_seq = [self.observations[idx + i] for i in range(self.seq_length)]
            action_seq = [self.actions[idx + i] for i in range(self.seq_length)]
            reward_seq = [self.rewards[idx + i] for i in range(self.seq_length)]
            done_seq = [self.dones[idx + i] for i in range(self.seq_length)]
            is_first_seq = [self.episode_starts[idx + i] for i in range(self.seq_length)]

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
        return len(self.observations)


# From https://github.com/Eclectic-Sheep/sheeprl/blob/main/sheeprl/utils/utils.py
# From https://github.com/danijar/dreamerv3/blob/8fa35f83eee1ce7e10f3dee0b766587d0a713a60/dreamerv3/jaxutils.py
def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x: Tensor) -> Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

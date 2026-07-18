from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import (
    Beta,
    Independent,
    OneHotCategoricalStraightThrough,
)

from configs.config_dreamer import DreamerConfig
from utils.utils_dreamer import (
    TwoHotEncodingDistribution,
    symexp,
    symlog,
    to_tensor,
)
from utils.vision import VisionDecoder, VisionEncoder


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
        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}")

        layers = []
        for i in range(num_layers):
            dim_in = input_dim if i == 0 else config.hidden_dim
            dim_out = output_dim if i == num_layers - 1 else config.hidden_dim
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(nn.RMSNorm(dim_out))
            layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class BlockLinear(nn.Module):
    """Linear projection whose weight matrix is block diagonal."""

    def __init__(
        self, input_size: int, output_size: int, blocks: int, bias: bool = True
    ) -> None:
        super().__init__()
        if input_size % blocks != 0:
            raise ValueError(
                f"input_size ({input_size}) must be divisible by blocks ({blocks})"
            )
        if output_size % blocks != 0:
            raise ValueError(
                f"output_size ({output_size}) must be divisible by blocks ({blocks})"
            )
        self.input_size = input_size
        self.output_size = output_size
        self.blocks = blocks
        self.input_block_size = input_size // blocks
        self.output_block_size = output_size // blocks
        self.weight = nn.Parameter(
            torch.empty(blocks, self.output_block_size, self.input_block_size)
        )
        self.bias = (
            nn.Parameter(torch.empty(blocks, self.output_block_size)) if bias else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for block_weight in self.weight:
            nn.init.orthogonal_(block_weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.input_size:
            raise ValueError(
                f"expected input width {self.input_size}, got {x.shape[-1]}"
            )
        grouped = x.reshape(*x.shape[:-1], self.blocks, self.input_block_size)
        output = torch.einsum("...gi,goi->...go", grouped, self.weight)
        if self.bias is not None:
            output = output + self.bias
        return output.reshape(*x.shape[:-1], self.output_size)


class BlockGRUCell(nn.Module):
    """DreamerV3 deterministic core with block-diagonal hidden and GRU weights."""

    def __init__(
        self,
        deter_size: int,
        stoch_size: int,
        action_size: int,
        hidden_size: int,
        blocks: int = 8,
        dyn_layers: int = 1,
    ) -> None:
        super().__init__()
        if deter_size % blocks != 0:
            raise ValueError(
                f"deter_size ({deter_size}) must be divisible by blocks ({blocks})"
            )
        if dyn_layers < 1:
            raise ValueError("dyn_layers must be at least 1")
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.blocks = blocks
        self.deter_block_size = deter_size // blocks

        def input_projection(input_size: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.RMSNorm(hidden_size, eps=1e-4),
                nn.SiLU(),
            )

        self.deter_input = input_projection(deter_size)
        self.stoch_input = input_projection(stoch_size)
        self.action_input = input_projection(action_size)

        # Each block receives its deterministic slice and a copy of the three
        # global input embeddings, matching the official DreamerV3 RSSM core.
        first_hidden_size = deter_size + blocks * 3 * hidden_size
        self.dynamic_layers = nn.ModuleList()
        for layer_index in range(dyn_layers):
            input_size = first_hidden_size if layer_index == 0 else deter_size
            self.dynamic_layers.append(
                nn.Sequential(
                    BlockLinear(input_size, deter_size, blocks),
                    nn.RMSNorm(deter_size, eps=1e-4),
                    nn.SiLU(),
                )
            )
        self.gate_projection = BlockLinear(deter_size, 3 * deter_size, blocks)

    def forward(self, stoch: Tensor, deter: Tensor, action: Tensor) -> Tensor:
        stoch = stoch.reshape(*stoch.shape[:-2], self.stoch_size)
        action_scale = torch.maximum(torch.ones_like(action), action.abs()).detach()
        action = action / action_scale

        global_features = torch.cat(
            [
                self.deter_input(deter),
                self.stoch_input(stoch),
                self.action_input(action),
            ],
            dim=-1,
        )
        repeated_features = global_features.unsqueeze(-2).expand(
            *global_features.shape[:-1], self.blocks, global_features.shape[-1]
        )
        grouped_deter = deter.reshape(
            *deter.shape[:-1], self.blocks, self.deter_block_size
        )
        x = torch.cat([grouped_deter, repeated_features], dim=-1).reshape(
            *deter.shape[:-1], -1
        )
        for layer in self.dynamic_layers:
            x = layer(x)

        gates = self.gate_projection(x).reshape(
            *deter.shape[:-1], self.blocks, 3 * self.deter_block_size
        )
        reset, candidate, update = gates.chunk(3, dim=-1)
        reset = reset.reshape(*deter.shape[:-1], self.deter_size).sigmoid()
        candidate = candidate.reshape(*deter.shape[:-1], self.deter_size)
        candidate = torch.tanh(reset * candidate)
        update = update.reshape(*deter.shape[:-1], self.deter_size)
        update = torch.sigmoid(update - 1.0)
        return update * candidate + (1.0 - update) * deter


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

        self.gru = BlockGRUCell(
            deter_size=self.recurrent_units,
            stoch_size=self.stochastic_size,
            action_size=config.action_space_dim,
            hidden_size=config.hidden_dim,
            blocks=config.rssm_blocks,
            dyn_layers=config.rssm_dyn_layers,
        )

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
                if len(param.shape) == 3:
                    for block_weight in param:
                        nn.init.orthogonal_(block_weight)
                elif len(param.shape) > 1:
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
        return self.gru(stochastic_state, recurrent_state, action)

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
            action: (batch, seq_len, action_dim) or (batch, action_dim). This is previous action
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
        # Check if there is any time dimension. If not, add a time dimension for a single step
        if len(action.shape) == 2:
            action = action.unsqueeze(1)
            embed = embed.unsqueeze(1)
            is_first = is_first.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False
        batch_size, seq_len, _ = action.shape

        # Add initial state if it is not present
        # Also create initial state for resets
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
        self.alpha = nn.Sequential(
            nn.Linear(config.hidden_dim, action_dim), nn.Softplus()
        )
        self.beta = nn.Sequential(
            nn.Linear(config.hidden_dim, action_dim), nn.Softplus()
        )

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
        alpha = self.alpha(x) + 1.0
        beta = self.beta(x) + 1.0
        dist = Beta(alpha, beta)
        if deterministic:
            action = dist.mode  # Beta mode in [0, 1] because alpha,beta > 1
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
    """Predict continuation probability from latent state."""

    def __init__(self, config: DreamerConfig):
        super().__init__()
        self.config = config
        input_dim = config.latent_dim
        output_dim = config.hidden_dim
        self.net = MLP(config, input_dim=input_dim, output_dim=output_dim, num_layers=1)
        self.continue_projection = nn.Sequential(nn.Linear(output_dim, 1), nn.Sigmoid())
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # nn.init.zeros_(self.continue_projection[0].weight)
        # nn.init.constant_(
        #     self.continue_projection[0].bias, self.config.continue_initial_logit
        # )

    def forward(self, latent):
        """Predict continuation probability from latent state."""
        latent = to_tensor(latent, self.config.device)
        return self.continue_projection(self.net(latent))


class WorldModel(nn.Module):
    """Thin orchestration module for Dreamer world-model components."""

    def __init__(self, config: DreamerConfig):
        super().__init__()
        self.config = config
        self.encoder = ObservationEncoder(config)
        self.rssm = RSSM(config)
        self.decoder = ObservationDecoder(config)
        self.reward_predictor = RewardPredictor(config)
        self.continue_predictor = ContinuePredictor(config)

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

    def predict_continue(self, latent):
        return self.continue_predictor(latent)

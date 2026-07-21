from dataclasses import dataclass
from configs.config_env import EnvConfig


@dataclass(frozen=True, kw_only=True)
class DreamerConfig(EnvConfig):
    # Model parameters
    hidden_dim: int = 64
    stochastic_units: int = 32

    rssm_unimix: float = 0.01
    rssm_blocks: int = 8
    rssm_dyn_layers: int = 1
    free_nats: float = 1.0
    mlp_n_layers: int = 3
    batch_size: int = 16
    batch_length: int = 64
    total_env_steps: int = 5_000_000
    replay_ratio: int = 32   # Was originally 512 to follow the paper for visual control but reduced 32 to match Minecraft since one episode is quite long too (to decay)
    # total_updates: int = 1_000
    adam_eps: float = 1e-5
    laprop_eps: float = 1e-20
    laprop_beta1: float = 0.9
    laprop_beta2: float = 0.99
    laprop_lr: float = 4e-5
    two_hot_bins: int = 255
    two_hot_high: float = 20.0
    two_hot_low: float = -20.0

    # Replay buffer parameters
    # replay_scaling: int = 64
    replay_capacity: int = 5_000_000  # paper is originally 5e6
    min_buffer_size_before_training: int | None = None

    # World model loss
    world_model_grad_norm_clip: float = 1000.0
    vision_reconstruction_weight: float = 1.0
    proprioception_reconstruction_weight: float = 1.0
    internal_state_reconstruction_weight: float = 1.0
    heat_sensor_reconstruction_weight: float = 1.0
    dyn_loss_weight: float = 1.0
    rep_loss_weight: float = 0.1
    reward_weight: float = 1.0
    continue_weight: float = 1.0
    # continue_initial_logit: float = 5.0

    # Imagination and optimisation
    imagine_horizon: int = 15
    imagine_last: int = 0
    imagine_batch_size: int = 16
    # world_model_grad_steps_per_update: int = 1
    world_model_lr: float = 1e-4
    ent_coef: float = 3e-4
    return_norm_rate: float = 0.01
    return_norm_limit: float = 1.0
    return_norm_percentile_low: float = 5.0
    return_norm_percentile_high: float = 95.0

    # Actor and critic
    actor_lr: float = 3e-5
    # actor_critic_grad_steps_per_update: int = 1
    actor_grad_norm_clip: float = 100.0

    critic_lr: float = 3e-5
    ema_critic_tau: float = 0.02
    critic_slow_regularization: float = 1.0
    critic_imagined_loss: float = 1.0
    critic_replay_loss: float = 0.3
    critic_grad_norm_clip: float = 100.0

    # Discount and GAE. DreamerV3 uses a horizon-based continuation discount
    # of 1 - 1 / horizon when contdisc is enabled.
    gamma: float = 0.997
    horizon: int = 333
    contdisc: bool = True
    gae_lambda: float = 0.95

    def __post_init__(self):
        if self.min_buffer_size_before_training is None:
            object.__setattr__(
                self,
                "min_buffer_size_before_training",
                int(self.batch_size * self.batch_length),
            )

    @property
    def encoder_dim(self) -> int:
        return self.base_cnn_channels * 64 + 2 * self.hidden_dim

    @property
    def stochastic_size(self) -> int:
        return self.stochastic_units * self.discrete_classes

    @property
    def latent_dim(self) -> int:
        return self.stochastic_size + self.recurrent_units

    @property
    def discount(self) -> float:
        if self.contdisc:
            return 1.0 - 1.0 / self.horizon
        return self.gamma

    @property
    def recurrent_units(self) -> int:
        return 8 * self.hidden_dim

    @property
    def base_cnn_channels(self) -> int:
        return self.hidden_dim // 16

    @property
    def discrete_classes(self) -> int:
        return self.hidden_dim // 16

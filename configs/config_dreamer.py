from dataclasses import dataclass
from configs.config_env import EnvConfig


@dataclass(frozen=True, kw_only=True)
class DreamerConfig(EnvConfig):
    # Model parameters
    hidden_dim: int = 256
    recurrent_units: int = 1024  # 8 * hidden_dim
    base_cnn_channels: int = 16  # hidden_dim // 16
    mlp_n_layers: int = 3
    stochastic_units: int = 32
    discrete_classes: int = 16  # hidden_dim // 16
    rssm_unimix: float = 0.01
    free_nats: float = 1.0
    batch_size: int = 16
    batch_length: int = 64
    total_env_steps: int = 5_000_000
    replay_ratio: int = 128   # Was originally 512 to follow the paper for visual control but reduced to speed things up
    total_updates: int = 1_000
    adam_eps: float = 1e-5
    two_hot_bins: int = 255
    two_hot_high: float = 20.0
    two_hot_low: float = -20.0

    # Replay buffer parameters
    replay_scaling: int = 64
    replay_capacity: int = 500_000  # paper is originally 5e6
    min_buffer_size_before_training: int | None = None
    online_batch_fraction: float = 0.5
    seed_steps: int = 50_000
    exploration_epsilon: float = 0.1

    # World model loss
    world_model_grad_norm_clip: float = 1000.0
    vision_reconstruction_weight: float = 1.0
    proprioception_reconstruction_weight: float = 1.0
    internal_state_reconstruction_weight: float = 1.0
    heat_sensor_reconstruction_weight: float = 1.0
    dyn_loss_weight: float = 1.0
    rep_loss_weight: float = 0.1
    kl_weight: float = 1.0
    reward_weight: float = 1.0
    terminal_weight: float = 1.0

    # Imagination and optimisation
    imagine_horizon: int = 15
    imagine_last: int = 0
    imagine_batch_size: int = 16
    world_model_grad_steps_per_update: int = 1
    world_model_lr: float = 1e-4
    ent_coef: float = 3e-4
    return_norm_rate: float = 0.01
    return_norm_limit: float = 1.0
    return_norm_percentile_low: float = 5.0
    return_norm_percentile_high: float = 95.0

    # Actor and critic
    actor_lr: float = 3e-5
    actor_critic_grad_steps_per_update: int = 1
    actor_grad_norm_clip: float = 100.0

    critic_lr: float = 3e-5
    ema_critic_tau: float = 0.02
    critic_slow_regularization: float = 1.0
    critic_imagined_loss: float = 1.0
    critic_replay_loss: float = 0.3
    critic_grad_norm_clip: float = 100.0

    # Discount and GAE
    gamma: float = 0.99
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

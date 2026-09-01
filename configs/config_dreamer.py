from dataclasses import dataclass
from typing import Literal
from configs.config_env import EnvConfig


@dataclass(frozen=True, kw_only=True)
class DreamerConfig(EnvConfig):
    # Initial setup
    total_env_steps: int = 3_000_000
    replay_ratio: int = 32
    batch_size: int = 16
    batch_length: int = 64
    mlp_n_layers: int = 3
    train_metrics_interval: int = 25_000
    rollout_metrics_interval: int = 100_000
    per_joint_metrics_interval: int = 500_000
    checkpoint_save_interval: int = 200_000

    # Replay buffer parameters
    replay_capacity: int = 1_000_000  # paper is originally 5e6 but reduced for memory reasons
    replay_context: int = 1
    min_buffer_size_before_training: int | None = None

    # Discount and GAE. DreamerV3 uses a horizon-based continuation discount
    # of 1 - 1 / horizon when contdisc is enabled.
    gamma: float = 0.997
    horizon: int = 333
    contdisc: bool = True
    gae_lambda: float = 0.95

    # Use Spatially-enhanced Recurrent Unit
    use_sru: bool = False

    # Weighted loss mode. ``all`` independently moderates
    # vision, proprioception, internal state, reward, and the combined KL.
    # ``all_except_kl`` learns the non-KL scales but retains Dreamer's fixed
    # KL strength. ``observations_only`` learns scales only for reconstruction
    # heads. ``weighted`` follows the paper's observation/reward/KL grouping.
    # ``none`` preserves the fixed-weight Dreamer objective.
    weighted_loss: Literal[
        "none", "harmony", "all", "all_except_kl", "observations_only"
    ] = "observations_only"

    # World Model
    hidden_dim: int = 512
    stochastic_units: int = 32
    rssm_unimix: float = 0.01
    rssm_blocks: int = 8
    rssm_dyn_layers: int = 1

    # Two hot
    two_hot_bins: int = 255
    # two_hot_high: float = 20.0
    # two_hot_low: float = -20.0

    # Optimiser
    laprop_eps: float = 1e-20
    laprop_beta1: float = 0.9
    laprop_beta2: float = 0.999
    laprop_lr: float = 4e-5
    warmup_steps: int = 1000
    use_amp: bool = True

    # Losses
    vision_reconstruction_weight: float = 1.0
    proprioception_reconstruction_weight: float = 1.0
    internal_state_reconstruction_weight: float = 1.0
    heat_sensor_reconstruction_weight: float = 1.0
    dyn_loss_weight: float = 1.0
    rep_loss_weight: float = 0.1
    reward_weight: float = 1.0
    continue_weight: float = 1.0
    free_nats: float = 1.0

    # Imagination
    imagine_horizon: int = 15
    imagine_last: int = 0
    imagine_batch_size: int = 16

    # Actor
    actor_policy: Literal["beta", "gaussian"] = "beta"
    actor_min_concentration: float = 1.0
    actor_outscale: float = 0.01
    actor_min_std: float = 0.1
    actor_max_std: float = 1.0
    ent_coef: float = 0.005

    # Critic
    return_norm_rate: float = 0.01
    return_norm_limit: float = 1.0
    return_norm_percentile_low: float = 5.0
    return_norm_percentile_high: float = 95.0
    ema_critic_tau: float = 0.02
    critic_slow_regularization: float = 1.0
    critic_imagined_loss: float = 1.0
    critic_replay_loss: float = 0.3

    def __post_init__(self):
        if self.weighted_loss not in (
            "none",
            "harmony",
            "all",
            "all_except_kl",
            "observations_only",
        ):
            raise ValueError(
                "weighted_loss must be one of 'none', 'harmony', 'all', "
                "'all_except_kl', or 'observations_only'"
            )
        if self.replay_context < 0:
            raise ValueError("replay_context must be non-negative")
        if self.actor_min_concentration < 1.0:
            raise ValueError(
                "actor_min_concentration must be at least 1.0 so the Beta "
                "policy has a finite interior mode"
            )
        if self.actor_outscale <= 0.0:
            raise ValueError("actor_outscale must be positive")
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

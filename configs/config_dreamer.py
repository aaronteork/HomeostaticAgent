from dataclasses import dataclass
from typing import Literal
from configs.config_env import EnvConfig


@dataclass(frozen=True, kw_only=True)
class DreamerConfig(EnvConfig):
    # Model parameters
    hidden_dim: int = 512
    stochastic_units: int = 32

    rssm_unimix: float = 0.01
    rssm_blocks: int = 8
    rssm_dyn_layers: int = 1
    free_nats: float = 1.0
    mlp_n_layers: int = 3
    batch_size: int = 16
    batch_length: int = 64
    total_env_steps: int = 5_000_000
    replay_ratio: int = 32
    # Keep scalar diagnostics useful without making SQLite telemetry a material
    # part of the training loop. The deterministic evaluation is deliberately
    # sparse because each worker runs one complete episode (up to
    # ``eval_max_steps``) outside the training trajectory.
    train_metrics_interval: int = 25_000
    rollout_metrics_interval: int = 50_000
    per_joint_metrics_interval: int = 500_000
    deterministic_probe_interval: int = 500_000
    # total_updates: int = 1_000
    adam_eps: float = 1e-5
    laprop_eps: float = 1e-20
    laprop_beta1: float = 0.9
    laprop_beta2: float = 0.999
    laprop_lr: float = 4e-5
    warmup_steps: int = 1000
    use_amp: bool = True
    two_hot_bins: int = 255
    two_hot_high: float = 20.0
    two_hot_low: float = -20.0

    # Replay buffer parameters
    # replay_scaling: int = 64
    replay_capacity: int = 1_000_000  # paper is originally 5e6
    replay_context: int = 1
    min_buffer_size_before_training: int | None = None

    # World model loss
    # world_model_grad_norm_clip: float = 1000.0
    # Reconstruction heads use per-element MSE before these weights are applied.
    # Equal values therefore give vision, proprioception, and internal state equal
    # head-level pressure rather than letting image pixel count dominate.
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
    # world_model_lr: float = 1e-4
    ent_coef: float = 0.005  # 3e-4
    return_norm_rate: float = 0.01
    return_norm_limit: float = 1.0
    return_norm_percentile_low: float = 5.0
    return_norm_percentile_high: float = 95.0

    # Actor and critic
    # The Beta actor is defined on [0, 1] then affinely mapped to the Ant's
    # native [-1, 1] action coordinates. This lower bound keeps its mode
    # defined and prevents boundary-singular concentrations.
    actor_min_concentration: float = 1.0
    actor_outscale: float = 0.01
    # actor_lr: float = 3e-5
    # actor_critic_grad_steps_per_update: int = 1
    # actor_grad_norm_clip: float = 100.0

    # critic_lr: float = 3e-5
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

    # Use Spatially-enhanced Recurrent Unit
    use_sru: bool = False

    # Rectified HarmonyDream loss mode. ``all`` independently moderates
    # vision, proprioception, internal state, reward, and the combined KL;
    # ``harmony`` follows the paper's observation/reward/KL grouping.
    # ``none`` preserves the fixed-weight Dreamer objective and is the default
    # so old checkpoints without harmonizer state remain loadable.
    harmony_dream_loss: Literal["none", "harmony", "all"] = "none"

    def __post_init__(self):
        if self.harmony_dream_loss not in ("none", "harmony", "all"):
            raise ValueError(
                "harmony_dream_loss must be one of 'none', 'harmony', or 'all'"
            )
        if self.replay_context < 0:
            raise ValueError("replay_context must be non-negative")
        if self.train_metrics_interval <= 0:
            raise ValueError("train_metrics_interval must be positive")
        if self.comparison_metrics_interval <= 0:
            raise ValueError("comparison_metrics_interval must be positive")
        if self.rollout_metrics_interval <= 0:
            raise ValueError("rollout_metrics_interval must be positive")
        if self.per_joint_metrics_interval <= 0:
            raise ValueError("per_joint_metrics_interval must be positive")
        if self.deterministic_probe_interval <= 0:
            raise ValueError("deterministic_probe_interval must be positive")
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

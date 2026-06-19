from dataclasses import dataclass, field
from configs.config_env import EnvConfig


@dataclass(frozen=True, kw_only=True)
class DreamerConfig(EnvConfig):
    # Model parameters
    hidden_dim: int = 256

    # RSSM parameters
    recurrent_units: int = 1024  # 8 * hidden_dim

    # Encoder details
    base_cnn_channels: int = 16  # hidden_dim // 16

    # MLP details
    mlp_n_layers: int = 3

    # Stochastic units
    stochastic_units: int = 16  ## hidden_dim // 16

    # Decoder parameters
    decoder_hidden_dim: int = 300
    vector_decoder_hidden_dim: int = 200

    # World model loss
    vision_reconstruction_weight: float = 1.0
    proprioception_reconstruction_weight: float = 1.0
    internal_state_reconstruction_weight: float = 1.0
    heat_sensor_reconstruction_weight: float = 1.0
    kl_weight: float = 0.1
    reward_weight: float = 0.1
    terminal_weight: float = 10.0

    # Imagination parameters
    imagine_horizon: int = 15
    imagine_batch_size: int = 1024

    # Actor-Critic parameters
    actor_hidden_dims: tuple = field(default_factory=lambda: (200, 100))
    critic_hidden_dims: tuple = field(default_factory=lambda: (200, 100))

    # Learning rates
    world_model_lr: float = 1e-4
    actor_lr: float = 1e-5
    critic_lr: float = 1e-4
    adam_eps: float = 1e-5
    max_grad_norm: float = 0.5

    # Replay buffer
    buffer_size: int = 2_000_000
    min_buffer_size_before_training: int = 10_000
    replay_buffer_batch_size: int = 64
    replay_buffer_seq_length: int = 50

    # Training
    rollout_steps: int = 10  # Short real world rollouts
    total_updates: int = 1_000
    world_model_epochs: int = 1
    world_model_grad_steps_per_update: int = 100

    # Actor-Critic training
    actor_critic_epochs: int = 4
    actor_critic_minibatch_size: int = 256
    ent_coef: float = 0.001
    vf_coef: float = 0.5

    # Discount and GAE
    gamma: float = 0.99
    gae_lambda: float = 0.95

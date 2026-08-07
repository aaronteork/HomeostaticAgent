from dataclasses import dataclass, field
from configs.config_env import EnvConfig


@dataclass(frozen=True, kw_only=True)
class PPOConfig(EnvConfig):
    total_updates: int = 500
    rollout_steps: int = 2_000
    minibatch_size: int = 10_000
    batch_size: int = field(init=False)
    frame_stack_size: int = 3
    frame_stack_key: str = "vision"
    gamma: float = 0.997
    gae_lambda: float = 0.95
    epochs: int = 30
    lr_start: float = 1e-4
    lr_end: float = 1e-5
    max_grad_norm: float = 0.5
    adam_eps: float = 1e-5
    clip_coef: float = 0.3
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    target_kl: float = 0.02
    image_latent_dim: int = 200
    checkpoint_steps: int = 5_000_000

    def __post_init__(self):
        if self.comparison_metrics_interval <= 0:
            raise ValueError("comparison_metrics_interval must be positive")
        temp_batch_size = self.rollout_steps * self.num_workers
        object.__setattr__(self, 'batch_size', temp_batch_size)

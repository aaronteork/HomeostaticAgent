import numpy as np
import torch
import torch.nn as nn
from torch.distributions.beta import Beta
from torchrl.data import ListStorage, TensorDictReplayBuffer
from tensordict import TensorDict
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

from configs.config_ppo import PPOConfig
from utils.vision import VisionEncoder


def create_replay_buffer(config):
    replay_buffer = TensorDictReplayBuffer(
        storage=ListStorage(device="cpu"),  # Store on CPU to avoid GPU memory accumulation
        sampler=SamplerWithoutReplacement(shuffle=False),
        # batch_size=config.minibatch_size,
    )
    return replay_buffer


def add_to_replay_buffer(
    replay_buffer,
    obs,
    action,
    log_prob,
    reward,
    terminated,
    truncated,
    valid,
    value,
):
    """
    Adds a flattened PPO batch to the replay buffer.
    Expects input shapes to be already flattened (batch_size, ...).
    """
    batch_size = action.shape[0]
    
    # Pack what PPO needs for the update epochs
    data = TensorDict(
        {
            "obs": obs,
            "actions": action,
            "log_probs": log_prob,
            "rewards": reward,
            "terminated": terminated,
            "truncated": truncated,
            "valid": valid,
            "values": value,
        },
        batch_size=[batch_size],
    )

    replay_buffer.extend(data)


def compute_gae_from_buffer(
    rewards,
    values,
    next_obs,
    terminations,
    truncations,
    agent,
    gamma=0.99,
    lam=0.95,
):
    """
    Computes GAE on structured (rollout_steps, num_workers) trajectories.

    Args:
        rewards: Tensor (T, N)
        values: Tensor (T, N) or (T, N, 1)
        next_obs: Dictionary containing vision, proprioception, internal_state of shape (N, ...)
        terminations: Tensor (T, N), true absorbing terminals
        truncations: Tensor (T, N), nonterminal episode boundaries such as
            time limits
        agent: The model used to get the final 'next_value'
        gamma: Discount factor
        lam: GAE lambda
    """
    # 1. Get the value of the final state reached in the rollout
    with torch.no_grad():
        _, _, _, next_value = agent(next_obs["vision"],
                                    next_obs["proprioception"],
                                    next_obs["internal_state"])
        next_value = next_value.detach().squeeze(-1) # (N,)

    # Squeeze values if it has extra dimension
    values = values.squeeze(-1) if len(values.shape) > 2 else values

    rollout_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)

    # Initialize lastgaelam as tensor of shape (num_workers,) to match batch dimension
    num_workers = rewards.shape[1] if len(rewards.shape) > 1 else 1
    lastgaelam = torch.zeros(num_workers, device=rewards.device, dtype=rewards.dtype)

    terminations = terminations.float()
    truncations = truncations.float()

    # print(f"[DEBUG GAE] rewards shape: {rewards.shape}, values shape: {values.shape}, dones shape: {dones.shape}")

    # 2. Iterate backwards
    for t in reversed(range(rollout_steps)):
        if t == rollout_steps - 1:
            nextvalues = next_value
        else:
            nextvalues = values[t + 1]

        # A true terminal suppresses value bootstrapping. A time-limit
        # truncation still bootstraps from its final observation, but both
        # kinds of episode boundary stop the GAE trace from entering the next
        # episode.
        bootstrap_mask = 1.0 - terminations[t]
        trace_mask = 1.0 - torch.maximum(terminations[t], truncations[t])
        # if t == rollout_steps - 1:
        #     print(f"[DEBUG GAE] t={t}: nextvalues shape={nextvalues.shape}, nextnonterminal shape={nextnonterminal.shape}, values[t] shape={values[t].shape}")
            
        delta = rewards[t] + gamma * nextvalues * bootstrap_mask - values[t]
        advantages[t] = lastgaelam = (
            delta + gamma * lam * trace_mask * lastgaelam
        )
        
    returns = advantages + values
    return advantages, returns


def reshape_trajectory(traj_dict, rollout_steps, num_workers):
    reshaped = {}
    for key, value in traj_dict.items():
        if isinstance(value, dict):  # obs is a dict
            reshaped[key] = {k: v.reshape(rollout_steps, num_workers, *v.shape[1:]) 
                           for k, v in value.items()}
        else:
            reshaped[key] = value.reshape(rollout_steps, num_workers, *value.shape[1:])
    return reshaped


def compute_explained_variance(predicted_values, returns):
    """Compute explained variance: 1 - (var(returns - pred) / var(returns))"""
    var_returns = torch.var(returns)
    residuals = returns - predicted_values
    var_residuals = torch.var(residuals)
    if var_returns.item() < 1e-10:
        return 0.0
    return (1 - var_residuals / var_returns).item()


class HomeostaticPPO(nn.Module):
    def __init__(self, cfg: PPOConfig):
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = VisionEncoder()
        
        # Calculate the output dimension of the vision encoder to set the input size of the projection layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, 12, 64, 64)  # Assuming input images are (C=12, H=64, W=64)
            vision_output = self.vision_encoder(dummy_input)
            vision_output_dim = vision_output.shape[-1]

        self.vision_proj = nn.Sequential(
            nn.Linear(vision_output_dim, cfg.image_latent_dim),
            nn.LayerNorm(cfg.image_latent_dim),
            nn.Tanh()
        )

        self.actor = PPOActorNetwork(cfg)
        self.critic = PPOCriticNetwork(cfg)

        self.initialize_weights()

    def forward(self, vision, proprioception, internal_state, deterministic=False, evaluate_actions=None):
        # Convert numpy to torch if needed
        if isinstance(vision, np.ndarray):
            vision = torch.from_numpy(vision).to(self.cfg.device)
        if isinstance(proprioception, np.ndarray):
            proprioception = torch.from_numpy(proprioception).to(self.cfg.device)
        if isinstance(internal_state, np.ndarray):
            internal_state = torch.from_numpy(internal_state).to(self.cfg.device)

        vision = self.vision_encoder(vision)
        vision = self.vision_proj(vision)

        if evaluate_actions is not None:
            # Training mode: evaluate actions and return (log_prob, entropy)
            log_prob, entropy = self.actor(vision, proprioception, internal_state, deterministic=deterministic, evaluate_actions=evaluate_actions)
            value = self.critic(vision, proprioception, internal_state)
            return log_prob, entropy, value
        else:
            # Rollout mode: sample actions and return (action, log_prob, entropy, value)
            action, log_prob, entropy = self.actor(vision, proprioception, internal_state, deterministic=deterministic)
            value = self.critic(vision, proprioception, internal_state)
            return action, log_prob, entropy, value

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class PPOActorNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        action_dim = self.cfg.action_space_dim
        internal_state_dim = 3 if self.cfg.num_heat > 0 else 2
        heat_sensor_dim = 3 if self.cfg.num_heat > 0 else 0
        input_dim = self.cfg.image_latent_dim + self.cfg.obs_space_dim + internal_state_dim + heat_sensor_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 300),
            nn.Tanh(),
            nn.Linear(300, 200),
            nn.Tanh(),
        )
        self.alpha = nn.Sequential(
            nn.Linear(200, action_dim),
            nn.Softplus()
        )
        self.beta = nn.Sequential(
            nn.Linear(200, action_dim),
            nn.Softplus()
        )

    def forward(self, vision, proprioception, internal_state, heat_sensor=None, deterministic=False, evaluate_actions=None):
        if self.cfg is not None and self.cfg.num_heat > 0:
            assert heat_sensor is not None
            x = torch.cat([vision.detach(), proprioception, internal_state, heat_sensor], dim=-1)
        else:
            x = torch.cat([vision.detach(), proprioception, internal_state], dim=-1)
        x = self.net(x)
        alpha = self.alpha(x) + 1
        beta = self.beta(x) + 1
        dist = Beta(alpha, beta)

        if evaluate_actions is not None:
            # Training mode: evaluate log_prob of given actions
            log_prob = dist.log_prob(evaluate_actions).sum(-1)
            entropy = dist.entropy().sum(-1)
            return log_prob, entropy
        else:
            # Rollout mode: sample actions
            if not deterministic:
                action = dist.rsample()
            else:
                action = dist.mode
            return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1)


class PPOCriticNetwork(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        if cfg is None:
            input_dim = cfg.image_latent_dim + cfg.obs_space_dim + 2
        else:
            internal_state_dim = 3 if cfg.num_heat > 0 else 2
            heat_sensor_dim = 3 if cfg.num_heat > 0 else 0
            input_dim = cfg.image_latent_dim + cfg.obs_space_dim + internal_state_dim + heat_sensor_dim
            
        self.net = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, vision, proprioception, internal_state, heat_sensor=None):
        if self.cfg is not None and self.cfg.num_heat > 0:
            assert heat_sensor is not None
            x = torch.cat([vision, proprioception, internal_state, heat_sensor], dim=-1)
        else:
            x = torch.cat([vision, proprioception, internal_state], dim=-1)
        return self.net(x)

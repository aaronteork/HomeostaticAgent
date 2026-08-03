from collections import deque
from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType
from gymnasium.spaces import Box, Dict
from gymnasium.vector.utils import create_empty_array
from gymnasium.wrappers import RecordEpisodeStatistics, RescaleAction
from gymnasium.wrappers.utils import create_zero_array

from envs.ant_env import HomeostaticAntEnv


class CustomFrameStackObservation(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(
        self,
        env: gym.Env,
        stack_key: str,
        stack_size: int,
        *,
        padding_type: str = "reset",
    ):
        gym.utils.RecordConstructorArgs.__init__(
            self, stack_key=stack_key, stack_size=stack_size, padding_type=padding_type
        )
        super().__init__(env)

        if not isinstance(env.observation_space, Dict):
            raise TypeError("This wrapper only supports Dict observation spaces.")
        
        self.stack_key = stack_key
        self.stack_size = stack_size
        self.padding_type = padding_type
        
        # Determine the shape of the single frame
        single_frame_shape = env.observation_space[stack_key].shape # e.g., (4, 64, 64)
        
        # New shape: (stack_size * C, H, W) -> e.g., (12, 64, 64)
        new_shape = (single_frame_shape[0] * stack_size, *single_frame_shape[1:])
        
        # Modify the observation space
        new_spaces = dict(env.observation_space.spaces)
        new_spaces[stack_key] = Box(
            low=env.observation_space[stack_key].low.min(),
            high=env.observation_space[stack_key].high.max(),
            shape=new_shape,
            dtype=env.observation_space[stack_key].dtype
        )
        self.observation_space = Dict(new_spaces)

        # Setup buffer
        self.padding_value = create_zero_array(env.observation_space[stack_key])
        self.obs_queue = deque([self.padding_value for _ in range(stack_size)], maxlen=stack_size)
        self.stacked_obs = create_empty_array(env.observation_space[stack_key], n=stack_size)

    def _process_obs(self, obs: dict) -> dict:
        obs = deepcopy(obs)
        self.obs_queue.append(obs[self.stack_key])
        # obs[self.stack_key] = concatenate(
        #     self.env.observation_space[self.stack_key], self.obs_queue, self.stacked_obs
        # )
        obs[self.stack_key] = np.concatenate(list(self.obs_queue), axis=0)
        return obs

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Reset buffer
        if self.padding_type == "reset":
            self.padding_value = obs[self.stack_key]
        self.obs_queue = deque([self.padding_value for _ in range(self.stack_size - 1)], maxlen=self.stack_size)
        return self._process_obs(obs), info

    def step(self, action: WrapperActType):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._process_obs(obs), reward, terminated, truncated, info


def create_env(config, multiple_env=False, rescale_action=True):
    """
    Creates either a single environment or an AsyncVectorEnv.
    """
    def _base_env_setup():
        env = HomeostaticAntEnv(config)
        env = RecordEpisodeStatistics(env)
        if rescale_action:
            env = RescaleAction(env, 0, 1)
        if hasattr(config, "frame_stack_key") and config.frame_stack_key:
            env = CustomFrameStackObservation(
                env, 
                stack_size=config.frame_stack_size, 
                stack_key=config.frame_stack_key
            )
        return env

    if multiple_env:
        # Define the factory function for the vector environment
        def make_env():
            env = _base_env_setup()
            return env
        env = gym.vector.AsyncVectorEnv([make_env for _ in range(config.num_workers)])
        return env

    # Return the single environment (without NumpyToTorch as per your original code)
    return _base_env_setup()

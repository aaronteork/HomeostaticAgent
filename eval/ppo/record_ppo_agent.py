# Create environment
import cv2
import pandas as pd
import torch
import numpy as np

from configs.config_ppo import PPOConfig
from utils.utils_env import create_env
from utils.utils_ppo import HomeostaticPPO


def record_ppo():
    # Create environment
    config = PPOConfig(is_training=False, image_size=(512, 512))
    env = create_env(config, multiple_env=False)
    obs, info = env.reset()

    # Set up model
    print(f"Using device: {config.device}")
    agent = HomeostaticPPO(config).to(config.device)
    checkpoint = torch.load("models/ppo_agent_2026-07-03_06-11-09.pt", map_location=config.device)
    agent.load_state_dict(checkpoint)
    agent.eval()

    # Set up video recorder
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_pov = cv2.VideoWriter("./eval/ppo/ppo_deterministic_pov_video.mp4", fourcc, 30, (512, 512))
    out_env = cv2.VideoWriter("./eval/ppo/ppo_deterministic_env_video.mp4", fourcc, 30, (512, 512))

    episode_reward = []
    episode_food = []
    episode_water = []
    episode_steps = 0

    episode_hunger = []
    episode_thirst = []

    done = False
    while not done and episode_steps < config.max_steps:
        # Prepare observation
        vision = np.transpose(obs["vision"], (1, 2, 0))
        vision = cv2.resize(vision, (64, 64), interpolation=cv2.INTER_AREA)
        vision = np.transpose(vision, (2, 0, 1))
        vision = torch.from_numpy(vision).unsqueeze(0).to(config.device)
        proprioception = (
            torch.from_numpy(obs["proprioception"])
            .unsqueeze(0)
            .to(config.device, dtype=torch.float32)
        )
        internal_state = (
            torch.from_numpy(obs["internal_state"]).unsqueeze(0).to(config.device)
        )

        # Get action from agent
        with torch.no_grad():
            action, _, _, _ = agent(vision, proprioception, internal_state, deterministic=True)
            action = action.cpu().numpy().squeeze(0)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        episode_reward.append(reward)
        episode_food.append(int(info["food_consumed"]))
        episode_water.append(int(info["water_consumed"]))
        episode_hunger.append(info["hunger"])
        episode_thirst.append(info["thirst"])
        episode_steps += 1

        # Capture pov and env frames from info
        out_pov.write(cv2.cvtColor(info["vision"], cv2.COLOR_RGB2BGR))
        out_env.write(cv2.cvtColor(info["environment"], cv2.COLOR_RGB2BGR))

        done = terminated or truncated
        if episode_steps % 100 == 0:
            print(f"Step: {episode_steps}", end="\r", flush=True)
    print("\n")
    print("Episode finished")
    env.close()

    # Save episode statistics
    episode_stats = pd.DataFrame(
        {
            "reward": episode_reward,
            "food_consumed": episode_food,
            "water_consumed": episode_water,
            "hunger": episode_hunger,
            "thirst": episode_thirst,
        }
    )
    episode_stats.to_csv("./eval/ppo/ppo_deterministic_episode_stats.csv", index=False)

    out_pov.release()
    out_env.release()


if __name__ == "__main__":
    record_ppo()

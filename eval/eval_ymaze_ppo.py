import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from gymnasium.wrappers import RecordEpisodeStatistics, RescaleAction

from configs.config_ppo import PPOConfig
from configs.config_ymaze import YMazeConfig
from envs.ymaze_test_env import YMazeTestEnv
from utils.utils_ppo import HomeostaticPPO
from utils.utils_env import CustomFrameStackObservation


def eval_ymaze_ppo():
    # Create configs
    ymaze_cfg = YMazeConfig(
        render_mode="rgbd_tuple", image_size=(512, 512), is_training=False
    )
    ppo_cfg = PPOConfig(is_training=False)

    # Create environment
    env = YMazeTestEnv(ymaze_cfg)
    env = RecordEpisodeStatistics(env)
    env = RescaleAction(env, 0.0, 1.0)
    env = CustomFrameStackObservation(env, stack_size=ppo_cfg.frame_stack_size, stack_key=ppo_cfg.frame_stack_key)

    # Create agent and load model
    agent = HomeostaticPPO(ppo_cfg)
    model = torch.load("./models/PPO_4000_rollout.pt")
    agent.load_state_dict(model)
    agent.to(ppo_cfg.device)
    agent.eval()

    # Set up video recording
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_pov = cv2.VideoWriter(
        "./eval/ppo/ppo_ymaze_pov_video.mp4", fourcc, 30, (512, 512)
    )
    out_env = cv2.VideoWriter(
        "./eval/ppo/ppo_ymaze_env_video.mp4", fourcc, 30, (512, 512)
    )

    # To store evaluation
    eval_results = {
        "episode": [],
        "final_hunger": [],
        "final_thirst": [],
        "food_consumed": [],
        "water_consumed": [],
        "termination_reason": [],
        "resources_consumed": [],
        "initial_hunger": [],
        "initial_thirst": [],
    }

    for episode in tqdm(range(ymaze_cfg.episodes_to_run), desc="Evaluating PPO agent"):
        obs, info = env.reset()
        done = False
        while not done:
            # Get model inputs
            vision = np.transpose(obs["vision"], (1, 2, 0))
            vision = cv2.resize(vision, (64, 64), interpolation=cv2.INTER_AREA)
            vision = np.transpose(vision, (2, 0, 1))
            vision = torch.from_numpy(vision).unsqueeze(0).to(ppo_cfg.device)
            proprioception = (
                torch.from_numpy(obs["proprioception"])
                .unsqueeze(0)
                .to(ppo_cfg.device, dtype=torch.float32)
            )
            internal_state = (
                torch.from_numpy(obs["internal_state"]).unsqueeze(0).to(ppo_cfg.device)
            )

            # Get action from agent
            with torch.no_grad():
                action, _, _, _ = agent(
                    vision, proprioception, internal_state, deterministic=False
                )
                action = action.cpu().numpy().squeeze(0)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Capture pov and env frames from info
            out_pov.write(cv2.cvtColor(info["vision"], cv2.COLOR_RGB2BGR))
            out_env.write(cv2.cvtColor(info["environment"], cv2.COLOR_RGB2BGR))

        # If episode is finished, log info
        eval_results["episode"].append(episode)
        eval_results["final_hunger"].append(info["hunger"])
        eval_results["final_thirst"].append(info["thirst"])
        eval_results["food_consumed"].append(info["food_consumed"])
        eval_results["water_consumed"].append(info["water_consumed"])
        eval_results["termination_reason"].append(info["termination_reason"])
        eval_results["resources_consumed"].append(info["resources_consumed"])
        eval_results["initial_hunger"].append(info["initial_hunger"])
        eval_results["initial_thirst"].append(info["initial_thirst"])

    env.close()
    out_pov.release()
    out_env.release()

    # Save episode information
    df = pd.DataFrame(eval_results)
    df.to_csv("./eval/ppo/ppo_ymaze_episode_stats.csv", index=False)

if __name__ == "__main__":
    eval_ymaze_ppo()

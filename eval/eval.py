import argparse
import datetime as dt
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import pandas as pd
import torch
from gymnasium.wrappers import RecordEpisodeStatistics, RescaleAction
from tqdm.auto import tqdm

# Dreamer
from configs.config_dreamer import DreamerConfig

# PPO
from configs.config_ppo import PPOConfig
from utils.utils_ppo import HomeostaticPPO

# Ymaze
from configs.config_ymaze import YMazeConfig
from envs.ymaze_test_env import YMazeTestEnv

# Original Training Environment
from utils.utils_env import CustomFrameStackObservation, create_env


DATETIME = dt.datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y-%m-%d_%H-%M-%S")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluating Agent")
    parser.add_argument(
        "--task",
        type=str,
        choices=["forage", "ymaze", "shift", "ymaze-shift"],
        help="Task to evaluate",
    )
    parser.add_argument("--model_path", type=str, help="Path to the trained model")
    parser.add_argument(
        "--model", type=str, choices=["ppo", "dreamer"], help="Algorithm to evaluate"
    )
    return parser.parse_args()


def setup_video_recording(model_name: str, task_name: str):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_pov = cv2.VideoWriter(
        f"./eval/{model_name}/{model_name}_{DATETIME}_{task_name}_pov_video.mp4",
        fourcc,
        30,
        (512, 512),
    )
    out_env = cv2.VideoWriter(
        f"./eval/{model_name}/{model_name}_{DATETIME}_{task_name}_env_video.mp4",
        fourcc,
        30,
        (512, 512),
    )
    return out_pov, out_env


def prep_obs(obs):
    # Prepare observation
    vision = np.transpose(obs["vision"], (1, 2, 0))
    vision = cv2.resize(vision, (64, 64), interpolation=cv2.INTER_AREA)
    vision = np.transpose(vision, (2, 0, 1))
    vision = torch.from_numpy(vision).unsqueeze(0)
    proprioception = torch.from_numpy(obs["proprioception"]).unsqueeze(0)
    internal_state = torch.from_numpy(obs["internal_state"]).unsqueeze(0)
    obs_dict = {
        "vision": vision,
        "proprioception": proprioception,
        "internal_state": internal_state,
    }
    return obs_dict


def ppo_get_action(model, obs, deterministic=True, device="cuda"):
    with torch.inference_mode():
        action, _, _, _ = model(
            obs["vision"].to(device),
            obs["proprioception"].to(device, dtype=torch.float32),
            obs["internal_state"].to(device),
            deterministic=deterministic,
        )
    return action


def create_ymaze(config):
    # Create environment
    env = YMazeTestEnv(config)
    env = RecordEpisodeStatistics(env)
    env = RescaleAction(env, 0.0, 1.0)
    return env


def task_forage(model, config, out_pov, out_env, model_name):

    # Create environment
    env = create_env(config, multiple_env=False)
    obs, info = env.reset()

    # Setup statistics collection
    episode_reward = []
    episode_food = []
    episode_water = []
    episode_steps = 0

    episode_hunger = []
    episode_thirst = []
    done = False
    while not done and episode_steps < config.max_steps:
        obs = prep_obs(obs)
        # Get action from agent
        if model_name == "ppo":
            action = ppo_get_action(model, obs, deterministic=True, device=config.device)
        action = action.cpu().numpy().squeeze(0)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Record stats and write frames
        episode_reward.append(reward)
        episode_food.append(int(info["food_consumed"]))
        episode_water.append(int(info["water_consumed"]))
        episode_hunger.append(info["hunger"])
        episode_thirst.append(info["thirst"])
        episode_steps += 1

        # Capture pov and env frames from info
        out_pov.write(cv2.cvtColor(info["vision"], cv2.COLOR_RGB2BGR))
        out_env.write(cv2.cvtColor(info["environment"], cv2.COLOR_RGB2BGR))
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
    episode_stats.to_csv(
        f"./eval/{model_name}/{model_name}_episode_stats.csv", index=False
    )



def task_shift(model, config, out_pov, out_env, model_name):

    # Create environment
    env = create_env(config, multiple_env=False)
    
    for episode in tqdm(range(10), desc="Evaluating agent"):
        obs, info = env.reset()
        # Setup statistics collection
        episode_steps = 0
        done = False
        while not done and episode_steps < config.max_steps:
            obs = prep_obs(obs)
            # Get action from agent
            if model_name == "ppo":
                action = ppo_get_action(model, obs, deterministic=True, device=config.device)
            action = action.cpu().numpy().squeeze(0)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Record stats and write frames
            episode_steps += 1

            # Capture pov and env frames from info
            out_pov.write(cv2.cvtColor(info["vision"], cv2.COLOR_RGB2BGR))
            out_env.write(cv2.cvtColor(info["environment"], cv2.COLOR_RGB2BGR))
            if episode_steps % 100 == 0:
                print(f"Step: {episode_steps}", end="\r", flush=True)
    env.close()


def task_ymaze(model, config, out_pov, out_env, model_name):

    ymaze_cfg = YMazeConfig(render_mode="rgbd_tuple", image_size=(512, 512), is_training=False)
    env = create_ymaze(ymaze_cfg)
    if model_name == "ppo":
        env = CustomFrameStackObservation(
            env,
            stack_size=config.frame_stack_size,
            stack_key=config.frame_stack_key,
        )

    # Setup statistics collection
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

    for episode in tqdm(range(ymaze_cfg.episodes_to_run), desc="Evaluating agent"):
        obs, info = env.reset()
        done = False
        while not done:
            # Get model inputs
            obs = prep_obs(obs)

            # Get action from agent
            if model_name == "ppo":
                action = ppo_get_action(model, obs, deterministic=True, device=config.device)
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

    # Save episode information
    df = pd.DataFrame(eval_results)
    df.to_csv(f"./eval/{model_name}/{model_name}_ymaze_episode_stats.csv", index=False)


def evaluate_agent(args):

    # Get model
    if args.model == "ppo":
        config = PPOConfig(is_training=False, image_size=(512, 512))
        model = HomeostaticPPO(config).to(config.device)
    # checkpoint = torch.load(args.model_path, map_location=config.device)
    # cleaned_state_dict = {
    #     key.replace("_orig_mod.", ""): value for key, value in checkpoint.items()
    # }
    # model.load_state_dict(cleaned_state_dict)
    model.eval()

    # Set up video recording
    out_pov, out_env = setup_video_recording(args.model, args.task)

    # Evaluate based on task
    if args.task == "forage":
        task_forage(model, config, out_pov, out_env, args.model)
    elif args.task == "ymaze":
        task_ymaze(model, config, out_pov, out_env, args.model)
    elif args.task == "shift":
        config = PPOConfig(is_training=False, image_size=(512, 512), shift=True)
        task_shift(model, config, out_pov, out_env, args.model)

    # Close video writers
    out_pov.release()
    out_env.release()


if __name__ == "__main__":
    args = parse_args()
    evaluate_agent(args)

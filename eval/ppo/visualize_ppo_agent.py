import cv2
import numpy as np
import torch

from configs.config_ppo import PPOConfig
from utils.utils_env import create_env
from utils.utils_ppo import HomeostaticPPO


def test_agent_training():
    """Test agent with visualization and resource consumption tracking."""

    # Setup configs
    ppo_cfg = PPOConfig(render_mode="human", image_size=(64, 64), is_training=False)

    # Create single environment for visualization
    env = create_env(ppo_cfg, multiple_env=False)
    # Create agent and replay buffer
    agent = HomeostaticPPO(ppo_cfg)
    model = torch.load("./models/PPO_final.pt")
    agent.load_state_dict(model)
    agent.to(ppo_cfg.device)
    agent.eval()

    obs, info = env.reset()

    print("\nAgent Training Test - Environment Visualization")
    print("=" * 50)
    print("Controls:")
    print("- Press 'q' to quit the visualization.")
    print("- The left window shows the Ant's First-Person Perspective (POV).")
    print("- The right window shows the Global Environment View.")
    print("=" * 50)

    episode_count = 0
    step_count = 0
    total_steps = 100_000

    try:
        while step_count < total_steps:
            # Agent forward pass - convert to tensors on device with batch dim
            with torch.no_grad():
                vision = torch.from_numpy(obs["vision"]).float().unsqueeze(0).to(ppo_cfg.device)
                proprio = torch.from_numpy(obs["proprioception"]).float().unsqueeze(0).to(ppo_cfg.device)
                internal = torch.from_numpy(obs["internal_state"]).float().unsqueeze(0).to(ppo_cfg.device)

                actions, log_prob, entropy, value = agent(vision, proprio, internal, deterministic=True)

            actions = actions.detach().cpu().numpy().squeeze(0)
            log_prob = log_prob.detach().cpu().numpy()
            entropy = entropy.detach().cpu().numpy()
            value = value.detach().cpu().numpy()

            # Step environment
            # print(actions)
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            done = terminations or truncations

            # Extract visualization
            pov_image = infos["vision"]
            env_image = infos["environment"]

            # Convert RGB to BGR for OpenCV display
            pov_bgr = cv2.cvtColor(pov_image, cv2.COLOR_RGB2BGR)
            env_bgr = cv2.cvtColor(env_image, cv2.COLOR_RGB2BGR)

            # Display windows side by side
            cv2.imshow("POV Perspective", pov_bgr)
            cv2.imshow("Global Environment", env_bgr)

            # Log resource consumption every 100 steps
            if step_count % 100 == 0:
                print(f"Step {step_count}/{total_steps} | Episode {episode_count}")
                print(f"  Food Consumed: {infos['food_consumed']}")
                print(f"  Water Consumed: {infos['water_consumed']}")
                print(f"  Hunger: {infos['hunger']:.3f}, Thirst: {infos['thirst']:.3f}")
                print(f"  Reward: {rewards:.3f}")

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nQuitting...")
                break

            # Episode reset
            if done:
                print(f"\nEpisode {episode_count} Ended:")
                print(f"  Total Food Consumed: {infos['food_consumed']}")
                print(f"  Total Water Consumed: {infos['water_consumed']}")
                print(f"  Final Hunger: {infos['hunger']:.3f}")
                print(f"  Final Thirst: {infos['thirst']:.3f}")
                print(f"  Termination Reason: {infos['termination_reason']}")

                obs, info = env.reset()
                episode_count += 1
            else:
                obs = next_obs

            step_count += 1

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        env.close()
        cv2.destroyAllWindows()
        print(f"\nTest Complete!")
        print(f"Total Steps: {step_count}")
        print(f"Total Episodes: {episode_count}")
        print("Environment closed.")


if __name__ == "__main__":
    test_agent_training()

from dataclasses import dataclass

from configs.config_env import EnvConfig


@dataclass(frozen=True, kw_only=True)
class YMazeConfig(EnvConfig):
    """Configuration for the Y-Maze test environment.

    This environment tests homeostatic RL agents on:
    1. Selective foraging: choosing the correct resource based on internal state
    2. Sequential planning: navigating to the second resource after consuming the first
    """

    # XML and environment
    env_name: str = "YMazeTestEnv"
    max_steps: int = 10_000 # More than decay for a whole episode
    render_mode: str = "rgb_array"

    # Y-maze geometry
    stem_length: float = 5.0          # Length of the stem corridor behind the junction
    arm_length: float = 6.0           # Length of each arm from the junction
    corridor_width: float = 4.0       # Width of the corridors
    fork_half_angle_deg: float = 40.0 # Half-angle of the Y fork (degrees from forward)
    wall_thickness: float = 0.2       # Thickness of the corridor walls
    resource_offset: float = 1.0      # Distance from arm end cap to resource center

    # Agent spawn position (fixed)
    spawn_y_offset: float = 1.5  # Agent spawns at y = -(stem_length - spawn_y_offset)

    # Internal state initialization
    # Primary need is more depleted (more negative), secondary is less depleted
    primary_need_low: float = -0.15  # -0.45
    primary_need_high: float = -0.11  # -0.25
    secondary_need_low: float = -0.1 # -0.20
    secondary_need_high: float = -0.05 # -0.05

    # Test settings
    randomize_arms: bool = True  # Whether to randomize which arm has food vs water each episode
    is_training: bool = False    # Override default; test environment renders debug images
    episodes_to_run: int = 10             # Number of episodes to run for evaluation

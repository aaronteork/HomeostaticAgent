from dataclasses import dataclass

from configs.config_base import BaseConfig


@dataclass(frozen=True, kw_only=True)
class EnvConfig(BaseConfig):
    xml_path: str = "ant_env.xml"
    env_name: str = "HomeostaticAntEnv"
    max_steps: float = float("inf")
    image_size: tuple[int, int] = (64, 64)
    num_workers: int = 16
    # A coarse survival heartbeat is sufficient for cross-trainer comparisons
    # and avoids frequent synchronous SQLite writes.
    comparison_metrics_interval: int = 50_000
    arena_size: float = 6.0
    shift: bool = False
    training_initial_bounds: float = 0.5
    day_night_cycle_len: int = 1
    object_spacing: float = 2.0
    object_interaction_dist: float = 1.0
    heat_sensor_range: float = 1.0
    num_food: int = 6
    num_water: int = 4
    num_heat: int = 0
    hunger_decay: float = 0.00015
    thirst_decay: float = 0.00015
    replenish_rate: float = 0.1
    action_heat_gain_rate: float = 0.0 if num_heat == 0 else 0.01  # TODO: Change placeholder values
    heat_source_gain_rate: float = 0.0 if num_heat == 0 else 0.01  # TODO: Change placeholder values
    night_cooling_rate: float = 0.0 if num_heat == 0 else 0.01     # TODO: Change placeholder values
    sweat_cooling_rate: float = 0.0 if num_heat == 0 else 0.01     # TODO: Change placeholder values
    sweat_thirst_cost: float = 0.0 if num_heat == 0 else 0.01   # TODO: Change placeholder values
    posture_drive_penalty: float = 0.0
    movement_penalty_weight: float = 0.001 # 
    posture_penalty_weight: float = 0.005 # 
    reward_scale: float = 100.0
    render_mode: str = "rgb_array"
    obs_space_dim: int = 27
    action_space_dim: int = 8 if num_heat == 0 else 9

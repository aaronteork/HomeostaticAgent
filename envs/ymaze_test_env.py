import math
from pathlib import Path

import mujoco
import numpy as np
import tempfile
import xml.etree.ElementTree as ET
from gymnasium import spaces
from gymnasium.envs.mujoco.ant_v5 import AntEnv
from gymnasium.utils import EzPickle

from configs.config_ymaze import YMazeConfig

DEFAULT_CAMERA_CONFIG = {}


def qtoeuler(q):
    """quaternion to Euler angle

    :param q: quaternion (w, x, y, z)
    :return: array of [roll, pitch, yaw]
    """
    phi = math.atan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
    theta = math.asin(2 * (q[0] * q[2] - q[3] * q[1]))
    psi = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    return np.array([phi, theta, psi])


class YMazeTestEnv(AntEnv, EzPickle):
    """Y-Maze test environment for evaluating homeostatic RL agents.

    This environment implements a classic Y-maze paradigm to test:
    1. Selective foraging: does the agent choose the resource matching its most depleted need?
    2. Sequential planning/memory: after consuming the first resource, does the agent
       navigate to the second resource based on prior visual memory?

    Maze layout:
        - A short stem corridor leads to a junction point
        - Two arms branch off at configurable angles (default 40° each side)
        - Each arm contains one resource (food or water) near its endpoint
        - The agent starts at the stem, facing the junction, with both resources visible

    Internal state initialization:
        - One variable (hunger or thirst) is set moderately negative (primary need)
        - The other is set slightly negative (secondary need)
        - Which variable is primary is randomized each episode

    Episode termination:
        - Both resources consumed (success)
        - Homeostatic limits exceeded (|hunger| or |thirst| > 1.0)
        - Agent flipped over
        - Max steps exceeded (truncation safety net)

    Resources do NOT respawn after consumption.
    """

    def __init__(
        self,
        cfg: YMazeConfig,
        **kwargs,
    ):
        """Initialize the Y-Maze test environment.

        Args:
            cfg (YMazeConfig): Configuration object for the Y-maze environment.
        """

        # -------------- Setup config -------------- #
        self.cfg = cfg

        # -------------- Precompute maze geometry -------------- #
        theta = np.radians(cfg.fork_half_angle_deg)
        self.theta = theta
        hw = cfg.corridor_width / 2.0

        # Arm direction unit vectors (from junction)
        self.d_left = np.array([-np.sin(theta), np.cos(theta)])
        self.d_right = np.array([np.sin(theta), np.cos(theta)])

        # Resource positions (slightly before arm end caps)
        self.left_arm_resource_pos = self.d_left * (cfg.arm_length - cfg.resource_offset)
        self.right_arm_resource_pos = self.d_right * (cfg.arm_length - cfg.resource_offset)

        # Divider tip y-coordinate (where inner walls converge)
        self._divider_tip_y = hw / np.sin(theta)

        # Agent spawn position
        self._spawn_x = 0.0
        self._spawn_y = -(cfg.stem_length - cfg.spawn_y_offset)

        # -------------- Build XML with maze walls -------------- #
        xml_file_path = Path(__file__).parent / self.cfg.xml_path
        tree = ET.parse(xml_file_path)
        worldbody = tree.find(".//worldbody")

        # Add Y-maze walls to XML
        self._add_maze_walls(worldbody)

        with tempfile.NamedTemporaryFile(mode='wt', suffix=".xml", delete=False) as f:
            tree.write(f.name)
            temp_xml_path = f.name

        # -------------- Initialize AntEnv -------------- #
        AntEnv.__init__(
            self,
            xml_file=temp_xml_path,
            width=self.cfg.image_size[0],
            height=self.cfg.image_size[1],
            render_mode=self.cfg.render_mode,
            **kwargs,
        )

        # -------------- Initialise parameters -------------- #
        self.hunger = 0.0
        self.thirst = 0.0
        self.posture = 0.0
        self.prev_drive = 0.0
        self.current_step = 0

        self.food_consumed = 0
        self.water_consumed = 0

        # Resource tracking (list of (type, x, y) tuples, same format as HomeostaticAntEnv)
        self.object = []

        # Ant body ID for position/orientation queries
        self.ant_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )

        # Test metrics
        self.resources_consumed = []

        # Store initial state for correctness evaluation
        self._initial_hunger = self.hunger
        self._initial_thirst = self.thirst

        # ------------ Observation and Action space -------------- #
        # Action space: 8 continuous dims (no temperature/sweat action)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        # Observation space: same structure as HomeostaticAntEnv (no temperature)
        self.observation_space = spaces.Dict({
            "proprioception": spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.cfg.obs_space_dim,), dtype=np.float32
            ),
            "vision": spaces.Box(
                low=0.0, high=1.0, shape=(4, self.cfg.image_size[0], self.cfg.image_size[1]), dtype=np.float32
            ),
            "internal_state": spaces.Box(
                low=-1.0, high=1.0, shape=(2,), dtype=np.float32
            ),
        })

        EzPickle.__init__(**locals())

    # ==================== Maze Construction ==================== #

    def _add_maze_walls(self, worldbody):
        """Add Y-maze wall geoms to the XML worldbody.

        Each wall segment (defined by two 2D endpoints) is converted into a
        rotated box geom with proper position, size, and euler rotation.
        """
        wall_segments = self._compute_wall_segments()
        wall_attrs = dict(
            type="box",
            rgba="0.5 0.5 0.5 1",
            conaffinity="1",
            condim="3",
        )

        for name, p1, p2 in wall_segments:
            center = (p1 + p2) / 2.0
            length = np.linalg.norm(p2 - p1)
            angle_rad = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
            angle_deg = np.degrees(angle_rad)

            t = self.cfg.wall_thickness

            ET.SubElement(
                worldbody, "geom",
                dict(wall_attrs,
                     name=name,
                     pos=f"{center[0]:.4f} {center[1]:.4f} 0.5000",
                     size=f"{length / 2:.4f} {t / 2:.4f} 2.0000",
                     euler=f"0 0 {angle_deg:.4f}",
                ),
            )

    def _compute_wall_segments(self):
        """Compute all Y-maze wall segments as (name, p1_2d, p2_2d) tuples.

        Geometry derivation:
            Junction at origin (0, 0). Stem extends backward (-Y). Arms extend
            forward at ±fork_half_angle from +Y axis.

            Outer walls of each arm connect to the stem walls at a computed junction
            point. Inner walls (divider) converge to a tip at (0, hw/sin(θ)).

        Returns:
            list of (name: str, p1: np.ndarray, p2: np.ndarray) tuples
        """
        theta = self.theta
        hw = self.cfg.corridor_width / 2.0
        L = self.cfg.arm_length
        S = self.cfg.stem_length

        sin_t = np.sin(theta)
        cos_t = np.cos(theta)

        # Connection point: where stem wall (x = ±hw) meets the arm outer wall line
        # Derived from: outer wall at parameter t has x = -sin(θ)*t - hw*cos(θ)
        # Setting x = -hw and solving for t gives t_connect = hw*(1-cos(θ))/sin(θ)
        t_connect = hw * (1 - cos_t) / sin_t
        # y at connection point: cos(θ)*t_connect - hw*sin(θ) = -t_connect
        connect_y = -t_connect

        # Divider tip: where inner walls of both arms converge at (0, hw/sin(θ))
        tip_y = hw / sin_t

        # Outer wall endpoints at arm end
        n_left_out = np.array([-cos_t, -sin_t])   # outward perpendicular of left arm
        n_right_out = np.array([cos_t, -sin_t])    # outward perpendicular of right arm
        n_left_in = np.array([cos_t, sin_t])       # inward perpendicular of left arm
        n_right_in = np.array([-cos_t, sin_t])     # inward perpendicular of right arm

        left_outer_end = self.d_left * L + hw * n_left_out
        right_outer_end = self.d_right * L + hw * n_right_out
        left_inner_end = self.d_left * L + hw * n_left_in
        right_inner_end = self.d_right * L + hw * n_right_in

        segments = []

        # 1. Back wall (closes the stem behind the agent)
        segments.append((
            "wall_back",
            np.array([-hw, -S]),
            np.array([hw, -S]),
        ))

        # 2. Left stem wall (vertical, from back wall to junction connection point)
        segments.append((
            "wall_stem_left",
            np.array([-hw, -S]),
            np.array([-hw, connect_y]),
        ))

        # 3. Right stem wall (mirror of left)
        segments.append((
            "wall_stem_right",
            np.array([hw, -S]),
            np.array([hw, connect_y]),
        ))

        # 4. Left arm outer wall (from stem connection to arm end)
        segments.append((
            "wall_left_outer",
            np.array([-hw, connect_y]),
            left_outer_end,
        ))

        # 5. Right arm outer wall (mirror of left)
        segments.append((
            "wall_right_outer",
            np.array([hw, connect_y]),
            right_outer_end,
        ))

        # 6. Left divider wall (from divider tip to left arm inner end)
        segments.append((
            "wall_divider_left",
            np.array([0.0, tip_y]),
            left_inner_end,
        ))

        # 7. Right divider wall (from divider tip to right arm inner end)
        segments.append((
            "wall_divider_right",
            np.array([0.0, tip_y]),
            right_inner_end,
        ))

        # 8. Divider tip cap (small segment to close the gap where inner walls meet)
        cap_hw = self.cfg.wall_thickness  # small cap
        segments.append((
            "wall_divider_tip",
            np.array([-cap_hw, tip_y]),
            np.array([cap_hw, tip_y]),
        ))

        # 9. Left arm end cap (connects outer and inner walls at arm end)
        segments.append((
            "wall_left_cap",
            left_outer_end,
            left_inner_end,
        ))

        # 10. Right arm end cap (connects outer and inner walls at arm end)
        segments.append((
            "wall_right_cap",
            right_outer_end,
            right_inner_end,
        ))

        return segments

    # ==================== Reset ==================== #

    def reset_model(self):
        """Reset the environment for a new test episode.

        - Agent spawns at a fixed position in the stem, facing +Y toward the junction
        - Internal state: one variable is moderately depleted (primary need),
          the other is slightly depleted (secondary need)
        - Resources placed at arm endpoints (optionally randomized)
        """
        self.current_step = 0
        self.food_consumed = 0
        self.water_consumed = 0

        # Reset test metrics
        self.resources_consumed = []

        # ---------- Internal state initialization ---------- #
        # Sample primary (more depleted) and secondary (less depleted) need values
        primary = self.np_random.uniform(self.cfg.primary_need_low, self.cfg.primary_need_high)
        secondary = self.np_random.uniform(self.cfg.secondary_need_low, self.cfg.secondary_need_high)

        # Randomly assign which variable (hunger/thirst) is the primary need
        if self.np_random.random() < 0.5:
            self.hunger = primary
            self.thirst = secondary
        else:
            self.hunger = secondary
            self.thirst = primary

        # Store initial state for determining "correct" first choice
        self._initial_hunger = self.hunger
        self._initial_thirst = self.thirst

        # ---------- Fixed spawn position ---------- #
        qpos = self.init_qpos.copy()
        qpos[0] = self._spawn_x  # x = 0 (center of stem)
        qpos[1] = self._spawn_y  # y = -(stem_length - spawn_y_offset)

        # Orientation: identity quaternion (w=1, x=0, y=0, z=0) = facing +Y (forward along stem)
        qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        # Small noise on joint angles only (not position/orientation) for numerical stability
        qpos[7:] += self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq - 7)

        qvel = self.init_qvel.copy()
        qvel += self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)

        self.set_state(qpos, qvel)
        self.posture = self._get_posture()

        # ---------- Place resources ---------- #
        self.object = []
        if self.cfg.randomize_arms:
            if self.np_random.random() < 0.5:
                self.object.append(("food", self.left_arm_resource_pos[0], self.left_arm_resource_pos[1]))
                self.object.append(("water", self.right_arm_resource_pos[0], self.right_arm_resource_pos[1]))
            else:
                self.object.append(("water", self.left_arm_resource_pos[0], self.left_arm_resource_pos[1]))
                self.object.append(("food", self.right_arm_resource_pos[0], self.right_arm_resource_pos[1]))
        else:
            # Fixed: food always in left arm, water always in right arm
            self.object.append(("food", self.left_arm_resource_pos[0], self.left_arm_resource_pos[1]))
            self.object.append(("water", self.right_arm_resource_pos[0], self.right_arm_resource_pos[1]))

        # Initialize previous drive for reward computation
        self.prev_drive = self._calculate_drive()

        return self._get_obs()

    # ==================== Core Dynamics ==================== #

    def _calculate_drive(self):
        """Homeostatic drive: sum of squared deviations from zero."""
        return (
            self.hunger ** 2
            + self.thirst ** 2
            + self.cfg.posture_drive_penalty * (self.posture ** 2)
        )

    def _get_posture(self):
        """Posture deviation from upright, using Euler angles (roll, pitch)."""
        return np.square(
            qtoeuler(self.data.qpos[3:7])[:2] - qtoeuler([1.0, 0.0, 0.0, 0.0])[:2]
        ).sum()

    def step(self, action):
        """Execute one timestep of the Y-maze test.

        Dynamics:
            - Hunger and thirst decay passively each step
            - Physical action is applied (8-dim joint torques)
            - Resource contact: if agent touches a resource, it is consumed (no respawn)
            - Reward: homeostatic drive reduction + physical penalties (same as training)

        Termination:
            - Both resources consumed (success)
            - Homeostatic limit exceeded (|hunger| or |thirst| > 0.99999)
            - Agent flipped over

        Truncation:
            - Max steps exceeded
        """
        # Passive decay
        self.hunger -= self.cfg.hunger_decay
        self.thirst -= self.cfg.thirst_decay

        # Apply physical action and simulate
        self.do_simulation(action, self.frame_skip)
        self.current_step += 1
        self.posture = self._get_posture()

        # ---------- Resource contact detection ---------- #
        ant_pos = self.data.xpos[self.ant_body_id][:2]

        # Check contact with resources (NO respawn after consumption)
        # Set the state to zero after consumption so that the agent will automatically need the other one
        for obj in list(self.object):
            type_gen, x, y = obj
            if np.linalg.norm(ant_pos - np.array([x, y])) < self.cfg.object_interaction_dist:
                if type_gen == "food":
                    self.hunger = 0.0
                    self.food_consumed += 1
                    self.object.remove(obj)
                    self.resources_consumed.append("food")
                elif type_gen == "water":
                    self.thirst = 0.0
                    self.water_consumed += 1
                    self.object.remove(obj)
                    self.resources_consumed.append("water")

        # ---------- State checks ---------- #
        up_vector_z = self.data.xmat[self.ant_body_id][8]
        z_pos = self.data.xpos[self.ant_body_id][2]
        action_magnitude = np.linalg.norm(action)

        limit_reached = (
            abs(self.hunger) > 0.99999
            or abs(self.thirst) > 0.99999
        )
        is_flipped = up_vector_z < 0.0
        both_consumed = self.food_consumed >= 1 and self.water_consumed >= 1

        # Termination reason codes:
        # 0 = not terminated, 1 = homeostatic limit, 2 = flipped, 4 = both consumed (success)
        term_reason = 0
        if is_flipped:
            term_reason = 2
        elif limit_reached:
            term_reason = 1
        elif both_consumed:
            term_reason = 4

        # Clip internal states
        self.hunger = np.clip(self.hunger, -1.0, 1.0)
        self.thirst = np.clip(self.thirst, -1.0, 1.0)

        # ---------- Reward ---------- #
        current_drive = self._calculate_drive()
        homeo_reward = self.cfg.reward_scale * (self.prev_drive - current_drive)
        physical_penalty = (
            (-1.0 * self.cfg.movement_penalty_weight * action_magnitude ** 2)
            + (-1.0 * self.cfg.posture_penalty_weight * self.posture ** 2)
        )
        reward = homeo_reward + physical_penalty
        self.prev_drive = current_drive

        # ---------- Observation ---------- #
        obs = self._get_obs()

        if self.render_mode == "human":
            self.render()

        # ---------- Info dict ---------- #
        # Determine "correct" first choice based on INITIAL internal state
        # More negative value = more depleted = should be prioritized
        info = {
            "timestep": np.array(self.current_step),
            "hunger": np.array(self.hunger),
            "thirst": np.array(self.thirst),
            "food_consumed": np.array(self.food_consumed),
            "water_consumed": np.array(self.water_consumed),
            "up_vector_z": np.array(up_vector_z),
            "z_pos": np.array(z_pos),
            "termination_reason": np.array(term_reason),
            # Test-specific metrics
            "resources_consumed": self.resources_consumed,
            # Initial conditions (for post-hoc analysis)
            "initial_hunger": np.array(self._initial_hunger),
            "initial_thirst": np.array(self._initial_thirst),
        }

        # Render debug images when not training (for analysis/recording)
        if not self.cfg.is_training:
            env_image_rgb, _ = self.mux_render(camera_name="environment")
            env_image_rgb = self._add_hud(env_image_rgb)
            pov_image_rgb, _ = self.mux_render(camera_name="pov")

            info["vision"] = pov_image_rgb
            info["environment"] = env_image_rgb

        return obs, reward, self.terminated, self.truncated, info

    # ==================== Termination ==================== #

    @property
    def terminated(self):
        """Episode terminates on homeostatic death OR both resources consumed (success)."""
        limit_reached = (
            abs(self.hunger) > 0.99999
            or abs(self.thirst) > 0.99999
        )
        # up_vector_z = self.data.xmat[self.ant_body_id][8]
        # is_flipped = up_vector_z < 0.0
        # both_consumed = self.food_consumed >= 1 and self.water_consumed >= 1

        return bool(limit_reached)

    @property
    def truncated(self):
        """Episode truncates on max steps (safety net)."""
        return self.current_step >= self.cfg.max_steps

    # ==================== Observation ==================== #

    def _get_obs(self):
        """Construct observation dict: proprioception + vision + internal state.

        Same structure as HomeostaticAntEnv but without temperature/heat_sensor.
        """
        # Proprioceptive observation from AntEnv (joint positions + velocities)
        proprio_obs = AntEnv._get_obs(self)[:self.cfg.obs_space_dim]

        # Render POV vision (RGB + Depth, normalized, CHW format)
        pov_image_rgb, pov_image_depth = self.mux_render(camera_name="pov")
        pov_image_rgb = pov_image_rgb.astype(np.float32) / 255.0
        pov_image_depth = pov_image_depth.astype(np.float32)
        pov_image = np.concatenate(
            [pov_image_rgb, np.expand_dims(pov_image_depth, axis=-1)], axis=-1
        )
        pov_image = np.transpose(pov_image, (2, 0, 1))

        obs = {
            "proprioception": proprio_obs,
            "vision": pov_image,
            "internal_state": np.array(
                [self.hunger, self.thirst], dtype=np.float32
            ),
        }
        return obs

    # ==================== Rendering ==================== #

    def mux_render(self, camera_name):
        """Render from a specific camera, adding resource markers.

        Same approach as HomeostaticAntEnv: markers are added per render call
        for food (green sphere) and water (blue sphere).
        """
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id == -1:
            return self.mujoco_renderer.render(render_mode="rgbd_tuple")

        old_cam_id = self.mujoco_renderer.camera_id
        self.mujoco_renderer.camera_id = cam_id

        # Always full daylight in the Y-maze (no day/night cycle)
        self.model.light_diffuse[0] = [0.9, 0.9, 0.9]

        self._add_markers(render_mode="rgbd_tuple")

        img = self.mujoco_renderer.render(render_mode="rgbd_tuple")
        self.mujoco_renderer.camera_id = old_cam_id
        return img

    def _add_markers(self, render_mode="rgbd_tuple"):
        viewer = self.mujoco_renderer._get_viewer(render_mode=render_mode)
        for obj in self.object:
            type_gen, x, y = obj
            if type_gen == "food":
                viewer.add_marker(pos=(x, y, 0.5), size=(0.5, 0.5, 0.5), rgba=(0, 1, 0, 1), label=" ", type=mujoco.mjtGeom.mjGEOM_SPHERE)
            elif type_gen == "water":
                viewer.add_marker(pos=(x, y, 0.5), size=(0.5, 0.5, 0.5), rgba=(0, 0, 1, 1), label=" ", type=mujoco.mjtGeom.mjGEOM_SPHERE)
            elif type_gen == "heat":
                viewer.add_marker(pos=(x, y, 0.5), size=(0.5, 0.5, 0.5), rgba=(1, 0, 0, 1), label=" ", type=mujoco.mjtGeom.mjGEOM_SPHERE)

    def render(self):
        self._add_markers(render_mode=self.render_mode)
        return self.mujoco_renderer.render(render_mode=self.render_mode)

    def _add_hud(self, img):
        """Add a debug HUD overlay to the environment camera image."""
        import cv2

        img = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1

        # Determine correct first resource for display
        if self._initial_hunger < self._initial_thirst:
            correct = "FOOD - WATER"
        else:
            correct = "WATER - FOOD"

        stats = [
            (f"Hunger: {self.hunger:.2f} (init: {self._initial_hunger:.2f})", (0, 255, 0)),
            (f"Thirst: {self.thirst:.2f} (init: {self._initial_thirst:.2f})", (0, 0, 255)),
            # (f"Step: {self.current_step}", (0, 0, 0)),
            (f"Food: {self.food_consumed}  Water: {self.water_consumed}", (0, 0, 0)),
            (f"Correct Resources: {correct}", (128, 0, 128)),
        ]

        for i, (text, color) in enumerate(stats):
            cv2.putText(img, text, (10, 20 + i * 20), font, scale, color, thickness)

        return img

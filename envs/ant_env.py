import math
from pathlib import Path

import mujoco
import numpy as np
import tempfile
import xml.etree.ElementTree as ET
import scipy.spatial.transform as st
from gymnasium import spaces
from gymnasium.envs.mujoco.ant_v5 import AntEnv
from gymnasium.utils import EzPickle

from configs.config_env import EnvConfig

DEFAULT_CAMERA_CONFIG = {}
RESOURCE_MARKER_RADIUS = 0.5


def qtoeuler(q):
    """ quaternion to Euler angle

    :param q: quaternion
    :return:
    """
    phi = math.atan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
    theta = math.asin(2 * (q[0] * q[2] - q[3] * q[1]))
    # theta = -np.pi/2 + 2*math.atan2(math.sqrt(1 + 2*(q[0]*q[2] - q[1]*q[3])), math.sqrt(1 - 2*(q[0]*q[2]-q[1]*q[3])))
    psi = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    return np.array([phi, theta, psi])


class HomeostaticAntEnv(AntEnv, EzPickle):
    def __init__(
        self,
        cfg: EnvConfig,
        **kwargs,
    ):
        """Gymanasium Ant environment with added homeostatic needs and vision-based observations.

        Args:
            cfg (EnvConfig): Configuration object for the environment.
        """

        # -------------- Setup config -------------- #
        self.cfg = cfg

        # ----------------- Environment Setup ----------------- #
        # Create environment
        xml_file_path = Path(__file__).parent / self.cfg.xml_path
        tree = ET.parse(xml_file_path)
        worldbody = tree.find(".//worldbody")

        # Wall
        wall_attrs = dict(type="box", rgba="0.5 0.5 0.5 1", conaffinity="1", condim="3")
        for name, pos, size in [
            ("wall_n", f"0 {self.cfg.arena_size} 0.5", f"{self.cfg.arena_size} 0.1 2.0"),
            ("wall_s", f"0 {-self.cfg.arena_size} 0.5", f"{self.cfg.arena_size} 0.1 2.0"),
            ("wall_e", f"{self.cfg.arena_size} 0 0.5", f"0.1 {self.cfg.arena_size} 2.0"),
            ("wall_w", f"{-self.cfg.arena_size} 0 0.5", f"0.1 {self.cfg.arena_size} 2.0"),
        ]:
            ET.SubElement(
                worldbody,
                "geom",
                dict(wall_attrs, name=name, pos=pos, size=size),
            )
        with tempfile.NamedTemporaryFile(mode='wt', suffix=".xml", delete=False) as f:
            tree.write(f.name)
            temp_xml_path = f.name
            # model = mujoco.MjModel.from_xml_path(f.name)

        # Initialize AntEnv
        # AntEnv v5 parameters
        AntEnv.__init__(
            self,
            xml_file=temp_xml_path,
            width=self.cfg.image_size[0],
            height=self.cfg.image_size[1],
            render_mode=self.cfg.render_mode,
            **kwargs,
        )

        # -------------- Initialise parameters ---------------
        self.food_consumed = 0
        self.water_consumed = 0
        self.heat_exposed_time = 0.0
        self.posture = 0.0

        self.hunger = 0.0
        self.thirst = 0.0
        self.temperature = 0.0
        self.prev_drive = 0.0

        # Track resources and agent
        self.object= []
        self.ant_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )
        self.pov_camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "pov"
        )
        if self.pov_camera_id == -1:
            raise ValueError("ant_env.xml must define a camera named 'pov'.")

        # ------------ Observation and Action space ------------------
        # Create action space and observation space
        # 8 continuous space if without temperature dynamics, 9 continuous space if with temperature dynamics (the last one is for sweating)
        # The action space have a range of (-1, 1)
        # Observation state depends if heat is present
        if self.cfg.num_heat == 0:
            action_dim = 8
            internal_state_dim = 2
            include_heat_sensor = False
        else:
            action_dim = 9
            internal_state_dim = 3
            include_heat_sensor = True

        # Action space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )

        # Observation space
        obs_dict = {
            "proprioception": spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.cfg.obs_space_dim,), dtype=np.float32
            ),
            # Return vision as CxHxW with RGB channels normalized to [0,1] and depth channel in meters normalized to [0,1]
            "vision": spaces.Box(
                low=0.0, high=1.0, shape=(4, self.cfg.image_size[0], self.cfg.image_size[1]), dtype=np.float32
            ),
            "internal_state": spaces.Box(
                low=-1.0, high=1.0, shape=(internal_state_dim,), dtype=np.float32
            ),
        }

        # Conditionally add the heat sensor if required
        if include_heat_sensor:
            obs_dict["heat_sensor"] = spaces.Box(
                low=-1.0, high=1.0, shape=(3,), dtype=np.float32
            )

        self.observation_space = spaces.Dict(obs_dict)

        EzPickle.__init__(**locals())

    def reset_model(self):
        # Reset initial states
        if self.cfg.is_training:
            # During training, we can randomize the initial homeostatic state
            self.hunger = self.np_random.uniform(-(1 / 6), (1 / 6))
            self.thirst = self.np_random.uniform(-(1 / 6), (1 / 6))
            self.temperature = self.np_random.uniform(-(1 / 6), (1 / 6))
        else:
            # For shifting internal state task
            if self.cfg.shift:
                if self.np_random.random() < 0.5:
                    self.hunger = self.np_random.uniform(-0.15, -0.11)
                    self.thirst = self.np_random.uniform(-0.1, -0.05)
                    self.temperature = 0.0  # placeholder value
                else:
                    self.hunger = self.np_random.uniform(-0.1, -0.05)
                    self.thirst = self.np_random.uniform(-0.15, -0.11)
                    self.temperature = 0.0  # placeholder value
            else:
                self.hunger = 0.0
                self.thirst = 0.0
                self.temperature = 0.0

        # Add if no heat
        if self.cfg.num_heat == 0:
            self.temperature = 0.0

        # Reset resource consumption tracking for new episode
        self.current_step = 0
        self.food_consumed = 0
        self.water_consumed = 0
        self.heat_exposed_time = 0.0

        # --------------- Agent's initial position and orientation ---------------
        # Ranomize agent's initial position and orientation, but keep it within the arena and not too close to the walls
        # self.init_qpos includes the x and y coordinates in the first 2 entries, different from observation space which does not
        if self.cfg.shift:
            qpos = self.init_qpos
            qpos[1] = -3.0
            qvel = self.init_qvel
        else:
            qpos = self.init_qpos + self.np_random.uniform(
                size=self.model.nq, low=-0.01, high=0.01
            )
            # Start at random positions in the arena, but not too close to the walls
            qpos[0] = self.np_random.uniform(-self.cfg.arena_size + 2, self.cfg.arena_size - 2)
            qpos[1] = self.np_random.uniform(-self.cfg.arena_size + 2, self.cfg.arena_size - 2)

            curr_w, curr_x, curr_y, curr_z = qpos[3:7]
            current_rot = st.Rotation.from_quat([curr_x, curr_y, curr_z, curr_w])
            random_yaw_angle = self.np_random.uniform(low=0, high=2 * np.pi)
            yaw_rot = st.Rotation.from_euler("z", random_yaw_angle)
            final_rot = yaw_rot * current_rot
            raw_quat = final_rot.as_quat()
            qpos[3:7] = [raw_quat[3], raw_quat[0], raw_quat[1], raw_quat[2]]

            qvel = self.init_qvel + self.np_random.uniform(
                low=-0.01, high=0.01, size=self.model.nv
            )
        self.set_state(qpos, qvel)
        self.posture = self._get_posture()

        # ------------------- Reset resources ------------------- #
        self.object = []
        existing_items = set()            
        if self.cfg.shift:
            # Add one food
            self.object.append(("food", -3, 4))
            # Add one water
            self.object.append(("water", 3, 4))
        else:
            # Food with random positions
            for i in range(self.cfg.num_food):
                regen = True
                while regen:
                    regen = False
                    x = self.np_random.uniform(-self.cfg.arena_size + 0.75, self.cfg.arena_size - 0.75)
                    y = self.np_random.uniform(-self.cfg.arena_size + 0.75, self.cfg.arena_size - 0.75)
                    if np.linalg.norm(np.array([x, y]) - np.array(qpos[:2])) < self.cfg.object_spacing:
                        regen = True
                        continue
                    for pos in existing_items:
                        if np.linalg.norm(np.array([x, y]) - np.array(pos)) < self.cfg.object_spacing:
                            regen = True
                            break
                existing_items.add((x, y))
                self.object.append(("food", x, y))
            # Water with random positions
            for i in range(self.cfg.num_water):
                regen = True
                while regen:
                    regen = False
                    x = self.np_random.uniform(-self.cfg.arena_size + 0.75, self.cfg.arena_size - 0.75)
                    y = self.np_random.uniform(-self.cfg.arena_size + 0.75, self.cfg.arena_size - 0.75)
                    if np.linalg.norm(np.array([x, y]) - np.array(qpos[:2])) < self.cfg.object_spacing:
                        regen = True
                        continue
                    for pos in existing_items:
                        if np.linalg.norm(np.array([x, y]) - np.array(pos)) < self.cfg.object_spacing:
                            regen = True
                            break
                existing_items.add((x, y))
                self.object.append(("water", x, y))
            # Heat with random positions
            for i in range(self.cfg.num_heat):
                regen = True
                while regen:
                    regen = False
                    x = self.np_random.uniform(-self.cfg.arena_size + 0.75, self.cfg.arena_size - 0.75)
                    y = self.np_random.uniform(-self.cfg.arena_size + 0.75, self.cfg.arena_size - 0.75)
                    if np.linalg.norm(np.array([x, y]) - np.array(qpos[:2])) < self.cfg.object_spacing:
                        regen = True
                        continue
                    for pos in existing_items:
                        if np.linalg.norm(np.array([x, y]) - np.array(pos)) < self.cfg.object_spacing:
                            regen = True
                            break
                existing_items.add((x, y))
                self.object.append(("heat", x, y))

        # Initialize previous drive for the paper's reward formula
        self.prev_drive = self._calculate_drive()

        return self._get_obs()

    def _calculate_drive(self):
        return self.hunger**2 + self.thirst**2 + self.temperature**2 + (self.cfg.posture_drive_penalty * (self.posture ** 2))

    def _get_posture(self):
        # Posture dynamics - Deviation from upright
        # Using Euler angles (roll, pitch) to calculate tilt
        # data.qpos[3:7] is torso orientation (w, x, y, z)
        return np.linalg.norm(qtoeuler(self.data.qpos[3:7])[:2] - qtoeuler([1.0, 0.0, 0.0, 0.0])[:2])

    def _generate_new_object(self, type_gen):
        existing = set()
        for obj in self.object:
            existing.add((obj[1], obj[2]))
        # Agent's position
        agent_pos = self.data.qpos[:2]
        regen = True
        while regen:
            regen = False
            x = self.np_random.uniform(-self.cfg.arena_size + 0.75, self.cfg.arena_size - 0.75)
            y = self.np_random.uniform(-self.cfg.arena_size + 0.75, self.cfg.arena_size - 0.75)
            if (x, y) in existing:
                regen = True
                continue
            if np.linalg.norm(np.array([x, y]) - agent_pos) < self.cfg.object_spacing:
                regen = True
                continue
            for pos in existing:
                if np.linalg.norm(np.array([x, y]) - np.array(pos)) < self.cfg.object_spacing:
                    regen = True
                    break
        return (type_gen, x, y)

    def _get_heat_sensor_obs(self):
        """
        Detects the nearest heat source within 1.0m and returns the local direction.
        Returns [0, 0] if no heat source is within range.
        """
        ant_pos = self.data.xpos[self.ant_body_id][:2]
        nearest_heat_dist = self.cfg.heat_sensor_range
        nearest_heat_pos = None

        for obj in self.object:
            type_gen, x, y = obj
            if type_gen != "heat":
                continue
            heat_pos = np.array([x, y])
            dist = np.linalg.norm(ant_pos - heat_pos)
            if dist < nearest_heat_dist:
                nearest_heat_dist = dist
                nearest_heat_pos = heat_pos

        if nearest_heat_pos is None:
            return np.array([0.0, 0.0, 0], dtype=np.float32)

        # Vector from Ant to Heat in world coordinates
        delta_world = nearest_heat_pos - ant_pos
        
        # Transform to local frame (torso's rotation)
        # xmat is a 9-element array (3x3 rotation matrix)
        rot_mat = self.data.xmat[self.ant_body_id].reshape(3, 3)
        # Local X (Forward) and Local Y (Left) projections
        local_x = np.dot(delta_world, rot_mat[:2, 0])
        local_y = np.dot(delta_world, rot_mat[:2, 1])
        
        # Normalize to unit vector for direction only
        direction = np.array([local_x, local_y], dtype=np.float32)
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction /= norm

        return np.array([direction[0], direction[1], 1.0], dtype=np.float32)

    def _get_obs(self):
        # This is the basic proprioceptive observation from AntEnv
        # OBS_SPACE_DIM shape about the body position, velocity, and joint angles
        # Only take the original proprioceptive part, not the resource positions we added in the XML file
        proprio_obs = AntEnv._get_obs(self)[:self.cfg.obs_space_dim]

        # Render vision observations
        pov_image_rgb, pov_image_depth = self.mux_render(camera_name="pov")
        # Normalize RGB to [0,1]
        # Depth should already be [0, 1]
        pov_image_rgb = pov_image_rgb.astype(np.float32) / 255.0
        pov_image_depth = pov_image_depth.astype(np.float32)
        pov_image = np.concatenate([pov_image_rgb, np.expand_dims(pov_image_depth, axis=-1)], axis=-1)
        pov_image = np.transpose(pov_image, (2, 0, 1))

        # Observation
        obs = {
            "proprioception": proprio_obs,
            "vision": pov_image,
        }
        if self.cfg.num_heat > 0:
            obs["internal_state"] = np.array(
                [self.hunger, self.thirst, self.temperature],
                dtype=np.float32,  # internal variables
            )
            obs["heat_sensor"] = self._get_heat_sensor_obs()
        else:
             obs["internal_state"] = np.array(
                [self.hunger, self.thirst],
                dtype=np.float32,  # internal variables
            )
        return obs

    def _add_hud(self, img):
        import cv2

        # Copy to avoid modifying the original if it's a view
        img = img.copy()

        # HUD settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1

        # Day/Night Status
        is_night = (self.current_step % self.cfg.day_night_cycle_len) >= (
            self.cfg.day_night_cycle_len / 2
        )
        time_text = "NIGHT" if is_night else "DAY"
        time_color = (0, 0, 0)

        stats = [
            (f"Hunger: {self.hunger:.2f}", (0, 255, 0)),  # Green
            (f"Thirst: {self.thirst:.2f}", (0, 0, 255)),  # Blue
            (f"Temp:   {self.temperature:.2f}", (255, 0, 0)),  # Red
            (f"Time:   {time_text}", time_color),
        ]

        for i, (text, color) in enumerate(stats):
            cv2.putText(img, text, (10, 20 + i * 20), font, scale, color, thickness)

        return img

    def _add_markers(self, render_mode="rgbd_tuple"):
        viewer = self.mujoco_renderer._get_viewer(render_mode=render_mode)
        for obj in self.object:
            type_gen, x, y = obj
            if type_gen == "food":
                viewer.add_marker(pos=(x, y, 0.5), size=(RESOURCE_MARKER_RADIUS,) * 3, rgba=(0, 1, 0, 1), label=" ", type=mujoco.mjtGeom.mjGEOM_SPHERE)
            elif type_gen == "water":
                viewer.add_marker(pos=(x, y, 0.5), size=(RESOURCE_MARKER_RADIUS,) * 3, rgba=(0, 0, 1, 1), label=" ", type=mujoco.mjtGeom.mjGEOM_SPHERE)
            elif type_gen == "heat":
                viewer.add_marker(pos=(x, y, 0.5), size=(RESOURCE_MARKER_RADIUS,) * 3, rgba=(1, 0, 0, 1), label=" ", type=mujoco.mjtGeom.mjGEOM_SPHERE)

    def mux_render(self, camera_name):
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id == -1:
            return self.mujoco_renderer.render(render_mode="rgbd_tuple")

        old_cam_id = self.mujoco_renderer.camera_id
        self.mujoco_renderer.camera_id = cam_id

        # Adjust lighting based on day/night before rendering
        is_night = (self.current_step % self.cfg.day_night_cycle_len) >= (
            self.cfg.day_night_cycle_len / 2
        )
        if is_night:
            self.model.light_diffuse[0] = [0.1, 0.1, 0.1]
        else:
            self.model.light_diffuse[0] = [0.9, 0.9, 0.9]

        self._add_markers(render_mode="rgbd_tuple")

        img = self.mujoco_renderer.render(render_mode="rgbd_tuple")

        self.mujoco_renderer.camera_id = old_cam_id
        return img

    def render(self):
        self._add_markers(render_mode=self.render_mode)
        return self.mujoco_renderer.render(render_mode=self.render_mode)

    def step(self, action):
        # Passive decay/gain
        self.hunger -= self.cfg.hunger_decay
        self.thirst -= self.cfg.thirst_decay

        physical_action = action[:8]
        if self.cfg.num_heat > 0:
            sweat_action = action[8]

        # Apply physical action and simulate
        self.do_simulation(physical_action, self.frame_skip)
        self.current_step += 1
        self.posture = self._get_posture()

        # Add shift after 50 steps
        # Change the secondary resource positions after 50 steps if shift is enabled
        if self.cfg.shift and self.current_step == 100:
            if self.hunger < self.thirst:
                self.thirst -= 0.2
            elif self.hunger > self.thirst:
                self.hunger -= 0.2
            else:
                # Randomly choose one to decrease if they are equal
                if self.np_random.random() < 0.5:
                    self.hunger -= 0.2
                else:
                    self.thirst -= 0.2

        # Homeostatic Dynamics
        is_night = (self.current_step % self.cfg.day_night_cycle_len) >= (
            self.cfg.day_night_cycle_len / 2
        )

        # Resource and Heat Contact Detection
        contact_heat = 0
        ant_pos = self.data.xpos[self.ant_body_id][:2]

        # Check contact with objects
        for obj in list(self.object):
            type_gen, x, y = obj
            if np.linalg.norm(ant_pos - np.array([x, y])) < self.cfg.object_interaction_dist:
                if type_gen == "food" and self._is_in_front(np.array([x, y])):
                    self.hunger += self.cfg.replenish_rate
                    self.food_consumed += 1
                    self.object.remove(obj)
                    if not self.cfg.shift:
                        self.object.append(self._generate_new_object(type_gen))
                elif type_gen == "water" and self._is_in_front(np.array([x, y])):
                    self.thirst += self.cfg.replenish_rate
                    self.water_consumed += 1
                    self.object.remove(obj)
                    if not self.cfg.shift:
                        self.object.append(self._generate_new_object(type_gen))
                elif type_gen == "heat":
                    # heat doesnt get consumed
                    contact_heat += 1

        action_magnitude = np.linalg.norm(physical_action)
        if self.cfg.num_heat > 0:
            # Temperature dynamics
            self.temperature += action_magnitude**2 * self.cfg.action_heat_gain_rate  # quadratic so small movements gain less heat than moving wildly
            if contact_heat:
                self.temperature += self.cfg.heat_source_gain_rate * contact_heat
                self.heat_exposed_time += 1.0 * contact_heat

            # Sweat changes
            if sweat_action > 0.0:
                self.temperature -= (sweat_action * self.cfg.sweat_cooling_rate)
                self.thirst -= (self.cfg.sweat_thirst_cost * sweat_action)
                self.model.geom_rgba[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "torso_geom")
                ] = [0.2, 0.6, 1.0, 1.0]  # Change color to indicate sweating
            else:
                self.model.geom_rgba[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "torso_geom")
                ] = [0.8, 0.6, 0.4, 1.0]  # Default color
            if is_night:
                self.temperature -= self.cfg.night_cooling_rate

        # Check if agent has flipped
        up_vector_z = self.data.xmat[self.ant_body_id][8]
        z_pos = self.data.xpos[self.ant_body_id][2]

        limit_reached = (
            abs(self.hunger) > 0.99999
            or abs(self.thirst) > 0.99999
            or abs(self.temperature) > 0.99999
        )
        is_flipped = up_vector_z < 0.0
        is_height_invalid = z_pos < 0.2 or z_pos > 1.0

        term_reason = 0
        # Check flipped first
        
        if is_flipped:
            term_reason = 2  # flipped
        elif limit_reached:
            term_reason = 1  # limit reached
        elif is_height_invalid:
            term_reason = 3  # height

        # Clipping state variables
        self.hunger = np.clip(self.hunger, -1.0, 1.0)
        self.thirst = np.clip(self.thirst, -1.0, 1.0)
        self.temperature = np.clip(self.temperature, -1.0, 1.0)

        # Homeostatic Reward (Paper: Reduction in Drive)
        current_drive = self._calculate_drive()

        homeo_reward = self.cfg.reward_scale * (self.prev_drive - current_drive)
        movement_penalty = -self.cfg.movement_penalty_weight * action_magnitude**2
        posture_penalty = -self.cfg.posture_penalty_weight * self.posture**2
        physical_penalty = movement_penalty + posture_penalty

        reward = homeo_reward + physical_penalty
        self.prev_drive = current_drive

        obs = self._get_obs()

        if self.render_mode == "human":
            self.render()

        info = {
            "timestep": np.array(self.current_step),
            "is_night": np.array(is_night),
            "hunger": np.array(self.hunger),
            "thirst": np.array(self.thirst),
            "food_consumed": np.array(self.food_consumed),
            "water_consumed": np.array(self.water_consumed),
            "up_vector_z": np.array(up_vector_z),
            "is_flipped": np.array(is_flipped),
            "z_pos": np.array(z_pos),
            "termination_reason": np.array(term_reason),
            "posture": np.array(self.posture),
            "action_magnitude": np.array(action_magnitude),
            "reward_homeostatic": np.array(homeo_reward),
            "reward_movement_penalty": np.array(movement_penalty),
            "reward_posture_penalty": np.array(posture_penalty),
        }

        # info = {
        #     "time": {"timestep": self.current_step, "is_night": is_night},
        #     "internal_state": {
        #         "hunger": self.hunger,
        #         "thirst": self.thirst,
        #     },
        #     "resources_consumed": {
        #         "food": self.food_consumed,
        #         "water": self.water_consumed,
        #     },
        #     "stability": {
        #         "up_vector_z": up_vector_z,
        #         "z_pos": z_pos,
        #         "termination_reason": term_reason,
        #         "posture": self.posture,
        #     },
        # }
        if self.cfg.num_heat > 0:
            # info["internal_state"]["temperature"] = self.temperature
            # info["resources_consumed"]["heat_exposure_time"] = self.heat_exposed_time
            # info["resources_consumed"]["sweating"] = 1.0 if sweat_action > 0.0 else 0.0
            info["temperature"] = self.temperature
            info["heat_exposed_time"] = self.heat_exposed_time
            info["sweating"] = 1.0 if sweat_action > 0.0 else 0.0

        # Return the environment image in info for vieweing
        # Add visual HUD to the environment image for debugging/monitoring
        # Dont need depth for viewing
        # Only do it for debugging
        if not self.cfg.is_training:
            env_image_rgb, _ = self.mux_render(camera_name="environment")
            env_image_rgb = self._add_hud(env_image_rgb)

            # Also return POV in infor for recording and viewing
            # Current vision - dont have any normalize or frame stack etc.
            pov_image_rgb, _ = self.mux_render(camera_name="pov")

            info["vision"] = pov_image_rgb
            info["environment"] = env_image_rgb

        return obs, reward, self.terminated, self.truncated, info

    @property
    def terminated(self):
        # Homeostatic limits check (+/- 1.0)
        limit_reached = (
            abs(self.hunger) > 0.99999
            or abs(self.thirst) > 0.99999
            or abs(self.temperature) > 0.99999
        )

        # # Orientation check (True Flip Check)
        # # xmat[8] is the world-Z component of the torso's local Z-axis (Up)
        # # 1.0 = upright, 0.0 = on side, -1.0 = upside down
        # up_vector_z = self.data.xmat[self.ant_body_id][8]
        # is_flipped = up_vector_z < 0.0

        # # Height check
        # z_pos = self.data.xpos[self.ant_body_id][2]
        # is_too_low = z_pos < 0.2 or z_pos > 1.0

        return bool(limit_reached)  # or is_flipped  # or is_too_low

    @property
    def truncated(self):
        return self.current_step >= self.cfg.max_steps

    def _is_in_front(self, target_pos, target_radius=RESOURCE_MARKER_RADIUS):
        """Return whether any part of a resource lies in the POV horizontal FOV.

        MuJoCo cameras look along their local ``-Z`` axis.  Deriving the
        direction from ``cam_xmat`` keeps the interaction rule aligned with the
        XML camera even if its mounting orientation changes.  This deliberately
        checks the horizontal cone only: resource interaction is defined on the
        arena plane, while the camera's vertical framing changes as the ant
        pitches and rolls.  The rendered resources are spheres, so their
        angular radius is included; otherwise a visibly protruding resource at
        the image edge could not be consumed merely because its centre was just
        outside the cone.
        """
        camera_axes = self.data.cam_xmat[self.pov_camera_id].reshape(3, 3)
        camera_forward_xy = -camera_axes[:, 2][:2]
        forward_norm = np.linalg.norm(camera_forward_xy)
        if forward_norm < 1e-6:
            return False
        camera_forward_xy /= forward_norm

        ant_pos = self.data.xpos[self.ant_body_id][:2]
        target_direction = np.asarray(target_pos, dtype=np.float64)[:2] - ant_pos
        target_distance = np.linalg.norm(target_direction)
        if target_distance < 1e-6:
            return True
        target_direction /= target_distance

        # ``fovy`` is vertical.  Convert it to a horizontal FOV for the render
        # aspect ratio so this condition follows the camera rather than a
        # duplicated hard-coded angle.  The current 64x64 render makes both
        # angles 120 degrees.
        vertical_fov = np.deg2rad(self.model.cam_fovy[self.pov_camera_id])
        width, height = self.cfg.image_size
        horizontal_fov = 2.0 * np.arctan(np.tan(vertical_fov / 2.0) * width / height)
        angular_radius = np.arcsin(min(1.0, max(0.0, target_radius) / target_distance))
        return bool(
            np.dot(camera_forward_xy, target_direction)
            >= np.cos(horizontal_fov / 2.0 + angular_radius)
        )

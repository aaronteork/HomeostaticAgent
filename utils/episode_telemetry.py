from __future__ import annotations

import numpy as np


def _as_worker_array(values, num_workers: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.shape[0] != num_workers:
        raise ValueError(
            f"{name} must have {num_workers} worker rows, got {array.shape}"
        )
    return array


class EpisodeTelemetry:
    """Accumulate per-episode diagnostics without writing every transition."""

    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self.episode_age = np.zeros(num_workers, dtype=np.int64)
        self._reset_workers(np.arange(num_workers))

    def _reset_workers(self, workers) -> None:
        workers = np.asarray(workers, dtype=np.int64)
        if workers.size == 0:
            return
        self.episode_age[workers] = 0
        for name in (
            "homeostatic_return",
            "movement_penalty_return",
            "posture_penalty_return",
            "observed_return",
            "raw_action_abs_sum",
            "executed_action_magnitude_sum",
            "actor_loc_abs_sum",
            "actor_std_sum",
        ):
            if not hasattr(self, name):
                setattr(self, name, np.zeros(self.num_workers, dtype=np.float64))
            else:
                getattr(self, name)[workers] = 0.0
        for name in (
            "action_coordinate_count",
            "action_clipped_coordinate_count",
            "action_step_count",
            "action_any_clipped_count",
            "actor_loc_count",
            "actor_std_count",
            "flipped_step_count",
        ):
            if not hasattr(self, name):
                setattr(self, name, np.zeros(self.num_workers, dtype=np.int64))
            else:
                getattr(self, name)[workers] = 0
        if not hasattr(self, "executed_action_magnitude_max"):
            self.executed_action_magnitude_max = np.zeros(
                self.num_workers, dtype=np.float64
            )
        else:
            self.executed_action_magnitude_max[workers] = 0.0
        if not hasattr(self, "first_flip_step"):
            self.first_flip_step = np.full(self.num_workers, -1, dtype=np.int64)
        else:
            self.first_flip_step[workers] = -1

    def add_transition(
        self,
        *,
        valid,
        rewards,
        infos,
        native_actions,
        executed_actions=None,
        actor_loc=None,
        actor_std=None,
    ) -> np.ndarray:
        """Observe one vector step and return workers flipping for the first time."""
        valid = _as_worker_array(valid, self.num_workers, "valid").astype(bool)
        rewards = _as_worker_array(rewards, self.num_workers, "rewards")
        raw_actions = _as_worker_array(
            native_actions, self.num_workers, "native_actions"
        )
        if executed_actions is None:
            executed_actions = np.clip(raw_actions, -1.0, 1.0)
        executed_actions = _as_worker_array(
            executed_actions, self.num_workers, "executed_actions"
        )

        indices = np.flatnonzero(valid)
        if indices.size == 0:
            return indices

        self.episode_age[indices] += 1
        self.observed_return[indices] += rewards[indices]
        component_names = (
            ("reward_homeostatic", "homeostatic_return"),
            ("reward_movement_penalty", "movement_penalty_return"),
            ("reward_posture_penalty", "posture_penalty_return"),
        )
        for info_name, accumulator_name in component_names:
            if info_name not in infos:
                raise KeyError(f"environment info is missing {info_name!r}")
            values = _as_worker_array(infos[info_name], self.num_workers, info_name)
            getattr(self, accumulator_name)[indices] += values[indices]

        raw = raw_actions[indices]
        executed = executed_actions[indices]
        clipped = np.abs(raw) > 1.0
        coordinate_count = raw.shape[1]
        self.raw_action_abs_sum[indices] += np.abs(raw).sum(axis=1)
        self.action_coordinate_count[indices] += coordinate_count
        self.action_clipped_coordinate_count[indices] += clipped.sum(axis=1)
        self.action_step_count[indices] += 1
        self.action_any_clipped_count[indices] += clipped.any(axis=1)

        magnitudes = np.linalg.norm(executed, axis=1)
        self.executed_action_magnitude_sum[indices] += magnitudes
        self.executed_action_magnitude_max[indices] = np.maximum(
            self.executed_action_magnitude_max[indices], magnitudes
        )

        if actor_loc is not None:
            loc = _as_worker_array(actor_loc, self.num_workers, "actor_loc")[indices]
            self.actor_loc_abs_sum[indices] += np.abs(loc).sum(axis=1)
            self.actor_loc_count[indices] += loc.shape[1]
        if actor_std is not None:
            std = _as_worker_array(actor_std, self.num_workers, "actor_std")[indices]
            self.actor_std_sum[indices] += std.sum(axis=1)
            self.actor_std_count[indices] += std.shape[1]

        if "is_flipped" in infos:
            flipped = _as_worker_array(
                infos["is_flipped"], self.num_workers, "is_flipped"
            ).astype(bool)
        else:
            up_vector_z = _as_worker_array(
                infos["up_vector_z"], self.num_workers, "up_vector_z"
            )
            flipped = up_vector_z < 0.0
        flipped &= valid
        self.flipped_step_count[flipped] += 1
        newly_flipped = flipped & (self.first_flip_step < 0)
        self.first_flip_step[newly_flipped] = self.episode_age[newly_flipped]
        return np.flatnonzero(newly_flipped)

    @staticmethod
    def _safe_ratio(numerator: float, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    def finish_episode(
        self,
        worker: int,
        *,
        episode_return: float,
        episode_length: int,
        food_consumed: float,
        water_consumed: float,
    ) -> dict[str, float]:
        age = int(self.episode_age[worker])
        tracked_return = (
            self.homeostatic_return[worker]
            + self.movement_penalty_return[worker]
            + self.posture_penalty_return[worker]
        )
        metrics = {
            "episode/reward_per_step": self._safe_ratio(episode_return, episode_length),
            "episode/homeostatic_return": float(self.homeostatic_return[worker]),
            "episode/movement_penalty_return": float(
                self.movement_penalty_return[worker]
            ),
            "episode/posture_penalty_return": float(
                self.posture_penalty_return[worker]
            ),
            "episode/reward_component_residual": float(episode_return - tracked_return),
            "episode/resource_per_step": self._safe_ratio(
                food_consumed + water_consumed, episode_length
            ),
            "episode/action_raw_abs_mean": self._safe_ratio(
                self.raw_action_abs_sum[worker],
                int(self.action_coordinate_count[worker]),
            ),
            "episode/action_clip_coordinate_fraction": self._safe_ratio(
                self.action_clipped_coordinate_count[worker],
                int(self.action_coordinate_count[worker]),
            ),
            "episode/action_any_clipped_fraction": self._safe_ratio(
                self.action_any_clipped_count[worker],
                int(self.action_step_count[worker]),
            ),
            "episode/action_magnitude_mean": self._safe_ratio(
                self.executed_action_magnitude_sum[worker],
                int(self.action_step_count[worker]),
            ),
            "episode/action_magnitude_max": float(
                self.executed_action_magnitude_max[worker]
            ),
            "episode/actor_mean_abs": self._safe_ratio(
                self.actor_loc_abs_sum[worker], int(self.actor_loc_count[worker])
            ),
            "episode/actor_std_mean": self._safe_ratio(
                self.actor_std_sum[worker], int(self.actor_std_count[worker])
            ),
            "episode/first_flip_step": float(self.first_flip_step[worker]),
            "episode/flipped_step_fraction": self._safe_ratio(
                self.flipped_step_count[worker], age
            ),
            "episode/ever_flipped": float(self.first_flip_step[worker] >= 0),
        }
        self._reset_workers([worker])
        return metrics

    def active_survival_metrics(self) -> dict[str, float]:
        ages = self.episode_age.astype(np.float64)
        return {
            "survival/active_episode_age_mean": float(ages.mean()),
            "survival/active_episode_age_median": float(np.median(ages)),
            "survival/active_episode_age_p95": float(np.percentile(ages, 95)),
            "survival/active_episode_age_max": float(ages.max()),
        }


class RolloutTelemetryWindow:
    """Store one logging window of rollout samples for distribution summaries."""

    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.reset()

    def reset(self) -> None:
        self._actor_loc_abs = []
        self._actor_std = []
        self._raw_action_abs = []
        self._action_magnitude = []
        self._step_flipped_fraction = []
        self._raw_actions = []
        self._executed_actions = []
        self._actor_locs = []
        self._actor_stds = []

    def add_transition(
        self,
        *,
        valid,
        raw_actions,
        executed_actions,
        actor_loc,
        actor_std,
        is_flipped,
    ) -> None:
        valid = np.asarray(valid, dtype=bool)
        if not valid.any():
            return
        raw = np.asarray(raw_actions)[valid]
        executed = np.asarray(executed_actions)[valid]
        loc = np.asarray(actor_loc)[valid]
        std = np.asarray(actor_std)[valid]
        flipped = np.asarray(is_flipped, dtype=bool)[valid]
        for name, values in (
            ("raw_actions", raw),
            ("executed_actions", executed),
            ("actor_loc", loc),
            ("actor_std", std),
        ):
            if values.ndim != 2 or values.shape[1] != self.action_dim:
                raise ValueError(
                    f"{name} must have shape (workers, {self.action_dim}), "
                    f"got {values.shape}"
                )
        self._raw_actions.append(raw)
        self._executed_actions.append(executed)
        self._actor_locs.append(loc)
        self._actor_stds.append(std)
        self._actor_loc_abs.append(np.abs(loc).reshape(-1))
        self._actor_std.append(std.reshape(-1))
        self._raw_action_abs.append(np.abs(raw).reshape(-1))
        self._action_magnitude.append(np.linalg.norm(executed, axis=1))
        self._step_flipped_fraction.append(float(flipped.mean()))

    @staticmethod
    def _distribution_metrics(prefix: str, chunks) -> dict[str, float]:
        values = np.concatenate(chunks)
        return {
            f"{prefix}_mean": float(values.mean()),
            f"{prefix}_p95": float(np.percentile(values, 95)),
            f"{prefix}_max": float(values.max()),
        }

    def summary(self, *, reset: bool = True) -> dict[str, float]:
        if not self._raw_actions:
            return {}
        raw = np.concatenate(self._raw_actions, axis=0)
        clipped = np.abs(raw) > 1.0
        flipped_fractions = np.asarray(self._step_flipped_fraction)
        metrics = {}
        metrics.update(
            self._distribution_metrics("rollout/actor_mean_abs", self._actor_loc_abs)
        )
        actor_locs = np.concatenate(self._actor_locs, axis=0)
        metrics["rollout/actor_mean_rms"] = float(
            np.sqrt(np.mean(np.square(actor_locs)))
        )
        metrics["rollout/actor_mean_near_bound_fraction"] = float(
            (np.abs(actor_locs) >= 0.95).mean()
        )
        metrics.update(
            self._distribution_metrics("rollout/actor_std", self._actor_std)
        )
        metrics["rollout/actor_std_min"] = float(
            np.concatenate(self._actor_std).min()
        )
        metrics.update(
            self._distribution_metrics(
                "rollout/raw_action_abs", self._raw_action_abs
            )
        )
        metrics["rollout/raw_action_rms"] = float(np.sqrt(np.mean(np.square(raw))))
        metrics.update(
            self._distribution_metrics(
                "rollout/action_magnitude", self._action_magnitude
            )
        )
        metrics.update(
            {
                "rollout/action_clip_coordinate_fraction": float(clipped.mean()),
                "rollout/action_any_clipped_fraction": float(
                    clipped.any(axis=1).mean()
                ),
                "rollout/workers_flipped_fraction_mean": float(
                    flipped_fractions.mean()
                ),
                "rollout/workers_flipped_fraction_p95": float(
                    np.percentile(flipped_fractions, 95)
                ),
                "rollout/workers_flipped_fraction_max": float(
                    flipped_fractions.max()
                ),
                "rollout/workers_flipped_fraction_current": float(
                    flipped_fractions[-1]
                ),
            }
        )
        if reset:
            self.reset()
        return metrics

    def per_joint_summary(self, *, reset: bool = True) -> dict[str, float]:
        if not self._raw_actions:
            return {}
        raw = np.concatenate(self._raw_actions, axis=0)
        executed = np.concatenate(self._executed_actions, axis=0)
        loc = np.concatenate(self._actor_locs, axis=0)
        std = np.concatenate(self._actor_stds, axis=0)
        metrics = {}
        for joint in range(self.action_dim):
            prefix = f"rollout_joint/{joint}"
            metrics.update(
                {
                    f"{prefix}_actor_mean_abs": float(np.abs(loc[:, joint]).mean()),
                    f"{prefix}_actor_std_mean": float(std[:, joint].mean()),
                    f"{prefix}_raw_action_abs_mean": float(
                        np.abs(raw[:, joint]).mean()
                    ),
                    f"{prefix}_executed_action_abs_mean": float(
                        np.abs(executed[:, joint]).mean()
                    ),
                    f"{prefix}_clip_fraction": float(
                        (np.abs(raw[:, joint]) > 1.0).mean()
                    ),
                }
            )
        if reset:
            self.reset()
        return metrics

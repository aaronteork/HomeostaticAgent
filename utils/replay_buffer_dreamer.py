from collections import deque

import numpy as np

from configs.config_dreamer import DreamerConfig


class SequenceReplayBuffer:
    """Replay buffer that stores same-environment sequences for RSSM training.

    The training loop adds one transition per worker at each environment step. This
    buffer groups those calls into one time-major row, so sampling can return
    contiguous chunks from a single worker trajectory instead of accidentally
    mixing workers within one RSSM sequence.
    """

    def __init__(self, config: DreamerConfig, device="cpu"):
        self.config = config
        self.device = device
        self.num_workers = config.num_workers
        if self.num_workers <= 0:
            raise ValueError(f"num_workers must be positive, got {self.num_workers}")
        # Maximum time steps per worker to store, ensuring total capacity is as specified in config
        self.replay_capacity = max(1, config.replay_capacity // self.num_workers)
        self.batch_length = config.batch_length
        self.replay_context = config.replay_context
        self.sequence_length = self.batch_length + self.replay_context
        self.batch_size = config.batch_size
        self._rng = np.random.default_rng(self.config.seed)
        self._total_rows_committed = 0
        self._online_queue = deque()

        # Each item in the list is a time step of all environments, so the shape of each item is [num_workers, ...].
        # The deques ensure we only keep the most recent replay_capacity time steps per worker.
        self.observations = deque(maxlen=self.replay_capacity)
        self.actions = deque(maxlen=self.replay_capacity)
        self.rewards = deque(maxlen=self.replay_capacity)
        self.terminals = deque(maxlen=self.replay_capacity)
        self.episode_ends = deque(maxlen=self.replay_capacity)
        self.episode_starts = deque(maxlen=self.replay_capacity)
        self.context_stochastics = deque(maxlen=self.replay_capacity)
        self.context_recurrents = deque(maxlen=self.replay_capacity)

        self._pending_observations = []
        self._pending_actions = []
        self._pending_rewards = []
        self._pending_terminals = []
        self._pending_episode_ends = []
        self._pending_episode_starts = []
        self._pending_context_stochastics = []
        self._pending_context_recurrents = []

    def add(
        self,
        obs_dict,
        action,
        reward,
        terminal,
        is_last,
        is_first=False,
        context_stochastic=None,
        context_recurrent=None,
    ):
        """Add one worker-aligned replay row.

        Calls are expected in worker order. Once all workers for a timestep have
        been added, the row is committed as [env0, env1, ...]. ``terminal`` is
        a true absorbing end; ``is_last`` also includes time-limit truncations.
        """
        if len(self._pending_observations) >= self.num_workers:
            raise RuntimeError(
                "Pending replay row already has num_workers entries; this indicates an add() ordering bug."
            )

        self._pending_observations.append(obs_dict)
        self._pending_actions.append(action)
        self._pending_rewards.append(reward)
        self._pending_terminals.append(terminal)
        self._pending_episode_ends.append(is_last)
        self._pending_episode_starts.append(is_first)
        self._pending_context_stochastics.append(context_stochastic)
        self._pending_context_recurrents.append(context_recurrent)

        if len(self._pending_observations) == self.num_workers:
            self.observations.append(self._pending_observations)
            self.actions.append(self._pending_actions)
            self.rewards.append(self._pending_rewards)
            self.terminals.append(self._pending_terminals)
            self.episode_ends.append(self._pending_episode_ends)
            self.episode_starts.append(self._pending_episode_starts)
            self.context_stochastics.append(self._pending_context_stochastics)
            self.context_recurrents.append(self._pending_context_recurrents)
            self._total_rows_committed += 1
            if self._total_rows_committed % self.sequence_length == 0:
                start_abs = self._total_rows_committed - self.sequence_length
                for env_idx in range(self.num_workers):
                    self._online_queue.append((start_abs, env_idx))

            self._pending_observations = []
            self._pending_actions = []
            self._pending_rewards = []
            self._pending_terminals = []
            self._pending_episode_ends = []
            self._pending_episode_starts = []
            self._pending_context_stochastics = []
            self._pending_context_recurrents = []

    def _build_sequence_batch(self, start_indices, env_indices, force_first_reset):
        """Build same-worker contiguous sequences and keep replay coordinates.

        Sequences may cross episode boundaries. Boundary rows carry
        is_first=True, and their previous action is zeroed so the RSSM reset
        does not receive an action from the previous episode.
        """
        obs_sequences = []
        action_sequences = []
        prev_action_sequences = []
        reward_sequences = []
        terminal_sequences = []
        is_last_sequences = []
        is_first_sequences = []
        context_stochastic_sequences = []
        context_recurrent_sequences = []
        replay_indices = []

        for start_idx, env_idx in zip(start_indices, env_indices):
            obs_seq = [
                self.observations[start_idx + i][env_idx]
                for i in range(self.sequence_length)
            ]
            action_seq = [
                self.actions[start_idx + i][env_idx] for i in range(self.sequence_length)
            ]
            prev_action_seq = []
            for i in range(self.sequence_length):
                item_idx = start_idx + i
                if force_first_reset and i == 0:
                    prev_action = np.zeros_like(self.actions[item_idx][env_idx])
                elif self.episode_starts[item_idx][env_idx]:
                    prev_action = np.zeros_like(self.actions[item_idx][env_idx])
                elif i == 0:
                    if start_idx > 0:
                        prev_action = self.actions[start_idx - 1][env_idx]
                    else:
                        prev_action = np.zeros_like(self.actions[item_idx][env_idx])
                else:
                    prev_action = self.actions[item_idx - 1][env_idx]
                prev_action_seq.append(prev_action)
            reward_seq = [
                self.rewards[start_idx + i][env_idx] for i in range(self.sequence_length)
            ]
            terminal_seq = [
                self.terminals[start_idx + i][env_idx]
                for i in range(self.sequence_length)
            ]
            is_last_seq = [
                self.episode_ends[start_idx + i][env_idx]
                for i in range(self.sequence_length)
            ]
            is_first_seq = [
                self.episode_starts[start_idx + i][env_idx]
                for i in range(self.sequence_length)
            ]
            context_stochastic_seq = [
                self.context_stochastics[start_idx + i][env_idx]
                for i in range(self.sequence_length)
            ]
            context_recurrent_seq = [
                self.context_recurrents[start_idx + i][env_idx]
                for i in range(self.sequence_length)
            ]
            if force_first_reset:
                is_first_seq[0] = True

            obs_sequences.append(obs_seq)
            action_sequences.append(action_seq)
            prev_action_sequences.append(prev_action_seq)
            reward_sequences.append(reward_seq)
            terminal_sequences.append(terminal_seq)
            is_last_sequences.append(is_last_seq)
            is_first_sequences.append(is_first_seq)
            context_stochastic_sequences.append(context_stochastic_seq)
            context_recurrent_sequences.append(context_recurrent_seq)
            oldest_abs = self._total_rows_committed - len(self.observations)
            replay_indices.append((int(oldest_abs + start_idx), int(env_idx)))

        return {
            "obs": obs_sequences,
            "actions": action_sequences,
            "prev_actions": prev_action_sequences,
            "rewards": reward_sequences,
            "terminals": terminal_sequences,
            "is_last": is_last_sequences,
            "is_first": is_first_sequences,
            "context_stochastic": context_stochastic_sequences,
            "context_recurrent": context_recurrent_sequences,
            "replay_indices": replay_indices,
        }

    def _sample_uniform(self, batch_size):
        """Sample fixed-length same-worker sequences from the whole buffer."""
        time_len = len(self.observations)
        if (
            batch_size <= 0
            or len(self) < self.config.min_buffer_size_before_training
            or time_len < self.sequence_length
        ):
            return None

        max_start = time_len - self.sequence_length + 1
        start_indices = [
            int(self._rng.integers(0, max_start)) for _ in range(batch_size)
        ]
        env_indices = [
            int(self._rng.integers(0, self.num_workers)) for _ in range(batch_size)
        ]

        return self._build_sequence_batch(
            start_indices, env_indices, force_first_reset=False
        )

    def _sample_online(self, batch_size):
        """Pop fresh complete chunks once before falling back to uniform replay."""
        time_len = len(self.observations)
        if batch_size <= 0 or time_len < self.sequence_length:
            return None

        oldest_abs = self._total_rows_committed - time_len

        starts = []
        envs = []
        while self._online_queue and len(starts) < batch_size:
            start_abs, env_idx = self._online_queue.popleft()
            start_idx = start_abs - oldest_abs
            if start_idx < 0 or start_idx + self.sequence_length > time_len:
                continue
            starts.append(start_idx)
            envs.append(env_idx)

        if not starts:
            return None
        return self._build_sequence_batch(starts, envs, force_first_reset=False)

    def _merge_batches(self, batches):
        batches = [batch for batch in batches if batch is not None and batch["obs"]]
        if not batches:
            return None
        merged = {}
        for key in batches[0].keys():
            merged[key] = []
            for batch in batches:
                merged[key].extend(batch[key])
        return merged

    def sample(self, batch_size=None):
        """Sample a uniform batch for callers that do not need online mixing."""
        return self._sample_uniform(batch_size or self.batch_size)

    def sample_mixed(self):
        """Sample fresh online chunks first, then fill the batch with uniform replay."""
        online_batch = self._sample_online(self.batch_size)
        online_count = len(online_batch["obs"]) if online_batch is not None else 0
        replay_batch = self._sample_uniform(self.batch_size - online_count)
        return self._merge_batches([online_batch, replay_batch])

    def update_contexts(self, batch_data, latents):
        """Refresh post-prefix RSSM entries by stable absolute replay row ID."""
        if batch_data is None or "replay_indices" not in batch_data:
            return
        stochastic_states, recurrent_states = (
            latents.detach()
            .cpu()
            .split(
                [self.config.stochastic_size, self.config.recurrent_units],
                dim=-1,
            )
        )
        stochastic_states = stochastic_states.view(
            *stochastic_states.shape[:-1],
            self.config.stochastic_units,
            self.config.discrete_classes,
        ).numpy()
        recurrent_states = recurrent_states.numpy()

        time_len = len(self.observations)
        oldest_abs = self._total_rows_committed - time_len
        for batch_idx, (start_abs, env_idx) in enumerate(batch_data["replay_indices"]):
            for offset in range(self.batch_length):
                target_abs = start_abs + self.replay_context + offset
                target_idx = target_abs - oldest_abs
                if target_idx < 0 or target_idx >= time_len:
                    continue
                self.context_stochastics[target_idx][env_idx] = stochastic_states[
                    batch_idx, offset
                ].copy()
                self.context_recurrents[target_idx][env_idx] = recurrent_states[
                    batch_idx, offset
                ].copy()

    def __len__(self):
        return len(self.observations) * self.num_workers + len(
            self._pending_observations
        )

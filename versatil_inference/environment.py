"""UR3 block-pushing environment manager for VersatIL policy evaluation."""

import csv
import datetime
import gc
import logging
import math
from collections import Counter
from pathlib import Path

import numpy as np
from tso_robotics_sockets import InferenceResponseKey, ServerStatus
from versatil_constants.ur3 import UR3ProprioKey

from ur3_sim import UR3BlockPushEnv
from versatil_inference.episode_recorder import EpisodeRecorder
from versatil_inference.socket_flags import (
    DEFAULT_CLIENT_NAME,
    MAX_STEPS,
    NO_OP_ACTION,
    UR3TrajectoryColumn,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

UR3_TASK_NAME = "ur3_blockpush"


class Environment:
    """Manages batched UR3 block-pushing environments."""

    def __init__(
        self,
        seed: int,
        num_trials: int,
        output_folder: str,
        max_parallel_envs: int = 10,
        record_video: bool = False,
        normalize_io: bool = False,
        stats_path: str | None = None,
    ):
        self.seed = seed
        self.num_trials = num_trials
        self.num_envs = num_trials
        self.output_folder = output_folder
        self.max_parallel_envs = max_parallel_envs
        self.record_video = record_video
        self.normalize_io = normalize_io
        self.stats_path = stats_path
        self.current_status = ServerStatus.CREATING_ENV.value
        self.client_name = DEFAULT_CLIENT_NAME
        self._rollout_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._episode_seeds: list[int] = [seed + i for i in range(num_trials)]

        self.active_environments = [False] * num_trials
        self.steps_counts = [0] * num_trials
        self.number_of_resets = [0] * num_trials
        self.environments_rewards = [0.0] * num_trials
        self.environments_p1 = [0.0] * num_trials
        self.environments_p2 = [0.0] * num_trials
        self.environments_behavior_order = ["none"] * num_trials
        self._episode_rewards: list[list[float]] = [[] for _ in range(num_trials)]
        self.recorders: list[EpisodeRecorder | None] = [None] * num_trials
        self.trajectory_columns = [col.value for col in UR3TrajectoryColumn]

        self._batch_global_indices: list[int] = []
        self._batch_envs: list[UR3BlockPushEnv] = []
        self.latest_observation: dict[int, dict] = {}
        self.recently_reset_indices: list[int] = []
        self._last_actions: dict[int, np.ndarray] = {}

        logging.info(
            f"UR3 evaluation: {num_trials} trials, "
            f"seeds=[{self._episode_seeds[0]}..{self._episode_seeds[-1]}], "
            f"max {max_parallel_envs} parallel, record_video={record_video}, "
            f"normalize_io={normalize_io}"
        )

    @property
    def rollout_directory(self) -> Path:
        client_path = Path(self.client_name)
        if self.output_folder:
            safe_client_name = self.client_name.strip("/").replace("/", "_")
            return (
                Path(self.output_folder)
                / safe_client_name
                / UR3_TASK_NAME
                / self._rollout_date
            )
        return (
            client_path.parent
            / "rollouts"
            / client_path.name
            / UR3_TASK_NAME
            / self._rollout_date
        )

    def initialize(self) -> None:
        batch_size = min(self.max_parallel_envs, self.num_envs)
        self._batch_global_indices = list(range(batch_size))
        self._create_batch_environments()
        self.current_status = ServerStatus.WAITING_ACTION.value

    def get_latest_observation(self) -> dict[int, dict]:
        return self.latest_observation

    def consume_reset_indices(self) -> list[int]:
        indices = self.recently_reset_indices
        self.recently_reset_indices = []
        return indices

    def step(self, actions: dict[int, list[float]]) -> None:
        if self.current_status == ServerStatus.FINISHED.value:
            return

        rollout_directory = self.rollout_directory
        self.recently_reset_indices = []
        new_latest_observation: dict[int, dict] = {}

        for local_index, global_index in enumerate(self._batch_global_indices):
            if not self.active_environments[global_index]:
                continue

            env = self._batch_envs[local_index]
            action = self._action_for_env(local_index, actions)
            gym_state, reward, done, info = env.step(action)
            self.steps_counts[global_index] += 1
            step_count = self.steps_counts[global_index]
            reward = float(reward)
            self._episode_rewards[global_index].append(reward)
            self._last_actions[global_index] = action

            full_obs = self._build_full_obs(gym_state=gym_state, step_count=step_count)
            frame = self._render_frame(env)
            self.recorders[global_index].add_observation(
                frame=frame,
                trajectory_row=self._build_trajectory_row(full_obs, action),
                reward=reward,
                output_directory=rollout_directory,
            )

            terminated = bool(done) or step_count >= MAX_STEPS
            if terminated:
                self._finalize_episode(
                    global_index=global_index,
                    step_count=step_count,
                    rollout_directory=rollout_directory,
                    completed_tasks=info.get("all_completions_ids", []),
                )
            else:
                new_latest_observation[local_index] = full_obs

        self.latest_observation = new_latest_observation
        self._advance_status_after_step()

    def close(self) -> None:
        for env in self._batch_envs:
            try:
                env.close()
            except Exception:
                logging.exception("Failed to close UR3 environment")
        self._batch_envs = []

    def _create_batch_environments(self) -> None:
        batch_size = len(self._batch_global_indices)
        logging.info(
            f"Creating batch: trials {self._batch_global_indices[0]}-"
            f"{self._batch_global_indices[-1]} ({batch_size} envs)"
        )
        self._batch_envs = []
        rollout_directory = self.rollout_directory
        new_observation: dict[int, dict] = {}

        for local_index, global_index in enumerate(self._batch_global_indices):
            episode_seed = self._episode_seeds[global_index]
            env = UR3BlockPushEnv(
                stats_path=self.stats_path,
                normalize_io=self.normalize_io,
                max_episode_steps=MAX_STEPS,
            )
            env.seed(episode_seed)
            gym_state = env.reset()
            self._batch_envs.append(env)

            self.active_environments[global_index] = True
            self.steps_counts[global_index] = 0
            self._episode_rewards[global_index] = []
            self.recorders[global_index] = EpisodeRecorder(
                environment_id=UR3_TASK_NAME,
                language_instruction=f"seed_{episode_seed}",
                trajectory_columns=self.trajectory_columns,
            )
            initial_action = np.asarray(gym_state[:2], dtype=np.float64)
            self._last_actions[global_index] = initial_action
            full_obs = self._build_full_obs(gym_state=gym_state, step_count=0)
            self.recorders[global_index].add_observation(
                frame=self._render_frame(env),
                trajectory_row=self._build_trajectory_row(full_obs, initial_action),
                reward=0.0,
                output_directory=rollout_directory,
            )
            new_observation[local_index] = full_obs

        self.latest_observation = new_observation
        self.recently_reset_indices = list(range(batch_size))

    def _advance_to_next_batch(self) -> bool:
        self.close()
        gc.collect()
        next_start = self._batch_global_indices[-1] + 1
        if next_start >= self.num_envs:
            return False
        end = min(next_start + self.max_parallel_envs, self.num_envs)
        self._batch_global_indices = list(range(next_start, end))
        self._create_batch_environments()
        return True

    def _advance_status_after_step(self) -> None:
        batch_active = any(
            self.active_environments[gi] for gi in self._batch_global_indices
        )
        if batch_active:
            self.current_status = ServerStatus.WAITING_ACTION.value
            return
        if self._advance_to_next_batch():
            self.current_status = ServerStatus.WAITING_ACTION.value
            return
        self._write_results_csv()
        self.current_status = ServerStatus.FINISHED.value

    def _finalize_episode(
        self,
        global_index: int,
        step_count: int,
        rollout_directory: Path,
        completed_tasks: list[str],
    ) -> None:
        unique_tasks = set(completed_tasks)
        reward = float(sum(self._episode_rewards[global_index]))
        p1 = 1.0 if "1" in unique_tasks else 0.0
        p2 = 1.0 if "2" in unique_tasks else 0.0
        behavior_order = self._behavior_order(completed_tasks)

        self.environments_rewards[global_index] = reward
        self.environments_p1[global_index] = p1
        self.environments_p2[global_index] = p2
        self.environments_behavior_order[global_index] = behavior_order
        self.number_of_resets[global_index] += 1

        logging.info(
            f"Trial {global_index} "
            f"(seed={self._episode_seeds[global_index]}): "
            f"done, reward={reward:.3f}, "
            f"p1={int(p1)}, p2={int(p2)}, "
            f"behavior_order={behavior_order}, steps={step_count}"
        )
        self.recorders[global_index].save(
            reward=reward,
            p1=p1,
            p2=p2,
            behavior_order=behavior_order,
            output_directory=rollout_directory,
        )
        self.active_environments[global_index] = False
        self.recorders[global_index] = None

    @staticmethod
    def _behavior_order(completed_tasks: list[str]) -> str:
        if not completed_tasks:
            return "none"
        if len(completed_tasks) == 1:
            return completed_tasks[0]
        return "->".join(completed_tasks[:2])

    def behavior_order_counts(self) -> Counter:
        return Counter(self.environments_behavior_order)

    def behavior_order_entropy(self) -> float:
        counts = Counter(
            label
            for label in self.environments_behavior_order
            if label in {"1->2", "2->1"}
        )
        total = sum(counts.values())
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in counts.values():
            probability = count / total
            entropy -= probability * math.log(probability)
        return float(math.exp(entropy))

    def _action_for_env(
        self,
        local_index: int,
        actions: dict[int, list[float]],
    ) -> np.ndarray:
        if local_index in actions:
            return np.asarray(actions[local_index], dtype=np.float64).reshape(-1)[:2]
        latest_obs = self.latest_observation.get(local_index, {})
        if UR3ProprioKey.EE_POS.value in latest_obs:
            return np.asarray(latest_obs[UR3ProprioKey.EE_POS.value], dtype=np.float64)
        return np.asarray(NO_OP_ACTION, dtype=np.float64)

    def _render_frame(self, env: UR3BlockPushEnv) -> np.ndarray | None:
        if not self.record_video:
            return None
        try:
            return env.render(mode="rgb_array")
        except TypeError:
            return env.render("rgb_array")
        except Exception:
            logging.exception("UR3 render failed")
            return None

    def _build_full_obs(self, gym_state: np.ndarray, step_count: int) -> dict:
        state = np.asarray(gym_state, dtype=np.float32).reshape(-1)
        if state.size < 6:
            raise ValueError(f"UR3 state must have at least 6 values, got {state}")
        return {
            UR3ProprioKey.EE_POS.value: state[0:2],
            UR3ProprioKey.BLOCK1_POS.value: state[2:4],
            UR3ProprioKey.BLOCK2_POS.value: state[4:6],
            InferenceResponseKey.TIMESTEP.value: step_count,
        }

    def _build_trajectory_row(
        self,
        full_obs: dict,
        action: np.ndarray,
    ) -> dict[str, float]:
        ee_pos = full_obs[UR3ProprioKey.EE_POS.value]
        block1_pos = full_obs[UR3ProprioKey.BLOCK1_POS.value]
        block2_pos = full_obs[UR3ProprioKey.BLOCK2_POS.value]
        return {
            UR3TrajectoryColumn.EE_POS_X.value: float(ee_pos[0]),
            UR3TrajectoryColumn.EE_POS_Y.value: float(ee_pos[1]),
            UR3TrajectoryColumn.BLOCK1_POS_X.value: float(block1_pos[0]),
            UR3TrajectoryColumn.BLOCK1_POS_Y.value: float(block1_pos[1]),
            UR3TrajectoryColumn.BLOCK2_POS_X.value: float(block2_pos[0]),
            UR3TrajectoryColumn.BLOCK2_POS_Y.value: float(block2_pos[1]),
            UR3TrajectoryColumn.ACTION_X.value: float(action[0]),
            UR3TrajectoryColumn.ACTION_Y.value: float(action[1]),
        }

    def _write_results_csv(self) -> None:
        output_directory = self.rollout_directory
        output_directory.mkdir(parents=True, exist_ok=True)
        csv_path = output_directory / "results.csv"

        total_trials = sum(self.number_of_resets)
        mean_reward = (
            sum(self.environments_rewards) / total_trials if total_trials > 0 else 0.0
        )
        mean_p1 = sum(self.environments_p1) / total_trials if total_trials > 0 else 0.0
        mean_p2 = sum(self.environments_p2) / total_trials if total_trials > 0 else 0.0
        mean_goals = mean_p1 + mean_p2
        behavior_entropy = self.behavior_order_entropy()

        with open(csv_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "trial",
                    "seed",
                    "reward",
                    "p1",
                    "p2",
                    "goals",
                    "behavior_order",
                ]
            )
            for i in range(self.num_envs):
                p1 = self.environments_p1[i]
                p2 = self.environments_p2[i]
                writer.writerow(
                    [
                        i,
                        self._episode_seeds[i],
                        f"{self.environments_rewards[i]:.4f}",
                        f"{p1:.1f}",
                        f"{p2:.1f}",
                        f"{p1 + p2:.1f}",
                        self.environments_behavior_order[i],
                    ]
                )
            writer.writerow([])
            writer.writerow(
                [
                    "mean",
                    f"{total_trials}",
                    f"{mean_reward:.4f}",
                    f"{mean_p1:.4f}",
                    f"{mean_p2:.4f}",
                    f"{mean_goals:.4f}",
                    f"behavior_order_entropy={behavior_entropy:.4f}",
                ]
            )
            writer.writerow([])
            writer.writerow(["behavior_order", "count"])
            for label, count in sorted(self.behavior_order_counts().items()):
                writer.writerow([label, count])
        logging.info(f"Results saved to {csv_path}")

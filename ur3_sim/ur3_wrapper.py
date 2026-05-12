"""UR3 block-pushing simulator adapted to raw VersatIL I/O.

The simulator setup follows the VQ-BeT UR3 environment, which traces back to
Kim et al., "Automating Reinforcement Learning with Example-Based Resets".
The only default behavioral change is that observations/actions are raw
simulator coordinates, because VersatIL checkpoints normalize incoming
observations and unnormalize predicted actions before sending them over the
socket.
"""

import logging
from collections import OrderedDict
from pathlib import Path

import gym
import gym_custom
import numpy as np
from gym_custom import spaces
from gym_custom.envs.custom.ur_utils import NullObjectiveBase
from gym_custom.envs.custom.ur_utils import URScriptWrapper_SingleUR3

from ur3_sim.normalizer import IdentityNormalizer, MinMaxNormalizer

logger = logging.getLogger(__name__)


class UprightConstraint(NullObjectiveBase):
    """Keep the UR3 end effector upright during IK."""

    def __init__(self) -> None:
        pass

    def _evaluate(self, so3: np.ndarray) -> float:
        axis_desired = np.array([0, 0, -1])
        axis_current = so3[:, 2]
        return float(1.0 - np.dot(axis_current, axis_desired))


class UR3BlockPushEnv(gym.Wrapper):
    """UR3 two-block pushing environment from the VQ-BeT UR3 setup."""

    def __init__(
        self,
        stats_path: str | None = None,
        goal_cond: bool = False,
        normalize_io: bool = False,
        max_episode_steps: int = 1000,
    ):
        super().__init__(gym_custom.make("single-ur3-xy-left-comb-larr-for-train-v0"))
        if normalize_io:
            if stats_path is None:
                raise ValueError("stats_path is required when normalize_io=True")
            self.normalizer = MinMaxNormalizer()
            self.normalizer.load_stats(Path(stats_path))
        else:
            self.normalizer = IdentityNormalizer()

        self.servoj_args, self.speedj_args = (
            {"t": None, "wait": None},
            {"a": 5, "t": None, "wait": None},
        )
        self.pid_gains = {
            "servoj": {"P": 1.0, "I": 0.5, "D": 0.2},
            "speedj": {"P": 0.20, "I": 10.0},
        }
        self.ur3_scale_factor = np.array([5, 5, 5, 5, 5, 5])
        self.gripper_scale_factor = np.array([1.0])
        self.env = URScriptWrapper_SingleUR3(
            self.env,
            self.pid_gains,
            self.ur3_scale_factor,
            self.gripper_scale_factor,
        )
        self.max_episode_steps = max_episode_steps
        self.command_limits = {
            "movej": [np.array([-0.04, -0.04, 0]), np.array([0.04, 0.04, 0])]
        }
        self.action_space = self._set_action_space()["movej"]
        self.env.wrapper_right.ur3_scale_factor[:6] = [
            24.52907494,
            24.02851783,
            25.56517597,
            14.51868608,
            23.78797503,
            21.61325463,
        ]
        self.null_obj_func = UprightConstraint()
        self.state: np.ndarray | None = None
        self.absolute_pos = True
        self.completed_tasks: list[str] = []
        self.goal_1 = np.array([0.0, -0.25])
        self.goal_2 = np.array([0.0, -0.40])
        self.goal_cond = goal_cond

    def convert_action_to_space(self, action_limits):
        if isinstance(action_limits, dict):
            return spaces.Dict(
                OrderedDict(
                    [
                        (key, self.convert_action_to_space(value))
                        for key, value in self.command_limits.items()
                    ]
                )
            )
        if isinstance(action_limits, list):
            low = action_limits[0]
            high = action_limits[1]
            return gym_custom.spaces.Box(low, high, dtype=action_limits[0].dtype)
        raise NotImplementedError(type(action_limits), action_limits)

    def _set_action_space(self):
        return self.convert_action_to_space({"_": self.command_limits})

    def seed(self, seed: int | None = None):
        if seed is not None:
            np.random.seed(seed)
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        return [seed]

    def set_task_goal(self, task_goal: np.ndarray) -> None:
        self.goal_1 = task_goal[2:4]
        self.goal_2 = task_goal[4:]

    def reset(self, *args, seed: int | None = None, **kwargs):
        if seed is not None:
            self.seed(seed)
        self.env.reset(*args, **kwargs)
        self.done = False
        self.goal1_achieved = False
        self.goal2_achieved = False
        self.episode_steps = 0
        self.dt = 1
        self.state = np.array([0.45, -0.325, 0.3, -0.25, 0.3, -0.40])
        self.completed_tasks = []
        self.goal_1 = np.array([0.0, -0.25])
        self.goal_2 = np.array([0.0, -0.40])
        return self._format_state(self.state.copy())

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size == 3:
            logger.info("Overwriting end effector height to 0.8")
            action = action[:2]
        if action.size != 2:
            raise ValueError(f"UR3 action must be 2D or 3D, got shape {action.shape}")

        action_xy = self.normalizer.unnormalize_action(action)
        action_xyz = np.concatenate([action_xy.squeeze(), [0.8]])
        q_right_des, _, _, _ = self.env.inverse_kinematics_ee(
            action_xyz,
            self.null_obj_func,
            arm="right",
        )
        qvel_right = (q_right_des - self.env.get_obs_dict()["right"]["qpos"]) / self.dt

        next_state, _, done, _ = self.env.step(
            {
                "right": {
                    "speedj": {
                        "qd": qvel_right,
                        "a": self.speedj_args["a"],
                        "t": self.speedj_args["t"],
                        "wait": self.speedj_args["wait"],
                    },
                    "move_gripper_force": {"gf": np.array([15.0])},
                }
            }
        )

        self.episode_steps += 1

        reward = 0
        if not self.goal1_achieved:
            self.goal1_achieved = self.check_goal1_achieved(next_state)
            if self.goal1_achieved:
                reward = 1
                self.completed_tasks.append("1")
        if not self.goal2_achieved:
            self.goal2_achieved = self.check_goal2_achieved(next_state)
            if self.goal2_achieved:
                reward = 1
                self.completed_tasks.append("2")
        done = (self.episode_steps >= self.max_episode_steps) or (
            self.goal1_achieved and self.goal2_achieved
        )
        self.state = np.asarray(next_state[:6], dtype=np.float64)

        if self.goal_cond:
            reward = self.calc_reward(next_state) if done else 0
        info = {"all_completions_ids": list(self.completed_tasks)}
        return self._format_state(self.state.copy()), reward, done, info

    def _format_state(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(self.normalizer.normalize_state(state), dtype=np.float32)

    def calc_reward(self, next_state: np.ndarray) -> float:
        block1_dist = np.linalg.norm(self.goal_1 - next_state[2:4], ord=1)
        block2_dist = np.linalg.norm(self.goal_2 - next_state[4:6], ord=1)
        return float(-(block1_dist + block2_dist))

    def check_goal1_achieved(self, next_state: np.ndarray) -> bool:
        return bool(np.linalg.norm(self.goal_1 - next_state[2:4]) < 0.05)

    def check_goal2_achieved(self, next_state: np.ndarray) -> bool:
        return bool(np.linalg.norm(self.goal_2 - next_state[4:6]) < 0.05)

    def __getattr__(self, attrname: str):
        return getattr(self.env, attrname)

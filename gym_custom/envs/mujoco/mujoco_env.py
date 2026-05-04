import os
from collections import OrderedDict
from os import path

import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from mujoco import MjData, MjModel

import gym_custom
from gym_custom import spaces
from gym_custom.utils import seeding

DEFAULT_SIZE = 500  # 1000


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(
            OrderedDict(
                [
                    (key, convert_observation_to_space(value))
                    for key, value in observation.items()
                ]
            )
        )
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float("inf"), dtype=np.float32)
        high = np.full(observation.shape, float("inf"), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class MujocoEnv(gym_custom.Env):
    def __init__(self, model_path, frame_skip):
        # Get full path to the model file
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)

        if not path.exists(fullpath):
            raise IOError(f"File {fullpath} does not exist")

        # Load the model and create the simulation
        self.frame_skip = frame_skip
        self.model = MjModel.from_xml_path(fullpath)
        self.data = MjData(self.model)
        self.sim = self.data  # Initialize `self.sim` here
        self.viewer = None
        self._viewers = {}
        self.renderer = None
        self.metadata = {
            "render.modes": ["human", "rgb_array", "depth_array"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        self._set_action_space()

        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)

        self.seed()

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset_model(self):
        """Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass."""
        raise NotImplementedError

    def viewer_setup(self):
        """This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth."""
        pass

    def reset(self):
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
        self.sim_forward()
        return self.reset_model()

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.sim_forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim_step()

    def render(
        self,
        mode="human",
        width=DEFAULT_SIZE,
        height=DEFAULT_SIZE,
        camera_id=None,
        camera_name=None,
    ):
        if self.renderer is None:
            self.renderer = MujocoRenderer(
                model=self.model, data=self.data, width=width, height=height
            )
        return self.renderer.render(render_mode=mode)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def sim_step(self):
        self.data.time += self.dt
        mujoco.mj_step(self.model, self.data)

    def sim_forward(self):
        mujoco.mj_forward(self.model, self.data)

    def get_body_com(self, body_name):
        body_id = self.model.body_name2id(body_name)
        return self.data.xpos[body_id]

    def state_vector(self):
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

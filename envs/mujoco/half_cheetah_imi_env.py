import pickle

import akro
import numpy as np
from matplotlib.patches import Ellipse

from envs.mujoco.ant_env import AntEnv
from envs.mujoco.half_cheetah_env import HalfCheetahEnv
from envs.mujoco.mujoco_utils import convert_observation_to_space
from gym import utils


class HalfCheetahImiEnv(HalfCheetahEnv):
    def __init__(
            self,
            max_path_length,
            num_skips=10,
            num_frames=20,
            **kwargs,
    ):
        self.max_path_length = max_path_length
        self.num_skips = num_skips
        self.num_frames = num_frames

        with open('data/hci_trajs.pkl', 'rb') as f:
            self.trajs = pickle.load(f)

        self.cur_traj_idx = None
        self.num_steps = 0

        info_idxs = np.linspace(0, 1, self.num_frames + 1)[1:]
        self.info_idxs = [int(i * 20) for i in info_idxs]

        self.len_info = num_frames
        self.cur_info = np.zeros(self.len_info)

        super().__init__(**kwargs)
        utils.EzPickle.__init__(self, max_path_length=max_path_length, num_skips=num_skips, num_frames=num_frames, **kwargs)

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        low = np.full((self.len_info,), -float('inf'), dtype=np.float32)
        high = np.full((self.len_info,), float('inf'), dtype=np.float32)
        return akro.concat(self.observation_space, akro.Box(low=low, high=high, dtype=self.observation_space.dtype))

    def reset(self, **kwargs):
        if 'goal' in kwargs:
            raise Exception('Unknown situation')
        return super().reset()

    def reset_model(self):
        self.cur_traj_idx = np.random.randint(0, len(self.trajs))
        self.num_steps = 0
        cur_traj = self.trajs[self.cur_traj_idx]
        cur_info = cur_traj[self.info_idxs]
        cur_info = cur_info[:, [0]]
        self.cur_info = cur_info.reshape(-1)

        return super().reset_model()

    def _get_obs(self):
        obs = super()._get_obs()
        obs = np.concatenate([obs, self.cur_info])

        return obs

    def _get_done(self):
        return self.num_steps == self.max_path_length

    def compute_reward(self, xposbefore, xposafter):
        self.num_steps += 1
        if self.num_steps % self.num_skips == 0 and (self.num_steps // self.num_skips) in self.info_idxs:
            cur_idx = self.num_steps // self.num_skips
            delta = np.linalg.norm(self.trajs[self.cur_traj_idx][cur_idx][0] - super()._get_obs()[0])

            reward = -(delta ** 2) / len(self.info_idxs) / 30
        else:
            reward = 0

        return reward

    def plot_trajectory(self, trajectory, color, ax):
        # XXX: Hacky
        ellipse = Ellipse(xy=trajectory[-1], width=0.2, height=0.2,
                          edgecolor=color, lw=1, facecolor=color, alpha=0.8)
        ax.add_patch(ellipse)

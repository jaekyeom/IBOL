import akro
import numpy as np
from matplotlib.patches import Ellipse

from envs.mujoco.ant_env import AntEnv
from envs.mujoco.mujoco_utils import convert_observation_to_space
from gym import utils


class AntGoalEnv(AntEnv):
    def __init__(
            self,
            max_path_length,
            goal_range=50.0,
            **kwargs,
    ):
        self.max_path_length = max_path_length

        self.goal_range = goal_range
        self.cur_goal = np.random.uniform(-self.goal_range, self.goal_range, (2,))
        self.cur_transient_goal = None
        self.num_steps = 0

        super().__init__(**kwargs)
        utils.EzPickle.__init__(self, max_path_length=max_path_length, goal_range=goal_range, **kwargs)

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        low = np.full((2,), -float('inf'), dtype=np.float32)
        high = np.full((2,), float('inf'), dtype=np.float32)
        return akro.concat(self.observation_space, akro.Box(low=low, high=high, dtype=self.observation_space.dtype))

    def reset(self, **kwargs):
        if 'goal' in kwargs:
            self.cur_transient_goal = kwargs['goal']
        return super().reset()

    def reset_model(self):
        if self.cur_transient_goal is not None:
            self.cur_goal = self.cur_transient_goal
            self.cur_transient_goal = None
        else:
            self.cur_goal = np.random.uniform(-self.goal_range, self.goal_range, (2,))
        self.num_steps = 0

        return super().reset_model()

    def _get_obs(self):
        obs = super()._get_obs()
        obs = np.concatenate([obs, self.cur_goal])

        return obs

    def _get_done(self):
        return self.num_steps == self.max_path_length

    def compute_reward(self, xposbefore, yposbefore, xposafter, yposafter):
        self.num_steps += 1
        delta = np.linalg.norm(self.cur_goal - np.array([xposafter, yposafter]))
        if self.num_steps == self.max_path_length:
            reward = -delta
        else:
            reward = 0.

        return reward

    def plot_trajectory(self, trajectory, color, ax):
        # XXX: Hacky
        ellipse = Ellipse(xy=trajectory[-1], width=0.2, height=0.2,
                          edgecolor=color, lw=1, facecolor=color, alpha=0.8)
        ax.add_patch(ellipse)

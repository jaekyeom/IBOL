# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import os

from gym import utils
import numpy as np
from gym.envs.mujoco import mujoco_env

from envs.mujoco.mujoco_utils import MujocoTrait


class HalfCheetahEnv(MujocoTrait, mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self,
                 expose_obs_idxs=None,
                 expose_all_qpos=True,
                 task='default',
                 target_velocity=None,
                 model_path=None,
                 fixed_initial_state=False):
        utils.EzPickle.__init__(**locals())

        if model_path is None:
            model_path = 'half_cheetah.xml'

        # Settings from
        # https://github.com/openai/gym/blob/master/gym/envs/__init__.py
        self._expose_obs_idxs = expose_obs_idxs
        self._expose_all_qpos = expose_all_qpos
        self._task = task
        self._target_velocity = target_velocity
        self.fixed_initial_state = fixed_initial_state

        xml_path = "envs/mujoco/assets/"
        model_path = os.path.abspath(os.path.join(xml_path, model_path))

        mujoco_env.MujocoEnv.__init__(
            self,
            model_path,
            5)

    def compute_reward(self, **kwargs):
        return None

    def step(self, action, render=False):
        obsbefore = self._get_obs()
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        obsafter = self._get_obs()
        xposafter = self.sim.data.qpos[0]
        xvelafter = self.sim.data.qvel[0]
        reward_ctrl = -0.1 * np.square(action).sum()

        reward = self.compute_reward(xposbefore=xposbefore, xposafter=xposafter)
        if reward is None:
            if self._task == 'default':
                reward_vel = 0.
                reward_run = (xposafter - xposbefore) / self.dt
                reward = reward_ctrl + reward_run
            elif self._task == 'target_velocity':
                reward_vel = -(self._target_velocity - xvelafter) ** 2
                reward = reward_ctrl + reward_vel
            elif self._task == 'run_back':
                reward_vel = 0.
                reward_run = (xposbefore - xposafter) / self.dt
                reward = reward_ctrl + reward_run

        done = False
        ob = self._get_obs()
        info = dict(
            # reward_run=reward_run,
            # reward_ctrl=reward_ctrl,
            # reward_vel=reward_vel,
            coordinates=np.array([xposbefore, 0.]),
            next_coordinates=np.array([xposafter, 0.]),
            ori_obs=obsbefore,
            next_ori_obs=obsafter,
        )

        if render:
            info['render'] = self.render(mode='rgb_array').transpose(2, 0, 1)

        return ob, reward, done, info

    def _get_obs(self):
        if self._expose_all_qpos:
            obs = np.concatenate(
                [self.sim.data.qpos.flat, self.sim.data.qvel.flat])
        else:
            obs = np.concatenate([
                self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat,
            ])

        if self._expose_obs_idxs is not None:
            obs = obs[self._expose_obs_idxs]

        return obs

    def reset_model(self):
        if self.fixed_initial_state:
            qpos = self.init_qpos
            qvel = self.init_qvel
        else:
            qpos = self.init_qpos + np.random.uniform(
                low=-.1, high=.1, size=self.sim.model.nq)
            qvel = self.init_qvel + np.random.randn(self.sim.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def plot_trajectory(self, trajectory, color, ax):
        from matplotlib.collections import LineCollection
        #linewidths = np.linspace(0.5, 1.5, len(trajectory))
        #linewidths = np.linspace(0.1, 0.8, len(trajectory))
        linewidths = np.linspace(0.2, 1.2, len(trajectory))
        points = np.reshape(trajectory, (-1, 1, 2))
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, linewidths=linewidths, color=color)
        ax.add_collection(lc)

    def _get_coordinates_trajectories(self, trajectories):
        coordinates_trajectories = super()._get_coordinates_trajectories(
                trajectories)
        for i, traj in enumerate(coordinates_trajectories):
            # Designed to fit in [-5, 5] * [-5, 5] roughly.
            traj[:, 1] = (i - len(coordinates_trajectories) / 2) / 25.0
        return coordinates_trajectories

    def calc_eval_metrics(self, trajectories, is_option_trajectories):
        eval_metrics = super().calc_eval_metrics(trajectories, is_option_trajectories)

        trajectory_eval_metrics = defaultdict(list)
        coordinates_trajectories = super()._get_coordinates_trajectories(
                trajectories)
        for trajectory, coordinates_trajectory in zip(trajectories, coordinates_trajectories):
            trajectory_eval_metrics['TotalMove'].append(
                    np.sum(np.abs(coordinates_trajectory[:, 0])))
            trajectory_eval_metrics['NetMove'].append(
                    np.abs(coordinates_trajectory[-1, 0] - coordinates_trajectory[0, 0]))
            trajectory_eval_metrics['FarthestMove'].append(
                    np.max(np.abs(coordinates_trajectory[:, 0])))
            trajectory_eval_metrics['FarthestPosDist'].append(
                    np.max(coordinates_trajectory[:, 0]))
            trajectory_eval_metrics['FarthestNegDist'].append(
                    np.min(coordinates_trajectory[:, 0]))
            trajectory_eval_metrics['LastPos'].append(
                    coordinates_trajectory[-1, 0])

        eval_metrics.update({
            'AvgTotalMove': np.mean(trajectory_eval_metrics['TotalMove']),
            'AvgNetMove': np.mean(trajectory_eval_metrics['NetMove']),
            'AvgFarthestMove': np.mean(trajectory_eval_metrics['FarthestMove']),
            'StdFarthestMove': np.std(trajectory_eval_metrics['FarthestMove']),
            'MaxFarthestMove': np.max(trajectory_eval_metrics['FarthestMove']),
            'MinFarthestMove': np.min(trajectory_eval_metrics['FarthestMove']),
            'AvgLastPos': np.mean(trajectory_eval_metrics['LastPos']),
            'StdLastPos': np.std(trajectory_eval_metrics['LastPos']),
            'MaxLastPos': np.max(trajectory_eval_metrics['LastPos']),
            'MinLastPos': np.min(trajectory_eval_metrics['LastPos']),
            'AvgFarthestPosDist': np.mean(trajectory_eval_metrics['FarthestPosDist']),
            'StdFarthestPosDist': np.std(trajectory_eval_metrics['FarthestPosDist']),
            'MaxFarthestPosDist': np.max(trajectory_eval_metrics['FarthestPosDist']),
            'AvgFarthestNegDist': np.mean(trajectory_eval_metrics['FarthestNegDist']),
            'StdFarthestNegDist': np.std(trajectory_eval_metrics['FarthestNegDist']),
            'MinFarthestNegDist': np.min(trajectory_eval_metrics['FarthestNegDist']),
        })
        eval_metrics.update({
            'DiffMaxMinLastPos': eval_metrics['MaxLastPos'] - eval_metrics['MinLastPos'],
            'DiffMaxFarthestPosDistMinFarthestNegDist': eval_metrics['MaxFarthestPosDist'] - eval_metrics['MinFarthestNegDist'],
        })
        eval_metrics.update({
            'LastPos': np.array(trajectory_eval_metrics['LastPos']),
        })
        return eval_metrics


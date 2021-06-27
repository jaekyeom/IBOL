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
import math
import os

from gym import utils
import numpy as np
from gym.envs.mujoco import mujoco_env

from envs.mujoco.mujoco_utils import MujocoTrait


def q_inv(a):
    return [a[0], -a[1], -a[2], -a[3]]


def q_mult(a, b):  # multiply two quaternion
    w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    i = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
    j = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
    k = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
    return [w, i, j, k]


# pylint: disable=missing-docstring
class AntEnv(MujocoTrait, mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self,
                 task="motion",
                 goal=None,
                 expose_obs_idxs=None,
                 expose_all_qpos=True,
                 expose_body_coms=None,
                 expose_body_comvels=None,
                 expose_foot_sensors=False,
                 use_alt_path=False,
                 model_path=None,
                 fixed_initial_state=False,
                 done_allowing_step_unit=None,
                 fixed_mpl=None,
                 original_env=False,
                 ):
        utils.EzPickle.__init__(**locals())

        if model_path is None:
            model_path = 'ant.xml'

        self._task = task
        self._goal = goal
        self._expose_obs_idxs = expose_obs_idxs
        self._expose_all_qpos = expose_all_qpos
        self._expose_body_coms = expose_body_coms
        self._expose_body_comvels = expose_body_comvels
        self._expose_foot_sensors = expose_foot_sensors
        self._body_com_indices = {}
        self._body_comvel_indices = {}
        self.fixed_initial_state = fixed_initial_state

        self._done_allowing_step_unit = done_allowing_step_unit
        self._fixed_mpl = fixed_mpl
        self._original_env = original_env

        # Settings from
        # https://github.com/openai/gym/blob/master/gym/envs/__init__.py

        xml_path = "envs/mujoco/assets/"
        model_path = os.path.abspath(os.path.join(xml_path, model_path))
        mujoco_env.MujocoEnv.__init__(self, model_path, 5)

    def compute_reward(self, **kwargs):
        return None

    def step(self, a, render=False):
        if hasattr(self, '_step_count'):
            self._step_count += 1

        obsbefore = self._get_obs()
        xposbefore = self.sim.data.qpos.flat[0]
        yposbefore = self.sim.data.qpos.flat[1]
        oribefore = self.get_ori()
        self.do_simulation(a, self.frame_skip)
        obsafter = self._get_obs()
        xposafter = self.sim.data.qpos.flat[0]
        yposafter = self.sim.data.qpos.flat[1]
        oriafter = self.get_ori()

        reward = self.compute_reward(xposbefore=xposbefore, yposbefore=yposbefore, xposafter=xposafter, yposafter=yposafter)
        if reward is None:
            forward_reward = (xposafter - xposbefore) / self.dt
            sideward_reward = (yposafter - yposbefore) / self.dt

            ctrl_cost = .5 * np.square(a).sum()
            survive_reward = 1.0
            if self._task == "forward":
                reward = forward_reward - ctrl_cost + survive_reward
            elif self._task == "backward":
                reward = -forward_reward - ctrl_cost + survive_reward
            elif self._task == "left":
                reward = sideward_reward - ctrl_cost + survive_reward
            elif self._task == "right":
                reward = -sideward_reward - ctrl_cost + survive_reward
            elif self._task == "goal":
                reward = -np.linalg.norm(np.array([xposafter, yposafter]) - self._goal)
            elif self._task == "motion":
                reward = np.max(np.abs(np.array([forward_reward, sideward_reward
                                                 ]))) - ctrl_cost + survive_reward

            def _get_gym_ant_reward():
                forward_reward = (xposafter - xposbefore)/self.dt
                ctrl_cost = .5 * np.square(a).sum()
                contact_cost = 0.5 * 1e-3 * np.sum(
                    np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
                survive_reward = 1.0
                reward = forward_reward - ctrl_cost - contact_cost + survive_reward
                return reward
            reward = _get_gym_ant_reward()


        #done = self._get_done()
        if self._fixed_mpl is None or self._fixed_mpl:
            done = False
        else:
            state = self.state_vector()
            notdone = (np.isfinite(state).all()
                       and state[2] >= 0.2 and state[2] <= 1.0)
            if hasattr(self, '_done_internally') and self._done_allowing_step_unit is not None:
                self._done_internally = (self._done_internally or (not notdone))
                done = (self._done_internally and self._step_count % self._done_allowing_step_unit == 0)
            else:
                done = (not notdone)

        ob = self._get_obs()
        info = dict(
            # reward_forward=forward_reward,
            # reward_sideward=sideward_reward,
            # reward_ctrl=-ctrl_cost,
            # reward_survive=survive_reward,
            coordinates=np.array([xposbefore, yposbefore]),
            next_coordinates=np.array([xposafter, yposafter]),
            ori=oribefore,
            next_ori=oriafter,
            ori_obs=obsbefore,
            next_ori_obs=obsafter,
        )

        if render:
            info['render'] = self.render(mode='rgb_array').transpose(2, 0, 1)

        return ob, reward, done, info

    def _get_obs(self):
        if self._original_env:
            return np.concatenate([
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ])


        # No crfc observation
        if self._expose_all_qpos:
            obs = np.concatenate([
                self.sim.data.qpos.flat[:15],
                self.sim.data.qvel.flat[:14],
            ])
        else:
            obs = np.concatenate([
                self.sim.data.qpos.flat[2:15],
                self.sim.data.qvel.flat[:14],
            ])

        if self._expose_body_coms is not None:
            for name in self._expose_body_coms:
                com = self.get_body_com(name)
                if name not in self._body_com_indices:
                    indices = range(len(obs), len(obs) + len(com))
                    self._body_com_indices[name] = indices
                obs = np.concatenate([obs, com])

        if self._expose_body_comvels is not None:
            for name in self._expose_body_comvels:
                comvel = self.get_body_comvel(name)
                if name not in self._body_comvel_indices:
                    indices = range(len(obs), len(obs) + len(comvel))
                    self._body_comvel_indices[name] = indices
                obs = np.concatenate([obs, comvel])

        if self._expose_foot_sensors:
            obs = np.concatenate([obs, self.sim.data.sensordata])

        if self._expose_obs_idxs is not None:
            obs = obs[self._expose_obs_idxs]

        return obs

    def _get_done(self):
        return False

    def reset_model(self):
        self._step_count = 0
        self._done_internally = False

        if self.fixed_initial_state:
            qpos = self.init_qpos
            qvel = self.init_qvel
        else:
            qpos = self.init_qpos + np.random.uniform(
                size=self.sim.model.nq, low=-.1, high=.1)
            qvel = self.init_qvel + np.random.randn(self.sim.model.nv) * .1

        if not self._original_env:
            qpos[15:] = self.init_qpos[15:]
            qvel[14:] = 0.

        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        # self.viewer.cam.distance = self.model.stat.extent * 2.5
        pass

    def get_ori(self):
        ori = [0, 1, 0, 0]
        rot = self.sim.data.qpos[3:7]  # take the quaternion
        ori = q_mult(q_mult(rot, ori), q_inv(rot))[1:3]  # project onto x-y plane
        return np.array(ori)
        # ori = math.atan2(ori[1], ori[0])
        # return ori

    @property
    def body_com_indices(self):
        return self._body_com_indices

    @property
    def body_comvel_indices(self):
        return self._body_comvel_indices

    def calc_eval_metrics(self, trajectories, is_option_trajectories):
        eval_metrics = super().calc_eval_metrics(trajectories, is_option_trajectories)

        trajectory_eval_metrics = defaultdict(list)
        coordinates_trajectories = super()._get_coordinates_trajectories(
                trajectories)
        for trajectory, coordinates_trajectory in zip(trajectories, coordinates_trajectories):
            moves = np.linalg.norm(
                    coordinates_trajectory[1:] - coordinates_trajectory[:-1],
                    axis=-1)
            trajectory_eval_metrics['TotalMove'].append(
                    np.sum(moves))
            trajectory_eval_metrics['NetMove'].append(
                    np.linalg.norm(coordinates_trajectory[-1] - coordinates_trajectory[0], axis=-1))
            trajectory_eval_metrics['LastPosX'].append(
                    coordinates_trajectory[-1, 0])
            trajectory_eval_metrics['LastPosY'].append(
                    coordinates_trajectory[-1, 1])
            trajectory_eval_metrics['LastDist'].append(
                np.linalg.norm([coordinates_trajectory[-1, 0], coordinates_trajectory[-1, 1]]))

        eval_metrics.update({
            'AvgTotalMove': np.mean(trajectory_eval_metrics['TotalMove']),
            'AvgNetMove': np.mean(trajectory_eval_metrics['NetMove']),
            'AvgLastPosX': np.mean(trajectory_eval_metrics['LastPosX']),
            'StdLastPosX': np.std(trajectory_eval_metrics['LastPosX']),
            'MaxLastPosX': np.max(trajectory_eval_metrics['LastPosX']),
            'MinLastPosX': np.min(trajectory_eval_metrics['LastPosX']),
            'AvgLastPosY': np.mean(trajectory_eval_metrics['LastPosY']),
            'StdLastPosY': np.std(trajectory_eval_metrics['LastPosY']),
            'MaxLastPosY': np.max(trajectory_eval_metrics['LastPosY']),
            'MinLastPosY': np.min(trajectory_eval_metrics['LastPosY']),
        })
        eval_metrics.update({
            'DiffMaxMinLastPosX': eval_metrics['MaxLastPosX'] - eval_metrics['MinLastPosX'],
            'DiffMaxMinLastPosY': eval_metrics['MaxLastPosY'] - eval_metrics['MinLastPosY'],
            'SumStdLastPosXY': eval_metrics['StdLastPosX'] + eval_metrics['StdLastPosY'],
        })
        eval_metrics.update({
            'SumDiffMaxMinLastPosXY': eval_metrics['DiffMaxMinLastPosX'] + eval_metrics['DiffMaxMinLastPosY'],
        })
        eval_metrics.update({
            'LastDist': trajectory_eval_metrics['LastDist'],
        })
        return eval_metrics


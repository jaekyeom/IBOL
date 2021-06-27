from collections import defaultdict

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from envs.mujoco.mujoco_utils import MujocoTrait


class HopperEnv(MujocoTrait, mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, num_action_repeats=None, fixed_initial_state=False):
        utils.EzPickle.__init__(**locals())

        self._num_action_repeats = num_action_repeats
        self.fixed_initial_state = fixed_initial_state

        mujoco_env.MujocoEnv.__init__(self, 'hopper.xml', 4)

    def compute_reward(self, **kwargs):
        return None

    def step(self, a, render=False):
        obsbefore = self._get_obs()
        posbefore = self.sim.data.qpos[0]
        if self._num_action_repeats is None:
            self.do_simulation(a, self.frame_skip)
        else:
            for i in range(self._num_action_repeats):
                self.do_simulation(a, self.frame_skip)
        obsafter = self._get_obs()
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = self.compute_reward(xposbefore=posbefore, xposafter=posafter)
        if reward is None:
            reward = (posafter - posbefore) / self.dt
            reward += alive_bonus
            reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = False
        ob = self._get_obs()
        info = dict(
            coordinates=np.array([posbefore, 0.]),
            next_coordinates=np.array([posafter, 0.]),
            ori_obs=obsbefore,
            next_ori_obs=obsafter,
        )
        if render:
            info['render'] = self.render(mode='rgb_array').transpose(2, 0, 1)
        return ob, reward, done, info

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        if self.fixed_initial_state:
            qpos = self.init_qpos
            qvel = self.init_qvel
        else:
            qpos = self.init_qpos + np.random.uniform(low=-.005, high=.005, size=self.model.nq)
            qvel = self.init_qvel + np.random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    def plot_trajectory(self, trajectory, color, ax):
        from matplotlib.collections import LineCollection
        linewidths = np.linspace(0.2, 1.2, len(trajectory))
        points = np.reshape(trajectory, (-1, 1, 2))
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, linewidths=linewidths, color=color)
        ax.add_collection(lc)

    def _get_coordinates_trajectories(self, trajectories):
        coordinates_trajectories = super()._get_coordinates_trajectories(
            trajectories)
        for i, traj in enumerate(coordinates_trajectories):
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

from collections import defaultdict

import akro
import gym.spaces.utils
import numpy as np
import torch

from garage.envs import EnvSpec
from garage.torch.distributions import TanhNormal

from iod.utils import get_torch_concat_obs


class ChildPolicyEnv(gym.Wrapper):
    def __init__(
            self,
            env,
            cp_dict,
            cp_action_range,
            cp_use_mean,
            cp_multi_step,
            cp_action_dims,
            cp_num_truncate_obs,
            cp_omit_obs_idxs=None,
    ):
        super().__init__(env)

        self.child_policy = cp_dict['policy']
        self.child_policy.eval()

        self.cp_obs_mean = cp_dict['normalizer_obs_mean']
        self.cp_obs_std = cp_dict['normalizer_obs_std']
        self.cp_dim_action_ori = cp_dict['dim_option']
        self.cp_action_range = cp_action_range
        assert self.cp_action_range is not None

        self.cp_use_mean = cp_use_mean
        self.cp_multi_step = cp_multi_step
        self.cp_action_dims = cp_action_dims
        self.cp_dim_action = self.cp_dim_action_ori if cp_action_dims is None else len(cp_action_dims)
        self.cp_num_truncate_obs = cp_num_truncate_obs
        self.cp_omit_obs_idxs = cp_omit_obs_idxs

        self.observation_space = self.env.observation_space
        self.action_space = akro.Box(low=-self.cp_action_range,
                                     high=self.cp_action_range,
                                     shape=(self.cp_dim_action,))

        self.last_obs = None

    @property
    def spec(self):
        return EnvSpec(action_space=self.action_space,
                       observation_space=self.observation_space)

    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)

        self.last_obs = ret

        return ret

    def step(self, cp_action, **kwargs):
        cp_action = cp_action.copy()
        sum_rewards = 0.
        acc_infos = defaultdict(list)

        if self.cp_action_dims is not None:
            cp_action_ori = np.zeros(self.cp_dim_action_ori)
            cp_action_ori[self.cp_action_dims] = cp_action
            cp_action = cp_action_ori

        done_final = False
        for i in range(self.cp_multi_step):
            cp_obs = self.last_obs
            cp_obs = torch.as_tensor(cp_obs)
            if self.cp_num_truncate_obs > 0:
                cp_obs = cp_obs[:-self.cp_num_truncate_obs]
            cp_obs = (cp_obs - self.cp_obs_mean) / self.cp_obs_std
            if self.cp_omit_obs_idxs is not None:
                cp_obs[self.cp_omit_obs_idxs] = 0

            cp_action = torch.as_tensor(cp_action)

            if hasattr(self.child_policy, '_option_info'):
                cp_input = get_torch_concat_obs(
                    cp_obs, cp_action,
                    self.child_policy._option_info['num_repeats'],
                    dim=0,
                ).float()
            else:
                # for backward compatibility
                cp_input = torch.cat([cp_obs, cp_action], dim=0).float()

            if self.cp_use_mean:
                # First try to use mode
                if hasattr(self.child_policy._module, 'forward_mode'):
                    # Beta
                    action = self.child_policy.get_mode_actions(cp_input.unsqueeze(dim=0))[0]
                else:
                    # Tanhgaussian
                    action_dist = self.child_policy(cp_input.unsqueeze(dim=0))[0]
                    action = action_dist.mean.detach().numpy()
            else:
                action_dist = self.child_policy(cp_input.unsqueeze(dim=0))[0]
                action = action_dist.sample().detach().numpy()
            action = action[0]
            lb, ub = self.env.action_space.low, self.env.action_space.high
            action = lb + (action + 1) * (0.5 * (ub - lb))
            action = np.clip(action, lb, ub)

            next_obs, reward, done, info = self.env.step(action, **kwargs)

            self.last_obs = next_obs

            sum_rewards += reward
            for k, v in info.items():
                acc_infos[k].append(v)

            if info.get('done_internal', False):
                done_final = True

            if done:
                done_final = True
                break

        infos = {}
        for k, v in acc_infos.items():
            if isinstance(v[0], np.ndarray):
                infos[k] = np.array(v)
            elif isinstance(v[0], tuple):
                infos[k] = np.array([list(l) for l in v])
            else:
                infos[k] = sum(v)

        return next_obs, sum_rewards, done_final, infos

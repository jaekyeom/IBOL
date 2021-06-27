import copy

import numpy as np
import torch

from iod.iod import IOD
from iod import sac_utils


class IODSAC(IOD):
    def __init__(
            self,
            *,
            qf1,
            qf2,
            log_alpha,
            tau,
            discount,
            scale_reward,
            replay_buffer,
            min_buffer_size,
            target_coef,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.qf1 = qf1.to(self.device)
        self.qf2 = qf2.to(self.device)

        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)

        self.log_alpha = log_alpha.to(self.device)

        self.param_modules.update(qf1=self.qf1, qf2=self.qf2)

        if scale_reward:
            self._reward_scale_factor = 1 / (self.alpha + 1e-12)
        else:
            self._reward_scale_factor = 1
        self._target_entropy = -np.prod(self._env_spec.action_space.shape).item() / 2. * target_coef

        self.tau = tau
        self.discount = discount

        self.replay_buffer = replay_buffer
        self.min_buffer_size = min_buffer_size

        if self.replay_buffer is not None:
            assert self._train_child_policy or self._train_downstream_policy
            assert self._trans_minibatch_size is not None

        if self._sp_use_lstm:
            raise NotImplementedError()

    def _train_once_inner(self, data):
        if self.replay_buffer is not None:
            # Add paths to the replay buffer
            for i in range(len(data['actions'])):
                path = {}
                for key in data.keys():
                    cur_list = data[key][i]
                    if isinstance(cur_list, torch.Tensor):
                        cur_list = cur_list.detach().cpu().numpy()
                    if cur_list.ndim == 1:
                        cur_list = cur_list[..., np.newaxis]
                    elif cur_list.ndim == 0:  # valids
                        continue
                    path[key] = cur_list
                self.replay_buffer.add_path(path)

        super()._train_once_inner(data)

        sac_utils.update_targets(self)

    def _train_with_minibatch(self, data, target):
        if self._trans_minibatch_size is None:
            return super()._train_with_minibatch(data, target)

        assert self._train_child_policy or self._train_downstream_policy  # Only allowed when training child or downstream policy
        assert target == 'all'
        if self.replay_buffer is None:
            tensors, internal_vars = self._compute_common_tensors(data)

            num_transitions = internal_vars['num_transitions']

            assert self._trans_optimization_epochs > 0

            for _ in range(self._trans_optimization_epochs):
                mini_tensors, mini_internal_vars = self._get_mini_tensors(
                    tensors, internal_vars, num_transitions, self._trans_minibatch_size
                )
                self._pre_optimization(mini_tensors, mini_internal_vars, target)

                loss_sp_key = self._update_loss_sp(mini_tensors, mini_internal_vars)

                self._gradient_descent(
                    mini_tensors[loss_sp_key],
                    optimizer_keys=['sampling_policy'],
                )

                self._post_optimization(mini_tensors, mini_internal_vars, target)
        else:
            if self.replay_buffer.n_transitions_stored >= self.min_buffer_size:
                optimization_epochs = self._trans_optimization_epochs
                if optimization_epochs == 0:
                    optimization_epochs = sum(data['valids'])
                assert optimization_epochs > 0
                for i in range(optimization_epochs):
                    # Sample from the replay buffer
                    samples = self.replay_buffer.sample_transitions(self._trans_minibatch_size)
                    data = {}
                    for key, value in samples.items():
                        if value.shape[1] == 1:
                            value = np.squeeze(value, axis=1)
                        data[key] = np.asarray([torch.from_numpy(value).float().to(self.device)], dtype=np.object)
                    data['valids'] = [self._trans_minibatch_size]

                    assert len(data['obs']) == 1
                    if self.normalizer_type == 'consistent':
                        data['obs'][0] = self.normalizer.normalize(data['ori_obs'][0].cpu()).to(self.device)
                        data['next_obs'][0] = self.normalizer.normalize(data['next_ori_obs'][0].cpu()).to(self.device)
                    elif self.normalizer_type == 'garage_ex':
                        data['obs'][0] = ((data['ori_obs'][0].cpu() - self._cur_obs_mean) / self._cur_obs_std).to(torch.float32).to(self.device)
                        data['next_obs'][0] = ((data['next_ori_obs'][0].cpu() - self._cur_obs_mean) / self._cur_obs_std).to(torch.float32).to(self.device)

                    tensors = {}
                    internal_vars = {
                        'maybe_no_grad': {},
                    }

                    self._update_inputs(data, tensors, internal_vars)
                    self._pre_optimization(tensors, internal_vars, target)

                    loss_sp_key = self._update_loss_sp(tensors, internal_vars)

                    self._gradient_descent(
                        tensors[loss_sp_key],
                        optimizer_keys=['sampling_policy'],
                    )

                    self._post_optimization(tensors, internal_vars, target)

    def _pre_optimization(self, tensors, internal_vars, target):
        if target not in ['all', 'sampler']:
            return

        self._update_loss_qf(tensors, internal_vars)

        self._gradient_descent(
            tensors['LossQf1'],
            optimizer_keys=['qf1'],
        )
        self._gradient_descent(
            tensors['LossQf2'],
            optimizer_keys=['qf2'],
        )

    def _post_optimization(self, tensors, internal_vars, target):
        if target not in ['all', 'sampler']:
            return

        self._update_loss_alpha(tensors, internal_vars)
        self._gradient_descent(
            tensors['LossAlpha'],
            optimizer_keys=['log_alpha'],
        )

    def _compute_common_tensors(self, data, *, compute_extra_metrics=False, op_compute_chunk_size=None):
        tensors, internal_vars = super()._compute_common_tensors(data, compute_extra_metrics=compute_extra_metrics)

        if compute_extra_metrics:
            # Calculate loss_qf and loss_sp only when compute_extra_metrics.
            # Otherwise, they are calculated in _optimize_sp (during training).
            self._update_loss_qf(tensors, internal_vars)
            self._update_loss_sp(tensors, internal_vars)
            self._update_loss_alpha(tensors, internal_vars)

        return tensors, internal_vars

    def _update_loss_qf(self, tensors, v):
        if not self._train_child_policy and not self._train_downstream_policy:
            self._update_pseudo_rewards_sp(tensors, v)
            pseudo_rewards = v['funcs']['pr_get_pseudo_rewards']()
        else:
            pseudo_rewards = v['rewards_flat']

            # Don't handle normalize_reward here since it is already handled in process_samples.

        pseudo_rewards_flat = pseudo_rewards.flatten()

        processed_cat_obs_flat = self.sampling_policy.process_observations(v['cat_obs_flat'])
        next_processed_cat_obs_flat = self.sampling_policy.process_observations(v['next_cat_obs_flat'])

        sac_utils.update_loss_qf(
            self, tensors, v,
            obs_flat=processed_cat_obs_flat,
            actions_flat=v['actions_flat'],
            next_obs_flat=next_processed_cat_obs_flat,
            dones_flat=v['dones_flat'],
            rewards_flat=pseudo_rewards_flat * self._reward_scale_factor,
            policy=self.sampling_policy,
        )

    def _update_loss_sp(self, tensors, v):
        processed_cat_obs_flat = self.sampling_policy.process_observations(v['cat_obs_flat'])

        sac_utils.update_loss_sacp(
            self, tensors, v,
            obs_flat=processed_cat_obs_flat,
            policy=self.sampling_policy,
        )
        return 'LossSacp'

    def _update_loss_alpha(self, tensors, v):
        sac_utils.update_loss_alpha(
            self, tensors, v,
        )

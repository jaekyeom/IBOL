from collections import deque
import copy
import gc
import numpy as np
import torch
from contextlib import nullcontext
from garage import TrajectoryBatch
from garage.np.algos.rl_algorithm import RLAlgorithm
from torch.distributions import Independent

import aim_wrapper
import dowel_wrapper
from dowel import Histogram
from garagei import log_performance_ex
from garagei.torch.optimizers.optimizer_group_wrapper import OptimizerGroupWrapper
from garagei.torch.utils import unsqueeze_expand_flat_dim0, unwrap_dist, compute_total_norm, get_outermost_dist_attr, TrainContext
from iod.utils import get_option_colors, FigManager, MeasureAndAccTime, valuewise_sequencify_dicts, RunningMeanStd
from iod.utils import get_torch_concat_obs

class IOD(RLAlgorithm):
    def __init__(
            self,
            *,
            env_spec,
            normalizer_type,
            normalizer_mean,
            normalizer_std,
            sampling_policy,
            option_policy,
            traj_encoder,
            option_prior_net,
            optimizer,
            alpha,
            beta,
            max_path_length,
            max_optimization_epochs,
            n_epochs_per_eval,
            n_epochs_per_first_n_eval,
            custom_eval_steps,
            n_epochs_per_tb,
            n_epochs_per_save,
            n_epochs_per_pt_save,
            dim_option,
            dim_xi,
            num_eval_options,
            eval_plot_axis,
            name='IOD',
            device=torch.device('cpu'),
            num_sampling_options=16,
            num_train_per_epoch=1,
            discount=0.99,
            xi_aux_coef=0.,
            clip_grad_norm=None,
            record_metric_difference=True,
            sp_use_lstm=False,
            sp_lstm_hidden_dim=None,
            sp_lstm_num_layers=None,
            sp_step_xi=False,
            sp_step_xi_beta_param=None,
            sp_lstm_xi=False,
            sp_lstm_xi_dim=None,
            num_alt_samples=100,
            pseudo_reward_discount=None,
            split_group=10000,
            log_prob_mean=False,
            op_use_lstm=False,
            train_child_policy=False,
            tcp_reward_alive=0.0,
            tcp_multi_step=None,
            tcp_dropout_prob=0.,
            tcp_split=False,
            train_downstream_policy=False,
            rms_reward=False,
            rms_keep_rate=1.,
            rms_init='zero_one',
            trans_minibatch_size=None,
            trans_optimization_epochs=None,
    ):
        self.discount = discount
        self.max_path_length = max_path_length
        self.max_optimization_epochs = max_optimization_epochs

        self.device = device
        self.normalizer_type = normalizer_type
        self.sampling_policy = sampling_policy.to(self.device)
        self.option_policy = option_policy.to(self.device)
        self.traj_encoder = traj_encoder.to(self.device)
        self.param_modules = {
            'sampling_policy': self.sampling_policy,
            'traj_encoder': self.traj_encoder,
            'option_policy': self.option_policy,
        }


        self.option_prior_net = option_prior_net
        if isinstance(self.option_prior_net, torch.nn.Module):
            self.option_prior_net.to(self.device)
        self.alpha = alpha
        self.beta = beta
        self.xi_aux_coef = xi_aux_coef

        self.name = name

        self.dim_option = dim_option
        self.dim_xi = dim_xi
        self.num_sampling_options = num_sampling_options

        self._num_train_per_epoch = num_train_per_epoch
        self._env_spec = env_spec

        self.n_epochs_per_eval = n_epochs_per_eval
        self.n_epochs_per_first_n_eval = n_epochs_per_first_n_eval
        self.custom_eval_steps = custom_eval_steps
        self.n_epochs_per_tb = n_epochs_per_tb
        self.n_epochs_per_save = n_epochs_per_save
        self.n_epochs_per_pt_save = n_epochs_per_pt_save
        self.num_eval_options = num_eval_options
        self.eval_plot_axis = eval_plot_axis

        assert isinstance(optimizer, OptimizerGroupWrapper)
        self._optimizer = optimizer
        self._clip_grad_norm = clip_grad_norm

        self._record_metric_difference = record_metric_difference

        self._cur_max_path_length = max_path_length  # Used in 1. train, 2. process_samples

        self._sp_use_lstm = sp_use_lstm
        self._sp_lstm_hidden_dim = sp_lstm_hidden_dim
        self._sp_lstm_num_layers = sp_lstm_num_layers
        self._sp_step_xi = sp_step_xi
        self._sp_step_xi_beta_param = sp_step_xi_beta_param
        self._sp_lstm_xi = sp_lstm_xi
        self._sp_lstm_xi_dim = sp_lstm_xi_dim

        self.num_alt_samples = num_alt_samples
        self.pseudo_reward_discount = pseudo_reward_discount
        self.split_group = split_group

        self._log_prob_mean = log_prob_mean
        self._op_use_lstm = op_use_lstm

        self._train_child_policy = train_child_policy
        self._tcp_reward_alive = tcp_reward_alive
        self._tcp_multi_step = tcp_multi_step
        self._tcp_dropout_prob = tcp_dropout_prob
        self._tcp_split = tcp_split

        self._train_downstream_policy = train_downstream_policy

        self.rms_reward = rms_reward
        if rms_reward:
            self._reward_rms = RunningMeanStd((), rms_keep_rate, rms_init)
        else:
            self._reward_rms = RunningMeanStd((), 1., 'zero_one')

        self._trans_minibatch_size = trans_minibatch_size  # Used only in IODSAC
        self._trans_optimization_epochs = trans_optimization_epochs

        self._cur_obs_mean = None
        self._cur_obs_std = None

        if self.normalizer_type == 'manual':
            self._cur_obs_mean = np.full(self._env_spec.observation_space.flat_dim, normalizer_mean)
            self._cur_obs_std = np.full(self._env_spec.observation_space.flat_dim, normalizer_std)
        else:
            # Set to the default value
            self._cur_obs_mean = np.full(self._env_spec.observation_space.flat_dim, 0.)
            self._cur_obs_std = np.full(self._env_spec.observation_space.flat_dim, 1.)

        self._cur_downstream_rewards = deque(maxlen=500)

        assert not (self._sp_lstm_xi and not self._sp_use_lstm)

        self.traj_encoder.eval()  # for batch norm

        if self._train_child_policy:
            self.n_epochs_per_save = 0

    @property
    def policy(self):
        if self._train_child_policy:
            return {
                'sampling_policy': self.sampling_policy,
            }
        else:
            return {
                'sampling_policy': self.sampling_policy,
                'option_policy': self.option_policy,
            }

    def all_parameters(self):
        for m in self.param_modules.values():
            for p in m.parameters():
                yield p

    def _call_option_prior_net(self, obs):
        return self.option_prior_net(obs)

    def train_once(self, itr, paths, runner, extra_scalar_metrics={}):
        """Train the algorithm once.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        Returns:
            numpy.float64: Calculated mean value of undiscounted returns.

        """
        data = self.process_samples(paths, training=True)

        time_computing_metrics = [0.0]
        time_training = [0.0]

        with torch.no_grad(), MeasureAndAccTime(time_computing_metrics):
            tensors_before, _ = self._compute_common_tensors(data, compute_extra_metrics=True,
                                                             op_compute_chunk_size=self._optimizer._minibatch_size)
            gc.collect()

        with MeasureAndAccTime(time_training):
            self._train_once_inner(data)

        with torch.no_grad(), MeasureAndAccTime(time_computing_metrics):
            tensors_after, _ = self._compute_common_tensors(data, compute_extra_metrics=True,
                                                            op_compute_chunk_size=self._optimizer._minibatch_size)
            gc.collect()

        prefix_tabular, prefix_aim = aim_wrapper.get_metric_prefixes()
        with dowel_wrapper.get_tabular().prefix(prefix_tabular + self.name + '/'), dowel_wrapper.get_tabular(
                'plot').prefix(prefix_tabular + self.name + '/'):
            def _record_scalar(key, val):
                dowel_wrapper.get_tabular().record(key, val)
                aim_wrapper.track(val, name=(prefix_aim + self.name + '__' + key), epoch=itr)

            def _record_histogram(key, val):
                dowel_wrapper.get_tabular('plot').record(key, Histogram(val))

            for k in tensors_before.keys():
                if tensors_before[k].numel() == 1:
                    _record_scalar(f'{k}Before', tensors_before[k].item())
                    if self._record_metric_difference:
                        _record_scalar(f'{k}After', tensors_after[k].item())
                        _record_scalar(f'{k}Decrease', (tensors_before[k] - tensors_after[k]).item())
                else:
                    _record_histogram(f'{k}Before', tensors_before[k].detach().cpu().numpy())
            with torch.no_grad():
                total_norm = compute_total_norm(self.all_parameters())
                _record_scalar('TotalGradNormAll', total_norm.item())
                for key, module in self.param_modules.items():
                    total_norm = compute_total_norm(module.parameters())
                    _record_scalar(f'TotalGradNorm{key.replace("_", " ").title().replace(" ", "")}', total_norm.item())
            for k, v in extra_scalar_metrics.items():
                _record_scalar(k, v)
            _record_scalar('TimeComputingMetrics', time_computing_metrics[0])
            _record_scalar('TimeTraining', time_training[0])

        undiscounted_returns = log_performance_ex(
            itr,
            TrajectoryBatch.from_trajectory_list(self._env_spec, paths),
            discount=self.discount,
        )['undiscounted_returns']

        if self._train_downstream_policy:
            # Additionally reward stats
            prefix_tabular, prefix_aim = aim_wrapper.get_metric_prefixes()
            with dowel_wrapper.get_tabular().prefix(prefix_tabular + self.name + '/'):
                # Log smoothed mean
                self._cur_downstream_rewards.append(np.mean(undiscounted_returns))
                rewards = list(self._cur_downstream_rewards)
                mean500 = np.mean(rewards[-500:])
                dowel_wrapper.get_tabular().record('SmoothedReward500', mean500)

        return np.mean(undiscounted_returns)

    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunnerTraj): LocalRunnerTraj is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        last_return = None

        with aim_wrapper.AimContext({'phase': 'train', 'policy': 'sampling'}):
            for _ in runner.step_epochs(
                    full_tb_epochs=0,
                    log_period=1,
                    tb_period=self.n_epochs_per_tb,
                    save_period=self.n_epochs_per_pt_save,
                    new_save_period=self.n_epochs_per_save,
            ):
                for p in self.policy.values():
                    p.eval()

                eval_policy = (
                        (self.n_epochs_per_eval != 0 and runner.step_itr % self.n_epochs_per_eval == 0)
                        or (self.n_epochs_per_eval != 0 and self.n_epochs_per_first_n_eval is not None
                            and runner.step_itr < self.n_epochs_per_eval and runner.step_itr % self.n_epochs_per_first_n_eval == 0)
                        or (self.custom_eval_steps is not None and runner.step_itr in self.custom_eval_steps)
                )

                if eval_policy:
                    self._evaluate_policy(runner)
                    self._log_eval_metrics(runner)

                for p in self.policy.values():
                    p.train()

                for _ in range(self._num_train_per_epoch):
                    time_sampling = [0.0]
                    with MeasureAndAccTime(time_sampling):
                        runner.step_path = self._get_train_trajectories(runner)
                    last_return = self.train_once(
                        runner.step_itr,
                        runner.step_path,
                        runner,
                        extra_scalar_metrics={
                            'TimeSampling': time_sampling[0],
                        },
                    )
                    gc.collect()

                runner.step_itr += 1

        return last_return

    def _get_trajectories(self,
                          runner,
                          sampler_key,
                          batch_size=None,
                          extras=None,
                          update_stats=False,
                          update_normalizer=False,
                          update_normalizer_override=False,
                          worker_update=None,
                          max_path_length_override=None,
                          env_update=None):
        if batch_size is None:
            batch_size = len(extras)
        policy_sampler_key = sampler_key[6:] if sampler_key.startswith('local_') else sampler_key
        time_get_trajectories = [0.0]
        with MeasureAndAccTime(time_get_trajectories):
            trajectories, infos = runner.obtain_exact_trajectories(
                runner.step_itr,
                sampler_key=sampler_key,
                batch_size=batch_size,
                agent_update=self._get_policy_param_values_cpu(policy_sampler_key),
                env_update=env_update,
                worker_update=worker_update,
                update_normalized_env_ex=update_normalizer if self.normalizer_type == 'garage_ex' else None,
                get_attrs=['env._obs_mean', 'env._obs_var'],
                extras=extras,
                max_path_length_override=max_path_length_override,
                update_stats=update_stats,
            )
        print(f'_get_trajectories({sampler_key}) {time_get_trajectories[0]}s')

        if self.normalizer_type == 'garage_ex' and update_normalizer:
            self._set_updated_normalized_env_ex_except_sampling_policy(runner, infos)

        return trajectories

    def _get_train_trajectories_kwargs(self, runner):
        extras = self._generate_sampling_extras(
            runner._train_args.batch_size,
        )

        return dict(
            extras=extras,
            sampler_key='sampling_policy',
        )

    def _get_train_trajectories(self, runner, burn_in=False):
        default_kwargs = dict(
            runner=runner,
            update_stats=not burn_in,
            update_normalizer=True,
            update_normalizer_override=burn_in,
            max_path_length_override=self._cur_max_path_length,
            worker_update=dict(
                    _deterministic_initial_state=False,
                    _deterministic_policy=False,
            ),
            env_update=dict(),
        )
        kwargs = dict(default_kwargs, **self._get_train_trajectories_kwargs(runner))

        paths = self._get_trajectories(**kwargs)

        return paths

    def _update_rms(self, rewards):
        self._reward_rms.update(np.array(rewards))

    def process_samples(self, paths, training=False):
        r"""Process sample data based on the collected paths."""

        if self._train_child_policy:
            # {{{
            obs_dims_slice = slice(0, self.dim_xi)

            for path in paths:
                start = 0
                global_end = min(self._cur_max_path_length, len(path['actions']))
                while start < global_end:
                    # {{{
                    end = min(start + self._tcp_multi_step - 1, global_end - 1)

                    xi = path['agent_infos']['step_xi'][start]

                    observation = path['observations'][start]
                    next_observation = path['next_observations'][end]

                    delta = next_observation[obs_dims_slice] - observation[obs_dims_slice]

                    reward = np.inner(delta, xi) / np.sqrt(self.dim_xi) / self._tcp_multi_step
                    for i in range(start, end + 1):
                        path['rewards'][i] = reward

                    start += self._tcp_multi_step
                    # }}}

                path['rewards'][:] += self._tcp_reward_alive
            # }}}

        rms_rewards = []
        for path in paths:
            rms_rewards.extend(path['rewards'])
        self._update_rms(rms_rewards)
        del rms_rewards

        def _to_torch_float32(x):
            if x.dtype == np.object:
                return np.array([torch.tensor(i, dtype=torch.float32, device=self.device) for i in x], dtype=np.object)
            return torch.tensor(x, dtype=torch.float32, device=self.device)

        valids = np.asarray([len(path['actions'][:self._cur_max_path_length]) for path in paths])
        obs = np.asarray(
            [_to_torch_float32(path['observations'][:self._cur_max_path_length])
             for path in paths], dtype=np.object)
        ori_obs = np.asarray(
            [_to_torch_float32(path['env_infos']['ori_obs'][:self._cur_max_path_length])
             for path in paths], dtype=np.object)
        next_obs = np.asarray(
            [_to_torch_float32(path['next_observations'][:self._cur_max_path_length])
             for path in paths], dtype=np.object)
        next_ori_obs = np.asarray(
            [_to_torch_float32(path['env_infos']['next_ori_obs'][:self._cur_max_path_length])
             for path in paths], dtype=np.object)
        actions = np.asarray(
            [_to_torch_float32(path['actions'][:self._cur_max_path_length])
             for path in paths], dtype=np.object)
        rewards = np.asarray(
            [_to_torch_float32(path['rewards'][:self._cur_max_path_length])
             for path in paths], dtype=np.object)
        dones = np.asarray(
            [_to_torch_float32(path['dones'][:self._cur_max_path_length])
             for path in paths], dtype=np.object)

        data = dict(
            obs=obs,
            ori_obs=ori_obs,
            next_obs=next_obs,
            next_ori_obs=next_ori_obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            valids=valids,
        )

        for key in paths[0]['agent_infos'].keys():
            data[key] = np.asarray([torch.tensor(path['agent_infos'][key][:self._cur_max_path_length],
                                                 dtype=torch.float32, device=self.device)
                                    for path in paths], dtype=np.object)
        for key in ['step_xi', 'option']:
            if key not in data:
                continue
            next_key = f'next_{key}'
            data[next_key] = copy.deepcopy(data[key])
            for i in range(len(data[next_key])):
                cur_data = data[key][i]
                data[next_key][i] = torch.cat([cur_data[1:], cur_data[-1:]], dim=0)

        return data

    def _get_policy_param_values_cpu(self, key):
        param_dict = self.policy[key].get_param_values()
        for k in param_dict.keys():
            param_dict[k] = param_dict[k].detach().cpu()
        return param_dict

    def _sample_step_xi(self, batch_shape=None):
        size = (batch_shape or []) + [self.dim_xi]
        if self._sp_step_xi_beta_param is not None:
            xi = (np.random.beta(self._sp_step_xi_beta_param, self._sp_step_xi_beta_param, size).astype(np.float32)
                  * 2.0 - 1.0)
        else:
            # Sample from N(0, I)
            xi = np.random.randn(*size).astype(np.float32)
            xi = xi * getattr(self, '_cur_xi_gaussian_std', 1.0)
        assert xi.shape == tuple(size)
        return xi

    def _generate_sampling_extras(self, num_extras, num_dims=None):
        def _generate_step_xi():
            xi = self._sample_step_xi()

            if num_dims is not None:
                xi[num_dims:] = 0.

            return xi

        extras = []
        for i in range(num_extras):
            extra = {}
            if self._sp_step_xi:
                if not self._train_child_policy:
                    extra['step_xi'] = _generate_step_xi()
                else:
                    extra['step_xi'] = []
                    cur_xi = _generate_step_xi()
                    for j in range(self._cur_max_path_length):
                        if j % self._tcp_multi_step == 0:
                            extra['step_xi'].append(cur_xi)
                        else:
                            extra['step_xi'].append(None)  # None means to reuse previous xi
            if self._sp_lstm_xi:
                # hidden, cell_state
                if self._sp_lstm_xi_dim is None:
                    extra['lstm_xi'] = (
                        np.random.randn(self._sp_lstm_num_layers, self._sp_lstm_hidden_dim),
                        np.random.randn(self._sp_lstm_num_layers, self._sp_lstm_hidden_dim),
                    )
                else:
                    # Configure only first _sp_lstm_xi_dim of cell_state
                    extra['lstm_xi'] = (
                        np.zeros((self._sp_lstm_num_layers, self._sp_lstm_hidden_dim)),
                        np.concatenate([
                            np.random.randn(self._sp_lstm_num_layers, self._sp_lstm_xi_dim),
                            np.zeros((self._sp_lstm_num_layers, self._sp_lstm_hidden_dim - self._sp_lstm_xi_dim)),
                        ], axis=1),
                    )
            extras.append(extra)

        return extras

    def _generate_option_extras(self, options):
        return [{'option': option} for option in options]

    def _train_once_inner(self, data):
        # for batch norm
        modules_with_bn = [self.traj_encoder]

        with TrainContext(modules_with_bn):
            for minibatch in self._optimizer.get_minibatch(data, max_optimization_epochs=self.max_optimization_epochs[0]):
                self._train_with_minibatch(minibatch, 'all')

    def _clip_gradient(self, optimizer_keys):
        if self._clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self._optimizer.target_parameters(keys=optimizer_keys),
                self._clip_grad_norm)

    def _train_with_minibatch(self, data, target):
        tensors, internal_vars = self._compute_common_tensors(data)

        self._pre_optimization(tensors, internal_vars, target)

        if target == 'all':
            # {{{
            loss_sp_key = self._update_loss_sp(tensors, internal_vars)
            if not self._train_child_policy and not self._train_downstream_policy:
                loss_te_op_sd_key = self._update_loss_te_op_sd(tensors, internal_vars)

            optimizer_keys = ['sampling_policy', 'traj_encoder', 'option_policy']
            self._optimizer.zero_grad(keys=optimizer_keys)

            tensors[loss_sp_key].backward()
            if not self._train_child_policy and not self._train_downstream_policy:
                tensors[loss_te_op_sd_key].backward()
            # }}}
        else:
            assert False

        self._clip_gradient(optimizer_keys)
        self._optimizer.step(keys=optimizer_keys)

        self._post_optimization(tensors, internal_vars, target)

    def _gradient_descent(self, loss, optimizer_keys):
        self._optimizer.zero_grad(keys=optimizer_keys)
        loss.backward()
        self._clip_gradient(optimizer_keys)
        self._optimizer.step(keys=optimizer_keys)

    def _pre_optimization(self, tensors, internal_vars, target):
        pass

    def _post_optimization(self, tensors, internal_vars, target):
        pass

    def _get_mini_tensors(self, tensors, internal_vars, num_transitions, trans_minibatch_size):
        idxs = np.random.choice(num_transitions, trans_minibatch_size)
        mini_tensors = {}
        mini_internal_vars = {}

        for k, v in tensors.items():
            try:
                if len(v) == num_transitions:
                    mini_tensors[k] = v[idxs]
            except TypeError:
                pass
        for k, v in internal_vars.items():
            try:
                if len(v) == num_transitions:
                    mini_internal_vars[k] = v[idxs]
            except TypeError:
                pass

        return mini_tensors, mini_internal_vars

    def _compute_common_tensors(self, data, *, compute_extra_metrics=False, op_compute_chunk_size=None):
        tensors = {}  # contains tensors to be logged, including losses.
        internal_vars = {  # contains internal variables.
            'maybe_no_grad': {},
        }

        self._update_inputs(data, tensors, internal_vars)

        if not self._train_child_policy and not self._train_downstream_policy:
            self._update_te(tensors, internal_vars, compute_extra_metrics=compute_extra_metrics)
        self._update_sp(tensors, internal_vars, compute_extra_metrics=compute_extra_metrics)
        if not self._train_child_policy and not self._train_downstream_policy:
            self._update_op(tensors, internal_vars, compute_extra_metrics=compute_extra_metrics,
                            compute_chunk_size=op_compute_chunk_size)

        if compute_extra_metrics:
            self._update_loss_sp(tensors, internal_vars)
            if not self._train_child_policy and not self._train_downstream_policy:
                self._update_loss_te_op_sd(tensors, internal_vars)
                self._update_extra_metrics_sp_op(tensors, internal_vars)
        return tensors, internal_vars

    def _get_concat_obs(self, obs, option, num_repeats=1):
        return get_torch_concat_obs(obs, option, num_repeats)

    def _update_inputs(self, data, tensors, v):
        obs = list(data['obs'])
        next_obs = list(data['next_obs'])
        actions = list(data['actions'])
        valids = list(data['valids'])
        dones = list(data['dones'])
        rewards = list(data['rewards'])
        if 'log_prob' in data:
            log_probs = list(data['log_prob'])
        else:
            log_probs = None

        num_trajs = len(obs)
        valids_t = torch.tensor(data['valids'], device=self.device)
        valids_t_f32 = valids_t.to(torch.float32)
        max_traj_length = valids_t.max().item()

        obs_flat = torch.cat(obs, dim=0)
        next_obs_flat = torch.cat(next_obs, dim=0)
        actions_flat = torch.cat(actions, dim=0)
        dones_flat = torch.cat(dones, dim=0).to(torch.int)
        rewards_flat = torch.cat(rewards, dim=0)
        if log_probs is not None:
            log_probs_flat = torch.cat(log_probs, dim=0)
        else:
            log_probs_flat = None

        if 'pre_tanh_value' in data:
            pre_tanh_values = list(data['pre_tanh_value'])
            pre_tanh_values_flat = torch.cat(pre_tanh_values, dim=0)

        if self._train_child_policy or self._train_downstream_policy:
            if self.rms_reward:
                rewards_flat = (rewards_flat - self._reward_rms.mean) / self._reward_rms.std
        if self._train_child_policy or self._train_downstream_policy:
            rewards_flat = rewards_flat

        dims_action = actions_flat.size()[1:]

        assert obs_flat.ndim == 2
        dim_obs = obs_flat.size(1)
        num_transitions = actions_flat.size(0)

        # Trajectory encoder kwargs
        traj_encoder_extra_kwargs = dict()

        # Sampling policy kwargs
        if self._sp_use_lstm:
            sampling_policy_kwargs = dict(
                seq_lengths=valids,
                actions=actions_flat,
            )

            if self._sp_lstm_xi:
                lstm_hidden_xis_t = torch.stack([lstm_xi[0, 0] for lstm_xi in data['lstm_xi']], dim=1)
                lstm_cell_state_xis_t = torch.stack([lstm_xi[0, 1] for lstm_xi in data['lstm_xi']], dim=1)
                sampling_policy_kwargs.update(dict(
                    hidden_cell_state_tuple=(lstm_hidden_xis_t, lstm_cell_state_xis_t),
                ))
            else:
                sampling_policy_kwargs.update(dict(
                    hidden_cell_state_tuple=None,
                ))
        else:
            sampling_policy_kwargs = dict()

        if self._sp_step_xi:
            step_xis = list(data['step_xi'])
            step_xis = [sx[:, :self.dim_xi] for sx in step_xis]
            traj_step_xis = torch.stack([x[0] for x in step_xis], dim=0)
            assert traj_step_xis.size() == (num_trajs, self.dim_xi)
            step_xis_flat = torch.cat(step_xis, dim=0)
            assert step_xis_flat.size() == (num_transitions, self.dim_xi)
            cat_obs_flat = self._get_concat_obs(obs_flat, step_xis_flat)

            next_step_xis = list(data['next_step_xi'])
            next_step_xis_flat = torch.cat(next_step_xis, dim=0)
            next_cat_obs_flat = self._get_concat_obs(next_obs_flat, next_step_xis_flat)
        else:
            cat_obs_flat = obs_flat
            next_cat_obs_flat = next_obs_flat

        sampling_policy_kwargs.update(dict(
            observations=cat_obs_flat,
        ))

        if self._train_child_policy or self._train_downstream_policy:
            tensors.update({
                'RewardMean': torch.tensor(self._reward_rms.mean, device=self.device),
                'RewardStd': torch.tensor(self._reward_rms.std, device=self.device),
            })

        v.update({
            'obs': obs,
            'obs_flat': obs_flat,
            'next_obs_flat': next_obs_flat,
            'cat_obs_flat': cat_obs_flat,
            'next_cat_obs_flat': next_cat_obs_flat,
            'actions_flat': actions_flat,
            'valids': valids,
            'valids_t': valids_t,
            'valids_t_f32': valids_t_f32,
            'dones_flat': dones_flat,
            'rewards_flat': rewards_flat,
            'log_probs_flat': log_probs_flat,

            'dim_obs': dim_obs,
            'dims_action': dims_action,
            'num_trajs': num_trajs,
            'num_transitions': num_transitions,
            'max_traj_length': max_traj_length,
            'traj_encoder_extra_kwargs': traj_encoder_extra_kwargs,
            'sampling_policy_kwargs': sampling_policy_kwargs,
        })

        if self._sp_step_xi:
            v.update({
                'step_xis_flat': step_xis_flat,
                'traj_step_xis': traj_step_xis,
            })

        if 'pre_tanh_value' in data:
            v.update({
                'pre_tanh_values_flat': pre_tanh_values_flat,
            })

    def _convert_trans_vals_to_per_traj_means(self, v, vals):
        if v['valids_t'].min() == v['valids_t'].max():
            return vals.view(v['num_trajs'], v['max_traj_length']).mean(dim=1)

        if vals.size() == (v['num_transitions'],):
            vals = vals.split(v['valids'], dim=0)
            vals = torch.nn.utils.rnn.pad_sequence(vals, batch_first=True, padding_value=0.0)
        assert vals.size() == (v['num_trajs'], v['max_traj_length'])

        return vals.sum(dim=1) / v['valids_t_f32']

    def _divide_trans_vals_to_trajs(self, v, vals):
        assert vals.size() == (v['num_transitions'],)
        assert v['valids_t'].min() == v['valids_t'].max()
        return vals.view(v['num_trajs'], v['max_traj_length'])

    def _update_te(self, tensors, v, compute_extra_metrics=False):
        (option_dists, option_dists_trans), _ = self.traj_encoder.forward_with_transform(
            v['obs'],
            transform=lambda x: x.repeat_interleave(v['valids_t'], dim=0),
            **v['traj_encoder_extra_kwargs']
        )
        assert isinstance(option_dists, torch.distributions.Distribution)
        assert option_dists.batch_shape == (v['num_trajs'],)
        assert self.dim_option == option_dists.event_shape[0]

        assert option_dists_trans.batch_shape == (v['num_transitions'],)
        assert option_dists_trans.event_shape == (self.dim_option,)

        option_priors_flat = self._call_option_prior_net(v['obs_flat'])

        option_penalty = torch.distributions.kl.kl_divergence(option_dists_trans, option_priors_flat)
        assert option_penalty.size() == (v['num_transitions'],)
        option_penalty = option_penalty.split(v['valids'], dim=0)
        option_penalty = torch.nn.utils.rnn.pad_sequence(option_penalty, batch_first=True, padding_value=0.0)

        assert option_penalty.size() == (v['num_trajs'], v['max_traj_length'])

        option_penalty_means = option_penalty.sum(dim=1) / v['valids_t_f32']
        assert option_penalty_means.size() == (v['num_trajs'],)
        option_penalty_mean = option_penalty_means.mean()

        # Sampling with reparameterization trick using rsample().
        option_samples = option_dists.rsample([self.num_sampling_options])
        assert option_samples.size() == (self.num_sampling_options, v['num_trajs'], self.dim_option)

        if self._sp_step_xi:
            if v['valids_t'].min() != v['valids_t'].max():
                step_xis = v['step_xis_flat'].split(v['valids'], dim=0)
                step_xis = torch.nn.utils.rnn.pad_sequence(step_xis, batch_first=True, padding_value=0.0)
            else:
                step_xis = v['step_xis_flat'].view(v['num_trajs'], -1, self.dim_xi)

            te_option_log_probs = option_dists.log_prob(step_xis[:, 0, :])
            te_option_log_prob_mean = te_option_log_probs.mean()

        tensors.update({
            'OptionPenaltyMean': option_penalty_mean,
        })

        v.update({
            'option_penalty': option_penalty,
            'option_penalty_means': option_penalty_means,
            'option_penalty_mean': option_penalty_mean,
        })

        v.update({
            'option_dists': option_dists,
            'option_samples': option_samples,
        })

        if self._sp_step_xi:
            tensors.update({
                'TeOptionLogProbMean': te_option_log_prob_mean,
            })

            v.update({
                'te_option_log_probs': te_option_log_probs,
                'te_option_log_prob_mean': te_option_log_prob_mean,
            })

    def _update_sp(self, tensors, v, compute_extra_metrics=False):
        if compute_extra_metrics:
            (sp_action_dists_flat, sp_action_dists_repeated), *_ = self.sampling_policy.forward_with_transform(
                **v['sampling_policy_kwargs'],
                transform=lambda x: unsqueeze_expand_flat_dim0(x, self.num_sampling_options),
            )
            assert sp_action_dists_repeated.batch_shape == (self.num_sampling_options * v['num_transitions'],)
        else:
            sp_action_dists_flat, *_ = self.sampling_policy(
                **v['sampling_policy_kwargs'],
            )
            sp_action_dists_repeated = None
        assert sp_action_dists_flat.batch_shape == (v['num_transitions'],)
        if 'pre_tanh_values_flat' in v:
            sp_action_log_probs = sp_action_dists_flat.log_prob(v['actions_flat'], v['pre_tanh_values_flat'])
        else:
            sp_action_log_probs = sp_action_dists_flat.log_prob(v['actions_flat'])
        assert sp_action_log_probs.size() == (v['num_transitions'],)
        sp_action_log_probs_transitions = sp_action_log_probs
        sp_action_log_probs = sp_action_log_probs.split(v['valids'], dim=0)
        sp_action_log_probs = torch.nn.utils.rnn.pad_sequence(sp_action_log_probs, batch_first=True, padding_value=0.0)

        assert sp_action_log_probs.size() == (v['num_trajs'], v['max_traj_length'])

        sp_action_log_prob_sums = sp_action_log_probs.sum(dim=1)
        sp_action_log_prob_means = sp_action_log_prob_sums / v['valids_t_f32']
        sp_action_log_prob_mean = sp_action_log_prob_means.mean()

        sp_action_dists_flat_entropy = sp_action_dists_flat.entropy()
        assert sp_action_dists_flat_entropy.size() == (v['num_transitions'],)

        tensors.update({
            'SpActionLogProbMean': sp_action_log_prob_mean,
            'SpActionLogProbMax': sp_action_log_probs.max(),
            'SpActionLogProbMin': sp_action_log_probs.min(),
        })

        v.update({
            'sp_action_dists_flat': sp_action_dists_flat,
            'sp_action_dists_repeated': sp_action_dists_repeated,
            'sp_action_log_probs': sp_action_log_probs,
            'sp_action_log_probs_transitions': sp_action_log_probs_transitions,
            'sp_action_log_prob_sums': sp_action_log_prob_sums,
            'sp_action_log_prob_means': sp_action_log_prob_means,
            'sp_action_log_prob_mean': sp_action_log_prob_mean,
            'sp_action_dists_flat_entropy': sp_action_dists_flat_entropy,
        })

        if self._sp_step_xi:
            with torch.no_grad():
                # Sp action log-probabilities for random xi's (sampled from prior).
                # {{{
                sampling_policy_kwargs_random_xi = dict(
                        v['sampling_policy_kwargs'],
                        observations=self._get_concat_obs(
                            v['obs_flat'],
                            torch.tensor(self._sample_step_xi([v['obs_flat'].size(0)]), device=v['obs_flat'].device),
                        ),
                )
                random_xi_sp_action_dists_flat, *_ = self.sampling_policy(
                    **sampling_policy_kwargs_random_xi,
                )
                random_xi_sp_action_log_prob_mean = random_xi_sp_action_dists_flat.log_prob(v['actions_flat']).mean()
                # }}}
                tensors.update({
                    'SpActionLogProbForRandomXiMean': random_xi_sp_action_log_prob_mean,
                })

                # Sp action log-probabilities for zero-filled xi's.
                # {{{
                sampling_policy_kwargs_zero_xi = dict(
                        v['sampling_policy_kwargs'],
                        observations=self._get_concat_obs(
                            v['obs_flat'],
                            torch.zeros(v['obs_flat'].size(0), self.dim_xi, device=v['obs_flat'].device),
                        ),
                )
                zero_xi_sp_action_dists_flat, *_ = self.sampling_policy(
                    **sampling_policy_kwargs_zero_xi,
                )
                zero_xi_sp_action_log_prob_mean = zero_xi_sp_action_dists_flat.log_prob(v['actions_flat']).mean()
                # }}}
                tensors.update({
                    'SpActionLogProbForZeroXiMean': zero_xi_sp_action_log_prob_mean,
                })

    def _update_op(self, tensors, v, compute_extra_metrics=False, compute_chunk_size=None):
        def _get_input_chunk(traj_slice, option_overrider=None):
            # {{{
            assert traj_slice.step is None
            trans_slice = slice(
                    v['valids_t'][:traj_slice.start].sum().item(),
                    v['valids_t'][:traj_slice.stop].sum().item())
            num_chunk_transitions = trans_slice.stop - trans_slice.start
            assert num_chunk_transitions >= 0

            expanded_pure_obs_flat = unsqueeze_expand_flat_dim0(v['obs_flat'][trans_slice],
                                                                self.num_sampling_options)
            assert expanded_pure_obs_flat.size() == (self.num_sampling_options * num_chunk_transitions,
                                                     v['dim_obs'])

            if option_overrider is None:
                # Sample option parameters self.num_sampling_options times for each trajectory.
                option_samples = v['option_samples'][:, traj_slice].repeat_interleave(v['valids_t'][traj_slice], dim=1)
                assert option_samples.size() == (self.num_sampling_options,
                                                 num_chunk_transitions,
                                                 self.dim_option)
                option_samples_flat = option_samples.reshape(self.num_sampling_options * num_chunk_transitions,
                                                             self.dim_option)
            else:
                option_samples_flat = option_overrider(expanded_pure_obs_flat)

            expanded_obs = self._get_concat_obs(expanded_pure_obs_flat, option_samples_flat)
            expanded_actions = v['actions_flat'][trans_slice].unsqueeze(dim=0).expand(
                self.num_sampling_options, -1,
                *((-1,) * len(v['dims_action']))
            ).reshape(self.num_sampling_options * num_chunk_transitions, *v['dims_action'])
            if 'pre_tanh_values_flat' in v:
                expanded_pre_tanh_values = v['pre_tanh_values_flat'][trans_slice].unsqueeze(dim=0).expand(
                    self.num_sampling_options, -1,
                    *((-1,) * len(v['dims_action']))
                ).reshape(self.num_sampling_options * num_chunk_transitions, *v['dims_action'])
            else:
                expanded_pre_tanh_values = None

            option_policy_kwargs = dict(
                observations=expanded_obs,
            )
            if self._op_use_lstm:
                option_policy_kwargs = dict(
                    option_policy_kwargs,
                    seq_lengths=v['valids'][traj_slice] * self.num_sampling_options,
                    actions=expanded_actions,
                    hidden_cell_state_tuple=None,
                )

            vars_to_export = ['option_policy_kwargs', 'trans_slice', 'expanded_actions', 'expanded_pre_tanh_values']

            locals_ref = locals()
            return dict(
                    (k, locals_ref[k])
                    for k in vars_to_export
            )
            # }}}

        if compute_chunk_size is None:
            chunking_results = _get_input_chunk(slice(0, v['num_trajs']))
            option_policy_kwargs = chunking_results['option_policy_kwargs']
            expanded_actions = chunking_results['expanded_actions']
            expanded_pre_tanh_values = chunking_results['expanded_pre_tanh_values']
            op_action_dists_flat = self.option_policy(**option_policy_kwargs)[0]

        else:
            traj_slices = [
                slice(start, min(start + compute_chunk_size, v['num_trajs']))
                for start in range(0, v['num_trajs'], compute_chunk_size)
            ]
            chunking_results = [_get_input_chunk(s) for s in traj_slices]

            chunked_input = valuewise_sequencify_dicts([
                d['option_policy_kwargs'] for d in chunking_results
            ])
            trans_slices = [d['trans_slice'] for d in chunking_results]
            expanded_actions = [d['expanded_actions'] for d in chunking_results]
            expanded_pre_tanh_values = [d['expanded_pre_tanh_values'] for d in chunking_results]
            del chunking_results

            def _merge(xs, batch_dim):
                # {{{
                assert len(xs) == len(traj_slices)
                assert (trans_slices[0].stop - trans_slices[0].start) != (traj_slices[0].stop - traj_slices[0].start) or traj_slices == trans_slices
                if xs[0].size(batch_dim) == self.num_sampling_options * (trans_slices[0].stop - trans_slices[0].start):
                    slices = trans_slices
                    pure_batch_size = v['num_transitions']
                elif xs[0].size(batch_dim) == self.num_sampling_options * (traj_slices[0].stop - traj_slices[0].start):
                    slices = traj_slices
                    pure_batch_size = v['num_trajs']
                else:
                    assert False
                xs = [
                    x.view(*x.size()[:batch_dim],
                           self.num_sampling_options,
                           s.stop - s.start,
                           *x.size()[batch_dim+1:])
                    for x, s in zip(xs, slices)
                ]
                xs = torch.cat(xs, dim=batch_dim+1)
                assert xs.size()[batch_dim:batch_dim+2] == (self.num_sampling_options, pure_batch_size)
                xs = xs.view(*xs.size()[:batch_dim],
                             self.num_sampling_options * pure_batch_size,
                             *xs.size()[batch_dim+2:])
                return xs
                # }}}

            op_action_dists_flat = self.option_policy.forward_with_chunks(
                    **chunked_input, merge=_merge)[0]
            expanded_actions = _merge(expanded_actions, batch_dim=0)
            if expanded_pre_tanh_values[0] is None:
                expanded_pre_tanh_values = None
            else:
                expanded_pre_tanh_values = _merge(expanded_pre_tanh_values, batch_dim=0)

        assert expanded_actions.size() == (self.num_sampling_options * v['num_transitions'], *v['dims_action'])
        assert op_action_dists_flat.batch_shape == (self.num_sampling_options * v['num_transitions'],)

        expanded_actions_orig = expanded_actions

        if expanded_pre_tanh_values is None:
            op_action_log_probs = op_action_dists_flat.log_prob(expanded_actions)
        else:
            op_action_log_probs = op_action_dists_flat.log_prob(expanded_actions, expanded_pre_tanh_values)
        op_action_log_probs = op_action_log_probs.view(
            self.num_sampling_options, v['num_transitions'])
        op_action_log_probs_transitions = op_action_log_probs.mean(dim=0)

        op_action_log_prob_means = self._convert_trans_vals_to_per_traj_means(v, op_action_log_probs_transitions)
        op_action_log_prob_mean = op_action_log_prob_means.mean()

        if True:
            # {{{
            obs_flat_repeated = unsqueeze_expand_flat_dim0(v['obs_flat'], self.num_alt_samples)
            num_total_alt_options = obs_flat_repeated.size(0)
            expanded_actions = unsqueeze_expand_flat_dim0(v['actions_flat'], self.num_alt_samples)
            if 'pre_tanh_values_flat' in v:
                expanded_pre_tanh_values = unsqueeze_expand_flat_dim0(v['pre_tanh_values_flat'], self.num_alt_samples)
            split_group = self.split_group
            next_alt_log_probs = []
            for i in range((num_total_alt_options + split_group - 1) // split_group):
                start_idx = i * split_group
                end_idx = min((i + 1) * split_group, num_total_alt_options)
                cur_slice = slice(start_idx, end_idx)

                alt_option_samples = torch.tensor(self._sample_step_xi([obs_flat_repeated[cur_slice].size(0)]), device=obs_flat_repeated.device)
                target_policy = self.sampling_policy
                num_option_repeat = 1

                context_manager = nullcontext
                with context_manager():
                    if 'pre_tanh_values_flat' in v:
                        next_alt_log_probs.append(
                            target_policy(
                                self._get_concat_obs(obs_flat_repeated[cur_slice], alt_option_samples, num_option_repeat),
                            )[0].log_prob(expanded_actions[cur_slice], expanded_pre_tanh_values[cur_slice])
                        )
                    else:
                        next_alt_log_probs.append(
                            target_policy(
                                self._get_concat_obs(obs_flat_repeated[cur_slice], alt_option_samples, num_option_repeat),
                            )[0].log_prob(expanded_actions[cur_slice])
                        )
            next_alt_log_probs = torch.cat(next_alt_log_probs, dim=0).view(
                    self.num_alt_samples, v['num_transitions'])

            sp_marginal_log_probs_over_xi = torch.logsumexp(next_alt_log_probs, dim=0) - np.log(self.num_alt_samples)
            sp_marginal_log_probs_over_xi_means = self._convert_trans_vals_to_per_traj_means(v, sp_marginal_log_probs_over_xi)

            v.update({
                'sp_marginal_log_probs_over_xi': sp_marginal_log_probs_over_xi,
                'sp_marginal_log_probs_over_xi_means': sp_marginal_log_probs_over_xi_means,
            })
            # }}}

        tensors.update({
            'OpActionLogProbMean': op_action_log_prob_mean,
        })

        v.update({
            'op_action_dists_flat': op_action_dists_flat,
            'op_action_log_probs': op_action_log_probs,
            'op_action_log_probs_transitions': op_action_log_probs_transitions,
            'op_action_log_prob_means': op_action_log_prob_means,
            'op_action_log_prob_mean': op_action_log_prob_mean,
        })


        if True:
            with torch.no_grad():
                # Op action log-probabilities for random options (sampled from prior).
                # {{{
                if compute_chunk_size is None:
                    chunking_results = _get_input_chunk(slice(0, v['num_trajs']), lambda x: self._call_option_prior_net(x).sample())
                    option_policy_kwargs = chunking_results['option_policy_kwargs']
                    random_option_op_action_dists_flat = self.option_policy(**option_policy_kwargs)[0]
                else:
                    chunking_results = [
                        _get_input_chunk(s, lambda x: self._call_option_prior_net(x).sample())
                        for s in traj_slices
                    ]

                    chunked_input = valuewise_sequencify_dicts([
                        d['option_policy_kwargs'] for d in chunking_results
                    ])
                    del chunking_results

                    random_option_op_action_dists_flat = self.option_policy.forward_with_chunks(
                            **chunked_input, merge=_merge)[0]

                random_option_op_action_log_prob_mean = random_option_op_action_dists_flat.log_prob(expanded_actions_orig).mean()
                # }}}
                tensors.update({
                    'OpActionLogProbForRandomOptionMean': random_option_op_action_log_prob_mean,
                })

                # Op action log-probabilities for zero-filled options.
                # {{{
                if compute_chunk_size is None:
                    chunking_results = _get_input_chunk(
                            slice(0, v['num_trajs']),
                            lambda x: torch.zeros(x.size(0), self.dim_option, device=x.device))
                    option_policy_kwargs = chunking_results['option_policy_kwargs']
                    zero_option_op_action_dists_flat = self.option_policy(**option_policy_kwargs)[0]
                else:
                    chunking_results = [
                        _get_input_chunk(s, lambda x: torch.zeros(x.size(0), self.dim_option, device=x.device))
                        for s in traj_slices
                    ]

                    chunked_input = valuewise_sequencify_dicts([
                        d['option_policy_kwargs'] for d in chunking_results
                    ])
                    del chunking_results

                    zero_option_op_action_dists_flat = self.option_policy.forward_with_chunks(
                            **chunked_input, merge=_merge)[0]

                zero_option_op_action_log_prob_mean = zero_option_op_action_dists_flat.log_prob(expanded_actions_orig).mean()
                # }}}
                tensors.update({
                    'OpActionLogProbForZeroOptionMean': zero_option_op_action_log_prob_mean,
                })

    def _update_pseudo_rewards_sp(self, tensors, v):
        pseudo_reward_inputs = []
        per_traj_pseudo_reward_inputs = []
        grad_term_inputs = []
        def _add_input(x, grad_term=False):
            if v['valids_t'].min() != v['valids_t'].max() and x.ndim == 1:
                x = x.split(v['valids'], dim=0)
                x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0.0)
                assert x.size() == (v['num_trajs'], v['max_traj_length'])
            else:
                x = x.view(v['num_trajs'], v['max_traj_length'])
            pseudo_reward_inputs.append(x.detach())
            if grad_term:
                grad_term_inputs.append(x)
        def _add_per_traj_input(x, grad_term=False):
            x = x.view(v['num_trajs'])
            per_traj_pseudo_reward_inputs.append(x.detach())
            if grad_term:
                raise NotImplementedError
        def _get_grad_mean_term():
            return sum(self._convert_trans_vals_to_per_traj_means(v, x)
                       for x in grad_term_inputs)
        def _get_pseudo_episode_returns():
            with torch.no_grad():
                return (sum(self._convert_trans_vals_to_per_traj_means(v, x)
                            for x in pseudo_reward_inputs)
                        + sum(per_traj_pseudo_reward_inputs))
        def _get_pseudo_rewards():
            with torch.no_grad():
                res = sum(x for x in pseudo_reward_inputs)
                assert res.size() == (v['num_trajs'], v['max_traj_length'])
                if len(per_traj_pseudo_reward_inputs) > 0:
                    res = res + sum(x[:, None].expand(-1, v['max_traj_length'])
                                    for x in per_traj_pseudo_reward_inputs)
                return res
        def _get_pseudo_reward_cumsum():
            with torch.no_grad():
                assert v['valids_t'].min() == v['valids_t'].max()
                if self.pseudo_reward_discount is None:
                    discount_factors = 1.0
                else:
                    discount_factors = torch.ones(v['max_traj_length'], device=self.device)
                    discount_factors[:] = self.pseudo_reward_discount
                    discount_factors = discount_factors.cumprod(0)
                res = ((sum(x for x in pseudo_reward_inputs) * discount_factors).flip([1]).cumsum(dim=1).flip([1])
                       / discount_factors)
                assert res.size() == (v['num_trajs'], v['max_traj_length'])
                if len(per_traj_pseudo_reward_inputs) > 0:
                    res = res + sum(x[:, None].expand(-1, v['max_traj_length'])
                                    for x in per_traj_pseudo_reward_inputs)
                return res

        v['funcs'] = v.get('funcs', {})
        v['funcs'].update({
            'pr_add_input': _add_input,
            'pr_add_per_traj_input': _add_per_traj_input,
            'pr_get_grad_mean_term': _get_grad_mean_term,
            'pr_get_pseudo_episode_returns': _get_pseudo_episode_returns,
            'pr_get_pseudo_rewards': _get_pseudo_rewards,
            'pr_get_pseudo_reward_cumsum': _get_pseudo_reward_cumsum,
        })

        _add_input(v['op_action_log_probs_transitions'])

        _add_input(self.alpha * (- v['sp_marginal_log_probs_over_xi']),
                   grad_term=False)

        # option_penalty_term: Compression term
        _add_input(- self.beta * v['option_penalty'])

        # xi_aux_term: Auxiliary term
        # {{{
        xi_aux_term = v['te_option_log_probs']
        _add_per_traj_input(self.xi_aux_coef * xi_aux_term)
        # }}}

    def _update_loss_sp(self, tensors, v):
        sp_action_log_prob_aggs = (v['sp_action_log_prob_means'] * self._cur_max_path_length) if self._log_prob_mean else v['sp_action_log_prob_sums']

        # Prevent loss_surrogate_sp.backward() from computing gradients w.r.t. option_policy and traj_encoder.
        if not self._train_child_policy and not self._train_downstream_policy:
            self._update_pseudo_rewards_sp(tensors, v)

            loss_surrogate_sp = (- (
                    sp_action_log_prob_aggs * v['funcs']['pr_get_pseudo_episode_returns']()
                    + v['funcs']['pr_get_grad_mean_term']()
            ).mean())

            pseudo_rewards = torch.cat([
                er[:v]
                for er, v in zip(v['funcs']['pr_get_pseudo_rewards'](), v['valids_t'])
            ], dim=0)
            assert pseudo_rewards.size() == (v['num_transitions'],)
            tensors.update({
                'SpPseudoRewardMean': pseudo_rewards.mean(),
                'SpPseudoRewardMax': pseudo_rewards.max(),
                'SpPseudoRewardMin': pseudo_rewards.min(),
            })
        else:
            rewards = (-self.alpha * v['sp_action_log_probs'].detach() + v['rewards_flat'].view(v['num_trajs'], v['max_traj_length'])).sum(dim=1)
            loss_surrogate_sp = (- (
                    sp_action_log_prob_aggs * rewards.detach()
                    - self.alpha * v['sp_action_log_prob_means']).mean())

        loss_key = 'LossSurrogateSp'
        tensors.update({
            loss_key: loss_surrogate_sp,
        })
        return loss_key

    def _update_loss_te_op_sd(self, tensors, v):
        # option_penalty_term: Compression term for decreasing
        option_penalty_term = (- self.beta * v['option_penalty_mean'])

        # xi_aux_term: Auxiliary term
        # {{{
        if self._sp_step_xi:
            xi_aux_term = v['te_option_log_prob_mean']
            xi_aux_term = self.xi_aux_coef * xi_aux_term
        else:
            xi_aux_term = 0
        # }}}

        # Prevent loss.backward() from computing gradients w.r.t. sampling_policy.
        loss = (- (- self.alpha * v['sp_action_log_prob_mean'].detach()
                   + v['op_action_log_prob_mean']
                   + option_penalty_term
                   + xi_aux_term))

        if self._train_child_policy or self._train_downstream_policy:
            loss = loss * 0.

        loss_key = 'Loss'
        tensors.update({
            loss_key: loss,
        })
        return loss_key

    def _update_extra_metrics_sp_op(self, tensors, v):
        with torch.no_grad():
            if v['dims_action'] == (2,):
                tensors['Actions0'] = v['actions_flat'][:, 0]
                tensors['Actions1'] = v['actions_flat'][:, 1]
            tensors['ActionMin'] = v['actions_flat'].min()
            tensors['ActionMax'] = v['actions_flat'].max()

            def _compute_and_log_entropy(sp_action_dists_flat, op_action_dists_flat):
                sp_action_dists_flat_entropy = v.get('sp_action_dists_flat_entropy', None)
                if sp_action_dists_flat_entropy is None:
                    sp_action_dists_flat_entropy = sp_action_dists_flat.entropy()
                tensors['SpActionEntropyMean'] = sp_action_dists_flat_entropy.mean()
                op_action_dists_flat_entropy = op_action_dists_flat.entropy()
                tensors['OpActionEntropyMean'] = op_action_dists_flat_entropy.mean()
                tensors['SpOpActionEntropyDiffMean'] = (
                        sp_action_dists_flat_entropy
                        - op_action_dists_flat_entropy.view(
                    self.num_sampling_options, v['num_transitions']).mean(dim=0)).mean()

            _compute_and_log_entropy(v['sp_action_dists_flat'], v['op_action_dists_flat'])

            def _compute_and_log_stddev(sp_action_dists_flat, op_action_dists_flat):
                tensors['SpOpActionStdDiffMean'] = (
                        sp_action_dists_flat.stddev
                        - op_action_dists_flat.stddev.view(
                    self.num_sampling_options, v['num_transitions'], *v['dims_action']).mean(dim=0)).mean()

            try:
                _compute_and_log_stddev(v['sp_action_dists_flat'], v['op_action_dists_flat'])
            except NotImplementedError:
                try:
                    _compute_and_log_stddev(Independent(unwrap_dist(v['sp_action_dists_flat']), 1),
                                            Independent(unwrap_dist(v['op_action_dists_flat']), 1))
                except NotImplementedError:
                    pass

            tensors['SpOpActionLogProbDiffMean'] = (
                    v['sp_action_log_probs_transitions'] - v['op_action_log_probs_transitions']).mean()
            try:
                tensors['SpOpActionKLMean'] = torch.distributions.kl.kl_divergence(
                    v['sp_action_dists_repeated'], v['op_action_dists_flat']).mean()
            except NotImplementedError:
                pass

            sp_beta_dist_c1 = get_outermost_dist_attr(v['sp_action_dists_flat'], 'concentration1')
            sp_beta_dist_c0 = get_outermost_dist_attr(v['sp_action_dists_flat'], 'concentration0')
            if sp_beta_dist_c1 is not None and sp_beta_dist_c0 is not None:
                tensors['SpBetaDistC1C0Sums'] = sp_beta_dist_c1 + sp_beta_dist_c0
                tensors['SpBetaDistC1C0Diffs'] = sp_beta_dist_c1 - sp_beta_dist_c0
                tensors['SpBetaDistC1C0SumMean'] = tensors['SpBetaDistC1C0Sums'].mean()
                tensors['SpBetaDistC1C0AbsDiffMean'] = tensors['SpBetaDistC1C0Diffs'].abs().mean()

            op_beta_dist_c1 = get_outermost_dist_attr(v['op_action_dists_flat'], 'concentration1')
            op_beta_dist_c0 = get_outermost_dist_attr(v['op_action_dists_flat'], 'concentration0')
            if op_beta_dist_c1 is not None and op_beta_dist_c0 is not None:
                tensors['OpBetaDistC1C0Sums'] = op_beta_dist_c1 + op_beta_dist_c0
                tensors['OpBetaDistC1C0Diffs'] = op_beta_dist_c1 - op_beta_dist_c0
                tensors['OpBetaDistC1C0SumMean'] = tensors['OpBetaDistC1C0Sums'].mean()
                tensors['OpBetaDistC1C0AbsDiffMean'] = tensors['OpBetaDistC1C0Diffs'].abs().mean()

    def _set_updated_normalized_env_ex_except_sampling_policy(self, runner, infos):
        mean = np.mean(infos['env._obs_mean'], axis=0)
        var = np.mean(infos['env._obs_var'], axis=0)

        self._cur_obs_mean = mean
        self._cur_obs_std = var ** 0.5

        runner.set_hanging_env_update(
            dict(
                _obs_mean=mean,
                _obs_var=var,
            ),
            sampler_keys=['option_policy', 'local_sampling_policy', 'local_option_policy'],
        )

    def _evaluate_policy(self, runner):
        if not self._train_child_policy and not self._train_downstream_policy:
            self._plot_op_from_preset(runner)
        elif self._train_child_policy:
            self._plot_sp(runner)

    def _plot_sp(self, runner):
        sp_sampling_extras = self._generate_sampling_extras(self.num_eval_options)
        sp_trajectories = self._get_trajectories(
            runner,
            sampler_key='sampling_policy',
            extras=sp_sampling_extras,
            update_normalizer=False,
            max_path_length_override=self._cur_max_path_length,
            worker_update=dict(
                _deterministic_initial_state=False,
                _deterministic_policy=False,
            ),
        )
        data = self.process_samples(sp_trajectories)
        sp_obs = data['obs']
        zero_options = np.zeros((len(sp_obs), self.dim_option))
        sp_option_means_all, sp_option_stddevs_all, sp_option_samples_all = zero_options, np.ones((len(sp_obs), self.dim_option)), zero_options
        sp_option_means_all = sp_option_means_all[:, None, :]
        sp_option_colors = get_option_colors(sp_option_means_all[:, 0])
        with FigManager(runner, f'EvalSp__TrajPlot') as fm:
            runner._env.render_trajectories(
                sp_trajectories, sp_option_colors, self.eval_plot_axis, fm.ax
            )

    def _plot_op_from_preset(self, runner):
        if self.dim_option == 2:
            eval_options = [[0., 0.]]
            for dist in [1.5, 3.0, 4.5, 0.75, 2.25, 3.75]:
                for angle in [0, 4, 2, 6, 1, 5, 3, 7]:
                    eval_options.append([dist * np.cos(angle * np.pi / 4), dist * np.sin(angle * np.pi / 4)])
            eval_options = eval_options[:self.num_eval_options]
            eval_options = np.array(eval_options)
            eval_options = np.repeat(eval_options, 2, axis=0)
            eval_option_colors = get_option_colors(eval_options)
            eval_options = eval_options / 4.5 * 3.0

            preset_op_trajectories = self._get_trajectories(
                runner,
                sampler_key='option_policy',
                extras=self._generate_option_extras(eval_options),
                max_path_length_override=self._cur_max_path_length,
                worker_update=dict(
                    _deterministic_initial_state=False,
                    _deterministic_policy=False,
                ),
            )

            with FigManager(runner, 'EvalOp__TrajPlotWithCFromPreset') as fm:
                runner._env.render_trajectories(
                    preset_op_trajectories, eval_option_colors, self.eval_plot_axis, fm.ax
                )

            # with aim_wrapper.AimContext({'phase': 'eval', 'policy': 'option'}):
            #     log_performance_ex(
            #         runner.step_itr,
            #         TrajectoryBatch.from_trajectory_list(self._env_spec, preset_op_trajectories),
            #         discount=self.discount,
            #         additional_records=preset_op_trajectories,
            #         additional_prefix=type(runner._env.unwrapped).__name__,
            #     )

    def _log_eval_metrics(self, runner):
        runner.eval_log_diagnostics()
        runner.plot_log_diagnostics()

import copy
import pathlib
import time

import dowel_wrapper
import akro
import numpy as np
import torch
from garage.envs import EnvSpec
from garage.misc.tensor_utils import discount_cumsum
from matplotlib import figure
from sklearn import decomposition

class EnvSpecEx(EnvSpec):
    def __init__(self,
                 observation_space,
                 action_space,
                 pure_observation_space,
                 option_space,
                 ):
        super().__init__(observation_space, action_space)
        self.pure_observation_space = pure_observation_space
        self.option_space = option_space


def make_env_spec_for_option_policy(env_spec, num_option_params, num_repeats=1, use_option=True):
    option_space = None
    if use_option:
        option_space = akro.Box(low=-np.inf, high=np.inf, shape=(num_option_params * num_repeats,))
        space = akro.concat(env_spec.observation_space, option_space)
    else:
        space = env_spec.observation_space
    new_spec = EnvSpecEx(
        action_space=env_spec.action_space,
        observation_space=space,
        pure_observation_space=env_spec.observation_space,
        option_space=option_space,
    )
    return new_spec


def get_torch_concat_obs(obs, option, num_repeats, dim=1):
    concat_obs = torch.cat([obs] + [option] * num_repeats, dim=dim)
    return concat_obs


def get_np_concat_obs(obs, option, num_repeats=1):
    concat_obs = np.concatenate([obs] + [option] * num_repeats)
    return concat_obs


def get_normalizer_preset(normalizer_type):
    if normalizer_type == 'off':
        normalizer_mean = np.array([0.])
        normalizer_std = np.array([1.])
    elif normalizer_type == 'half_cheetah_preset':
        normalizer_mean = np.array(
            [-0.07861924, -0.08627162, 0.08968642, 0.00960849, 0.02950368, -0.00948337, 0.01661406, -0.05476654,
             -0.04932635, -0.08061652, -0.05205841, 0.04500197, 0.02638421, -0.04570961, 0.03183838, 0.01736591,
             0.0091929, -0.0115027])
        normalizer_std = np.array(
            [0.4039283, 0.07610687, 0.23817, 0.2515473, 0.2698137, 0.26374814, 0.32229397, 0.2896734, 0.2774097,
             0.73060024, 0.77360505, 1.5871304, 5.5405455, 6.7097645, 6.8253727, 6.3142195, 6.417641, 5.9759197])
    elif normalizer_type == 'ant_preset':
        normalizer_mean = np.array(
            [0.00486117, 0.011312, 0.7022248, 0.8454677, -0.00102548, -0.00300276, 0.00311523, -0.00139029,
             0.8607109, -0.00185301, -0.8556998, 0.00343217, -0.8585605, -0.00109082, 0.8558013, 0.00278213,
             0.00618173, -0.02584622, -0.00599026, -0.00379596, 0.00526138, -0.0059213, 0.27686235, 0.00512205,
             -0.27617684, -0.0033233, -0.2766923, 0.00268359, 0.27756855])
        normalizer_std = np.array(
            [0.62473416, 0.61958003, 0.1717569, 0.28629342, 0.20020866, 0.20572574, 0.34922406, 0.40098143,
             0.3114514, 0.4024826, 0.31057045, 0.40343934, 0.3110796, 0.40245822, 0.31100526, 0.81786263, 0.8166509,
             0.9870919, 1.7525449, 1.7468817, 1.8596431, 4.502961, 4.4070187, 4.522444, 4.3518476, 4.5105968,
             4.3704205, 4.5175962, 4.3704395])
    elif normalizer_type == 'hopper_preset':
        normalizer_mean = np.array(
            [-0.06534887, 1.16330826, -0.43862215, -0.30512918, -0.27676193, 0.13715465,
             -0.37676255, -0.64648255, -2.84027584, -1.9038601, -1.81660088, 0.78045179])
        normalizer_std = np.array(
            [0.09497205, 0.11141675, 0.49826073, 0.41885291, 0.39282394, 0.26305757,
             0.56293281, 0.82921643, 2.68054553, 2.57605108, 2.37854084, 2.03616333])
    elif normalizer_type == 'humanoid_preset':
        normalizer_mean = np.array(
            [-8.1131503e-02, -7.3915249e-04, 9.5715916e-01, 9.5207644e-01, 2.0175683e-03, -6.3051097e-02,
             -1.2828799e-02, -5.4687279e-04, -2.4450898e-01, 7.7590477e-03, -3.2982033e-02, -1.7136147e-02,
             -1.7263800e-01, -1.6152242e+00, -3.4986842e-02, -3.4458160e-02, -1.6019167e-01, -1.5958424e+00,
             3.0278003e-01, -2.7908441e-01, -3.4809363e-01, -2.9139769e-01, 2.8643531e-01, -3.4040874e-01,
             -3.8491020e-01, 2.6394178e-05, -1.2304888e+00, 3.6492027e-02, -6.8305099e-01, -8.6309865e-02,
             9.3602976e-03, -5.4201365e-01, 1.1908096e-02, -9.6945368e-02, -4.0906958e-02, -3.0476081e-01,
             -3.3397417e+00, -8.6432390e-02, -6.1523411e-02, -2.6818362e-01, -3.3175933e+00, 7.4578458e-01,
             -9.6735454e-01, -1.1773691e+00, -7.7269357e-01, 9.5517111e-01, -1.1721193e+00])
        normalizer_std = np.array(
            [0.12630117, 0.09309318, 0.31789413, 0.07312579, 0.12920779, 0.21994449, 0.1426761, 0.18718153,
             0.43414274, 0.32560128, 0.1282181, 0.23556797, 0.4009979, 0.97610635, 0.12872458, 0.23611404,
             0.4062315, 0.9686742, 0.3580939, 0.42217487, 0.49625927, 0.3586807, 0.4218451, 0.50105387, 0.5517619,
             0.43790612, 0.8357725, 1.3804333, 2.4758842, 2.2540345, 3.15485, 4.4246655, 2.8681147, 2.6601605,
             3.5328803, 5.8904147, 6.434801, 2.6590736, 3.5234997, 5.899381, 6.412176, 2.5906591, 3.0781884,
             3.3108664, 2.5866294, 3.0885093, 3.2871766])
    elif normalizer_type == 'dkitty_randomized_preset':
        normalizer_mean = np.array([
            1.1274621e-03, 4.5083303e-02, -2.3361841e-02, -2.3531685e-02,
            -3.5072537e-04, 7.8615658e-03, -4.6682093e-02, 5.9095848e-01,
            -6.4748323e-01, 4.3181144e-02, 5.9336978e-01, -6.4504164e-01,
            5.5207919e-02, 5.3012931e-01, -6.6083997e-01, -4.5732595e-02,
            5.2520913e-01, -6.7066163e-01, -7.6035224e-02, 6.0535645e-01,
            -1.8569539e-01, 6.8125479e-02, 6.1711168e-01, -1.8589365e-01,
            1.0672496e-01, 3.3246967e-01, -2.1466903e-01, -9.7962923e-02,
            3.2937735e-01, -2.1739161e-01, -2.2122373e-03, -3.1226452e-03,
            6.3065747e-03, 1.9571438e-05, 4.8977748e-04, 4.8178816e-03,
            8.9417556e-03, -5.5894599e-04, 6.3485024e-03, 1.1505619e-02,
            -5.1445374e-03, 1.5553882e-03, 9.9187011e-01,
        ])
        normalizer_std = np.array([
            0.04832589, 0.05530648, 0.02311891, 0.0888353,
            0.09049863, 0.26108277, 0.16047981, 0.3712226,
            0.43055698, 0.16101387, 0.37294742, 0.43025646,
            0.15507433, 0.34794778, 0.43562445, 0.15434338,
            0.34626138, 0.4418219, 1.1267132, 1.0703346,
            0.9746261, 1.1161678, 1.0643424, 0.96674275,
            1.1675166, 1.0648345, 0.91643167, 1.1538163,
            1.0698338, 0.92239964, 0.5807821, 0.571994,
            0.5790013, 0.5788797, 0.57733214, 0.57801986,
            0.57584053, 0.57899916, 0.5765792, 0.5754438,
            0.58059597, 0.57773757, 0.01137926,
        ])

    return normalizer_mean, normalizer_std


def get_2d_colors(points, min_point, max_point):
    points = np.array(points)
    min_point = np.array(min_point)
    max_point = np.array(max_point)

    colors = (points - min_point) / (max_point - min_point)
    colors = np.hstack((
        colors,
        (2 - np.sum(colors, axis=1, keepdims=True)) / 2,
    ))
    colors = np.clip(colors, 0, 1)
    colors = np.c_[colors, np.full(len(colors), 0.8)]

    return colors


def get_option_colors(options, color_range=4):
    num_options = options.shape[0]
    dim_option = options.shape[1]

    if dim_option <= 2:
        # Use a predefined option color scheme
        if dim_option == 1:
            options_2d = []
            d = 2.
            for i in range(len(options)):
                option = options[i][0]
                if option < 0:
                    abs_value = -option
                    options_2d.append((d - abs_value * d, d))
                else:
                    abs_value = option
                    options_2d.append((d, d - abs_value * d))
            options = np.array(options_2d)
            # options = np.c_[options, options]
        option_colors = get_2d_colors(options, (-color_range, -color_range), (color_range, color_range))
    else:
        if dim_option > 3 and num_options >= 3:
            pca = decomposition.PCA(n_components=3)
            # Add random noises to break symmetry.
            pca_options = np.vstack((options, np.random.randn(dim_option, dim_option)))
            pca.fit(pca_options)
            option_colors = np.array(pca.transform(options))
        elif dim_option > 3 and num_options < 3:
            option_colors = options[:, :3]
        elif dim_option == 3:
            option_colors = options

        max_colors = np.array([color_range] * 3)
        min_colors = np.array([-color_range] * 3)
        if all((max_colors - min_colors) > 0):
            option_colors = (option_colors - min_colors) / (max_colors - min_colors)
        option_colors = np.clip(option_colors, 0, 1)

        option_colors = np.c_[option_colors, np.full(len(option_colors), 0.8)]

    return option_colors


class FigManager:
    def __init__(self, runner, label, extensions=None, subplot_spec=None):
        self.runner = runner
        self.label = label
        self.fig = figure.Figure()
        if subplot_spec is not None:
            self.ax = self.fig.subplots(*subplot_spec).flatten()
        else:
            self.ax = self.fig.add_subplot()

        if extensions is None:
            self.extensions = ['png']
        else:
            self.extensions = extensions

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        plot_paths = [(pathlib.Path(self.runner._snapshotter.snapshot_dir)
                       / 'plots'
                       / f'{self.label}_{self.runner.step_itr}.{extension}') for extension in self.extensions]
        plot_paths[0].parent.mkdir(parents=True, exist_ok=True)
        for plot_path in plot_paths:
            self.fig.savefig(plot_path, dpi=300)
        dowel_wrapper.get_tabular('plot').record(self.label, self.fig)

class MeasureAndAccTime:
    def __init__(self, target):
        assert isinstance(target, list)
        assert len(target) == 1
        self._target = target

    def __enter__(self):
        self._time_enter = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._target[0] += (time.time() - self._time_enter)

class Timer:
    def __init__(self):
        self.t = time.time()

    def __call__(self, msg='', *args, **kwargs):
        print(f'{msg}: {time.time() - self.t:.20f}')
        self.t = time.time()

def valuewise_sequencify_dicts(dicts):
    result = dict((k, []) for k in dicts[0].keys())
    for d in dicts:
        for k, v in d.items():
            result[k].append(v)
    return result

def zip_dict(d):
    keys = list(d.keys())
    values = [d[k] for k in keys]
    for z in zip(*values):
        yield dict((k, v) for k, v in zip(keys, z))

def split_paths(paths, chunking_points):
    assert 0 in chunking_points
    assert len(chunking_points) >= 2
    if len(chunking_points) == 2:
        return

    orig_paths = copy.copy(paths)
    paths.clear()
    for path in orig_paths:
        ei = path
        for s, e in zip(chunking_points[:-1], chunking_points[1:]):
            assert len(set(
                len(v)
                for k, v in path.items()
                if k not in ['env_infos', 'agent_infos']
            )) == 1
            new_path = {
                k: v[s:e]
                for k, v in path.items()
                if k not in ['env_infos', 'agent_infos']
            }
            new_path['dones'][-1] = True

            assert len(set(
                len(v)
                for k, v in path['env_infos'].items()
            )) == 1
            new_path['env_infos'] = {
                k: v[s:e]
                for k, v in path['env_infos'].items()
            }

            assert len(set(
                len(v)
                for k, v in path['agent_infos'].items()
            )) == 1
            new_path['agent_infos'] = {
                k: v[s:e]
                for k, v in path['agent_infos'].items()
            }

            paths.append(new_path)


class RunningMeanStd(object):
    def __init__(self, shape, keep_rate, init):
        # keep_rate < 0 means cumulative average
        # keep_rate >= 0 means exponential moving average

        if keep_rate < 0 or init == 'zero_one':
            self._mean = np.zeros(shape, np.float64)
            self._var = np.ones(shape, np.float64)
        else:
            self._mean = None
            self._var = None
        self.count = 0

        self.keep_rate = keep_rate
        self.init = init

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0, dtype=np.float64)
        batch_var = np.var(arr, axis=0, dtype=np.float64)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        if self.keep_rate < 0:
            delta = batch_mean - self._mean
            tot_count = self.count + batch_count

            new_mean = self._mean + delta * batch_count / tot_count
            m_a = self._var * self.count
            m_b = batch_var * batch_count
            m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
            new_var = m_2 / (self.count + batch_count)

            new_count = batch_count + self.count

            self._mean = new_mean
            self._var = new_var
            self.count = new_count
        else:
            if self._mean is None:
                self._mean = batch_mean
                self._var = batch_var
            else:
                self._mean = self._mean * self.keep_rate + batch_mean * (1 - self.keep_rate)
                self._var = self._var * self.keep_rate + batch_var * (1 - self.keep_rate)

    @property
    def mean(self):
        return self._mean.astype(np.float32)

    @property
    def var(self):
        return self._var.astype(np.float32)

    @property
    def std(self):
        return np.sqrt(self._var).astype(np.float32)

def compute_traj_batch_performance(batch, discount):
    # From log_performance_ex()
    returns = []
    undiscounted_returns = []
    for trajectory in batch.split():
        returns.append(discount_cumsum(trajectory.rewards, discount))
        undiscounted_returns.append(sum(trajectory.rewards))

    return dict(
        undiscounted_returns=undiscounted_returns,
        discounted_returns=[rtn[0] for rtn in returns],
    )


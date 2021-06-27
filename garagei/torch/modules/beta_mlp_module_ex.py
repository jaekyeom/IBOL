import abc

import torch
from torch import nn
from torch.distributions import Beta
from torch.distributions.independent import Independent

from garage.torch.distributions import TanhNormal
from garage.torch.modules.mlp_module import MLPModule
from garage.torch.modules.multi_headed_mlp_module import MultiHeadedMLPModule

from garagei.torch.distributions.transformed_distribution_ex import TransformedDistributionEx


class BetaMLPBaseModuleEx(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 alpha_hidden_sizes=(32, 32),
                 alpha_hidden_nonlinearity=torch.tanh,
                 alpha_hidden_w_init=nn.init.xavier_uniform_,
                 alpha_hidden_b_init=nn.init.zeros_,
                 alpha_output_nonlinearity=torch.nn.Softplus,
                 alpha_output_w_init=nn.init.xavier_uniform_,
                 alpha_output_b_init=nn.init.zeros_,
                 beta_hidden_sizes=(32, 32),
                 beta_hidden_nonlinearity=torch.tanh,
                 beta_hidden_w_init=nn.init.xavier_uniform_,
                 beta_hidden_b_init=nn.init.zeros_,
                 beta_output_nonlinearity=torch.nn.Softplus,
                 beta_output_w_init=nn.init.xavier_uniform_,
                 beta_output_b_init=nn.init.zeros_,
                 min_alpha=1.05,
                 min_beta=1.05,
                 layer_normalization=False,
                 beta_distribution_cls=Beta,
                 distribution_transformations=None):
        super().__init__()

        self._input_dim = input_dim
        self._action_dim = output_dim

        self._alpha_hidden_sizes = alpha_hidden_sizes
        self._alpha_hidden_nonlinearity = alpha_hidden_nonlinearity
        self._alpha_hidden_w_init = alpha_hidden_w_init
        self._alpha_hidden_b_init = alpha_hidden_b_init
        self._alpha_output_nonlinearity = alpha_output_nonlinearity
        self._alpha_output_w_init = alpha_output_w_init
        self._alpha_output_b_init = alpha_output_b_init

        self._beta_hidden_sizes = beta_hidden_sizes
        self._beta_hidden_nonlinearity = beta_hidden_nonlinearity
        self._beta_hidden_w_init = beta_hidden_w_init
        self._beta_hidden_b_init = beta_hidden_b_init
        self._beta_output_nonlinearity = beta_output_nonlinearity
        self._beta_output_w_init = beta_output_w_init
        self._beta_output_b_init = beta_output_b_init

        self._layer_normalization = layer_normalization
        self._beta_dist_class = beta_distribution_cls
        self._distribution_transformations = distribution_transformations

        if min_alpha is not None:
            self.register_buffer('min_alpha', torch.Tensor([min_alpha]))
        if min_beta is not None:
            self.register_buffer('min_beta', torch.Tensor([min_beta]))

    def _maybe_move_distribution_transformations(self):
        device = next(self.parameters()).device
        if self._distribution_transformations is not None:
            self._distribution_transformations = [
                t.maybe_clone_to_device(device)
                for t in self._distribution_transformations
            ]
    # Parent module's .to(), .cpu(), and .cuda() call children's ._apply().
    def _apply(self, *args, **kwargs):
        ret = super()._apply(*args, **kwargs)
        self._maybe_move_distribution_transformations()
        return ret

    @property
    def min_alpha(self):
        return self.named_buffers().get('min_alpha', None)
    @property
    def min_beta(self):
        return self.named_buffers().get('min_beta', None)

    @abc.abstractmethod
    def _get_alpha_and_beta(self, *inputs):
        pass

    def forward(self, *inputs):
        alpha, beta = self._get_alpha_and_beta(*inputs)

        if self.min_alpha:
            alpha = alpha + self.min_alpha
        if self.min_beta:
            beta = beta + self.min_beta

        dist = self._beta_dist_class(alpha, beta)
        if self._distribution_transformations is not None:
            dist = TransformedDistributionEx(
                    dist,
                    self._distribution_transformations)

        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pretanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist, TanhNormal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)

        return dist

    def forward_mode(self, *inputs):
        alpha, beta = self._get_alpha_and_beta(*inputs)

        if self.min_alpha:
            alpha = alpha + self.min_alpha
        if self.min_beta:
            beta = beta + self.min_beta

        assert torch.all(alpha > 1).item() and torch.all(beta > 1).item()

        mode_samples = (alpha - 1.0) / (alpha + beta - 2.0)
        if self._distribution_transformations is not None:
            for transform in self._distribution_transformations:
                mode_samples = transform(mode_samples)
        return mode_samples

    def forward_with_transform(self, *inputs, transform):
        alpha, beta = self._get_alpha_and_beta(*inputs)

        if self.min_alpha:
            alpha = alpha + self.min_alpha
        if self.min_beta:
            beta = beta + self.min_beta

        dist = self._beta_dist_class(alpha, beta)
        if self._distribution_transformations is not None:
            dist = TransformedDistributionEx(
                    dist,
                    self._distribution_transformations)

        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pretanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist, TanhNormal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)

        alpha = transform(alpha)
        beta = transform(beta)

        dist_transformed = self._beta_dist_class(alpha, beta)
        if self._distribution_transformations is not None:
            dist_transformed = TransformedDistributionEx(
                    dist_transformed,
                    self._distribution_transformations)

        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pretanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist_transformed, TanhNormal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist_transformed.batch_shape samples.
            dist_transformed = Independent(dist_transformed, 1)

        return dist, dist_transformed

    def forward_with_chunks(self, *inputs, merge):
        alpha = []
        beta = []
        for chunk_inputs in zip(*inputs):
            chunk_alpha, chunk_beta = self._get_alpha_and_beta(*chunk_inputs)
            alpha.append(chunk_alpha)
            beta.append(chunk_beta)
        alpha = merge(alpha, batch_dim=0)
        beta = merge(beta, batch_dim=0)

        if self.min_alpha:
            alpha = alpha + self.min_alpha
        if self.min_beta:
            beta = beta + self.min_beta

        dist = self._beta_dist_class(alpha, beta)
        if self._distribution_transformations is not None:
            dist = TransformedDistributionEx(
                    dist,
                    self._distribution_transformations)

        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pretanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist, TanhNormal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)

        return dist



class BetaMLPIndependentModulesEx(BetaMLPBaseModuleEx):
    def __init__(self,
                 input_dim,
                 output_dim,
                 alpha_hidden_sizes=(32, 32),
                 alpha_hidden_nonlinearity=torch.tanh,
                 alpha_hidden_w_init=nn.init.xavier_uniform_,
                 alpha_hidden_b_init=nn.init.zeros_,
                 alpha_output_nonlinearity=torch.nn.Softplus,
                 alpha_output_w_init=nn.init.xavier_uniform_,
                 alpha_output_b_init=nn.init.zeros_,
                 beta_hidden_sizes=(32, 32),
                 beta_hidden_nonlinearity=torch.tanh,
                 beta_hidden_w_init=nn.init.xavier_uniform_,
                 beta_hidden_b_init=nn.init.zeros_,
                 beta_output_nonlinearity=torch.nn.Softplus,
                 beta_output_w_init=nn.init.xavier_uniform_,
                 beta_output_b_init=nn.init.zeros_,
                 min_alpha=1.05,
                 min_beta=1.05,
                 layer_normalization=False,
                 beta_distribution_cls=Beta,
                 distribution_transformations=None):
        super(BetaMLPIndependentModulesEx,
              self).__init__(input_dim=input_dim,
                             output_dim=output_dim,
                             alpha_hidden_sizes=alpha_hidden_sizes,
                             alpha_hidden_nonlinearity=alpha_hidden_nonlinearity,
                             alpha_hidden_w_init=alpha_hidden_w_init,
                             alpha_hidden_b_init=alpha_hidden_b_init,
                             alpha_output_nonlinearity=alpha_output_nonlinearity,
                             alpha_output_w_init=alpha_output_w_init,
                             alpha_output_b_init=alpha_output_b_init,
                             beta_hidden_sizes=beta_hidden_sizes,
                             beta_hidden_nonlinearity=beta_hidden_nonlinearity,
                             beta_hidden_w_init=beta_hidden_w_init,
                             beta_hidden_b_init=beta_hidden_b_init,
                             beta_output_nonlinearity=beta_output_nonlinearity,
                             beta_output_w_init=beta_output_w_init,
                             beta_output_b_init=beta_output_b_init,
                             min_alpha=min_alpha,
                             min_beta=min_beta,
                             layer_normalization=layer_normalization,
                             beta_distribution_cls=beta_distribution_cls,
                             distribution_transformations=distribution_transformations)

        self._alpha_module = MLPModule(
            input_dim=self._input_dim,
            output_dim=self._action_dim,
            hidden_sizes=self._alpha_hidden_sizes,
            hidden_nonlinearity=self._alpha_hidden_nonlinearity,
            hidden_w_init=self._alpha_hidden_w_init,
            hidden_b_init=self._alpha_hidden_b_init,
            output_nonlinearity=self._alpha_output_nonlinearity,
            output_w_init=self._alpha_output_w_init,
            output_b_init=self._alpha_output_b_init,
            layer_normalization=self._layer_normalization)

        self._beta_module = MLPModule(
            input_dim=self._input_dim,
            output_dim=self._action_dim,
            hidden_sizes=self._beta_hidden_sizes,
            hidden_nonlinearity=self._beta_hidden_nonlinearity,
            hidden_w_init=self._beta_hidden_w_init,
            hidden_b_init=self._beta_hidden_b_init,
            output_nonlinearity=self._beta_output_nonlinearity,
            output_w_init=self._beta_output_w_init,
            output_b_init=self._beta_output_b_init,
            layer_normalization=self._layer_normalization)

    def _get_alpha_and_beta(self, *inputs):
        return self._alpha_module(*inputs), self._beta_module(*inputs)



class BetaMLPTwoHeadedModuleEx(BetaMLPBaseModuleEx):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=torch.nn.Softplus,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 min_alpha=1.05,
                 min_beta=1.05,
                 layer_normalization=False,
                 beta_distribution_cls=Beta,
                 distribution_transformations=None):
        super(BetaMLPTwoHeadedModuleEx,
              self).__init__(input_dim=input_dim,
                             output_dim=output_dim,
                             alpha_hidden_sizes=hidden_sizes,
                             alpha_hidden_nonlinearity=hidden_nonlinearity,
                             alpha_hidden_w_init=hidden_w_init,
                             alpha_hidden_b_init=hidden_b_init,
                             alpha_output_nonlinearity=output_nonlinearity,
                             alpha_output_w_init=output_w_init,
                             alpha_output_b_init=output_b_init,
                             min_alpha=min_alpha,
                             min_beta=min_beta,
                             layer_normalization=layer_normalization,
                             beta_distribution_cls=beta_distribution_cls,
                             distribution_transformations=distribution_transformations)

        self._shared_alpha_beta_network = MultiHeadedMLPModule(
            n_heads=2,
            input_dim=self._input_dim,
            output_dims=self._action_dim,
            hidden_sizes=self._alpha_hidden_sizes,
            hidden_nonlinearity=self._alpha_hidden_nonlinearity,
            hidden_w_init=self._alpha_hidden_w_init,
            hidden_b_init=self._alpha_hidden_b_init,
            output_nonlinearities=self._alpha_output_nonlinearity,
            output_w_inits=self._alpha_output_w_init,
            output_b_inits=self._alpha_output_b_init,
            layer_normalization=self._layer_normalization)

    def _get_alpha_and_beta(self, *inputs):
        return self._shared_alpha_beta_network(*inputs)


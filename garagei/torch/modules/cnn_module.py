import copy
import itertools
import numpy as np
import torch
from torch import nn

from iod.utils import zip_dict


class CNNModule(nn.Module):
    def __init__(self,
                 *,
                 input_dim,
                 hidden_sizes,
                 kernel_sizes,
                 hidden_nonlinearity=nn.ReLU,
                 post_cnn_module,
                 num_reduced_obs=None,
                 use_delta=False,
                 omit_obs_idxs=None,
                 ):
        super().__init__()

        self.input_dim = input_dim

        modules = []
        last_dim = input_dim
        for i, hs, ks in zip(itertools.count(), hidden_sizes, kernel_sizes):
            if i > 0:
                modules.append(hidden_nonlinearity())
            modules.append(nn.Conv1d(
                    last_dim, hs, kernel_size=ks))
        self.cnn = nn.Sequential(*modules)

        self.post_cnn_module = post_cnn_module
        self.num_reduced_obs = num_reduced_obs
        self.use_delta = use_delta
        self.omit_obs_idxs = omit_obs_idxs

    def _forward_cnn(self, x):
        # x: (Batch (list), Time, Dim)

        # TODO: Refactor below
        batch_size = len(x)

        x = copy.copy(x)

        if self.omit_obs_idxs is not None:
            for i in range(batch_size):
                x[i] = x[i].clone()
                x[i][:, self.omit_obs_idxs] = 0

        if self.num_reduced_obs is not None:
            # Reduce into "self.num_reduced_obs" equally spaced obs
            for i in range(batch_size):
                if x[i].size(0) <= self.num_reduced_obs:
                    continue
                # Sample from reversed linspace to ensure sampling the last element when num_reduced_obs == 1
                idxs = np.flip(np.round(
                    np.linspace(x[i].size(0) - 1, 0, self.num_reduced_obs)
                ).astype(int)).copy()
                x[i] = x[i][idxs]

        if self.use_delta:
            for i in range(batch_size):
                x[i] = x[i][1:] - x[i][:-1]

        x = torch.stack(x, dim=0)  # Assume that all of the lengths are same
        x = x.transpose(1, 2)
        cnn_out = self.cnn(x)
        cnn_out_mean = cnn_out.mean(dim=2)

        return cnn_out_mean

    def forward(self, x, **kwargs):
        cnn_out = self._forward_cnn(x)
        out = self.post_cnn_module(cnn_out)
        return out, None

    def forward_with_transform(self, x, *, transform, **kwargs):
        cnn_out = self._forward_cnn(x)
        out = self.post_cnn_module.forward_with_transform(cnn_out, transform=transform)
        return out, None

    def forward_with_chunks(self, x, *, merge, **kwargs):
        cnn_out = []
        for chunk_x, chunk_kwargs in zip(x, zip_dict(kwargs)):
            chunk_cnn_out = self._forward_cnn(chunk_x)
            cnn_out.append(chunk_cnn_out)
        out = self.post_cnn_module.forward_with_chunks(cnn_out, merge=merge)
        return out, None

    def forward_force_only_last(self, x, **kwargs):
        cnn_out = self._forward_cnn(x)
        out = self.post_cnn_module(cnn_out)
        return out, None



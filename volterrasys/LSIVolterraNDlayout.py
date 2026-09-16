import math

import torch
from torch import nn
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter


def _glorot_uniform_(tensor):
    """Keras GlorotUniform initialization for channels-last kernels."""
    shape = tuple(tensor.shape)
    if len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in, fan_out = shape
    else:
        receptive_field_size = math.prod(shape[:-2])
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size

    limit = math.sqrt(6.0 / (fan_in + fan_out))
    with torch.no_grad():
        tensor.uniform_(-limit, limit)


class LSIVolterraNDlayout(LazyModuleMixin, nn.Module):
    """Linear shift invariant multiresolution Volterra kernel layout for D-dimensional data.

    Present support: 1D, 2D, and 3D channels-last tensors.

    VolterraSys-PyTorch: Multidimensional linear and nonlinear Volterra kernel layers in wavelet and natural bases.
    Copyright 2026 Kishore Kumar Tarafdar.
    Licensed under the Apache License, Version 2.0. See LICENSE for details.

    Linear (m=1) shift invariant multiresolution Volterra kernel layout for D-dimensional input.
    Input: general D-dimensional signal x[n1,n2,...,nD]
    Output: Linear shift invariant D-dimensional monomial y1[n1,n2,...,nD]
    """

    def __init__(self, filters=1, kernel_size=4, wave='haar', **kwargs):
        super().__init__()
        self.L = kernel_size
        self.filters = filters
        self.wave = wave
        if self.wave is None:
            self.mra = False
        else:
            self.mra = True

        self.N = None
        self.axes = None
        self.dim = None
        self.channels = None
        self.h1 = UninitializedParameter()

    def initialize_parameters(self, x):
        if self.N is None or self.has_uninitialized_params():
            self.build(x.shape, device=x.device, dtype=self.h1.dtype)

    def build(self, input_shape, device=None, dtype=None):
        if device is None:
            device = self.h1.device
        if dtype is None:
            dtype = self.h1.dtype

        self.N = int(input_shape[1])                 # sequence length
        self.axes = list(range(1, len(input_shape) - 1))
        self.dim = self.axes[-1]
        self.channels = int(input_shape[-1])         # number of channels

        if not isinstance(self.h1, UninitializedParameter):
            return self

        # Multi-channel kernel: ([L] * dim, channels, filters)
        kernel_shape = [self.L] * self.dim + [self.channels, self.filters]
        self.h1.materialize(kernel_shape, device=device, dtype=dtype)
        _glorot_uniform_(self.h1)
        return self

    def forward(self, x):
        pass

    def get_config(self):
        return {
            'kernel_size': self.L,
            'wave': self.wave,
            'filters': self.filters,
        }

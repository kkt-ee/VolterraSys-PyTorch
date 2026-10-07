import math

import torch
from torch import nn
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter

from dwt.DWT1D import DWT1D, IDWT1D
from dwt.multilevel.dwt1 import dwt_packed_axis, idwt_packed_axis
from ._wavelet_ops import (
    natural_to_wavelet_kernel,
    validate_backend,
    validate_level,
    wavelet_to_natural_kernel,
)


def _glorot_uniform_(tensor):
    """Keras GlorotUniform initialization for a channels-last kernel."""
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


class LinearVolterra1D(LazyModuleMixin, nn.Module):
    """Linear (m=1) multiresolution Volterra kernel (shift variant).

    VolterraSys-PyTorch: Multidimensional linear and nonlinear Volterra kernel layers in wavelet and natural bases.
    Copyright 2026 Kishore Kumar Tarafdar.
    Licensed under the Apache License, Version 2.0. See LICENSE for details.

    Input: sequence x[n]
    Output: Linear monomial y1 (shift variant)
    """

    def __init__(
        self,
        filters=1,
        Ny=16,
        wave='haar',
        backend='matrix',
        level=1,
        **kwargs,
    ):
        super().__init__()
        self.wave = wave
        if self.wave is None:
            self.mra = False
        else:
            self.mra = True
        self.filters = filters
        self.Ny = Ny
        self.level = validate_level(level)
        self.backend = validate_backend(backend)

        self.N = None
        self.channels = None
        self.h1 = UninitializedParameter()

        if self.mra:
            self.dwt_input = DWT1D(
                self.wave,
                clean=False,
                backend=self.backend,
            )
            self.idwt_output = IDWT1D(
                self.wave,
                clean=False,
                backend=self.backend,
            )
        else:
            self.dwt_input = None
            self.idwt_output = None

        self.x = None

    def initialize_parameters(self, x):
        if self.N is None or self.has_uninitialized_params():
            self.build(x.shape, device=x.device, dtype=self.h1.dtype)

    def build(self, input_shape, device=None, dtype=None):
        if device is None:
            device = self.h1.device
        if dtype is None:
            dtype = self.h1.dtype

        self.N = int(input_shape[1])
        self.channels = int(input_shape[-1])

        if not isinstance(self.h1, UninitializedParameter):
            return self

        kernel_shape = (self.Ny, self.N, self.channels, self.filters)
        self.h1.materialize(kernel_shape, device=device, dtype=dtype)
        _glorot_uniform_(self.h1)
        if self.mra:
            with torch.no_grad():
                self.h1.copy_(natural_to_wavelet_kernel(
                    self.h1,
                    self.wave,
                    order=1,
                    level=self.level,
                ))
        return self

    def forward(self, x):
        self.x = x
        if self.mra is True:
            return self.__compute_output_with_mra_kernel(x)
        else:
            return self.__compute_output_in_natural_domain(x)

    def __compute_output_with_mra_kernel(self, x):
        HT = self.__make_H()

        # H[v,l] and alpha[l].
        if self.level == 1:
            alpha = self.dwt_input(x)
        else:
            alpha = dwt_packed_axis(
                x,
                level=self.level,
                wave=self.wave,
                axis=1,
                backend=self.backend,
            )
        beta = torch.einsum('bic,uico->buo', alpha, HT)
        if self.level == 1:
            y = self.idwt_output(beta)
        else:
            y = idwt_packed_axis(
                beta,
                level=self.level,
                wave=self.wave,
                axis=1,
                backend=self.backend,
            )
        return y

    def __compute_output_in_natural_domain(self, x):
        h = self.__make_h()
        ynatural = torch.einsum('bic,uico->buo', x, h)
        return ynatural

    def sanity_check(self):
        return (
            self.__compute_output_with_mra_kernel(self.x),
            self.__compute_output_in_natural_domain(self.x),
        )

    def __make_H(self):
        return self.h1

    def __make_h(self):
        if not self.mra:
            return self.h1
        return wavelet_to_natural_kernel(
            self.h1,
            self.wave,
            order=1,
            level=self.level,
        )

    def get_config(self):
        return {
            'Ny': self.Ny,
            'wave': self.wave,
            'level': self.level,
            'filters': self.filters,
            'backend': self.backend,
        }


if __name__ == '__main__':
    N, channels = 4, 1
    filters, Ny = 1, 16
    layer = LinearVolterra1D(filters=filters, Ny=Ny, wave='haar')
    x = torch.randn(1, N, channels)
    y = layer(x)
    print('x shape:', x.shape)
    print('y shape:', y.shape)

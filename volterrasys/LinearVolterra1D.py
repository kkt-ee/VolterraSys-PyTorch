import math

import torch
from torch import nn
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter

from dwt.DWT1D import DWT1D, IDWT1D


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

    def __init__(self, filters=1, Ny=16, wave='haar', **kwargs):
        super().__init__()
        self.wave = wave
        if self.wave is None:
            self.mra = False
        else:
            self.mra = True
        self.filters = filters
        self.Ny = Ny

        self.N = None
        self.channels = None
        self.h1 = UninitializedParameter()

        if self.mra:
            self.S_input = UninitializedBuffer()
            self.idwt_input = IDWT1D(self.wave, clean=False)
            self.dwt_input = DWT1D(self.wave, clean=False)
            self.dwt_output = DWT1D(self.wave, clean=False)
            self.idwt_output = IDWT1D(self.wave, clean=False)
        else:
            self.register_buffer('S_input', None)
            self.idwt_input = None
            self.dwt_input = None
            self.dwt_output = None
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

        if self.mra and isinstance(self.S_input, UninitializedBuffer):
            S = self.idwt_input._get_S(self.N, device).to(dtype=dtype)
            self.S_input.materialize((self.N, self.N), device=device, dtype=dtype)
            with torch.no_grad():
                self.S_input.copy_(S)

        if not isinstance(self.h1, UninitializedParameter):
            return self

        kernel_shape = (self.Ny, self.N, self.channels, self.filters)
        self.h1.materialize(kernel_shape, device=device, dtype=dtype)
        _glorot_uniform_(self.h1)
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
        alpha = self.dwt_input(x)
        beta = torch.einsum('bic,uico->buo', alpha, HT)
        y = self.idwt_output(beta)
        return y

    def __compute_output_in_natural_domain(self, x):
        ynatural = torch.einsum('bic,uico->buo', x, self.h1)
        return ynatural

    def sanity_check(self):
        return (
            self.__compute_output_with_mra_kernel(self.x),
            self.__compute_output_in_natural_domain(self.x),
        )

    def __make_H(self):
        def mra_kernel(h):
            shape = h.shape
            Ny, N, channels, filters = shape

            # Combine channels and filters: (Ny, N, channels * filters).
            h1_reshaped = self.h1.reshape(Ny, N, -1)

            # Apply synthesis on the input axis and DWT on the output axis.
            dwt_once = torch.einsum(
                'uic,il->ulc', h1_reshaped, self.S_input
            )
            dwt_once_T = dwt_once.permute(1, 0, 2)
            dwt_twice = self.dwt_output(dwt_once_T)

            # Return to (Ny, N, channels * filters).
            tmp_H = dwt_twice.permute(1, 0, 2)

            # Restore (Ny, N, channels, filters).
            H = tmp_H.reshape(
                self.h1.shape[0],
                self.h1.shape[1],
                self.h1.shape[2],
                self.h1.shape[3],
            )
            return H

        if self.mra is True:
            return mra_kernel(self.h1)
        else:
            return self.h1

    def get_config(self):
        return {
            'Ny': self.Ny,
            'wave': self.wave,
            'filters': self.filters,
        }


if __name__ == '__main__':
    N, channels = 4, 1
    filters, Ny = 1, 16
    layer = LinearVolterra1D(filters=filters, Ny=Ny, wave='haar')
    x = torch.randn(1, N, channels)
    y = layer(x)
    print('x shape:', x.shape)
    print('y shape:', y.shape)

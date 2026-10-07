import math
import string

import torch
from torch import nn
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter

from dwt.DWT1D import DWT1D, IDWT1D
from dwt.multilevel.dwt1 import dwt_packed_axis, idwt_packed_axis
from ._wavelet_ops import (
    analysis_axis,
    natural_to_wavelet_kernel,
    synthesis_axis,
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


class Volterra1D(LazyModuleMixin, nn.Module):
    """General mth-order multiresolution Volterra kernel (shift variant).

    VolterraSys-PyTorch: Multidimensional linear and nonlinear Volterra kernel layers in wavelet and natural bases.
    Copyright 2026 Kishore Kumar Tarafdar.
    Licensed under the Apache License, Version 2.0. See LICENSE for details.

    Input: sequence x[n]
    Output: mth-order monomial ym (shift variant)
    """

    def __init__(
        self,
        m=1,
        filters=1,
        Ny=16,
        wave='haar',
        backend='matrix',
        level=1,
        **kwargs,
    ):
        super().__init__()
        if isinstance(m, bool) or not isinstance(m, int) or m < 1:
            raise ValueError("m must be a positive integer.")

        self.m = m
        self.wave = wave
        self.mra = self.wave is not None
        self.filters = filters
        self.Ny = Ny
        self.level = validate_level(level)
        self.backend = validate_backend(backend)

        self._make_einsum_equations()

        self.N = None
        self.channels = None
        self.hm = UninitializedParameter()

        if self.mra and self.backend == 'matrix':
            self.A_input = UninitializedBuffer()
            self.S_output = UninitializedBuffer()
        else:
            self.register_buffer('A_input', None)
            self.register_buffer('S_output', None)

        self.x = None

    def _make_einsum_equations(self):
        labels = [label for label in string.ascii_letters if label not in 'buo']
        if 3 * self.m > len(labels):
            raise ValueError("m is too large for a single vectorized einsum.")

        input_indices = labels[:self.m]
        channel_indices = labels[self.m:2 * self.m]
        coefficient_indices = labels[2 * self.m:3 * self.m]

        input_terms = [
            f'b{input_index}{channel_index}'
            for input_index, channel_index in zip(
                input_indices, channel_indices
            )
        ]
        kernel_term = (
            'u'
            + ''.join(input_indices)
            + ''.join(channel_indices)
            + 'o'
        )
        self._contraction_equation = (
            ','.join(input_terms + [kernel_term]) + '->buo'
        )

        synthesis_terms = [
            input_index + coefficient_index
            for input_index, coefficient_index in zip(
                input_indices, coefficient_indices
            )
        ]
        transformed_kernel_term = (
            'u'
            + ''.join(coefficient_indices)
            + ''.join(channel_indices)
            + 'o'
        )
        self._kernel_transform_equation = (
            ','.join([kernel_term] + synthesis_terms)
            + '->'
            + transformed_kernel_term
        )

    def initialize_parameters(self, x):
        if self.N is None or self.has_uninitialized_params():
            self.build(x.shape, device=x.device, dtype=self.hm.dtype)

    def build(self, input_shape, device=None, dtype=None):
        if input_shape[1] is None or input_shape[-1] is None:
            raise ValueError("The input length and channel count must be known.")

        if device is None:
            device = self.hm.device
        if dtype is None:
            dtype = self.hm.dtype

        self.N = int(input_shape[1])
        self.channels = int(input_shape[-1])

        if (
            self.mra
            and self.backend == 'matrix'
            and isinstance(self.A_input, UninitializedBuffer)
        ):
            if self.level == 1:
                dwt = DWT1D(self.wave, clean=False)
                idwt = IDWT1D(self.wave, clean=False)
                A_input = dwt._get_A(self.N, device)
                S_output = idwt._get_S(self.Ny, device)
            else:
                A_input = dwt_packed_axis(
                    torch.eye(self.N, device=device, dtype=dtype),
                    level=self.level,
                    wave=self.wave,
                    axis=0,
                    backend='matrix',
                )
                S_output = idwt_packed_axis(
                    torch.eye(self.Ny, device=device, dtype=dtype),
                    level=self.level,
                    wave=self.wave,
                    axis=0,
                    backend='matrix',
                )
            matrices = (
                ('A_input', A_input),
                ('S_output', S_output),
            )
            for name, matrix in matrices:
                buffer = getattr(self, name)
                buffer.materialize(
                    tuple(matrix.shape), device=device, dtype=dtype
                )
                with torch.no_grad():
                    buffer.copy_(matrix.to(dtype=dtype))

        if not isinstance(self.hm, UninitializedParameter):
            return self

        # hm[output, input_1, ..., input_m,
        #    channel_1, ..., channel_m, filter].
        kernel_shape = (
            (self.Ny,)
            + (self.N,) * self.m
            + (self.channels,) * self.m
            + (self.filters,)
        )
        self.hm.materialize(kernel_shape, device=device, dtype=dtype)
        _glorot_uniform_(self.hm)
        if self.mra:
            with torch.no_grad():
                self.hm.copy_(natural_to_wavelet_kernel(
                    self.hm,
                    self.wave,
                    order=self.m,
                    level=self.level,
                ))
        return self

    def forward(self, x):
        self.x = x
        if self.mra:
            return self.__compute_output_with_mra_kernel(x)
        return self.__compute_output_in_natural_domain(x)

    def __compute_output_with_mra_kernel(self, x):
        H = self.__make_H()
        if self.backend == 'matrix':
            alpha = torch.einsum('li,bic->blc', self.A_input, x)
        else:
            alpha = analysis_axis(
                x,
                self.wave,
                axis=1,
                level=self.level,
            )
        beta = torch.einsum(
            self._contraction_equation,
            *([alpha] * self.m),
            H,
        )
        if self.backend == 'matrix':
            return torch.einsum('uv,bvo->buo', self.S_output, beta)
        return synthesis_axis(
            beta,
            self.wave,
            axis=1,
            level=self.level,
        )

    def __compute_output_in_natural_domain(self, x):
        h = self.__make_h()
        return torch.einsum(
            self._contraction_equation,
            *([x] * self.m),
            h,
        )

    def sanity_check(self):
        return (
            self.__compute_output_with_mra_kernel(self.x),
            self.__compute_output_in_natural_domain(self.x),
        )

    def __make_H(self):
        return self.hm

    def __make_h(self):
        if not self.mra:
            return self.hm
        return wavelet_to_natural_kernel(
            self.hm,
            self.wave,
            order=self.m,
            level=self.level,
        )

    def get_config(self):
        return {
            'm': self.m,
            'Ny': self.Ny,
            'wave': self.wave,
            'level': self.level,
            'filters': self.filters,
            'backend': self.backend,
        }


if __name__ == '__main__':
    N, channels = 4, 1
    filters, Ny, m = 1, 16, 3
    layer = Volterra1D(m=m, filters=filters, Ny=Ny, wave='haar')
    x = torch.randn(1, N, channels)
    y = layer(x)
    print('x shape:', x.shape)
    print('y shape:', y.shape)

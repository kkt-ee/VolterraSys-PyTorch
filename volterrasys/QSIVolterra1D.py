import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter

from dwt.DWT1D import DWT1D, IDWT1D
from dwt.DWT2D import DWT2D, IDWT2D
from ._wavelet_ops import synthesis_transpose_axis, validate_backend


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


class QSIVolterra1D(LazyModuleMixin, nn.Module):
    """Quadratic (m=2) shift invariant multiresolution Volterra kernel for sequences.

    VolterraSys-PyTorch: Multidimensional linear and nonlinear Volterra kernel layers in wavelet and natural bases.
    Copyright 2026 Kishore Kumar Tarafdar.
    Licensed under the Apache License, Version 2.0. See LICENSE for details.
    """

    def __init__(
        self,
        filters=1,
        kernel_size=4,
        wave='haar',
        backend='matrix',
        **kwargs,
    ):
        super().__init__()
        self.L = kernel_size
        self.filters = filters
        self.wave = wave
        if self.wave is not None:
            self.mra = True
        else:
            self.mra = False
        self.backend = validate_backend(backend)

        self.N = None
        self.channels = None
        self.h = UninitializedParameter()

        if self.mra:
            if self.backend == 'matrix':
                self.S_input = UninitializedBuffer()
            else:
                self.register_buffer('S_input', None)
            self.dwt1 = DWT1D(
                self.wave,
                clean=False,
                backend=self.backend,
            )
            self.idwt1 = IDWT1D(
                self.wave,
                clean=False,
                backend=self.backend,
            )
            self.dwt2 = DWT2D(
                self.wave,
                clean=False,
                backend=self.backend,
            )
        else:
            self.register_buffer('S_input', None)
            self.dwt1 = None
            self.idwt1 = None
            self.dwt2 = None

        self.x2 = None
        self.h2 = None
        self.h2_shifted = None

    def initialize_parameters(self, x):
        if self.N is None or self.has_uninitialized_params():
            self.build(x.shape, device=x.device, dtype=self.h.dtype)

    def build(self, input_shape, device=None, dtype=None):
        if device is None:
            device = self.h.device
        if dtype is None:
            dtype = self.h.dtype

        self.N = int(input_shape[1])
        self.channels = int(input_shape[-1])

        if (
            self.mra
            and self.backend == 'matrix'
            and isinstance(self.S_input, UninitializedBuffer)
        ):
            S = self.idwt1._get_S(self.N, device).to(dtype=dtype)
            self.S_input.materialize((self.N, self.N), device=device, dtype=dtype)
            with torch.no_grad():
                self.S_input.copy_(S)

        if not isinstance(self.h, UninitializedParameter):
            return self

        kernel_shape = (self.L, self.channels, self.filters)
        self.h.materialize(kernel_shape, device=device, dtype=dtype)
        _glorot_uniform_(self.h)
        return self

    def make_mra_h2(self):
        # Quadratic kernel outer product: (L, channels, filters)^2.
        h2 = torch.einsum('ico,jco->ijco', self.h, self.h)
        self.h2 = h2

        # Pad the two lag axes to (N, N, channels, filters).
        paddings = (
            0, 0,
            0, 0,
            0, self.N - self.L,
            0, self.N - self.L,
        )
        h2_padded = F.pad(h2, paddings, mode='constant', value=0)

        # Create h2[n-k1, n-k2] with circular indices.
        n = torch.arange(self.N, device=self.h.device)
        row_indices = torch.remainder(n[:, None] - n, self.N)
        h2_rows_shifted = h2_padded[row_indices]
        column_indices = row_indices[:, None, :, None, None].expand_as(
            h2_rows_shifted
        )
        h2_columns_shifted = torch.gather(
            h2_rows_shifted, dim=2, index=column_indices
        )
        self.h2_shifted = h2_columns_shifted

        def make_mra_kernel(h_shifted):
            shape = h_shifted.shape
            x = h_shifted.reshape(
                shape[0], shape[1], shape[2], shape[3] * shape[4]
            )
            if self.backend == 'matrix':
                x = torch.einsum('nijc,il->nljc', x, self.S_input)
                x = torch.einsum('nljc,jm->nlmc', x, self.S_input)
            else:
                x = synthesis_transpose_axis(x, self.wave, axis=1)
                x = synthesis_transpose_axis(x, self.wave, axis=2)
            h_init = x.reshape(
                shape[0], shape[1], shape[2], shape[3], shape[4]
            )

            # DWT of the output-index axis.
            shape = h_init.shape
            x = h_init.reshape(
                shape[0], shape[1], shape[2] * shape[3] * shape[4]
            )
            x = x.permute(1, 0, 2)
            x = self.dwt1(x)
            x = x.permute(1, 0, 2)
            H = x.reshape(
                shape[0], shape[1], shape[2], shape[3], shape[4]
            )
            return H

        if self.mra is True:
            return make_mra_kernel(self.h2_shifted)
        else:
            return self.h2_shifted

    def forward(self, x):
        # x: (batch, N, channels)
        self.x2 = torch.einsum('bic,bjc->bijc', x, x)

        if self.mra is False:
            self.make_mra_h2()
            return self.__call_compute_in_natural_domain(self.x2)
        else:
            return self.__call_compute_with_mra_kernel(self.x2)

    def __call_compute_with_mra_kernel(self, x2):
        alpha2 = self.dwt2(x2)
        H = self.make_mra_h2()
        beta = torch.einsum('ijkco,bjkc->bio', H, alpha2)
        y2 = self.idwt1(beta)
        return y2

    def __call_compute_in_natural_domain(self, x2):
        y2 = torch.einsum('ijkco,bjkc->bio', self.h2_shifted, x2)
        return y2

    def sanity_check(self):
        return (
            self.__call_compute_with_mra_kernel(self.x2),
            self.__call_compute_in_natural_domain(self.x2),
        )

    def get_filter(self):
        return self.h2

    def get_config(self):
        return {
            'kernel_size': self.L,
            'wave': self.wave,
            'filters': self.filters,
            'backend': self.backend,
        }


if __name__ == '__main__':
    N, channels, filters = 8, 1, 1
    layer = QSIVolterra1D(filters=filters)
    x = torch.randn(1, N, channels)
    y = layer(x)
    print('x shape:', x.shape)
    print('y shape:', y.shape)

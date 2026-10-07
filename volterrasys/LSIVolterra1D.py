import torch
import torch.nn.functional as F
from torch.nn.parameter import UninitializedBuffer

from dwt.DWT1D import DWT1D, IDWT1D
from .LSIVolterraNDlayout import LSIVolterraNDlayout
from ._wavelet_ops import synthesis_transpose_axis, validate_backend


class LSIVolterra1D(LSIVolterraNDlayout):
    """LSI 1D wavelet and natural basis kernel.

    VolterraSys-PyTorch: Multidimensional linear and nonlinear Volterra kernel layers in wavelet and natural bases.
    Copyright 2026 Kishore Kumar Tarafdar.
    Licensed under the Apache License, Version 2.0. See LICENSE for details.

    Linear (m=1) shift invariant multiresolution Volterra kernel for sequences.
    Input: sequence x[n]
    Output: Linear shift invariant sequence monomial y1[n]
    """

    def __init__(
        self,
        filters=1,
        kernel_size=4,
        wave='haar',
        backend='matrix',
        **kwargs,
    ):
        super().__init__(filters=filters, kernel_size=kernel_size, wave=wave, **kwargs)
        self.backend = validate_backend(backend)
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
        else:
            self.register_buffer('S_input', None)
            self.dwt1 = None
            self.idwt1 = None

        self.x = None
        self.row_indices = None
        self.h1_shifted = None

    def build(self, input_shape, device=None, dtype=None):
        if device is None:
            device = self.h1.device
        if dtype is None:
            dtype = self.h1.dtype

        if (
            self.mra
            and self.backend == 'matrix'
            and isinstance(self.S_input, UninitializedBuffer)
        ):
            N = int(input_shape[1])
            S = self.idwt1._get_S(N, device).to(dtype=dtype)
            self.S_input.materialize((N, N), device=device, dtype=dtype)
            with torch.no_grad():
                self.S_input.copy_(S)

        super().build(input_shape, device=device, dtype=dtype)
        return self

    def __makeH(self):
        paddings = (
            0, 0,
            0, 0,
            0, self.N - self.L,
        )
        h_padded = F.pad(self.h1, paddings, mode='constant', value=0)

        # Creating x[n-k] for all n.
        n = torch.arange(self.N, device=self.h1.device)
        self.row_indices = torch.remainder(n[:, None] - n, self.N)

        h1_shifted = h_padded[self.row_indices]
        self.h1_shifted = h1_shifted

        if self.mra is False:
            return h1_shifted                    # natural shifted kernel
        else:
            # Vectorized: flatten the channel and filter axes.
            tmp_h_for_dwt = h1_shifted.reshape(
                self.N, self.N, self.channels * self.filters)

            # Synthesis transform of the input index.
            if self.backend == 'matrix':
                x = torch.einsum(
                    'ric,il->rlc', tmp_h_for_dwt, self.S_input
                )
            else:
                x = synthesis_transpose_axis(
                    tmp_h_for_dwt,
                    self.wave,
                    axis=1,
                )

            # Transpose to swap axes 0 and 1 (rows and columns).
            xT = x.permute(1, 0, 2)

            # DWT1D along the original output index.
            x2 = self.dwt1(xT)

            # Transpose back to the original orientation.
            x2T = x2.permute(1, 0, 2)

            # Restore (N, N, input channels, output filters).
            HT = x2T.reshape(
                self.N, self.N, self.channels, self.filters)
            return HT

    def forward(self, x):
        self.x = x
        if self.mra is True:
            return self._call_with_mra_kernel(x)
        else:
            return self._call_in_natural_domain(x)

    def _call_with_mra_kernel(self, x):
        # H[v,l]
        α = self.dwt1(x)
        HT = self.__makeH()
        print('HT and α', HT.shape, α.shape)
        β = torch.einsum('bic,uico->buo', α, HT)
        print('β', β.shape)
        y = self.idwt1(β)
        return y

    def _call_in_natural_domain(self, x):
        h1_shifted = self.__makeH()              # dummy, retained from TensorFlow source
        return torch.einsum('bic,rico->bro', x, self.h1_shifted)

    def sanity_check(self):
        return self._call_with_mra_kernel(self.x), self._call_in_natural_domain(self.x)

    def get_config(self):
        config = super().get_config()
        config.update({'backend': self.backend})
        return config


if __name__ == '__main__':
    N, channels, filters = 128, 1, 1
    layer = LSIVolterra1D(filters=filters, wave='bior1.3')
    x = torch.randn(1, N, channels)
    y = layer(x)
    print('x shape:', x.shape)
    print('y shape:', y.shape)
    y, ynat = layer.sanity_check()
    print('Natural/MRA maximum difference:', (y - ynat).abs().max().item())

import os
import sys

import pytest
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from volterrasys.LinearVolterra1D import LinearVolterra1D
from volterrasys.Volterra1D import Volterra1D
from volterrasys._wavelet_ops import (
    natural_to_wavelet_kernel,
    wavelet_to_natural_kernel,
)


CASES = (
    ('linear', LinearVolterra1D, {}, 1),
    ('volterra_m1', Volterra1D, {'m': 1}, 1),
    ('volterra_m2', Volterra1D, {'m': 2}, 2),
    ('volterra_m3', Volterra1D, {'m': 3}, 3),
)


def _kernel(layer):
    if isinstance(layer, LinearVolterra1D):
        return layer.h1
    return layer.hm


def _assert_close(actual, expected, tolerance=4e-5):
    torch.testing.assert_close(
        actual,
        expected,
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize('level', [2, 3])
@pytest.mark.parametrize('wave', ['haar', 'bior2.2'])
@pytest.mark.parametrize('name,layer_class,extra,order', CASES)
def test_multilevel_direct_H_matches_matrix_filterbank_and_natural(
    name,
    layer_class,
    extra,
    order,
    wave,
    level,
):
    del name
    torch.manual_seed(610 + level + order)
    x = torch.randn(2, 32, 1)
    common = {
        'filters': 1,
        'Ny': 32,
        'wave': wave,
        'level': level,
    }
    matrix = layer_class(backend='matrix', **common, **extra)
    filterbank = layer_class(backend='filterbank', **common, **extra)
    natural = layer_class(
        filters=1,
        Ny=32,
        wave=None,
        level=level,
        **extra,
    )

    matrix(x)
    filterbank(x)
    natural(x)
    h = torch.randn_like(_kernel(natural)) * 0.01
    H = natural_to_wavelet_kernel(h, wave, order, level=level)
    with torch.no_grad():
        _kernel(matrix).copy_(H)
        _kernel(filterbank).copy_(H)
        _kernel(natural).copy_(h)

    assert torch.equal(_kernel(matrix), H)
    assert torch.equal(_kernel(filterbank), H)
    _assert_close(
        wavelet_to_natural_kernel(H, wave, order, level=level),
        h,
    )
    _assert_close(matrix(x), filterbank(x))
    _assert_close(matrix(x), natural(x))
    _assert_close(filterbank(x), natural(x))


def test_multilevel_filterbank_gradients_match_matrix():
    torch.manual_seed(620)
    x = torch.randn(2, 16, 1)
    probe = torch.randn(2, 16, 1)
    matrix = Volterra1D(
        m=3,
        filters=1,
        Ny=16,
        wave='bior2.2',
        backend='matrix',
        level=2,
    )
    filterbank = Volterra1D(
        m=3,
        filters=1,
        Ny=16,
        wave='bior2.2',
        backend='filterbank',
        level=2,
    )
    matrix(x)
    filterbank(x)
    H = torch.randn_like(matrix.hm) * 0.01
    with torch.no_grad():
        matrix.hm.copy_(H)
        filterbank.hm.copy_(H)

    def gradients(layer):
        variable = x.detach().clone().requires_grad_(True)
        return torch.autograd.grad(
            (layer(variable) * probe).sum(),
            (variable, layer.hm),
        )

    for matrix_gradient, filterbank_gradient in zip(
        gradients(matrix),
        gradients(filterbank),
    ):
        _assert_close(matrix_gradient, filterbank_gradient)


def test_multilevel_supports_different_input_and_output_lengths():
    torch.manual_seed(630)
    x = torch.randn(1, 32, 1)
    matrix = Volterra1D(
        m=2,
        filters=1,
        Ny=16,
        wave='bior2.2',
        backend='matrix',
        level=2,
    )
    filterbank = Volterra1D(
        m=2,
        filters=1,
        Ny=16,
        wave='bior2.2',
        backend='filterbank',
        level=2,
    )
    natural = Volterra1D(m=2, filters=1, Ny=16, wave=None, level=2)
    matrix(x)
    filterbank(x)
    natural(x)
    h = torch.randn_like(natural.hm) * 0.01
    H = natural_to_wavelet_kernel(h, 'bior2.2', 2, level=2)
    with torch.no_grad():
        matrix.hm.copy_(H)
        filterbank.hm.copy_(H)
        natural.hm.copy_(h)

    assert matrix(x).shape == (1, 16, 1)
    _assert_close(matrix(x), filterbank(x))
    _assert_close(matrix(x), natural(x))


@pytest.mark.parametrize(
    'layer',
    [
        LinearVolterra1D(level=3),
        Volterra1D(m=1, level=3),
        Volterra1D(m=2, level=3),
        Volterra1D(m=3, level=3),
    ],
)
def test_level_round_trips_through_config(layer):
    restored = type(layer)(**layer.get_config())
    assert restored.level == 3


def test_multilevel_state_dict_round_trip():
    torch.manual_seed(640)
    x = torch.randn(1, 16, 1)
    source = Volterra1D(
        m=3,
        filters=1,
        Ny=16,
        wave='bior2.2',
        backend='filterbank',
        level=2,
    )
    expected = source(x)
    state = source.state_dict()
    assert set(state) == {'hm'}

    restored = Volterra1D(**source.get_config())
    restored.load_state_dict(state)
    assert restored.level == 2
    _assert_close(restored(x), expected, tolerance=0.0)


@pytest.mark.parametrize('invalid', [0, -1, True, 1.5])
@pytest.mark.parametrize(
    'layer_class,extra',
    [
        (LinearVolterra1D, {}),
        (Volterra1D, {'m': 2}),
    ],
)
def test_level_is_validated(layer_class, extra, invalid):
    with pytest.raises(ValueError, match='level must be a positive integer'):
        layer_class(level=invalid, **extra)


def test_existing_positional_backend_argument_is_preserved():
    assert LinearVolterra1D(1, 8, 'haar', 'filterbank').backend == 'filterbank'
    assert Volterra1D(2, 1, 8, 'haar', 'filterbank').backend == 'filterbank'


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason='MPS is unavailable',
)
@pytest.mark.parametrize('backend', ['matrix', 'filterbank'])
def test_multilevel_direct_H_forward_and_backward_on_mps(backend):
    device = torch.device('mps')
    x = torch.randn(1, 32, 1, device=device, requires_grad=True)
    layer = Volterra1D(
        m=3,
        filters=1,
        Ny=32,
        wave='bior2.2',
        backend=backend,
        level=3,
    ).to(device)
    output = layer(x)
    output.square().sum().backward()
    torch.mps.synchronize()

    assert torch.isfinite(output).all()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.hm.grad).all()

import os
import sys

import pytest
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from volterrasys.LSIVolterra1D import LSIVolterra1D
from volterrasys.LinearVolterra1D import LinearVolterra1D
from volterrasys.QSIVolterra1D import QSIVolterra1D
from volterrasys.Volterra1D import Volterra1D


CASES = (
    'lsi',
    'qsi',
    'linear',
    'volterra_m1',
    'volterra_m2',
    'volterra_m3',
)


def _make_layer(case, wave, backend, ny=8):
    common = {'filters': 3, 'wave': wave, 'backend': backend}
    if case == 'lsi':
        return LSIVolterra1D(kernel_size=3, **common)
    if case == 'qsi':
        return QSIVolterra1D(kernel_size=3, **common)
    if case == 'linear':
        return LinearVolterra1D(Ny=ny, **common)
    if case.startswith('volterra_m'):
        return Volterra1D(m=int(case[-1]), Ny=ny, **common)
    raise AssertionError(f'Unknown test case: {case}')


def _kernel(layer):
    if isinstance(layer, (LSIVolterra1D, LinearVolterra1D)):
        return layer.h1
    if isinstance(layer, QSIVolterra1D):
        return layer.h
    return layer.hm


def _transformed_kernel(layer):
    if isinstance(layer, LSIVolterra1D):
        return layer._LSIVolterra1D__makeH()
    if isinstance(layer, QSIVolterra1D):
        return layer.make_mra_h2()
    if isinstance(layer, LinearVolterra1D):
        return layer._LinearVolterra1D__make_H()
    return layer._Volterra1D__make_H()


def _assert_close(actual, expected, tolerance=3e-5):
    torch.testing.assert_close(
        actual,
        expected,
        rtol=tolerance,
        atol=tolerance,
    )


def _build_with_shared_kernel(case, wave, ny=8, device='cpu'):
    torch.manual_seed(500)
    x = torch.randn(2, 8, 2, device=device)
    matrix = _make_layer(case, wave, 'matrix', ny=ny).to(device)
    filterbank = _make_layer(case, wave, 'filterbank', ny=ny).to(device)
    natural = _make_layer(case, None, 'matrix', ny=ny).to(device)

    matrix(x)
    filterbank(x)
    natural(x)
    values = torch.randn_like(_kernel(matrix)) * 0.1
    with torch.no_grad():
        _kernel(matrix).copy_(values)
        _kernel(filterbank).copy_(values)
        _kernel(natural).copy_(values)
    return x, matrix, filterbank, natural


@pytest.mark.parametrize('wave', ['haar', 'db2', 'bior2.2', 'rbio2.2'])
@pytest.mark.parametrize('case', CASES)
def test_filterbank_matches_matrix_and_natural_domain(case, wave):
    x, matrix, filterbank, natural = _build_with_shared_kernel(case, wave)

    _assert_close(_transformed_kernel(filterbank), _transformed_kernel(matrix))
    matrix_output = matrix(x)
    filterbank_output = filterbank(x)
    natural_output = natural(x)
    _assert_close(filterbank_output, matrix_output)
    _assert_close(filterbank_output, natural_output)

    assert set(filterbank.state_dict()) == {_kernel_name(filterbank)}
    assert len(matrix.state_dict()) > 1


def _kernel_name(layer):
    if isinstance(layer, (LSIVolterra1D, LinearVolterra1D)):
        return 'h1'
    if isinstance(layer, QSIVolterra1D):
        return 'h'
    return 'hm'


@pytest.mark.parametrize('case', CASES)
def test_filterbank_gradients_match_matrix(case):
    x, matrix, filterbank, _ = _build_with_shared_kernel(case, 'bior2.2')
    torch.manual_seed(501)
    probe = torch.randn_like(matrix(x))

    def gradients(layer):
        variable = x.detach().clone().requires_grad_(True)
        output = layer(variable)
        return torch.autograd.grad(
            (output * probe).sum(),
            (variable, _kernel(layer)),
        )

    matrix_gradients = gradients(matrix)
    filterbank_gradients = gradients(filterbank)
    for filterbank_gradient, matrix_gradient in zip(
        filterbank_gradients, matrix_gradients
    ):
        _assert_close(filterbank_gradient, matrix_gradient, tolerance=1e-4)


@pytest.mark.parametrize(
    'layer_class, extra',
    [
        (LSIVolterra1D, {'kernel_size': 3}),
        (QSIVolterra1D, {'kernel_size': 3}),
        (LinearVolterra1D, {'Ny': 8}),
        (Volterra1D, {'m': 3, 'Ny': 8}),
    ],
)
def test_filterbank_backend_round_trips_through_config(layer_class, extra):
    layer = layer_class(
        filters=3,
        wave='bior2.2',
        backend='filterbank',
        **extra,
    )
    restored = layer_class(**layer.get_config())
    assert restored.backend == 'filterbank'


def test_filterbank_state_dict_round_trip():
    torch.manual_seed(502)
    x = torch.randn(2, 8, 2)
    source = Volterra1D(
        m=3,
        filters=3,
        Ny=16,
        wave='bior2.2',
        backend='filterbank',
    )
    expected = source(x)
    state = source.state_dict()
    assert set(state) == {'hm'}

    restored = Volterra1D(
        m=3,
        filters=3,
        Ny=16,
        wave='bior2.2',
        backend='filterbank',
    )
    restored.load_state_dict(state)
    _assert_close(restored(x), expected)


def test_filterbank_supports_a_different_output_length():
    x, matrix, filterbank, natural = _build_with_shared_kernel(
        'volterra_m3', 'bior2.2', ny=16
    )
    _assert_close(_transformed_kernel(filterbank), _transformed_kernel(matrix))
    _assert_close(filterbank(x), matrix(x))
    _assert_close(filterbank(x), natural(x))
    assert filterbank(x).shape == (2, 16, 3)


def test_backend_is_validated():
    with pytest.raises(ValueError, match='backend must be one of'):
        Volterra1D(backend='unknown')


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason='MPS is unavailable',
)
def test_filterbank_forward_and_backward_on_mps():
    device = torch.device('mps')
    torch.manual_seed(503)
    x = torch.randn(1, 8, 1, device=device, requires_grad=True)
    layer = Volterra1D(
        m=3,
        filters=1,
        Ny=8,
        wave='bior2.2',
        backend='filterbank',
    ).to(device)
    output = layer(x)
    output.square().sum().backward()
    torch.mps.synchronize()

    assert torch.isfinite(output).all()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.hm.grad).all()

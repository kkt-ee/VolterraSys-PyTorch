from functools import lru_cache

from dwt.dwt_op import analysis_filterbank_axis, synthesis_filterbank_axis
from dwt.filters import FetchAnalysisSynthesisFilters


_VALID_BACKENDS = ('matrix', 'filterbank')


def validate_backend(backend):
    if backend not in _VALID_BACKENDS:
        choices = ', '.join(repr(value) for value in _VALID_BACKENDS)
        raise ValueError(f'backend must be one of: {choices}.')
    return backend


@lru_cache(maxsize=None)
def _filters(wave):
    filters = FetchAnalysisSynthesisFilters(wave)
    analysis = tuple(tuple(values) for values in filters.analysis())
    synthesis = tuple(tuple(values) for values in filters.synthesis())
    return analysis, synthesis


def analysis_axis(x, wave, axis):
    """Apply the wavelet analysis operator A along one tensor axis."""
    analysis, _ = _filters(wave)
    return analysis_filterbank_axis(x, *analysis, axis=axis)


def synthesis_axis(x, wave, axis):
    """Apply the wavelet synthesis operator S along one tensor axis."""
    _, synthesis = _filters(wave)
    return synthesis_filterbank_axis(x, *synthesis, axis=axis)


def synthesis_transpose_axis(x, wave, axis):
    """Apply S^T along a natural-domain kernel input axis."""
    _, synthesis = _filters(wave)
    return analysis_filterbank_axis(x, *synthesis, axis=axis)

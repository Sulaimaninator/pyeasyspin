"""Calculate autocorrelation function using the FFT."""
from .._registry import F


def autocorrfft(*varargin):
    """autocorrfft(x[, dim, normalized, vector, centered]) -> AutoCorr = crosscorrfft(x, x, ...)."""
    nargin = len(varargin)
    if not 1 <= nargin <= 5:
        raise ValueError('Wrong number of input arguments.')
    x = varargin[0]
    dim = varargin[1] if nargin >= 2 else []
    normalized = varargin[2] if nargin >= 3 else True
    vector = varargin[3] if nargin >= 4 else 0
    centered = varargin[4] if nargin >= 5 else True
    return F.crosscorrfft(x, x, dim, normalized, vector, centered)

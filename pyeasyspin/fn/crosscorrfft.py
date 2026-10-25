"""Calculate cross-correlation function using the FFT."""
import numpy as np
from .._mat import isempty


def _nd(a, n):
    """Pad a's shape with trailing singletons up to n dimensions (MATLAB size(a, k) = 1 for k > ndims)."""
    return a.reshape(a.shape + (1,) * (n - a.ndim)) if a.ndim < n else a


def crosscorrfft(*varargin):
    """crosscorrfft(x, y[, dim, normalized, vector, centered]) -> CrossCorr (same layout as x, length N along
    dim; summed over dimension `vector` if nonzero). dim is 1-based; default the first non-singleton
    dimension of y."""
    nargin = len(varargin)
    if not 2 <= nargin <= 6:
        raise ValueError('Wrong number of input arguments.')
    x, y = varargin[0], varargin[1]
    dim = varargin[2] if nargin >= 3 else []
    normalized = varargin[3] if nargin >= 4 else 1
    vector = varargin[4] if nargin >= 5 else 0
    centered = varargin[5] if nargin >= 6 else 1
    x = np.asarray(x)
    y = np.asarray(y)
    Dimsx = max(x.ndim, 2)
    Dimsy = max(y.ndim, 2)
    if Dimsx != Dimsy:
        raise ValueError('Input arrays must be the same size.')
    if dim is None or isempty(dim):
        sizey = y.shape if y.ndim >= 2 else (1, y.size)
        idx = [k for k, s in enumerate(sizey) if s > 1]
        if idx:
            dim = idx[0] + 1
        else:
            raise ValueError('No non-singleton dimensions were detected for input.')
    dim = int(dim)
    nd = max(Dimsy, dim)
    x = _nd(x if x.ndim >= 2 else x.reshape(1, -1), nd)
    y = _nd(y if y.ndim >= 2 else y.reshape(1, -1), nd)
    ax = dim - 1
    same = x.shape == y.shape and np.array_equal(x, y)
    if centered:
        x = x - np.mean(x, axis=ax, keepdims=True)
        y = x if same else y - np.mean(y, axis=ax, keepdims=True)
    N = y.shape[ax]
    Fx = np.fft.fft(x, 2 * N, axis=ax)
    Fy = Fx if same else np.fft.fft(y, 2 * N, axis=ax)
    r = np.fft.ifft(Fx * np.conj(Fy), axis=ax)
    CrossCorr = np.real(np.take(r, np.arange(N), axis=ax))
    if vector:
        CrossCorr = np.sum(CrossCorr, axis=int(vector) - 1, keepdims=True)
    n = N * np.ones(N) - np.arange(1, N + 1) + 1
    sizen = [1] * CrossCorr.ndim
    sizen[ax] = N
    CrossCorr = CrossCorr / n.reshape(sizen)
    if normalized:
        CrossCorr = CrossCorr / np.take(CrossCorr, [0], axis=ax)
    return CrossCorr

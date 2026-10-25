"""Add noise to a signal"""
import numpy as np


def addnoise(y=None, SNR=None, noisemodel=None, rng=None):
    """addnoise(y, SNR, noisemodel) -> y + noise; noisemodel 'u' (uniform), 'n' (normal) or 'f' (1/f).
    Noise is scaled to unit std, then to (max(y)-min(y))/SNR. numpy RNG (keyword rng), so values
    differ from MATLAB's rand/randn."""
    if SNR is None:
        raise ValueError('Second input must be the desired signal-to-noise ratio (SNR).')
    if noisemodel is None:
        raise ValueError("Third input must be the desired noise type ('f', 'n', or 'u').")
    if np.size(SNR) != 1 or SNR <= 0:
        raise ValueError('The second input, the signal-to-noise ratio (SNR), must be a positive number.')
    rng = np.random.default_rng() if rng is None else rng
    y = np.asarray(y)
    dims = y.shape if y.ndim >= 2 else (1, y.size)
    if noisemodel == 'u':
        noisefun = lambda: rng.random(dims) - 0.5
    elif noisemodel == 'n':
        noisefun = lambda: rng.standard_normal(dims)
    elif noisemodel == 'f':
        noisefun = lambda: _oneoverfnoise(dims, 1, rng)
    else:
        raise ValueError("Unknown noise type '%s'. Use 'n', 'u', or 'f'." % noisemodel)
    if np.iscomplexobj(y):
        noise = noisefun() + 1j * noisefun()
    else:
        noise = noisefun()
    noise = noise / np.std(noise.ravel(order='F'), ddof=1)
    noise = noise.reshape(dims, order='F')
    sl = np.max(y) - np.min(y) if not np.iscomplexobj(y) else _cmax(y) - _cmin(y)
    signallevel = max(np.real(sl), np.imag(sl))
    noiselevel = signallevel / SNR if signallevel > 0 else 1
    yn = y.reshape(dims) + noise * noiselevel
    return yn


def _cmax(y):
    """MATLAB max for complex arrays: by abs, ties broken by angle."""
    v = y.ravel(order='F')
    return v[np.lexsort((np.angle(v), np.abs(v)))[-1]]


def _cmin(y):
    v = y.ravel(order='F')
    return v[np.lexsort((np.angle(v), np.abs(v)))[0]]


def _oneoverfnoise(dims, alpha, rng):
    nPoints = int(np.prod(dims))
    freq = np.arange(nPoints // 2 + 1, dtype=float)
    with np.errstate(divide='ignore'):
        Amplitudes = freq ** (-alpha / 2)
    Amplitudes[0] = 1
    Phases = np.exp(2j * np.pi * rng.random(Amplitudes.shape))
    FD = Amplitudes * Phases
    if nPoints % 2:
        d = np.concatenate([FD, np.conj(FD[:0:-1])])
    else:
        d = np.concatenate([FD, np.conj(FD[-2:0:-1])])
    noise = np.real(np.fft.ifft(d))
    if dims[0] > dims[1]:
        return noise.reshape(-1, 1)
    return noise.reshape(1, -1)

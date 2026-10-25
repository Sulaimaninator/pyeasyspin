"""Spectrum convolution with line shapes"""
import warnings
import numpy as np
from .._registry import F
from .._mat import size, vec


def _expandfull(v, n):
    v = vec(np.asarray(v, dtype=float))
    if v.size >= n:
        return v.copy()
    return np.concatenate([v, np.full(n - v.size, v[0])])


def convspec(spec=None, steps=None, fwhm=None, deriv=None, alpha=None, phase=None):
    """convspec(spec, steps, fwhm, deriv=0, alpha=1, phase=0) -> spectrum convolved (FFT,
    zero-padded to 2*N+1 per dimension) with Gaussian/Lorentzian line shapes."""
    if spec is None or steps is None or fwhm is None:
        raise ValueError('Wrong number of input arguments.')
    if deriv is None:
        deriv = 0
    if alpha is None:
        alpha = 1
    if phase is None:
        phase = 0
    spec = np.asarray(spec)
    if np.any(np.isnan(spec)):
        warnings.warn('The input data array (spec) contains NaN entries.')
    d = np.asarray(deriv)
    if np.all(d != -1) and np.all(d != 0) and np.all(d != 1) and np.all(d != 2):
        raise ValueError('deriv must be 0, 1 or 2.')
    a = np.asarray(alpha)
    if np.any(a < 0) or np.any(a > 1):
        raise ValueError('alpha must satisfy 0 <= alpha <= 1.')
    sz_original = size(spec)
    sz = [s for s in sz_original if s != 1]
    nDims = len(sz)
    spec = np.reshape(spec, sz, order='F')
    if nDims > 2 and spec.size > 200 ** 3:
        warnings.warn('Very big data array! convspec might take very long!')
    steps = _expandfull(steps, nDims)
    fwhm = _expandfull(fwhm, nDims)
    deriv = _expandfull(deriv, nDims)
    alpha = _expandfull(alpha, nDims)
    phase = _expandfull(phase, nDims)
    NN = 2 * np.array(sz) + 1
    mid = np.floor(NN / 2 + 0.5).astype(int) + 1
    fwhm = fwhm / steps
    mask = fwhm == 0              # MATLAB steps(mask) = 1 grows steps (with zeros) if mask is longer
    if mask.size > steps.size:
        if mask[steps.size:].any():
            steps = np.concatenate([steps, np.zeros(int(np.nonzero(mask)[0][-1]) + 1 - steps.size)])
        mask = mask[:steps.size]
    steps[mask[:steps.size]] = 1
    if np.any((fwhm < 1) & (fwhm > 0) & (deriv == 1)):
        raise ValueError(('The linewidth is smaller (%gx) than the increment. '
                          'The convolution with such a narrow derivative lineshape does not work. '
                          'Either increase linewidth or increase axis resolution.') % np.min(fwhm[fwhm != 0]))
    decays = []
    for i in range(nDims):
        n = int(NN[i])
        if fwhm[i] > 0:
            Line = vec(F.lshape(np.arange(1, n + 1, dtype=float).reshape(1, -1), float(mid[i]), fwhm[i],
                                deriv[i], alpha[i], phase[i]))
            m = int(mid[i])
            decays.append(np.fft.ifft(np.concatenate([Line[m - 1:], Line[:m - 1]])))
        else:
            decays.append(np.ones(n) / n)
    kern = decays[0]
    for dvec in decays[1:]:
        kern = np.multiply.outer(kern, dvec)
    out = np.fft.fftn(np.fft.ifftn(spec, s=[int(v) for v in NN], axes=tuple(range(nDims))) * kern)
    out = np.prod(NN.astype(float)) / np.prod(steps ** deriv) * out
    out = out[tuple(slice(0, s) for s in sz)]
    out = np.reshape(out, sz_original, order='F')
    if not np.iscomplexobj(spec):
        out = np.real(out)
    return out

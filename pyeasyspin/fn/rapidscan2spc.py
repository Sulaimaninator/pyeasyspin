"""Conversion of sinusoidal rapid-scan signal to EPR spectrum"""
import math
import numpy as np
from .._registry import F, nout
from .._mat import vec, isnumeric


def _v(x):
    return x[0] if isinstance(x, tuple) else x


@nout(2)
def rapidscan2spc(M=None, rsAmp=None, rsFreq=None, g=None, plot=False):
    """rapidscan2spc(M, rsAmp, rsFreq, g=gfree) -> (dB, spc): complex rapid-scan signal M (one full
    period, even number of points), scan amplitude rsAmp (mT, peak-to-peak), scan frequency rsFreq (kHz)
    -> field offset dB (mT) and the deconvolved spectrum."""
    nargin = sum(a is not None for a in (M, rsAmp, rsFreq, g))
    if nargin == 0:
        return None
    if nargin < 3:
        raise ValueError('At least three inputs are expected: M, rsAmp, and rsFreq')
    if g is None:
        g = _v(F.gfree())
    Ma = np.asarray(M)
    if not isnumeric(Ma) or not (Ma.ndim <= 1 or (Ma.ndim == 2 and min(Ma.shape) == 1)):
        raise ValueError('First input (rapid-scan signal) must be a vector.')
    if not np.iscomplexobj(Ma):
        raise ValueError('First input (rapid-scan signal) must contain both absorption and dispersion.')
    M = vec(Ma)
    N = M.size
    if N % 2:
        raise ValueError('First input (rapid-scan signal) must contain an even number of points.')
    if np.size(rsAmp) != 1 or np.iscomplexobj(rsAmp) or rsAmp <= 0:
        raise ValueError('Second input (modulation amplitude) must be a positive number.')
    if np.size(rsFreq) != 1 or np.iscomplexobj(rsFreq) or rsFreq <= 0:
        raise ValueError('Third input (modulation frequency) must be a positive number.')
    rsFreq_kHz = float(rsFreq)
    rsFreq = rsFreq_kHz * 1e3
    Tperiod = 1 / rsFreq
    t = np.arange(N) / N * Tperiod
    rsAmp_mT = float(rsAmp)
    rsAmp = rsAmp_mT / 1e3
    A = rsAmp / 2
    gamma = g * _v(F.bmagn()) / _v(F.hbar())
    A = gamma * A
    om = 2 * math.pi * rsFreq
    ph = A * np.sin(om * t) / om
    phf = np.exp(-1j * ph)
    Mph = M * phf

    def fourierdeconv(sl):
        return np.fft.fftshift(np.fft.fft(Mph[sl]) / np.fft.fft(phf[sl]))
    h = N // 2
    spc = fourierdeconv(slice(0, h)) + fourierdeconv(slice(h, N))
    nu = vec(F.fdaxis(t[:h].reshape(-1, 1))) / 1e6
    dB = -vec(F.unitconvert(nu.reshape(1, -1), 'MHz->mT', g))
    idx = np.abs(dB) <= rsAmp_mT / 2
    dB = dB[idx].reshape(1, -1)
    spc = spc[idx].reshape(-1, 1)
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1)
        t_us = t * 1e6
        ax[0].plot(t_us, M.real, t_us, M.imag)
        ax[0].set_xlabel('time (µs)')
        ax[1].plot(dB.ravel(), spc.real.ravel(), dB.ravel(), spc.imag.ravel())
        ax[1].set_xlabel('field offset (mT)')
        plt.show()
    return dB, spc

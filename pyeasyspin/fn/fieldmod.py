"""Field modulation (pseudo-modulation of a spectrum)."""
import numpy as np
from scipy.special import jv
from .._registry import nout


@nout(1)
def fieldmod(B=None, spc=None, ModAmp=None, Harmonic=None, plot=False):
    """fieldmod(B, spc, ModAmp (peak-to-peak, units of B), Harmonic=1) -> in-phase modulated
    spectrum, same orientation as spc. FFT/Bessel algorithm (kernel i^h J_h(pi*ModAmp*k)) with
    EasySpin's padding (2N+1) and normalization."""
    if B is None or spc is None or ModAmp is None:
        raise ValueError('Wrong number of input arguments!')
    if Harmonic is None:
        Harmonic = 1
    if np.size(Harmonic) != 1 or Harmonic < 0 or np.iscomplexobj(Harmonic) or np.mod(Harmonic, 1):
        raise ValueError('Harmonic must be a positive integer (1, 2, 3, etc)!')
    if ModAmp <= 0:
        raise ValueError('Modulation amplitude (3rd argument) must be positive.')
    Harmonic = int(np.ravel(Harmonic)[0])
    Bv = np.ravel(np.asarray(B, dtype=float))
    spc = np.asarray(spc)
    n = Bv.size
    if spc.size != n and max(spc.shape, default=0) != n:
        raise ValueError('x and y must have the same length!')
    sz = spc.shape if spc.ndim >= 2 else (1, spc.size)
    if all(s != 1 for s in sz):
        raise ValueError('spc (2nd input) must be a row or column vector.')
    isRowVector = sz[0] == 1
    s = spc.ravel(order='F')
    dx = Bv[1] - Bv[0]
    Ampl = ModAmp / 2 / dx
    NN = 2 * n + 1
    ffty = np.fft.fft(s, NN)
    ffty[int(np.ceil(NN / 2)):] = 0
    S = np.arange(NN) / NN
    yMod = np.fft.ifft(ffty * jv(Harmonic, 2 * np.pi * Ampl * S))
    yMod = yMod[:n]
    yMod = (1j) ** Harmonic * yMod
    out = np.real(yMod)
    out = out.reshape(1, -1) if isRowVector else out.reshape(-1, 1)
    if plot:
        import matplotlib.pyplot as plt
        plt.clf()
        plt.subplot(2, 1, 1); plt.plot(Bv, s); plt.xlim(Bv.min(), Bv.max()); plt.title('Original spectrum')
        plt.subplot(2, 1, 2); plt.plot(Bv, out.ravel()); plt.xlim(Bv.min(), Bv.max())
        plt.xlabel('magnetic field (mT)')
        plt.title('Modulated spectrum, harmonic %d, modulation amplitude %g mT' % (Harmonic, ModAmp))
    return out

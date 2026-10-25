"""Cross-term averaged FFT"""
import numpy as np
from .._mat import vec


def ctafft(TD=None, Averages=None, N=None):
    """ctafft(TD, Averages, N=size(TD,1)) -> sqrt of the mean power spectrum over shifted starts."""
    TD = np.asarray(TD)
    if TD.ndim < 2:
        TD = TD.reshape(1, -1)
    if N is None:
        N = TD.shape[0]
    N = int(N)
    Av = vec(np.asarray(Averages, dtype=float))
    if np.any(Av < 0) or np.any(np.mod(Av, 1)):
        raise ValueError('Averages must contain positive integers!')
    if Av.size == 1:
        Av = np.arange(1, Av[0] + 1)
    RowVec = TD.size == TD.shape[1]
    if RowVec:
        TD = TD.T
    if np.any(Av >= TD.shape[0]):
        raise ValueError('Entries in Averages cannot be larger than the length of TD!')
    FD = np.zeros((N, TD.shape[1]))
    for a in Av:
        newFD = np.fft.fft(TD[int(a) - 1:, :], n=N, axis=0)
        FD = FD + np.abs(newFD) ** 2
    FD = np.sqrt(FD / Av.size)
    if RowVec:
        FD = FD.T
    return FD

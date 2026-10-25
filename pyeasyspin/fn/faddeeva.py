"""Faddeeva function w(z) = exp(-z^2)*erfc(-iz) (Weideman's rational expansion, as EasySpin)."""
import numpy as np


def faddeeva(z=None, N=None):
    """faddeeva(z, N=38) -> w, same shape as z."""
    if N is None:
        N = 38
    N = int(N)
    M = 2 * N
    M2 = 2 * M
    k = np.arange(-M + 1, M, dtype=float)
    L = np.sqrt(N / np.sqrt(2))
    theta = k * np.pi / M
    t = L * np.tan(theta / 2)
    f = np.exp(-t ** 2) * (L ** 2 + t ** 2)
    f = np.concatenate(([0.0], f))
    a = np.real(np.fft.fft(np.fft.fftshift(f))) / M2
    a = a[1:N + 1][::-1]
    z = np.asarray(z)
    Z = (L + 1j * z) / (L - 1j * z)
    p = np.zeros(np.shape(Z), dtype=complex)
    for c in a:            # Horner, as MATLAB polyval
        p = Z * p + c
    w = 2 * p / (L - 1j * z) ** 2 + (1 / np.sqrt(np.pi)) / (L - 1j * z)
    return w

"""Splitting pattern of n equivalent spins I."""
import numpy as np


def equivsplit(I=None, n=None):
    """Relative intensities of the EPR splitting pattern of n equivalent nuclei with spin I."""
    if I is None or n is None:
        raise ValueError(f'You gave {int(I is not None) + int(n is not None)} inputs, but 2 are expected (I and n).')
    if n < 1 or n % 1:
        raise ValueError('The number of equivalent nuclei n (second input argument), must be an integer greater than 0.')
    if np.size(I) != 1 or np.iscomplexobj(I) or (float(I) % 0.5) or I < 0:
        raise ValueError('The spin quantum number I (first input argument) must be a positive multiple of 1/2.')
    E = np.ones(int(round(2 * I + 1)))
    Intensity = E
    for _ in range(2, int(n) + 1):
        Intensity = np.convolve(Intensity, E)
    return Intensity

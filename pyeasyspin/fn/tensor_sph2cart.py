"""Rank-2 cartesian tensor from irreducible spherical tensors"""
import numpy as np


def tensor_sph2cart(T0=None, T1=None, T2=None):
    """tensor_sph2cart(T0, T1 (3, q=1,0,-1), T2 (5, q=2..-2)) -> T (3x3, complex)."""
    if T0 is None or T1 is None or T2 is None:
        raise ValueError('Three inputs (T0, T1, T2) are required.')
    if np.size(T0) != 1:
        raise ValueError('First input (T0) must be a single number.')
    if np.iscomplexobj(T0):
        raise ValueError('First input (T0) must be a real-valued number.')
    if np.size(T1) != 3:
        raise ValueError('Second input (T1) must be an array with 3 numbers.')
    if np.size(T2) != 5:
        raise ValueError('Third input (T2) must be an array with 5 numbers.')
    T0 = float(np.asarray(T0).ravel()[0])
    t1 = np.asarray(T1).ravel(order='F')
    t2 = np.asarray(T2).ravel(order='F')
    a = lambda q: t1[1 - q]
    b = lambda q: t2[2 - q]
    x, y, z = 0, 1, 2
    T = np.zeros((3, 3), dtype=complex)
    T[x, z] = 0.5 * a(1) + 0.5 * a(-1) - 0.5 * b(1) + 0.5 * b(-1)
    T[z, x] = -0.5 * a(1) - 0.5 * a(-1) - 0.5 * b(1) + 0.5 * b(-1)
    T[y, z] = -0.5j * a(1) + 0.5j * a(-1) + 0.5j * b(1) + 0.5j * b(-1)
    T[z, y] = 0.5j * a(1) - 0.5j * a(-1) + 0.5j * b(1) + 0.5j * b(-1)
    T[x, x] = -1 / np.sqrt(3) * T0 + 0.5 * b(2) - 1 / np.sqrt(6) * b(0) + 0.5 * b(-2)
    T[y, y] = -1 / np.sqrt(3) * T0 - 0.5 * b(2) - 1 / np.sqrt(6) * b(0) - 0.5 * b(-2)
    T[z, z] = -1 / np.sqrt(3) * T0 + np.sqrt(2 / 3) * b(0)
    T[x, y] = 1j / np.sqrt(2) * a(0) - 0.5j * b(2) + 0.5j * b(-2)
    T[y, x] = -1j / np.sqrt(2) * a(0) - 0.5j * b(2) + 0.5j * b(-2)
    return T

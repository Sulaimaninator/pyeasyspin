"""Irreducbile spherical tensors from rank-2 cartesian tensor"""
import numpy as np
from .._registry import nout


@nout(3)
def tensor_cart2sph(Tc=None):
    """tensor_cart2sph(Tc (scalar, 3-vector principal values, or 3x3)) -> (T0, T1 (3x1), T2 (5x1))."""
    Tc = np.asarray(Tc)
    if not np.issubdtype(Tc.dtype, np.number) or np.iscomplexobj(Tc):
        raise ValueError('Input must be a real-valued scalar, a 3x1 array or a 3x3 array.')
    Tc = Tc.astype(float)
    T0 = 0.0
    T1 = np.zeros((3, 1))
    T2 = np.zeros((5, 1))
    if Tc.size == 1:
        T0 = -np.sqrt(1 / 3) * (3 * float(Tc.ravel()[0]))
    elif Tc.size == 3:
        t = Tc.ravel(order='F')
        x, y, z = t
        T0 = -np.sqrt(1 / 3) * (x + y + z)
        T2[0] = +0.5 * (x - y)
        T2[4] = +0.5 * (x - y)
        T2[2] = np.sqrt(2 / 3) * (z - 0.5 * (x + y))
    elif Tc.size == 9:
        T = Tc.reshape(3, 3, order='F')
        x, y, z = 0, 1, 2
        T1 = T1.astype(complex)
        T2 = T2.astype(complex)
        T0 = -np.sqrt(1 / 3) * (T[x, x] + T[y, y] + T[z, z])
        T1[0] = -0.5 * (T[z, x] - T[x, z] + 1j * (T[z, y] - T[y, z]))
        T1[2] = -0.5 * (T[z, x] - T[x, z] - 1j * (T[z, y] - T[y, z]))
        T1[1] = -(1j / np.sqrt(2)) * (T[x, y] - T[y, x])
        T2[0] = 0.5 * ((T[x, x] - T[y, y]) + 1j * (T[x, y] + T[y, x]))
        T2[4] = 0.5 * ((T[x, x] - T[y, y]) - 1j * (T[x, y] + T[y, x]))
        T2[1] = -0.5 * ((T[x, z] + T[z, x]) + 1j * (T[y, z] + T[z, y]))
        T2[3] = +0.5 * ((T[x, z] + T[z, x]) - 1j * (T[y, z] + T[z, y]))
        T2[2] = np.sqrt(2 / 3) * (T[z, z] - 0.5 * (T[x, x] + T[y, y]))
    else:
        raise ValueError('Cartesian tensor has wrong size. It needs to be 1x1, 3x1, 1x3, or 3x3.')
    return float(T0), T1, T2

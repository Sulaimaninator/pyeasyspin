"""Irreducbile spherical tensor operator from 2 vector operators"""
import numpy as np
from .._registry import nout


def _mul(a, b):
    """MATLAB a*b: matrix product for matrices, scalar product otherwise."""
    if np.ndim(a) >= 2 and np.ndim(b) >= 2 and np.size(a) > 1 and np.size(b) > 1:
        return a @ b
    return a * b


@nout(3)
def istotensor(a=None, b=None):
    """istotensor(a, b): a, b = 3-vectors or lists of 3 operators -> (T0, T1 list(3), T2 list(5))."""
    if isinstance(a, (list, tuple)):
        ax, ay, az = a[0], a[1], a[2]
    else:
        aa = np.asarray(a).ravel(order='F'); ax, ay, az = aa[0], aa[1], aa[2]
    if isinstance(b, (list, tuple)):
        bx, by, bz = b[0], b[1], b[2]
    else:
        bb = np.asarray(b).ravel(order='F'); bx, by, bz = bb[0], bb[1], bb[2]
    m = _mul
    T0 = -(1 / np.sqrt(3)) * (m(ax, bx) + m(ay, by) + m(az, bz))
    T1 = [-0.5 * ((m(ax, bz) - m(az, bx)) + 1j * (m(ay, bz) - m(az, by))),
          -(1j / np.sqrt(2)) * (m(ay, bx) - m(ax, by)),
          -0.5 * ((m(ax, bz) - m(az, bx)) - 1j * (m(ay, bz) - m(az, by)))]
    T2 = [0.5 * ((m(ax, bx) - m(ay, by)) + 1j * (m(ax, by) + m(ay, bx))),
          -0.5 * ((m(ax, bz) + m(az, bx)) + 1j * (m(ay, bz) + m(az, by))),
          np.sqrt(2 / 3) * (m(az, bz) - 0.5 * (m(ax, bx) + m(ay, by))),
          0.5 * ((m(ax, bz) + m(az, bx)) - 1j * (m(ay, bz) + m(az, by))),
          0.5 * ((m(ax, bx) - m(ay, by)) - 1j * (m(ax, by) + m(ay, bx)))]
    return T0, T1, T2

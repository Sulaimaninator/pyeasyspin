"""Projection of a triangulated function onto a spectrum (port of EasySpin's projecttriangles mex).

    Spectrum = projecttriangles(Tri, Areas, Fun, Amplitude, x)

Tri (3 x N, 1-based vertex indices), Areas (N), Fun (positions, M), Amplitude (M or 1) and the
x axis. Bin indices are truncated toward zero as in the C source. Only the real part of the
amplitudes is used (mxGetPr), also as in the mex. Any integer array is accepted for Tri; the mex
insists on uint32.
"""
import numpy as np
from numba import njit

from .._mat import vec


@njit(cache=True)
def _triproject(spec, tri, wei, fun, amp, x0, delta, nPoints, isoInt):
    Amplitude = amp[0]
    for iTri in range(tri.shape[1]):
        idx1 = tri[0, iTri] - 1
        idx2 = tri[1, iTri] - 1
        idx3 = tri[2, iTri] - 1
        left = fun[idx1]
        middle = fun[idx2]
        right = fun[idx3]
        Area = wei[iTri]
        if np.isnan(left) or np.isnan(middle) or np.isnan(right):
            continue
        if right < left:
            left, right = right, left
        if middle < left:
            left, middle = middle, left
        elif middle > right:
            right, middle = middle, right
        left = (left - x0) / delta
        middle = (middle - x0) / delta
        right = (right - x0) / delta
        Width = right - left
        Width1 = middle - left
        Width2 = right - middle
        if not isoInt:
            Amplitude = (amp[idx1] + amp[idx2] + amp[idx3]) / 3
        first1 = int(left)
        last1 = int(middle)
        first2 = last1
        last2 = int(right)
        if Width1 > 0:
            if first1 < nPoints and last1 >= 0:
                f0 = (2 * Amplitude * Area / Width) / Width1 / delta
                if first1 == last1:
                    spec[first1] += f0 * Width1 * Width1 / 2
                else:
                    if first1 >= 0:
                        spec[first1] += f0 * (first1 + 1 - left) / 2 * (first1 + 1 - left)
                    else:
                        first1 = -1
                    if last1 < nPoints:
                        spec[last1] += f0 * ((last1 + middle) / 2 - left) * (middle - last1)
                    else:
                        last1 = nPoints
                    left -= 0.5
                    for idx in range(first1 + 1, last1):
                        spec[idx] += f0 * (idx - left)
        if Width2 > 0:
            if first2 < nPoints and last2 >= 0:
                f0 = (2 * Amplitude * Area / Width) / Width2 / delta
                if first2 == last2:
                    spec[first2] += f0 * Width2 * Width2 / 2
                else:
                    if first2 >= 0:
                        spec[first2] += f0 * (first2 + 1 - middle) * (right - (middle + first2 + 1) / 2)
                    else:
                        first2 = -1
                    if last2 < nPoints:
                        spec[last2] += f0 * (right - last2) * (right - last2) / 2
                    else:
                        last2 = nPoints
                    right -= 0.5
                    for idx in range(first2 + 1, last2):
                        spec[idx] += f0 * (right - idx)
        if Width1 == 0 and Width2 == 0:
            first1 = int(left)
            if first1 >= 0 and first1 < nPoints:
                spec[first1] += Amplitude * Area / delta


def projecttriangles(Tri=None, Areas=None, Fun=None, Amplitude=None, x=None):
    """Spectrum (1 x numel(x))."""
    if x is None:
        raise ValueError('Wrong number of input arguments!')
    Tri = np.asarray(Tri)
    if Tri.ndim != 2 or Tri.shape[0] != 3:
        raise ValueError('Triangulation array must have 3 rows!')
    if Tri.dtype.kind not in 'iu':
        raise ValueError('Triangulation array must be uint32!')
    tri = np.ascontiguousarray(Tri, dtype=np.int64)
    wei = vec(np.asarray(Areas, dtype=float))
    if wei.size != tri.shape[1]:
        raise ValueError('Number of triangles and areas must match!')
    fun = np.ascontiguousarray(vec(np.real(np.asarray(Fun))), dtype=float)
    amp = np.ascontiguousarray(vec(np.real(np.asarray(Amplitude))), dtype=float)
    isoInt = amp.size == 1
    if not isoInt and amp.size != fun.size:
        raise ValueError('Number of line positions and amplitudes must match!')
    xv = vec(np.asarray(x, dtype=float))
    spec = np.zeros(xv.size)
    _triproject(spec, tri, np.ascontiguousarray(wei), fun, amp, xv[0], xv[1] - xv[0], xv.size, isoInt)
    return spec.reshape(1, -1)

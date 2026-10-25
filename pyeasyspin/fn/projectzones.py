"""Integrative projection of a piecewise-linear function onto a spectrum (port of EasySpin's
projectzones mex).

    Spectrum = projectzones(Position, Ampl, SegWeights, x)

Bin indices are truncated toward zero as in the C source. Complex amplitudes give a complex
spectrum, with real and imaginary parts computed separately, as in the mex.
"""
import numpy as np
from numba import njit

from .._mat import vec


@njit(cache=True)
def _zones(spec, pos, amp, aniso, segw, x0, delta, nPoints):
    meanAmp = amp[0] / delta
    for iSeg in range(segw.size):
        left = pos[iSeg]
        right = pos[iSeg + 1]
        if np.isnan(left) or np.isnan(right):
            continue
        if left > right:
            left, right = right, left
        left = (left - x0) / delta
        right = (right - x0) / delta
        first = int(left)
        last = int(right)
        if first >= nPoints or last < 0:
            continue
        if aniso:
            meanAmp = (amp[iSeg] + amp[iSeg + 1]) / 2 / delta
        if first == last:
            spec[first] += meanAmp * segw[iSeg]
        else:
            Height = meanAmp * segw[iSeg] / (right - left)
            if first >= 0:
                spec[first] += Height * (first + 1 - left)
            else:
                first = -1
            if last < nPoints:
                spec[last] += Height * (right - last)
            else:
                last = nPoints
            for idx in range(first + 1, last):
                spec[idx] += Height


def projectzones(Position=None, Ampl=None, SegWeights=None, x=None):
    """Spectrum (1 x numel(x)), complex if Ampl is complex."""
    if x is None:
        raise ValueError('Wrong number of input parameters!')
    P = np.asarray(Position)
    A = vec(np.asarray(Ampl))
    W = np.asarray(SegWeights)
    pos = np.ascontiguousarray(vec(P), dtype=float) if not np.iscomplexobj(P) else None
    segw = np.ascontiguousarray(vec(W), dtype=float) if not np.iscomplexobj(W) else None
    xv = vec(np.asarray(x, dtype=float))
    nPeaks = P.size
    if W.size != nPeaks - 1:
        raise ValueError('Wrong number of segment weights. Should be nPeaks-1')
    if A.size != 1 and A.size != nPeaks:
        raise ValueError('Wrong number of amplitudes! Should be 1 or nPeaks.')
    if pos is None:
        raise ValueError('Peak positions are complex! Only real are allowed.')
    if segw is None:
        raise ValueError('Segment weights are complex! Only real are allowed.')
    aniso = A.size != 1
    x0, delta = xv[0], xv[1] - xv[0]
    if np.iscomplexobj(A):
        sr, si = np.zeros(xv.size), np.zeros(xv.size)
        _zones(sr, pos, np.ascontiguousarray(A.real, dtype=float), aniso, segw, x0, delta, xv.size)
        _zones(si, pos, np.ascontiguousarray(A.imag, dtype=float), aniso, segw, x0, delta, xv.size)
        return (sr + 1j * si).reshape(1, -1)
    spec = np.zeros(xv.size)
    _zones(spec, pos, np.ascontiguousarray(A, dtype=float), aniso, segw, x0, delta, xv.size)
    return spec.reshape(1, -1)

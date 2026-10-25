"""Voigtian line shape function"""
import numpy as np
from .._registry import F, nout
from .._mat import vec


def _conv_same(a, b):
    """MATLAB conv(a,b,'same') for vectors (orientation of a kept)."""
    a = np.asarray(a)
    av, bv = vec(a), vec(b)
    full = np.convolve(av, bv)
    start = int(np.ceil((bv.size - 1) / 2))   # MATLAB: central part, size of a
    out = full[start:start + av.size]
    return out.reshape(a.shape, order='F')


@nout(-1)
def voigtian(x=None, x0=None, fwhmGL=None, diff=None, phase=None, nargout=None):
    """voigtian(x, x0, [fwhmG fwhmL], diff=0, phase=0) -> y; nargout=2 -> (y1, y2)."""
    if x is None or x0 is None or fwhmGL is None:
        raise ValueError('Not enough input arguments!')
    if np.size(x0) != 1:
        raise ValueError('x0 (2nd input) must be a single number.')
    fwhmGL = vec(np.asarray(fwhmGL))
    if fwhmGL.size != 2:
        raise ValueError('fwhmGL (3rd input) must contain two numbers!')
    if np.iscomplexobj(fwhmGL) or np.any(fwhmGL < 0):
        raise ValueError('Both numbers in fwhmGL (3rd input) must be non-negative.')
    if np.all(fwhmGL == 0):
        raise ValueError('At least one value in fwhmGL (3rd input) must be nonzero.')
    if diff is None:
        diff = 0
    if phase is None:
        phase = 0
    calculateQuadrature = nargout is not None and nargout > 1
    fwhmG, fwhmL = float(fwhmGL[0]), float(fwhmGL[1])
    x = np.asarray(x, dtype=float)
    xv = vec(x)
    x0center = xv[int(np.ceil(xv.size / 2)) - 1]
    if fwhmG > fwhmL:
        diffG, phaseG, x0G = diff, phase, x0
        diffL, phaseL, x0L = 0, 0, x0center
    else:
        diffG, phaseG, x0G = 0, 0, x0center
        diffL, phaseL, x0L = diff, phase, x0
    yG = yGquad = yL = yLquad = None
    if calculateQuadrature:
        if fwhmG > 0:
            yG, yGquad = F.gaussian(x, x0G, fwhmG, diffG, phaseG, nargout=2)
        if fwhmL > 0:
            yL, yLquad = F.lorentzian(x, x0L, fwhmL, diffL, phaseL, nargout=2)
    else:
        if fwhmG > 0:
            yG = F.gaussian(x, x0G, fwhmG, diffG, phaseG, nargout=1)
            yGquad = np.zeros((0, 0))
        if fwhmL > 0:
            yL = F.lorentzian(x, x0L, fwhmL, diffL, phaseL, nargout=1)
            yLquad = np.zeros((0, 0))
    if fwhmL > 0 and fwhmG > 0:
        y1 = _conv_same(yG, yL)
        y2 = None
        if calculateQuadrature:
            y2 = _conv_same(yG, yLquad) if fwhmL > fwhmG else _conv_same(yGquad, yL)
        dx = xv[1] - xv[0]
        y1 = y1 * dx
        if calculateQuadrature:
            y2 = y2 * dx
    elif fwhmL > 0:
        y1, y2 = yL, yLquad
    else:
        y1, y2 = yG, yGquad
    if calculateQuadrature:
        return y1, y2
    return y1

"""multinucstick mex: stick spectrum of all nuclear sub-lines of one EPR line, binned onto an
axis. Line-by-line numba port of easyspin/private/multinucstick.c."""
import numpy as np
from numba import njit
from .._mat import vec, a2


@njit(cache=True)
def _stick(centerPos, nStates, shifts, M, startPos, deltaPos, nPoints):
    nNuclei = nStates.size
    centerPos = (centerPos - startPos) / deltaPos
    sh = shifts / deltaPos                      # the mex divides its input array in place
    spectrum = np.zeros(nPoints)
    iState = np.zeros(nNuclei, dtype=np.int64)
    position = np.zeros(nNuclei)
    iNuc = 0
    nPeaks = 0
    while iNuc >= 0:
        if iNuc == nNuclei:                     # vertex: bin, then backtrack
            idx = int(position[iNuc - 1])       # C cast: truncation toward zero
            if idx >= 0 and idx < nPoints:
                spectrum[idx] += 1
                nPeaks += 1
            iNuc -= 1
        if iState[iNuc] < nStates[iNuc]:        # go deeper, next sibling
            p_ = position[iNuc - 1] if iNuc > 0 else centerPos
            position[iNuc] = p_ + sh[iNuc * M + iState[iNuc]]
            iState[iNuc] += 1
            iNuc += 1
        else:                                   # last state: reset and backtrack
            iState[iNuc] = 0
            iNuc -= 1
    return spectrum


def multinucstick(*varargin):
    """multinucstick(B0, nStates, shifts, startPos, deltaPos, nPoints) -> 1 x nPoints spectrum.
    shifts: max(2I+1) x nNuclei array (column iNuc holds the nStates(iNuc) sub-line shifts)."""
    centerPos, nStates, shifts, startPos, deltaPos, nPoints = varargin[:6]
    nStates = vec(np.asarray(nStates, dtype=float))
    nNuclei = nStates.size
    sh = a2(np.asarray(shifts, dtype=float))
    if sh.shape[1] != nNuclei:
        raise ValueError('Wrong size of shifts array!')
    deltaPos = float(deltaPos)
    if deltaPos < 0:
        raise ValueError('delta cannot be negative.')
    n = int(float(nPoints))
    spc = _stick(float(centerPos), nStates, vec(sh).copy(), sh.shape[0], float(startPos), deltaPos, n)
    return spc.reshape(1, -1)

"""lisum1i mex (LInear SUMmation, 1D): accumulate line shapes into a spectrum by interpolated
differencing of an integral template. Line-by-line port of easyspin/private/lisum1i.c (numba)."""
import numpy as np
from numba import njit
from .._mat import vec


@njit(cache=True)
def _lisum(T, PosT, WidT, Pos, Amp, Wid, x):
    nT = T.size
    nx = x.size
    y = np.zeros(nx)
    delta = x[1] - x[0]
    beta = delta / 1.0
    flags = np.zeros(3, dtype=np.bool_)   # inf, nan, negative width
    for k in range(Pos.size):
        if np.isnan(Pos[k]) or np.isnan(Amp[k]) or np.isnan(Wid[k]):
            flags[1] = True
            continue
        if np.isinf(Pos[k]) or np.isinf(Amp[k]) or np.isinf(Wid[k]):
            flags[0] = True
            continue
        WidS = Wid[k]
        if WidS < 0:
            flags[2] = True
            WidS = 0.0
        if WidS < delta / 100:
            WidS = delta / 100
        alpha = WidT / WidS * beta
        left = (Pos[k] - x[0]) / delta - PosT / alpha
        left -= 0.5
        right = left + (nT - 1) / alpha
        if left >= nx - 1 or right <= 0:
            continue
        first = 0 if left < 0 else int(np.ceil(left))
        last = nx - 1 if right > nx - 1 else int(np.ceil(right))
        Amplitude = Amp[k] / beta
        pT = alpha * (first - left)
        idxT = int(pT - alpha)          # C cast: truncation toward zero
        remT = pT - alpha - idxT
        if idxT < 0:
            ValueOld = T[0]
        else:
            ValueOld = T[idxT] * (1 - remT) + T[idxT + 1] * remT
        idx = first
        while idx < last:
            idxT = int(pT)
            remT = pT - idxT
            Value = T[idxT] * (1 - remT) + T[idxT + 1] * remT
            y[idx] += Amplitude * (Value - ValueOld)
            idx += 1
            pT += alpha
            ValueOld = Value
        idxT = int(pT)
        remT = pT - idxT
        if idxT < nT - 2:
            Value = T[idxT] * (1 - remT) + T[idxT + 1] * remT
        else:
            Value = T[nT - 1]
        y[idx] += Amplitude * (Value - ValueOld)
    return y, flags


def lisum1i(*varargin):
    """lisum1i(T, PosT, WidT, Pos, Amp, Wid, x) -> y (1 x numel(x)).
    T: integral line-shape template on 1:numel(T); PosT: its 1-based center; WidT: its width;
    Pos/Amp/Wid: line positions, amplitudes, widths (same numel); x: spectral axis."""
    if len(varargin) != 7:
        raise ValueError('Wrong number of input parameters!')
    T, PosT, WidT, Pos, Amp, Wid, x = varargin
    f = lambda a: np.ascontiguousarray(vec(np.asarray(a, dtype=float)))
    Pos, Amp, Wid = f(Pos), f(Amp), f(Wid)
    if not (Pos.size == Amp.size == Wid.size):
        raise ValueError('Pos, Amp, Wid must have the same number of elements!')
    y, flags = _lisum(f(T), float(np.ravel(PosT)[0]) - 1, float(np.ravel(WidT)[0]), Pos, Amp, Wid, f(x))
    if flags[0]:
        print('********** lisum1ic: Inf encountered!! Please report! **************')
    if flags[2]:
        print('********** lisum1ic: Negative width encountered!! Please report! **************')
    return y.reshape(1, -1)

"""(mex routine) Peak binning for saffron's fast frequency-domain ESEEM/HYSCORE algorithm.

numba line-by-line port of sf_peaks.c (EasySpin 6.0.12): same loop nesting, same real/imaginary
arithmetic and the same accumulation order into the spectral buffers. Like the mex file, it has
no outputs: it adds the peak amplitudes into bufferRe/bufferIm in place."""
import math

import numpy as np
from numba import njit

from .._registry import nout
from .._mat import size

# Set to True to reproduce EasySpin's typo in case 15 ([1 2 2 1], 2D-CP), which uses the imaginary
# part of T2r instead of T3r in the last factor (iAmp = rAmp3*iT2r[op] + iAmp3*rT3r[op]).
EASYSPIN_CASE15_TYPO = False


@njit(cache=True)
def _bin(nu, dt, n):
    id1 = np.fmod(-nu * dt, 1.0) * n
    if id1 >= 0:
        id1 = math.floor(id1)
    else:
        id1 = math.ceil(id1)
        if id1 < 0:
            id1 += n
    return id1


@njit(cache=True)
def _case1(rS, iS, n1, dt1, E1l, E1r, rG, iG, rD, iD, N):
    nPeaks = 0
    for j in range(N):
        for k in range(N):
            jk = j + k * N
            kj = k + j * N
            nu = E1l[j] - E1r[k]
            idx = int(_bin(nu, dt1, n1))
            rAmp = rD[kj] * rG[jk] - iD[kj] * iG[jk]
            iAmp = rD[kj] * iG[jk] + iD[kj] * rG[jk]
            rS[idx] += rAmp
            iS[idx] += iAmp
            nPeaks += 1
    return nPeaks


@njit(cache=True)
def _case2(rS, iS, n1, dt1, E1l, E1r, E2l, E2r, rG, iG, rD, iD, rT1l, iT1l, rT1r, iT1r, N, minus):
    # minus=False: case 2 ([1 1]); minus=True: case 3 ([1 -1])
    nPeaks = 0
    for j in range(N):
        for k in range(N):
            jk = j + k * N
            for i in range(N):
                ij = i + j * N
                rAmp = rT1l[ij] * rG[jk] - iT1l[ij] * iG[jk]
                iAmp = rT1l[ij] * iG[jk] + iT1l[ij] * rG[jk]
                for l in range(N):
                    kl = k + l * N
                    li = l + i * N
                    if minus:
                        nu = -E2l[i] + E1l[j] - E1r[k] + E2r[l]
                    else:
                        nu = (E2l[i] - E2r[l]) + (E1l[j] - E1r[k])
                    idx = int(_bin(nu, dt1, n1))
                    rAmp1 = rAmp * rT1r[kl] - iAmp * iT1r[kl]
                    iAmp1 = rAmp * iT1r[kl] + iAmp * rT1r[kl]
                    rAmp2 = rAmp1 * rD[li] - iAmp1 * iD[li]
                    iAmp2 = rAmp1 * iD[li] + iAmp1 * rD[li]
                    rS[idx] += rAmp2
                    iS[idx] += iAmp2
                    nPeaks += 1
    return nPeaks


@njit(cache=True)
def _case11(rS, iS, n1, n2, dt1, dt2, E1l, E1r, E2l, E2r, rG, iG, rD, iD, rT1l, iT1l, rT1r, iT1r, N):
    nu1 = np.zeros(N * N, dtype=np.int64)
    nu2 = np.zeros(N * N, dtype=np.int64)
    idx = 0
    for j in range(N):
        for i in range(N):
            id1 = np.fmod(-(E1l[i] - E1r[j]) * dt1, 1.0) * n1
            id2 = np.fmod(-(E2l[i] - E2r[j]) * dt2, 1.0) * n2
            if id1 >= 0:
                id1 = math.floor(id1)
            else:
                id1 = math.ceil(id1)
                if id1 < 0:
                    id1 += n1
            if id2 >= 0:
                id2 = math.floor(id2)
            else:
                id2 = math.ceil(id2)
                if id2 < 0:
                    id2 += n2
            nu1[idx] = int(id1)
            nu2[idx] = int(id2)
            if id1 < 0 or id1 > n1 - 1:
                return -1
            if id2 < 0 or id2 > n2 - 1:
                return -2
            idx += 1
    nPeaks = 0
    for j in range(N):
        for k in range(N):
            jk = j + k * N
            for i in range(N):
                ij = i + j * N
                rAmp = rT1l[ij] * rG[jk] - iT1l[ij] * iG[jk]
                iAmp = rT1l[ij] * iG[jk] + iT1l[ij] * rG[jk]
                for l in range(N):
                    kl = k + l * N
                    li = l + i * N
                    il = i + l * N
                    rAmp1 = rAmp * rT1r[kl] - iAmp * iT1r[kl]
                    iAmp1 = rAmp * iT1r[kl] + iAmp * rT1r[kl]
                    rAmp2 = rAmp1 * rD[li] - iAmp1 * iD[li]
                    iAmp2 = rAmp1 * iD[li] + iAmp1 * rD[li]
                    idx = nu1[jk] + nu2[il] * n1
                    rS[idx] += rAmp2
                    iS[idx] += iAmp2
                    nPeaks += 1
    return nPeaks


@njit(cache=True)
def _case12_14(rS, iS, n1, n2, dt1, dt2, E1l, E2l, E3l, E1r, E2r, E3r, rG, iG, rD, iD,
               rT1l, iT1l, rT2l, iT2l, rT1r, iT1r, rT2r, iT2r, N, scheme14):
    # case 12 ([1 2 1]) and case 14 ([1 1 2]); Amp = D[ni]*T2l[ij]*T1l[jk]*G[kl]*T1r[lm]*T2r[mn]
    nPeaks = 0
    for k in range(N):
        for l in range(N):
            kl = k + l * N
            for j in range(N):
                jk = j + k * N
                rAmp0 = rT1l[jk] * rG[kl] - iT1l[jk] * iG[kl]
                iAmp0 = rT1l[jk] * iG[kl] + iT1l[jk] * rG[kl]
                for m in range(N):
                    lm = l + m * N
                    rAmp1 = rAmp0 * rT1r[lm] - iAmp0 * iT1r[lm]
                    iAmp1 = rAmp0 * iT1r[lm] + iAmp0 * rT1r[lm]
                    if scheme14:
                        nu1_ = (E1l[k] - E1r[l]) + (E2l[j] - E2r[m])
                        id1 = _bin(nu1_, dt1, n1)
                        id2 = 0.0
                    else:
                        nu2_ = (E2l[j] - E2r[m])
                        id2 = _bin(nu2_, dt2, n2)
                        id1 = 0.0
                    for i in range(N):
                        ij = i + j * N
                        rAmp2 = rT2l[ij] * rAmp1 - iT2l[ij] * iAmp1
                        iAmp2 = rT2l[ij] * iAmp1 + iT2l[ij] * rAmp1
                        for n in range(N):
                            mn = m + n * N
                            ni = n + i * N
                            rAmp3 = rAmp2 * rT2r[mn] - iAmp2 * iT2r[mn]
                            iAmp3 = rAmp2 * iT2r[mn] + iAmp2 * rT2r[mn]
                            rAmp = rD[ni] * rAmp3 - iD[ni] * iAmp3
                            iAmp = rD[ni] * iAmp3 + iD[ni] * rAmp3
                            if scheme14:
                                nu2_ = (E3l[i] - E3r[n])
                                id2 = _bin(nu2_, dt2, n2)
                            else:
                                nu1_ = (E1l[k] - E1r[l]) + (E3l[i] - E3r[n])
                                id1 = _bin(nu1_, dt1, n1)
                            idx = int(id1 + id2 * n1)
                            if idx < 0 or idx > n1 * n2:
                                return -3
                            rS[idx] += rAmp
                            iS[idx] += iAmp
                            nPeaks += 1
    return nPeaks


@njit(cache=True)
def _case15(rS, iS, n1, n2, dt1, dt2, E1l, E2l, E3l, E4l, E1r, E2r, E3r, E4r, rG, iG, rD, iD,
            rT1l, iT1l, rT2l, iT2l, rT3l, iT3l, rT1r, iT1r, rT2r, iT2r, rT3r, iT3r, N, typo):
    # [1 2 2 1], e.g. 2D-CP; Amp = D[pi]*T3l[ij]*T2l[jk]*T1l[kl]*G[lm]*T1r[mn]*T2r[no]*T3r[op]
    nPeaks = 0
    iT3r_last = iT2r if typo else iT3r
    for l in range(N):
        for m in range(N):
            lm = l + m * N
            for i in range(N):
                for p in range(N):
                    pi = p + i * N
                    nu1_ = (E1l[l] - E1r[m]) + (E4l[i] - E4r[p])
                    id1 = _bin(nu1_, dt1, n1)
                    for k in range(N):
                        kl = k + l * N
                        rAmp0 = rT1l[kl] * rG[lm] - iT1l[kl] * iG[lm]
                        iAmp0 = rT1l[kl] * iG[lm] + iT1l[kl] * rG[lm]
                        for n in range(N):
                            mn = m + n * N
                            rAmp1 = rAmp0 * rT1r[mn] - iAmp0 * iT1r[mn]
                            iAmp1 = rAmp0 * iT1r[mn] + iAmp0 * rT1r[mn]
                            for j in range(N):
                                ij = i + j * N
                                jk = j + k * N
                                rAmp2 = rT2l[jk] * rAmp1 - iT2l[jk] * iAmp1
                                iAmp2 = rT2l[jk] * iAmp1 + iT2l[jk] * rAmp1
                                rAmp3 = rT3l[ij] * rAmp2 - iT3l[ij] * iAmp2
                                iAmp3 = rT3l[ij] * iAmp2 + iT3l[ij] * rAmp2
                                rAmp2 = rD[pi] * rAmp3 - iD[pi] * iAmp3
                                iAmp2 = rD[pi] * iAmp3 + iD[pi] * rAmp3
                                for o in range(N):
                                    no = n + o * N
                                    op = o + p * N
                                    rAmp3 = rAmp2 * rT2r[no] - iAmp2 * iT2r[no]
                                    iAmp3 = rAmp2 * iT2r[no] + iAmp2 * rT2r[no]
                                    rAmp = rAmp3 * rT3r[op] - iAmp3 * iT3r[op]
                                    iAmp = rAmp3 * iT3r_last[op] + iAmp3 * rT3r[op]
                                    nu2_ = (E2l[k] - E2r[n]) + (E3l[j] - E3r[o])
                                    id2 = _bin(nu2_, dt2, n2)
                                    if id1 < 0 or id1 > n1 - 1:
                                        return -1
                                    if id2 < 0 or id2 > n2 - 1:
                                        return -2
                                    idx = int(id1 + id2 * n1)
                                    rS[idx] += rAmp
                                    iS[idx] += iAmp
                                    nPeaks += 1
    return nPeaks


@njit(cache=True)
def _case17(rS, iS, n1, n2, dt1, dt2, E1l, E2l, E3l, E4l, E1r, E2r, E3r, E4r, rG, iG, rD, iD,
            rT1l, iT1l, rT2l, iT2l, rT3l, iT3l, rT1r, iT1r, rT2r, iT2r, rT3r, iT3r, N):
    # [1 1 2 2], e.g. 2D refocused primary ESEEM
    nPeaks = 0
    for l in range(N):
        for m in range(N):
            lm = l + m * N
            for k in range(N):
                kl = k + l * N
                rAmp0 = rT1l[kl] * rG[lm] - iT1l[kl] * iG[lm]
                iAmp0 = rT1l[kl] * iG[lm] + iT1l[kl] * rG[lm]
                for n in range(N):
                    mn = m + n * N
                    nu1_ = (E1l[l] - E1r[m]) + (E2l[k] - E2r[n])
                    id1 = _bin(nu1_, dt1, n1)
                    rAmp1 = rAmp0 * rT1r[mn] - iAmp0 * iT1r[mn]
                    iAmp1 = rAmp0 * iT1r[mn] + iAmp0 * rT1r[mn]
                    for j in range(N):
                        jk = j + k * N
                        rAmp2 = rT2l[jk] * rAmp1 - iT2l[jk] * iAmp1
                        iAmp2 = rT2l[jk] * iAmp1 + iT2l[jk] * rAmp1
                        for o in range(N):
                            no = n + o * N
                            rAmp3 = rAmp2 * rT2r[no] - iAmp2 * iT2r[no]
                            iAmp3 = rAmp2 * iT2r[no] + iAmp2 * rT2r[no]
                            for i in range(N):
                                ij = i + j * N
                                rAmp4 = rT3l[ij] * rAmp3 - iT3l[ij] * iAmp3
                                iAmp4 = rT3l[ij] * iAmp3 + iT3l[ij] * rAmp3
                                for p in range(N):
                                    op = o + p * N
                                    pi = p + i * N
                                    rAmp5 = rAmp4 * rT3r[op] - iAmp4 * iT3r[op]
                                    iAmp5 = rAmp4 * iT3r[op] + iAmp4 * rT3r[op]
                                    rAmp = rD[pi] * rAmp5 - iD[pi] * iAmp5
                                    iAmp = rD[pi] * iAmp5 + iD[pi] * rAmp5
                                    nu2_ = (E3l[j] - E3r[o]) + (E4l[i] - E4r[p])
                                    id2 = _bin(nu2_, dt2, n2)
                                    idx = int(id1 + id2 * n1)
                                    if idx < 0 or idx > n1 * n2:
                                        return -3
                                    rS[idx] += rAmp
                                    iS[idx] += iAmp
                                    nPeaks += 1
    return nPeaks


_SCHEMES = {1: (1, 1), 2: (2, 1), 3: (2, 1), 11: (2, 2), 12: (3, 2), 13: (3, 2), 14: (3, 2),
            15: (4, 2), 17: (4, 2)}
_NARGS = {1: 10, 2: 12, 3: 14, 4: 16}
_MIXNAMES = {2: ('M1l', 'M1r'), 3: ('M1l', 'M2l', 'M1r', 'M2r'),
             4: ('M1l', 'M2l', 'M3l', 'M1r', 'M2r', 'M3r')}


def _ri(x, nStates, name):
    """Real and imaginary parts (column-major, flat) of a matrix argument (mxGetPr/mxGetPi)."""
    a = x.toarray() if hasattr(x, 'toarray') else np.asarray(x)
    if a.size != nStates * nStates:
        raise ValueError(f'{name} has wrong size!')
    a = a.ravel(order='F')
    r = np.ascontiguousarray(np.real(a), dtype=np.float64)
    i = np.ascontiguousarray(np.imag(a), dtype=np.float64) if np.iscomplexobj(a) else np.zeros(a.size)
    return r, i


def _flat_view(buf):
    """Flat column-major float64 view of a buffer (or a copy plus a write-back flag)."""
    if isinstance(buf, np.ndarray) and buf.dtype == np.float64:
        v = buf.reshape(-1, order='F')
        if np.shares_memory(v, buf) and v.flags.c_contiguous:
            return v, False
        return np.ascontiguousarray(v), True
    return np.ascontiguousarray(np.asarray(buf, dtype=np.float64).reshape(-1, order='F')), False


@nout(0)
def sf_peaks(*varargin):
    """sf_peaks(IncSchemeID, bufferRe, bufferIm, dt, idxFreeL, idxFreeR, Ea, Eb, G, D, M1l, M1r, ...):
    bins the amplitudes of all peaks of one coherence transfer pathway (frequencies from the
    nuclear sub-level energies Ea/Eb, MHz; dt in us) into bufferRe/bufferIm (1 x n or n1 x n2,
    float64, modified in place). Schemes 1 [1], 2 [1 1], 3 [1 -1], 11 [1 2], 12 [1 2 1],
    14 [1 1 2], 15 [1 2 2 1], 17 [1 1 2 2]; 13 is recognized but not implemented (as in EasySpin).
    Case 15 uses Im(T3r) in the last product (EasySpin's mex uses Im(T2r); DEVIATIONS.md)."""
    nrhs = len(varargin)
    if nrhs < 10:
        raise ValueError('Insufficient number of input arguments.')
    a0 = varargin[0]
    if np.size(a0) != 1:
        raise ValueError('IncSchemeID must be a scalar!')
    IncSchemeID = int(np.ravel(np.asarray(a0, dtype=float))[0])     # mxGetScalar -> int
    if IncSchemeID not in _SCHEMES:
        raise ValueError('Unrecognized incrementation scheme.')
    nFree, nDim = _SCHEMES[IncSchemeID]
    bufRe, bufIm = varargin[1], varargin[2]
    s = size(bufRe)
    if nDim == 1:
        n1 = s[0]
        if n1 == 1:
            n1 = s[1]
        n2 = 1
    else:
        n1, n2 = s[0], s[1]
    if np.size(bufIm) != n1 * n2:
        raise ValueError('buffIm has wrong number of elements!')
    dt = np.ravel(np.asarray(varargin[3], dtype=float))
    if dt.size != nDim:
        raise ValueError('dt has wrong number of elements!')
    dt1 = float(dt[0])
    dt2 = float(dt[1]) if nDim == 2 else 0.0
    freeL = np.ravel(np.asarray(varargin[4], dtype=float))
    if freeL.size != nFree:
        raise ValueError('Wrong number of left evolution intervals.')
    freeR = np.ravel(np.asarray(varargin[5], dtype=float))
    if freeR.size != nFree:
        raise ValueError('Wrong number of right evolution intervals.')
    Ea = np.asarray(varargin[6])
    if np.iscomplexobj(Ea):
        raise ValueError('Ea must be real!')
    Ea = np.ascontiguousarray(np.ravel(Ea, order='F'), dtype=np.float64)
    nStates = Ea.size
    Eb = np.asarray(varargin[7])
    if np.iscomplexobj(Eb):
        raise ValueError('Eb must be real!')
    Eb = np.ascontiguousarray(np.ravel(Eb, order='F'), dtype=np.float64)
    if Eb.size != nStates:
        raise ValueError('Ea and Eb must have the same number of elements.')
    rG, iG = _ri(varargin[8], nStates, 'G')
    rD, iD = _ri(varargin[9], nStates, 'D')
    if nrhs != _NARGS[nFree]:
        raise ValueError('Wrong number of input arguments.')
    mix = [_ri(varargin[10 + q], nStates, nm) for q, nm in enumerate(_MIXNAMES.get(nFree, ()))]

    EL = [Ea if freeL[q] == 1 else Eb for q in range(nFree)]
    ER = [Ea if freeR[q] == 1 else Eb for q in range(nFree)]
    rS, wbRe = _flat_view(bufRe)
    iS, wbIm = _flat_view(bufIm)
    N = nStates
    if IncSchemeID == 1:
        res = _case1(rS, iS, n1, dt1, EL[0], ER[0], rG, iG, rD, iD, N)
    elif IncSchemeID in (2, 3):
        (rT1l, iT1l), (rT1r, iT1r) = mix
        res = _case2(rS, iS, n1, dt1, EL[0], ER[0], EL[1], ER[1], rG, iG, rD, iD,
                     rT1l, iT1l, rT1r, iT1r, N, IncSchemeID == 3)
    elif IncSchemeID == 11:
        (rT1l, iT1l), (rT1r, iT1r) = mix
        res = _case11(rS, iS, n1, n2, dt1, dt2, EL[0], ER[0], EL[1], ER[1], rG, iG, rD, iD,
                      rT1l, iT1l, rT1r, iT1r, N)
    elif IncSchemeID in (12, 14):
        (rT1l, iT1l), (rT2l, iT2l), (rT1r, iT1r), (rT2r, iT2r) = mix
        res = _case12_14(rS, iS, n1, n2, dt1, dt2, EL[0], EL[1], EL[2], ER[0], ER[1], ER[2],
                         rG, iG, rD, iD, rT1l, iT1l, rT2l, iT2l, rT1r, iT1r, rT2r, iT2r, N,
                         IncSchemeID == 14)
    elif IncSchemeID in (15, 17):
        (rT1l, iT1l), (rT2l, iT2l), (rT3l, iT3l), (rT1r, iT1r), (rT2r, iT2r), (rT3r, iT3r) = mix
        args = (rS, iS, n1, n2, dt1, dt2, EL[0], EL[1], EL[2], EL[3], ER[0], ER[1], ER[2], ER[3],
                rG, iG, rD, iD, rT1l, iT1l, rT2l, iT2l, rT3l, iT3l, rT1r, iT1r, rT2r, iT2r,
                rT3r, iT3r, N)
        res = _case15(*args, EASYSPIN_CASE15_TYPO) if IncSchemeID == 15 else _case17(*args)
    else:   # 13
        raise ValueError('Incrementation scheme is currently not supported.')
    if wbRe:
        bufRe[...] = rS.reshape(bufRe.shape, order='F')
    if wbIm:
        bufIm[...] = iS.reshape(bufIm.shape, order='F')
    if res == -1:
        raise ValueError('id1 is out of range.')
    if res == -2:
        raise ValueError('id2 is out of range.')
    if res == -3:
        raise ValueError('idx out of range.')
    return None

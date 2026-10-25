"""- Fresnel cosine integrals, C(X), C1(X), or C2(X)"""
import math
import os
import numpy as np

from .._matlab import ppval as _ppval

_PP = {}


def _load_pp(name):
    """EasySpin's piecewise-polynomial table (_Fresnel_data_.mat: FCspl, FSspl; order-8 Hermite pp on
    [0, 7.5]). Read from pyeasyspin/data.
    Returns None if the file is absent (callers then use scipy.special.fresnel)."""
    if name in _PP:
        return _PP[name]
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cands = [os.path.join(here, 'data', '_Fresnel_data_.mat')]
    pp = None
    for c in cands:
        if os.path.exists(c):
            import scipy.io
            d = scipy.io.loadmat(c, squeeze_me=True, struct_as_record=False)
            s = d[name]
            pp = (np.asarray(s.breaks, dtype=float).ravel(), np.atleast_2d(np.asarray(s.coefs, dtype=float)))
            break
    _PP[name] = pp
    return pp


def _fresnel_core(X, fresnelType, which, fname):
    if fresnelType is None or (np.size(fresnelType) == 0):
        fresnelType = 0
    else:
        ft = np.asarray(fresnelType)
        if ft.dtype.kind not in 'iuf' or np.size(ft) != 1 or float(ft) not in (0.0, 1.0, 2.0):
            raise ValueError('fresnelType must be scalar, one of {0,1,2} if supplied.')
        fresnelType = int(float(ft))
    Xa = np.asarray(X)
    scalar = Xa.ndim == 0
    X = np.array(Xa, dtype=complex if np.iscomplexobj(Xa) else float, ndmin=1)
    if np.iscomplexobj(X):
        if np.any((X.imag != 0) & (X.real != 0)):
            raise ValueError('X should be real or imaginary.')
        idxImag = (X.real == 0) & (X.imag != 0)
        Xr = X.real.copy()
        Xr[idxImag] = X.imag[idxImag]
    else:
        idxImag = np.zeros(X.shape, dtype=bool)
        Xr = X.copy()
    out = np.zeros(Xr.shape)
    S = Xr < 0
    Xr[S] = -Xr[S]
    if fresnelType == 1:
        Xr = math.sqrt(2 / math.pi) * Xr
    elif fresnelType == 2:
        Xr = np.sqrt(2 * Xr / math.pi)
    Xlim = 7.5
    klim = (Xr >= Xlim) if which == 'C' else (Xr > Xlim)
    pi = math.pi
    if np.any(klim):
        xk = Xr[klim]
        P = (1 - 3 / pi ** 2 / xk ** 4 + 105 / pi ** 4 / xk ** 8 - 10395 / pi ** 6 / xk ** 12 + 2027025 / pi ** 8 / xk ** 16)
        Qp = (1 - 15 / pi ** 2 / xk ** 4 + 945 / pi ** 4 / xk ** 8 - 135135 / pi ** 6 / xk ** 12
              + 34459425 / pi ** 8 / xk ** 16)
        if which == 'C':
            out[klim] = 0.5 + P * np.sin(pi / 2 * xk ** 2) / (pi * xk) - Qp * np.cos(pi / 2 * xk ** 2) / (pi ** 2 * xk ** 3)
        else:
            out[klim] = 0.5 - P * np.cos(pi / 2 * xk ** 2) / (pi * xk) - Qp * np.sin(pi / 2 * xk ** 2) / (pi ** 2 * xk ** 3)
    klim = ~klim
    if np.any(klim):
        pp = _load_pp(fname)
        if pp is not None:
            out[klim] = _ppval(pp, Xr[klim])
        else:
            from scipy.special import fresnel
            s_, c_ = fresnel(Xr[klim])
            out[klim] = c_ if which == 'C' else s_
    out[S] = -out[S]
    if np.any(idxImag):
        out = out.astype(complex)
        out[idxImag] = (1j if which == 'C' else -1j) * out[idxImag]
    if scalar:
        return out[0]
    return out.reshape(Xa.shape)


def fresnelC(X=None, fresnelType=None):
    """fresnelC(X, fresnelType=0): C(X) = int_0^X cos(pi t^2/2) dt (type 0); type 1 and 2 are the
    C1/C2 variants (argument sqrt(2/pi)*X and sqrt(2X/pi)). EasySpin's spline table on [0, 7.5),
    asymptotic series above."""
    if X is None:
        raise ValueError('1 or 2 arguemtns are required.')
    return _fresnel_core(X, fresnelType, 'C', 'FCspl')

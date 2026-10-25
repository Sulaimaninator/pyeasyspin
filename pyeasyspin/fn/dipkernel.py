"""Kernel matrix for DEER"""
import math
import numpy as np
from .._registry import F
from .._mat import vec


def _v(x):
    return x[0] if isinstance(x, tuple) else x


def dipkernel(t=None, r=None, gAB=None):
    """dipkernel(t, r[, [gA gB]]) -> K (numel(t) x numel(r)): DEER kernel, t in µs, r in nm;
    multiplied by the distance increment dr when r has several (equidistant) values."""
    nargin = sum(a is not None for a in (t, r, gAB))
    if nargin < 2:
        raise ValueError('At least two inputs are required: t and r.')
    if gAB is not None:
        if np.size(gAB) != 2:
            raise ValueError('Third input (g12) must be an array with two elements, [g1 g2].')
        gA, gB = vec(np.asarray(gAB, dtype=float))
    else:
        gA = _v(F.gfree())
        gB = _v(F.gfree())
    t = vec(np.asarray(t, dtype=float)).reshape(-1, 1)
    r = vec(np.asarray(r, dtype=float)).reshape(-1, 1)
    D = (_v(F.mu0()) / 4 / math.pi) * _v(F.bmagn()) ** 2 * gA * gB
    D = D / _v(F.planck()) * 1e21
    wdd = 2 * math.pi * D / r ** 3
    ph = wdd.T * np.abs(t)
    kappa = np.sqrt(6 * ph / math.pi)
    with np.errstate(divide='ignore', invalid='ignore'):
        C = F.fresnelC(kappa) / kappa
        S = F.fresnelS(kappa) / kappa
        K = np.cos(ph) * C + np.sin(ph) * S
    K[np.isnan(K)] = 1
    if r.size > 1:
        dr = r[1, 0] - r[0, 0]
        if abs(dr / np.mean(np.diff(r[:, 0])) - 1) > 1e-8:
            raise ValueError('Distance vector must have equidistant increments.')
        K = dr * K
    return K

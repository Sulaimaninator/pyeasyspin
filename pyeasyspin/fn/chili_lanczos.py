"""Computes spectrum via Lanczos tridiagonalization"""
import math

import numpy as np
import scipy.sparse as sp

from .._registry import F, nout
from .._matlab import linspace as _linspace


def _dotu(a, b):
    """a.'*b without conjugation, summed pairwise by numpy (deterministic: BLAS dot results
    depend on the thread count, and neither reproduces MATLAB's MKL zdotu bit for bit)."""
    return np.sum(a * b)


def _col(b):
    if sp.issparse(b):
        b = b.toarray()
    return np.asarray(b).reshape(-1, order='F')


@nout(3)
def chili_lanczos(A=None, b=None, z=None, Opt=None):
    """chili_lanczos(A, b, z, Opt) -> (spec, converged, specchange): Lanczos tridiagonalization of
    the complex-symmetric A (bilinear inner product q.'*y, no conjugation, no reorthogonalization)
    from the starting vector b, with the spectral function 1/(z + alpha(1) - beta(1)^2/(z + ...))
    evaluated at every z. Opt.Lentz true: modified Lentz recursion (left to right), convergence
    when max|Delta-1| < Opt.Threshold, tested every min(10, ceil(N/20)) steps. Opt.Lentz false:
    right-to-left evaluation on a 201-point test axis for the convergence test; the returned
    spectrum is then evaluated on z (EasySpin returned the 201-point test spectrum, DEVIATIONS.md).
    spec has the shape of z; specchange holds the tested changes."""
    zz = np.asarray(z)
    zshape = zz.shape
    zv = zz.reshape(-1, order='F')
    bv = _col(b)
    A = sp.csc_matrix(A) if sp.issparse(A) else np.asarray(A)
    N = bv.size
    alpha = np.zeros(N, dtype=complex)
    beta = np.zeros(N, dtype=complex)
    specchange = np.full(N, np.nan)
    oldspec = np.inf
    errorRecomputationInterval = min(10, math.ceil(N / 20))
    useLentzMethod = bool(Opt.Lentz)
    if useLentzMethod:
        tiny = 1e-30
        spec = tiny
        C = spec
        D = 0
    else:
        zTest = _linspace(zv[0], zv[-1], 201)
    converged = False
    # pseudonorm b.'*b (no conjugation)
    q = bv / np.sqrt(_dotu(bv, bv))
    bq = 0
    k = 0
    for k in range(1, N + 1):
        y = A @ q
        alpha[k - 1] = _dotu(q, y)
        y = y - alpha[k - 1] * q - bq
        beta[k - 1] = np.sqrt(_dotu(y, y))
        bq = beta[k - 1] * q
        with np.errstate(divide='ignore', invalid='ignore'):
            q = y / beta[k - 1]
        if useLentzMethod:
            Cold = C
            Dold = D
            specold = spec
            bb = alpha[k - 1] + zv
            a = 1 if k == 1 else -(beta[k - 2] * beta[k - 2])
            D = bb + a * Dold
            C = bb + a / Cold
            D = 1 / D
            Delta = C * D
            spec = specold * Delta
            if k % errorRecomputationInterval == 0:
                specchange[k - 1] = np.max(np.abs(Delta - 1))
                converged = bool(specchange[k - 1] < Opt.Threshold)
        else:
            if k % errorRecomputationInterval == 0:
                spec_ = F.chili_contfracspec(zTest, alpha, beta, k)
                respec = np.real(spec_)
                specchange[k - 1] = np.max(np.abs(respec - oldspec)) / np.max(respec)
                converged = bool(specchange[k - 1] < Opt.Threshold)
                oldspec = respec
        if converged:
            break
        if beta[k - 1] == 0:
            specchange[k - 1] = 0
            break
    if not useLentzMethod:
        # EasySpin returns the 201-point test spectrum here (or nothing if no test was made);
        # the spectrum belongs on z (DEVIATIONS.md)
        spec = F.chili_contfracspec(zv, alpha, beta, k)
    spec = np.asarray(spec) * np.ones(zv.size)
    specchange = specchange[:k]
    specchange = specchange[~np.isnan(specchange)]
    return spec.reshape(zshape, order='F'), converged, specchange.reshape(1, -1)

"""Conjugate gradients algorithm"""
import numpy as np
import scipy.sparse as sp

from .._registry import nout


def _dotu(a, b):
    """a.'*b without conjugation, summed pairwise by numpy (deterministic: BLAS dot results
    depend on the thread count, and neither reproduces MATLAB's MKL zdotu bit for bit)."""
    return np.sum(a * b)


def _col(b):
    if sp.issparse(b):
        b = b.toarray()
    return np.asarray(b).reshape(-1, order='F')


@nout(5)
def chili_conjgrad(A=None, b=None, shift=None):
    """chili_conjgrad(A, b, shift=0) -> (x, T0, T1, err, stepsDone): conjugate gradients for the
    complex-symmetric system (A + shift*I) x = b with the bilinear (non-conjugating) inner
    product r.'*r, stopped when sum(abs(r)) < 1e-8 or after numel(b) steps. T0 (diagonal) and T1
    (off-diagonal) are the equivalent Lanczos tridiagonal matrix, shifted back by -shift; x is
    numel(b) x 1."""
    if shift is None:
        shift = 0
    Tolerance = 1e-8
    bv = _col(b)
    A = sp.csc_matrix(A) if sp.issparse(A) else np.asarray(A)
    N = bv.size
    x = np.zeros(N, dtype=complex)
    r = bv - A @ x
    p = r
    rr = _dotu(r, r)
    alpha, beta = [], []
    err = np.inf
    iStep = 0
    for iStep in range(1, N + 1):
        rr_old = rr
        rr = _dotu(r, r)
        if iStep == 1:
            beta_ = 0
        else:
            beta_ = rr / rr_old
            p = r + beta_ * p
        Ap = A @ p + shift * p
        alpha_ = rr / _dotu(p, Ap)
        x = x + alpha_ * p
        r = r - alpha_ * Ap
        alpha.append(alpha_)
        beta.append(beta_)
        err = float(np.sum(np.abs(r)))
        if err < Tolerance:
            break
    stepsDone = iStep
    alpha = np.array(alpha, dtype=complex)
    beta = np.array(beta, dtype=complex)
    T0 = np.zeros(stepsDone, dtype=complex)
    T1 = np.zeros(stepsDone, dtype=complex)
    for k in range(1, stepsDone):
        t1 = np.sqrt(beta[k]) / alpha[k - 1]
        T1[k - 1] = t1 * np.sign(t1.real)
        T0[k] = 1 / alpha[k] + beta[k] / alpha[k - 1]
    T0[0] = 1 / alpha[0]
    T0 = T0 - shift
    T1[stepsDone - 1] = 0
    return x.reshape(-1, 1), T0.reshape(1, -1), T1.reshape(1, -1), err, stepsDone

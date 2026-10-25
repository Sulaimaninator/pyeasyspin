"""Irreducible spherical tensor operators (ISTOs)"""
import numpy as np
import scipy.sparse as sps
from .._registry import F


def _isto_single(J, k, q):
    if J == 0:
        return sps.csr_matrix(np.ones((1, 1)))
    Jp = sps.csr_matrix(F.sop(J, '+', 'sparse'))
    Jm = sps.csr_matrix(F.sop(J, '-', 'sparse'))
    Nkk = (-1) ** k / 2 ** (k / 2)
    T = sps.identity(Jp.shape[0], format='csr')
    for _ in range(k):
        T = T @ Jp
    T = Nkk * T
    for q_ in range(k, abs(q), -1):
        T = (Jm @ T - T @ Jm) / np.sqrt(k * (k + 1) - q_ * (q_ - 1))
    if q < 0:
        T = (-1) ** q * T.conj().T
    return sps.csr_matrix(T)


def isto(J=None, kq=None, sparseOpt=None):
    """isto(J or Sys, kq (n x 2 [k q] per spin, or rows [k q i])[, 'sparse']) -> T."""
    if sparseOpt is None:
        sparseOpt = 'full'
    sparseOutput = sparseOpt == 'sparse'
    if isinstance(J, dict):
        if 'S' not in J:
            raise ValueError('Sys.S is missing.')
        J = J['S']
    else:
        Ja = np.asarray(J)
        if not np.issubdtype(Ja.dtype, np.number) or np.any((2 * Ja) % 1) or np.any(Ja < 0):
            raise ValueError('First input must be a spin system structure, or an array of angular momentum '
                             'quantum numbers (0, 1/2, 1, ...).')
    J = np.asarray(J, dtype=float).ravel(order='F')
    kq = np.atleast_2d(np.asarray(kq, dtype=float))
    if kq.shape[1] == 2:
        if kq.shape[0] != J.size:
            raise ValueError('Provide k and q for each angular momentum.')
    elif kq.shape[1] == 3:
        k, q, i = kq[:, 0], kq[:, 1], kq[:, 2].astype(int)
        if np.any(i > J.size):
            raise ValueError('index in kqi is larger than number of angular momenta.')
        kq = np.zeros((J.size, 2))
        kq[i - 1, 0] = k
        kq[i - 1, 1] = q
    k = kq[:, 0]
    q = kq[:, 1]
    if np.any(np.abs(q) > k):
        raise ValueError('q must satisfy -k<=q<=k.')
    if np.any(k % 1) or np.any(k < 0):
        raise ValueError('k must be a non-negative integer (0, 1, 2, ...).')
    if np.any(q % 1):
        raise ValueError('q must be a integer.')
    T = sps.csr_matrix(np.ones((1, 1)))
    for j in range(J.size):
        T = sps.kron(T, _isto_single(J[j], int(k[j]), int(q[j])), format='csr')
    if not sparseOutput:
        T = T.toarray()
    return T

"""Small helpers shared by the ham_* ports (not EasySpin functions)."""
import itertools
import numpy as np
import scipy.sparse as sps
from ._mat import vec, isempty


def spzeros(n):
    n = int(n)
    return sps.csr_matrix((n, n))


def finish(H, sparse):
    """Return H as scipy CSR if sparse else as a dense ndarray."""
    if sparse:
        return sps.csr_matrix(H)
    return H.toarray() if sps.issparse(H) else np.asarray(H)


def dense(H):
    return H.toarray() if sps.issparse(H) else np.asarray(H)


def add(A, B):
    """A + B without numpy.matrix results (dense if either is dense)."""
    if sps.issparse(A) and sps.issparse(B):
        return (A + B).tocsr()
    return dense(A) + dense(B)


def herm(H):
    """(H + H')/2"""
    return ((H + H.conj().T) / 2).tocsr() if sps.issparse(H) else (H + H.conj().T) / 2


def idxlist(x, default):
    """1-based index list from a MATLAB index argument ([] -> default)."""
    if x is None or isempty(x):
        return list(default)
    return [int(v) for v in vec(x)]


def rotate(T, ang, F):
    """R_M2T' * T * R_M2T for Euler angles ang (EasySpin frame convention), via F.erot."""
    R_M2T = np.asarray(F.erot(ang))
    R = R_M2T.T
    return R @ T @ R.T


def nchoosek2(n):
    return list(itertools.combinations(range(1, n + 1), 2))


def perms(v):
    """MATLAB perms: all permutations in reverse lexicographic order of positions."""
    v = list(v)
    return [[v[i] for i in p] for p in list(itertools.permutations(range(len(v))))[::-1]]

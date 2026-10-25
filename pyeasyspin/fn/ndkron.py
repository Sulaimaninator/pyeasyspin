"""Kronecker product of two N-D arrays (over the first two dimensions)."""
import numpy as np
from .._registry import F
from .._mat import isnumeric, reshape


def ndkron(A=None, B=None):
    """kron over dims 1-2, elementwise over dims 3,... (column-major layout as in MATLAB)."""
    if not isnumeric(A) or not isnumeric(B):
        raise ValueError('Inputs must be of numeric type.')
    A, B = np.asarray(A), np.asarray(B)
    if max(A.ndim, 2) != max(B.ndim, 2):
        raise ValueError('Inputs should have the same number of dimensions.')
    if A.ndim <= 2 and B.ndim <= 2:
        return np.kron(np.atleast_2d(A), np.atleast_2d(B))
    sA, sB = A.shape, B.shape
    ma, na, mb, nb = sA[0], sA[1], sB[0], sB[1]
    if na != mb:
        raise ValueError('size(A,2) and size(B,1) must be equal.')
    elif not F.allclose(np.array(sA[2:], float), np.array(sB[2:], float)):
        raise ValueError('Input dimensions (3,4,...) must be equal.')
    A = reshape(A, [1, ma, 1, na] + list(sA[2:]))
    B = reshape(B, [mb, 1, nb, 1] + list(sB[2:]))
    return reshape(A * B, [ma * mb, na * nb] + list(sA[2:]))

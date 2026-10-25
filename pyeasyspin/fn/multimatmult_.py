"""(mex routine) Page-wise real matrix multiplication C(:,:,k) = A(:,:,k)*B(:,:,k)."""
import numpy as np
from numba import njit


@njit(cache=True)
def _mulpages(A, B, C, rA, cA, rB, cB, N):
    # line-by-line port of mulMatMat in multimatmult_.c (same summation order, so bit-identical)
    sA = rA * cA
    sB = rB * cB
    sC = rA * cB
    for p in range(N):
        oa = sA * p
        ob = sB * p
        oc = sC * p
        for i in range(cB):
            c0 = oc + i * rA
            for k in range(cA):
                tmp = B[ob + i * cA + k]
                a0 = oa + k * rA
                for j in range(rA):
                    C[c0 + j] += tmp * A[a0 + j]


def multimatmult_(*varargin):
    """multimatmult_(A, B): page-wise product of two real N-D arrays (MATLAB layout). Only the
    real parts are used, as in the mex file (mxGetPr); multimatmult handles complex inputs."""
    if len(varargin) != 2:
        raise ValueError('Only 2 inputs can be accepted.')
    A = np.asarray(varargin[0])
    B = np.asarray(varargin[1])
    if A.dtype.kind not in 'fc' or B.dtype.kind not in 'fc':
        raise TypeError("Only inputs of type 'double' are supported.")
    A = np.real(A).astype(np.float64)
    B = np.real(B).astype(np.float64)
    Ad = A.shape if A.ndim >= 2 else (1,) * (2 - A.ndim) + A.shape
    Bd = B.shape if B.ndim >= 2 else (1,) * (2 - B.ndim) + B.shape
    if len(Ad) != len(Bd):
        raise ValueError('Inputs should have the same number of dimensions.')
    rA, cA = Ad[0], Ad[1]
    rB, cB = Bd[0], Bd[1]
    pages = tuple(Ad[2:])
    N = int(np.prod(pages)) if pages else 1
    C = np.zeros(rA * cB * N)
    _mulpages(A.ravel(order='F'), B.ravel(order='F'), C, rA, cA, rB, cB, N)
    return C.reshape((rA, cB) + pages, order='F')

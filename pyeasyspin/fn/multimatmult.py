"""Matlab gateway function to 3D matrix multiplication C"""
import numpy as np
from .._registry import F


def multimatmult(A=None, B=None):
    """Page-wise matrix product of two equally sized N-D arrays; 2-D inputs give A*B. Complex
    inputs use three real products (Gauss trick), as EasySpin does."""
    A = A.toarray() if hasattr(A, 'toarray') else np.asarray(A)
    B = B.toarray() if hasattr(B, 'toarray') else np.asarray(B)
    mode = '2D' if (A.ndim <= 2 or B.ndim <= 2) else 'ND'
    if A.shape != B.shape:
        raise ValueError('Size of input arrays must be equal.')
    iscomplex = np.iscomplexobj(A) or np.iscomplexobj(B)
    if mode == '2D':
        return A @ B
    if not iscomplex:
        return F.multimatmult_(A, B)
    ArBr = F.multimatmult_(np.real(A), np.real(B))
    AiBi = F.multimatmult_(np.imag(A), np.imag(B))
    ArpAiBrpBi = F.multimatmult_(np.real(A) + np.imag(A), np.real(B) + np.imag(B))
    C = np.empty(ArBr.shape, dtype=complex)
    C.real = ArBr - AiBi
    C.imag = ArpAiBrpBi - ArBr - AiBi
    return C

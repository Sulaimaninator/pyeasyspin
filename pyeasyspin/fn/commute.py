"""Commutator of two matrices."""
import numpy as np
from .._mat import isnumeric, size


def commute(A=None, B=None):
    """[A,B] = A*B - B*A."""
    if not isnumeric(A) or not isnumeric(B):
        raise ValueError('A and B must be matrices!')
    if size(A) != size(B):
        raise ValueError('A and B must have the same size!')
    if size(A, 1) != size(A, 2):
        raise ValueError('A and B must be square matrices!')
    if not hasattr(A, 'toarray'):
        A = np.asarray(A)
    if not hasattr(B, 'toarray'):
        B = np.asarray(B)
    return A @ B - B @ A

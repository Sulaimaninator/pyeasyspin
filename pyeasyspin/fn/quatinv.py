"""Invert a unit quaternion."""
import numpy as np


def quatinv(q=None):
    """quatinv(q (4,...)) -> conjugate quaternion."""
    q = np.asarray(q)
    if not np.issubdtype(q.dtype, np.number) or q.shape[0] != 4:
        raise ValueError('Input must be numeric with size (4,...).')
    inverter = np.array([1., -1., -1., -1.]).reshape((4,) + (1,) * (q.ndim - 1))
    return inverter * q

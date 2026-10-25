"""Rotate a vector using a unit quaternion."""
import numpy as np
from .._registry import F


def quatvecmult(q=None, v=None):
    """quatvecmult(q (4,...), v (3,...)) -> rotated vectors (3,...)."""
    q = np.asarray(q)
    v = np.asarray(v)
    if q.ndim == 1:
        q = q.reshape(-1, 1)
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    if q.shape[0] != 4:
        raise ValueError('Size of first dimension of the quaternion must equal 4.')
    if v.shape[0] != 3:
        raise ValueError('Size of first dimension of the vector must equal 3.')
    qmag = np.sum(q * q, axis=0)
    if np.any(1.0 - qmag > 1e-5):
        raise ValueError('Input quaternion is not normalized.')
    qv = np.zeros(q.shape, dtype=np.result_type(q.dtype, v.dtype, float))
    qv[1:] = v.reshape(qv[1:].shape, order='F')
    qinv = q * np.array([1., -1., -1., -1.]).reshape((4,) + (1,) * (q.ndim - 1))
    r = F.quatmult(F.quatmult(q, qv), qinv)
    return np.asarray(r)[1:]

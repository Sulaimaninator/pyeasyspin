"""Perform quaternion multiplication."""
import numpy as np


def quatmult(q=None, r=None):
    """quatmult(q, r) -> t = q*r (quaternions along the first dimension)."""
    q = np.asarray(q)
    r = np.asarray(r)
    if q.shape[0] != 4 or r.shape[0] != 4 or not np.issubdtype(q.dtype, np.number) \
            or not np.issubdtype(r.dtype, np.number):
        raise ValueError('q and r must be arrays of size (4,...)')
    if q.shape != r.shape:
        raise ValueError('Size input arrays must be equal.')
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    r0, r1, r2, r3 = r[0], r[1], r[2], r[3]
    t = np.zeros(q.shape, dtype=np.result_type(q.dtype, r.dtype, float))
    t[0] = r0 * q0 - r1 * q1 - r2 * q2 - r3 * q3
    t[1] = r0 * q1 + r1 * q0 - r2 * q3 + r3 * q2
    t[2] = r0 * q2 + r1 * q3 + r2 * q0 - r3 * q1
    t[3] = r0 * q3 - r1 * q2 + r2 * q1 + r3 * q0
    return t

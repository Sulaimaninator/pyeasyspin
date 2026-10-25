"""Convert unit quaternions to rotation matrices."""
import numpy as np


def quat2rotmat(q=None):
    """quat2rotmat(q (4,...)) -> mat (3,3,...)."""
    q = np.asarray(q)
    if q.ndim == 1:
        q = q.reshape(-1, 1)
    if not np.issubdtype(q.dtype, np.number) or q.shape[0] != 4:
        raise ValueError('q must be an array of size (4,...)')
    d = 1 - np.sqrt(np.sum(q * q, axis=0))
    if np.any(np.abs(d) > 1e-10):
        raise ValueError('Input is not normalized.')
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    mat = np.zeros((3, 3) + q.shape[1:], dtype=np.result_type(q.dtype, float))
    mat[0, 0] = 1 - 2 * q2 * q2 - 2 * q3 * q3
    mat[0, 1] = 2 * q1 * q2 - 2 * q0 * q3
    mat[0, 2] = 2 * q1 * q3 + 2 * q0 * q2
    mat[1, 0] = 2 * q1 * q2 + 2 * q0 * q3
    mat[1, 1] = 1 - 2 * q1 * q1 - 2 * q3 * q3
    mat[1, 2] = 2 * q2 * q3 - 2 * q0 * q1
    mat[2, 0] = 2 * q1 * q3 - 2 * q0 * q2
    mat[2, 1] = 2 * q2 * q3 + 2 * q0 * q1
    mat[2, 2] = 1 - 2 * q1 * q1 - 2 * q2 * q2
    while mat.ndim > 2 and mat.shape[-1] == 1:   # MATLAB drops trailing singleton dims
        mat = mat[..., 0]
    return mat

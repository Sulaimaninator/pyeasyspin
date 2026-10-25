"""Convert rotation matrices to unit quaternion."""
import numpy as np


def rotmat2quat(mat=None):
    """rotmat2quat(mat (3,3[,n[,k]])) -> q (4,n,k) with trailing singleton dims dropped."""
    mat = np.asarray(mat)
    if mat.ndim < 2 or mat.shape[0] != 3 or mat.shape[1] != 3 or not np.issubdtype(mat.dtype, np.number):
        raise ValueError('mat must be an array of size (3,3,...)')
    if mat.ndim > 4:
        raise ValueError('Arrays with greater than 4 dimensions are not supported.')
    m4 = mat.reshape((3, 3) + mat.shape[2:] + (1,) * (4 - mat.ndim))
    # orthogonality check (MATLAB uses multimatmult here; computed inline)
    d = np.abs(np.eye(3)[:, :, None, None] - np.einsum('ijnk,ljnk->ilnk', m4, m4))
    if np.any(d > 1e-10):
        raise ValueError('The rotation matrices are not orthogonal.')
    nn, kk = m4.shape[2], m4.shape[3]
    q = np.zeros((4, nn, kk))
    for k in range(kk):
        for n in range(nn):
            m = m4[:, :, n, k]
            itr = m[0, 0] + m[1, 1] + m[2, 2]
            m11, m12, m13 = m[0]
            m21, m22, m23 = m[1]
            m31, m32, m33 = m[2]
            if itr > 0:
                S = 2 * np.sqrt(itr + 1)
                qw = S / 4; qx = (m32 - m23) / S; qy = (m13 - m31) / S; qz = (m21 - m12) / S
            elif m11 > m22 and m11 > m33:
                S = 2 * np.sqrt(1 + m11 - m22 - m33)
                qw = (m32 - m23) / S; qx = S / 4; qy = (m12 + m21) / S; qz = (m13 + m31) / S
            elif m22 > m33:
                S = 2 * np.sqrt(1 + m22 - m11 - m33)
                qw = (m13 - m31) / S; qx = (m12 + m21) / S; qy = S / 4; qz = (m23 + m32) / S
            else:
                S = 2 * np.sqrt(1 + m33 - m11 - m22)
                qw = (m21 - m12) / S; qx = (m13 + m31) / S; qy = (m23 + m32) / S; qz = S / 4
            q[:, n, k] = [qw, qx, qy, qz]
    if np.any(np.abs(1.0 - np.sum(q * q, axis=0)) > 1e-10):
        raise ValueError('Output q is not normalized.')
    while q.ndim > 2 and q.shape[-1] == 1:
        q = q[..., 0]
    return q

"""Convert a unit quaternion into a sequence of Euler angles"""
import numpy as np
from .._registry import F, nout


@nout(-1)
def quat2euler(*varargin, nargout=None):
    """quat2euler(q[, 'passive'|'active']) -> angles (3,...); nargout=3 -> (alpha, beta, gamma)."""
    nargin = len(varargin)
    if nargin == 1:
        q = F.quatinv(varargin[0])
    elif nargin == 2:
        if not isinstance(varargin[1], str):
            raise ValueError('If two arguments are given, the second must be a string.')
        o = varargin[1].lower()
        if o == 'passive':
            q = F.quatinv(varargin[0])
        elif o == 'active':
            q = varargin[0]
        else:
            raise ValueError("The second argument must specify 'active' or 'passive'.")
    else:
        raise ValueError('Too many input arguments.')
    q = np.asarray(q)
    if q.ndim == 1:
        q = q.reshape(-1, 1)
    if q.shape[0] != 4 or not np.issubdtype(q.dtype, np.number):
        raise ValueError('q must be an array of size (4,...)')
    q = q.astype(float)
    d = np.abs(1.0 - np.sqrt(np.sum(q * q, axis=0)))
    if np.any(d > 1e-10):
        raise ValueError('q is not normalized.')
    q = np.where(q[0:1] < 0, -q, q)
    q0, q1, q2, q3 = q[0:1], q[1:2], q[2:3], q[3:4]
    sy = 2 * np.sqrt((q2 * q3 + q1 * q0) ** 2 + (q1 * q3 - q2 * q0) ** 2)
    alpha = np.arctan2(2. * (q2 * q3 - q0 * q1), 2. * (q1 * q3 + q0 * q2))
    alpha = np.where(alpha < 0, alpha + 2 * np.pi, alpha)
    beta = np.real(np.arctan2(sy, 1 - 2 * q1 ** 2 - 2 * q2 ** 2))
    gamma = np.arctan2(2. * (q2 * q3 + q0 * q1), -2. * (q1 * q3 - q0 * q2))
    gamma = np.where(gamma < 0, gamma + 2 * np.pi, gamma)
    if nargout in (None, 0, 1):
        return np.concatenate([alpha, beta, gamma], axis=0)
    if nargout == 3:
        return alpha, beta, gamma
    raise ValueError('Incorrect number of output arguments.')

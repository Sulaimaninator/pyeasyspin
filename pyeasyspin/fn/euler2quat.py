"""Convert Euler angles to unit quaternions."""
import numpy as np


def _isvector(a):
    s = a.shape if a.ndim >= 2 else (1, a.size)
    return len(s) == 2 and (s[0] == 1 or s[1] == 1)


def euler2quat(*varargin):
    """euler2quat(angles[, 'passive'|'active']) or euler2quat(alpha, beta, gamma[, opt]) -> q (4,...)."""
    nargin = len(varargin)
    if nargin in (1, 2):
        angles = np.asarray(varargin[0], dtype=float)
        if _isvector(angles):
            if angles.size != 3:
                raise ValueError('Input must be 3 separate angles, a 3-vector, or a (3,...) shaped array.')
            a = angles.ravel(order='F')
            alpha, beta, gamma = (np.array([[v]]) for v in a)
        elif angles.shape[0] == 3:
            alpha, beta, gamma = angles[0:1], angles[1:2], angles[2:3]
        else:
            raise ValueError('Input must be 3 separate angles, a 3-vector, or a (3,...) shaped array.')
        option = varargin[1] if nargin == 2 else ''
    elif nargin in (3, 4):
        v3 = [np.asarray(v, dtype=float) for v in varargin[:3]]
        if not _isvector(v3[0]) or not _isvector(v3[1]):
            raise ValueError('If three arguments are given, they must be scalars or vectors.')
        alpha, beta, gamma = (v.reshape(1, -1, order='F') for v in v3)
        option = varargin[3] if nargin == 4 else ''
    else:
        raise ValueError('Not enough input arguments.')
    if alpha.size != beta.size or beta.size != gamma.size:
        raise ValueError('Inputs must have the same number of elements.')
    opt = option.lower()
    if opt in ('', 'passive'):
        alpha, beta, gamma = -gamma, -beta, -alpha
    elif opt != 'active':
        raise ValueError("Last argument must be a string, either 'passive' or 'active'.")
    q0 = np.cos(beta / 2) * np.cos((gamma + alpha) / 2)
    q1 = np.sin(beta / 2) * np.sin((gamma - alpha) / 2)
    q2 = np.sin(beta / 2) * np.cos((gamma - alpha) / 2)
    q3 = np.cos(beta / 2) * np.sin((gamma + alpha) / 2)
    q = np.concatenate([q0, q1, q2, q3], axis=0)
    q = np.where(q[0:1] < 0, -q, q)
    return q

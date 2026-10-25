"""Rotation/transformation matrix from Euler angles"""
import numpy as np
from .._registry import nout
from .._mat import size


@nout(-1)
def erot(*varargin, nargout=None):
    """erot(angles[, 'rows'|'cols']) or erot(alpha, beta, gamma[, opt]) -> R (3x3);
    with an option, nargout=3 -> three columns/rows (3x1 each)."""
    nargin = len(varargin)
    if nargin in (1, 2):
        angles = varargin[0]
        if np.size(angles) != 3:
            s = size(np.asarray(angles))
            raise ValueError('  Three angles (either separately or in a 3-element array) expected.\n'
                             f'  You gave an {s[0]}x{s[1]} array with {np.size(angles)} elements.')
        a = np.asarray(angles, dtype=float).ravel(order='F')
        alpha, beta, gamma = a[0], a[1], a[2]
        option = varargin[1] if nargin == 2 else ''
    elif nargin in (3, 4):
        alpha, beta, gamma = (float(np.asarray(v, dtype=float).ravel()[0]) for v in varargin[:3])
        option = varargin[3] if nargin == 4 else ''
    else:
        raise ValueError('Wrong number of input arguments!')
    if not isinstance(option, str):
        raise ValueError("Last argument must be a string, either 'rows' or 'cols'.")
    if option == '':
        returnRows = False
    elif option == 'rows':
        returnRows = True
    elif option == 'cols':
        returnRows = False
    else:
        raise ValueError("Last argument must be a string, either 'rows' or 'cols'.")
    if nargout is None:
        nargout = 3 if option else 1
    if nargout == 3 and not option:
        raise ValueError('Please specify whether erot() should return the 3 columns or the 3 rows of the matrix.')
    if option and nargout != 3:
        raise ValueError("3 outputs required if you specify 'rows' or 'cols'.")
    if nargout not in (0, 1, 3):
        raise ValueError('Wrong number of outputs!')
    if np.isnan(alpha) or np.isnan(beta) or np.isnan(gamma):
        raise ValueError('At least one of the angles is NaN. Angles must be numbers.')
    sa, ca = np.sin(alpha), np.cos(alpha)
    sb, cb = np.sin(beta), np.cos(beta)
    sg, cg = np.sin(gamma), np.cos(gamma)
    R = np.array([[cg * cb * ca - sg * sa, cg * cb * sa + sg * ca, -cg * sb],
                  [-sg * cb * ca - cg * sa, -sg * cb * sa + cg * ca, sg * sb],
                  [sb * ca, sb * sa, cb]])
    if nargout == 3:
        if returnRows:
            return R[0, :].reshape(3, 1), R[1, :].reshape(3, 1), R[2, :].reshape(3, 1)
        return R[:, [0]], R[:, [1]], R[:, [2]]
    return R

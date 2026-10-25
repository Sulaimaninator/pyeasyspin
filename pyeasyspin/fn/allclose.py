"""Check whether all entries of two arrays agree within an absolute tolerance."""
import numpy as np
from .._mat import isnumeric


def allclose(*varargin):
    """allclose(A, B, tol=eps): all |A-B| < tol."""
    if len(varargin) == 2:
        A, B = varargin
        tol = np.finfo(float).eps
    elif len(varargin) == 3:
        A, B, tol = varargin
        if np.size(tol) != 1 or np.iscomplexobj(tol) or not isinstance(tol, (float, np.floating)):
            raise ValueError('Tol must be a real scalar of type double.')
    else:
        raise ValueError('Invalid number of inputs!')
    if not isnumeric(A) or not isnumeric(B):
        raise ValueError('Inputs A and B must be numeric arrays.')
    d = (np.asarray(A) - np.asarray(B)).ravel()
    if np.any(np.isnan(d)):
        raise ValueError('NaNs detected in difference!')
    return bool(np.all(np.abs(d) < tol))

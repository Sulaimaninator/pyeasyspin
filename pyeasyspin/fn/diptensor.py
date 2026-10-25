"""Calculate dipolar coupling tensor between two spins"""
import math
import numpy as np
from .._registry import F
from .._mat import vec, isnumeric


def _v(x):
    return x[0] if isinstance(x, tuple) else x


def _is_cell(x):
    """A MATLAB cell arrives as a list; a plain list of numbers counts as a numeric array."""
    if isinstance(x, (list, tuple)):
        return not all(isinstance(e, (int, float, complex, np.number)) and not isinstance(e, bool) for e in x)
    return False


def diptensor(spin1=None, spin2=None, rvec=None, *extra):
    """diptensor(spin1, spin2, rvec) -> T (3x3, MHz): dipolar coupling tensor. Each spin is a g value
    (scalar, 3 principal values, 3x3 matrix), a list [gpv, gFrame], or a nuclear isotope string;
    rvec in nm."""
    nargin = sum(a is not None for a in (spin1, spin2, rvec)) + len(extra)
    if nargin == 0:
        return None
    if nargin < 3:
        raise ValueError('Three inputs are required: spin1, spin2, and rvec.')
    if nargin > 3:
        raise ValueError('Only three inputs are allowed: spin1, spin2, and rvec.')
    if not isnumeric(rvec) or np.size(rvec) != 3:
        raise ValueError('Third input (rvec) must be a 3-element vector.')
    spins = [spin1, spin2]
    mug = [None, None]
    for iSpin in (2, 1):
        inp = spins[iSpin - 1]
        if not _is_cell(inp) and isnumeric(inp):
            a = np.asarray(inp, dtype=float)
            if a.size == 1:
                g = float(a.ravel()[0]) * np.eye(3)
            elif a.size == 3:
                g = np.diag(vec(a))
            elif a.shape == (3, 3):
                g = a
            else:
                raise ValueError('Input %d must be a 3-element array or a 3x3 array.' % iSpin)
            mug[iSpin - 1] = +_v(F.bmagn()) * g
        elif _is_cell(inp):
            if len(inp) == 2:
                gpv = inp[0]
                gFrame = inp[1]
                R_M2g = np.asarray(F.erot(gFrame))
                R_g2M = R_M2g.T
                g = R_g2M @ np.diag(vec(np.asarray(gpv, dtype=float))) @ R_g2M.T
                mug[iSpin - 1] = +_v(F.bmagn()) * g
        else:
            try:
                gn = F.nucgval(inp)
            except Exception:
                raise ValueError("Could not determine gn value of nuclear isotope '%s'." % inp) from None
            if np.size(gn) != 1:
                raise ValueError('Provide a single nuclear isotope in input %d.' % iSpin)
            mug[iSpin - 1] = -_v(F.nmagn()) * float(np.ravel(gn)[0])
    if mug[0] is None or mug[1] is None:
        raise ValueError('Index exceeds the number of array elements.')
    rvec = vec(np.asarray(rvec, dtype=float)) * 1e-9
    r = np.linalg.norm(rvec)
    n = (rvec / r).reshape(3, 1)
    d = 3 * (n @ n.T) - np.eye(3)
    m1 = mug[0].T if isinstance(mug[0], np.ndarray) else mug[0]
    pre = -_v(F.mu0()) / (4 * math.pi) * r ** -3
    T = (pre * m1) @ d if isinstance(m1, np.ndarray) else pre * m1 * d
    T = T @ mug[1] if isinstance(mug[1], np.ndarray) else T * mug[1]
    T = T / _v(F.planck()) / 1e6
    return T

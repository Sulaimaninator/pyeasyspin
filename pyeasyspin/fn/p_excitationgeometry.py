"""Microwave excitation geometry from Exp.mwMode."""
import numpy as np

from .._registry import F, nout
from .._struct import Struct
from .._mat import vec, isempty, numel


def _kori(k, what):
    """[phi theta] of the propagation direction k (letter, 3-vector, 2 angles or 1 angle)."""
    if isinstance(k, str):
        return vec(np.asarray(F.vec2ang(F.letter2vec(k), nargout=1), dtype=float))
    if numel(k) == 3:
        return vec(np.asarray(F.vec2ang(np.asarray(k, dtype=float), nargout=1), dtype=float))
    if numel(k) == 2:
        return vec(np.asarray(k, dtype=float))
    if numel(k) == 1 and what.endswith('mwMode'):   # only p_excitationgeometry accepts a single angle
        return np.array([0.0, float(vec(np.asarray(k, dtype=float))[0])])
    raise ValueError(f'k ({what}) must be a letter designating a direction, a 3-vector, or an array with two angles.')


@nout(6)
def p_excitationgeometry(mwMode=None):
    """(xi1, xik, nB1, nk, nB0, mode): angles between B0 and B1/k, lab-frame unit vectors (3x1)
    and the mode flags (struct)."""
    if mwMode is None or isempty(mwMode):
        mwMode = 'perpendicular'
    unpolarizedMode = circpolarizedMode = linearpolarizedMode = parallelMode = False
    circSense = np.nan
    if isinstance(mwMode, str):
        linearpolarizedMode = True
        if mwMode == 'perpendicular':
            k, alpha, logstr = 'y', -np.pi / 2, 'perpendicular'
        elif mwMode == 'parallel':
            k, alpha, logstr = 'y', np.pi, 'parallel'
            parallelMode = True
        else:
            raise ValueError(f"Unrecognized Exp.mwMode: '{mwMode}'.")
    elif isinstance(mwMode, (list, tuple)) and len(mwMode) == 2:
        k, Mode2 = mwMode
        if not isinstance(Mode2, str):
            linearpolarizedMode = True
            alpha = float(vec(np.asarray(Mode2, dtype=float))[0])
        else:
            if Mode2 == 'unpolarized':
                unpolarizedMode = True
            elif Mode2 == 'circular+':
                circpolarizedMode, circSense = True, 1.0
            elif Mode2 == 'circular-':
                circpolarizedMode, circSense = True, -1.0
            else:
                raise ValueError(f"Unrecognized 2nd element in Exp.mwMode: '{Mode2}'")
            alpha = 0.0
        logstr = 'user-specified'
    else:
        raise ValueError("Exp.mwMode must be 'perpendicular' or 'parallel' or a 2-element cell array.")
    F.logmsg(1, '  microwave excitation mode: %s', logstr)
    kOri = _kori(k, 'first element in mwMode')
    nB1, _, nk = F.erot(np.array([kOri[0], kOri[1], alpha]), 'rows', nargout=3)
    nB1, nk = np.asarray(nB1, dtype=float).reshape(3, 1), np.asarray(nk, dtype=float).reshape(3, 1)
    nB0 = np.array([[0.0], [0.0], [1.0]])
    xik = float(nk[:, 0] @ nB0[:, 0])
    xi1 = float(nB1[:, 0] @ nB0[:, 0])
    mode = Struct(unpolarizedMode=unpolarizedMode, circpolarizedMode=circpolarizedMode,
                  linearpolarizedMode=linearpolarizedMode, circSense=circSense, parallelMode=parallelMode)
    return xi1, xik, nB1, nk, nB0, mode

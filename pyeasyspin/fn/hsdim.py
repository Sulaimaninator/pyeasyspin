"""Dimension of the spin state space."""
import numpy as np
from .._registry import F
from .._mat import vec, isempty


def hsdim(SpinSystem=None):
    """Hilbert-space dimension of a spin system (Struct) or of a vector of spin quantum numbers."""
    if isinstance(SpinSystem, dict):
        Sys, err = F.validatespinsys(SpinSystem)
        if err:
            raise ValueError(err)
        S, I, L = Sys.S, Sys.I, Sys.L
        nEquivI = SpinSystem.get('n', None)
    else:
        S, L, I, nEquivI = SpinSystem, None, None, None
    v = lambda x: np.zeros(0) if x is None else vec(x).astype(float)
    if isempty(nEquivI):
        J = np.concatenate([v(S), v(I), v(L)])
        return float(np.prod(2 * J + 1))
    return float(np.prod(2 * v(S) + 1) * np.prod((2 * v(I) + 1) ** v(nEquivI)) * np.prod(2 * v(L) + 1))

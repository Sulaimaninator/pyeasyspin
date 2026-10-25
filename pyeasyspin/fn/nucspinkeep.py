"""Keep only selected nuclear spins in a spin system."""
import numpy as np
from .._registry import F
from .._mat import isempty, vec


def nucspinkeep(Sys=None, keepidx=None):
    """Spin system with only the nuclei keepidx (1-based) retained."""
    NewSys = Sys
    if keepidx is None or isempty(keepidx):
        return NewSys
    if 'nn' in Sys and not isempty(Sys['nn']) and np.any(np.asarray(Sys['nn']) != 0):
        raise ValueError('nucspinkeep does not work if Sys.nn is given.')
    Nucs = F.nucstring2list(Sys['Nucs'])
    nNuclei = len(Nucs)
    k = vec(keepidx)
    if np.any(k > nNuclei) or np.any(k <= 0):
        raise ValueError(f'There are only {nNuclei} nuclei in the spin system. Index out of range.')
    rmvidx = np.array([i for i in range(1, nNuclei + 1) if i not in set(k.astype(int).tolist())], float)
    return F.nucspinrmv(Sys, rmvidx)

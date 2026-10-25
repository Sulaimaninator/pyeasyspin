"""Extract the sub-system responsible for anisotropy (used by salt for orientation pre-selection)."""
import numpy as np
from .._registry import F
from .._mat import a2, vec


def _first(x):
    return x[0] if isinstance(x, tuple) else x


def anisosubsys(Sys=None, lw=None):
    """Spin system with all nuclei whose hyperfine splitting is below min(lw) removed and folded
    into Sys.HStrain (lw defaults to Sys.HStrain, else [0 0 0])."""
    subSys = _first(F.validatespinsys(Sys))
    if lw is None:
        lw = Sys['HStrain'] if 'HStrain' in Sys else np.array([[0.0, 0.0, 0.0]])
    nNuclei = int(subSys.nNuclei)
    if nNuclei == 0:
        return subSys
    lw = np.asarray(lw, dtype=float)
    splitting = 2 * vec(subSys.I) * np.max(np.abs(a2(subSys.A)), axis=1)
    rmv = splitting < np.min(lw)
    splitting[~rmv] = 0
    subSys = type(subSys)(subSys)
    subSys.HStrain = np.sqrt(lw ** 2 + np.sum(splitting ** 2))
    subSys = F.nucspinrmv(subSys, (np.nonzero(rmv)[0] + 1).astype(float).reshape(1, -1))
    subSys = type(subSys)(subSys)
    subSys.processed = 0.0
    subSys = _first(F.validatespinsys(subSys))
    return subSys

"""Larmor frequencies of nuclear spins."""
import numpy as np
from .._registry import F
from .._mat import isempty, vec


def larmorfrq(Nucs=None, Fields=None):
    """Larmor frequencies (MHz), one row per field (mT), one column per isotope."""
    if Nucs is None or Fields is None:
        raise ValueError('Wrong number of input arguments!')
    if isempty(Nucs) or isempty(Fields):
        return np.zeros((0, 0))
    if isinstance(Nucs, (list, tuple)):
        Nucs = F.nuclist2string(Nucs)
    gn = np.abs(vec(F.nucgval(Nucs)))
    prefac = gn * F.nmagn()[0] / 1e3 / F.planck() / 1e6
    return np.outer(vec(Fields), prefac)

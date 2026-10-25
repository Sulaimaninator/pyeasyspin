"""Remove nuclear spins from a spin system."""
import numpy as np
from .._registry import F
from .._struct import Struct
from .._mat import a2, vec, isempty, size


def _delrows(v, idx):
    v = a2(v)
    return np.delete(v, idx, axis=0)


def _dellin(v, idx):
    a = np.asarray(v)
    if a.ndim == 2 and a.shape[1] == 1 and a.shape[0] > 1:
        return np.delete(a, idx, axis=0)
    return np.delete(vec(a), idx).reshape(1, -1)


def _del3(v, rmv):
    v = a2(v)
    rows = [r for i in rmv for r in range(3 * (i - 1), 3 * i)]
    return np.delete(v, rows, axis=0)


def nucspinrmv(Sys=None, rmvidx=None):
    """Spin system with the nuclei rmvidx (1-based) removed."""
    New = Struct(Sys)
    if rmvidx is None or isempty(rmvidx):
        return Sys
    if 'nn' in Sys and not isempty(Sys['nn']) and np.any(vec(Sys['nn']) != 0):
        raise ValueError('nucspinrmv does not work if Sys.nn is given.')
    Nucs = list(F.nucstring2list(New.Nucs))
    nNuclei = len(Nucs)
    r = [int(i) for i in vec(rmvidx)]
    if any(i > nNuclei or i <= 0 for i in r):
        raise ValueError(f'There are only {nNuclei} nuclei in the spin system. Index out of range.')
    i0 = [i - 1 for i in r]
    Nucs = [n for k, n in enumerate(Nucs) if k not in set(i0)]
    s = F.nuclist2string(Nucs)
    if isempty(s):
        del New['Nucs']
    else:
        New.Nucs = s

    def full(flag, fld):
        if flag not in New:
            return size(a2(New[fld]), 1) == 3 * nNuclei if fld in New else False
        return bool(New[flag])

    def setor(fld, v):
        if isempty(v):
            del New[fld]
        else:
            New[fld] = v

    for flag, fields, main in (('fullsigma', ('sigma', 'sigmaFrame'), 'sigma'),
                               ('fullA', ('A', 'AFrame', 'A_'), 'A'),
                               ('fullQ', ('Q', 'QFrame'), 'Q')):
        if not full(flag, main):
            for fld in fields:
                if fld in New:
                    setor(fld, _delrows(New[fld], i0))
        else:
            v = _del3(New[main], sorted(r, reverse=True))
            if main == 'A' and np.any(np.isnan(v)):
                v = vec(v)[~np.isnan(vec(v))].reshape(1, -1)
            setor(main, v)
            fr = main + 'Frame'
            if fr in New:
                del New[fr]
        if main == 'sigma':
            for fld in ('n', 'gnscale'):
                if fld in New:
                    setor(fld, _dellin(New[fld], i0))
    return New

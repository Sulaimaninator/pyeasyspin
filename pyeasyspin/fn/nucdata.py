"""Nuclear spin data from EasySpin's isotopedata.txt."""
import os
import numpy as np
from .._registry import F, nout
from .._struct import Struct

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'isotopedata.txt')
_LIST = None


def _load():
    global _LIST
    if _LIST is None:
        if not os.path.exists(_DATA):
            raise ValueError(f'Could not open nuclear data file {_DATA}')
        cols = [[] for _ in range(9)]
        with open(_DATA) as fh:
            for line in fh:
                line = line.split('%', 1)[0].split()
                if len(line) < 9:
                    continue
                for k in range(9):
                    cols[k].append(line[k])
        num = lambda c: np.array([float(v) for v in c]).reshape(-1, 1)
        L = Struct()
        L.Protons, L.Nucleons = num(cols[0]), num(cols[1])
        L.Radioactive, L.Element, L.Name = cols[2], cols[3], cols[4]
        L.Spins, L.gns, L.Abundances, L.qms = num(cols[5]), num(cols[6]), num(cols[7]), num(cols[8])
        L.Symbols = ['%d%s' % (n, e) for n, e in zip(L.Nucleons[:, 0], L.Element)]
        _LIST = L
    return _LIST


@nout(5)
def nucdata(Isotopes=None, *_unused):
    """(Spin, gn, qm, Abund, Nucs) for an isotope string ('1H,14N') or list. Without input,
    returns the whole isotope table (Struct with column arrays) as a single value."""
    L = _load()
    if Isotopes is None:
        return L
    if isinstance(Isotopes, (list, tuple)) and len(Isotopes) == 1:
        Isotopes = Isotopes[0]
    if not isinstance(Isotopes, (str, list, tuple)):
        raise ValueError('Argument must be a string or a cell array!')
    Nucs = F.nucstring2list(Isotopes) if isinstance(Isotopes, str) else list(Isotopes)
    if len(Isotopes) == 0:
        e = np.zeros((0, 0))
        return e, e, e, e, e
    sym = L.Symbols
    for k in range(len(Nucs) - 1, -1, -1):
        if Nucs[k] not in sym:
            req = ''.join(ch for ch in Nucs[k] if ch.isalpha())
            iidx = [i for i, e in enumerate(L.Element) if e == req]
            if iidx:
                msg = (f"Problem in isotopes list, entry {k + 1} ('{Nucs[k]}'): Please specify one of "
                       + ', '.join(sym[i] for i in iidx) + '.')
            else:
                msg = f"Problem in isotopes list, entry {k + 1} ('{Nucs[k]}'): Unknown element '{req}'."
            raise ValueError(msg)
    idx = [sym.index(n) for n in Nucs]
    Spin = L.Spins[idx, 0]
    gn = L.gns[idx, 0]
    qm = L.qms[idx, 0]
    Abund = L.Abundances[idx, 0] / 100
    return Spin, gn, qm, Abund, Nucs

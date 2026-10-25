"""Hyperfine and quadrupole reference isotopes of an element."""
import numpy as np
from .._registry import F, nout
from .._struct import Struct


@nout(2)
def referenceisotope(el=None):
    """(gref, qref): most abundant isotope with I>=1/2 (gn, symbol, I) and with I>=1 (qm, symbol, I)."""
    no = F.elementsymbol2no(el) if isinstance(el, str) else el
    data = F.nucdata()
    if isinstance(data, tuple):
        data = data[0]
    idx = np.nonzero(data.Protons[:, 0] == no)[0]
    ab = data.Abundances[idx, 0]
    I = data.Spins[idx, 0]
    gn, qm = data.gns[idx, 0], data.qms[idx, 0]
    symb = [data.Symbols[i] for i in idx]
    a = np.where(I < 0.5, 0, ab)
    gref = qref = None
    if a.size and a.max() > 0:
        i = int(np.argmax(a))
        gref = Struct(gn=gn[i], symbol=symb[i], I=I[i])
    a = np.where(I < 1, 0, ab)
    if a.size and a.max() > 0:
        i = int(np.argmax(a))
        qref = Struct(qm=qm[i], symbol=symb[i], I=I[i])
    return gref, qref

"""Spin-orbit interaction Hamiltonian."""
import numpy as np
from .._registry import F
from .._mat import vec, isempty, a2
from .._ham import spzeros, finish


def ham_so(System=None, eSpins=None, opt=None):
    """Spin-orbit Hamiltonian sum_k soc(k) (S.L)^k (MHz)."""
    if opt is None:
        opt = ''
    if not isinstance(opt, str):
        raise ValueError("Third input must be a string, 'sparse'.")
    sparseResult = opt == 'sparse'
    System, err = F.validatespinsys(System)
    if err:
        raise ValueError(err)
    H = spzeros(System.nStates)
    if int(System.nL) == 0 or np.all(np.asarray(System.soc) == 0):
        return finish(H, sparseResult)
    nE = int(System.nElectrons)
    eS = list(range(1, nE + 1)) if eSpins is None or isempty(eSpins) else [int(s) for s in vec(eSpins)]
    if len(eS) < 1:
        raise ValueError('eSpins (2nd argument) must contain at least 1 values!')
    if any(s < 1 or s > nE for s in eS):
        raise ValueError(f'eSpins (2nd argument) must contain values between 1 and {nE}!')
    if len(set(eS)) != len(eS):
        raise ValueError('eSpins (2nd argument) contains double entries!')
    shift = nE + int(System.nNuclei)
    soc = a2(System.soc)
    L = vec(System.L)
    for iS in eS:
        iL = iS + shift
        soc_ = soc[iS - 1, :]
        if not np.any(soc_):
            continue
        if L[iS - 1] == 0:
            continue
        SL = spzeros(System.nStates)
        for c in range(1, 4):
            SL = SL + F.sop(System, np.array([[iS, c], [iL, c]], float), 'sparse')
        for order in np.nonzero(soc_)[0] + 1:
            P = SL
            for _ in range(int(order) - 1):
                P = P @ SL
            H = H + soc_[order - 1] * P
    return finish(H, sparseResult)

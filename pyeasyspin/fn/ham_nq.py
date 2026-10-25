"""Nuclear quadrupole interaction Hamiltonian."""
import numpy as np
from .._registry import F
from .._ham import spzeros, finish, idxlist, rotate, herm


def ham_nq(SpinSystem=None, Nuclei=None, Options=None):
    """Nuclear quadrupole Hamiltonian (MHz) for the given 1-based nuclei."""
    if Options is None:
        Options = ''
    if not isinstance(Options, str):
        raise ValueError("Third input must be a string, 'sparse'.")
    useSparse = Options == 'sparse'
    Sys, err = F.validatespinsys(SpinSystem)
    if err:
        raise ValueError(err)
    nN = int(Sys.nNuclei)
    nuc = idxlist(Nuclei, range(1, nN + 1))
    if any(n > nN or n < 1 for n in nuc):
        raise ValueError('Nuclear spin index/indices (2nd argument) out of range!')
    H = spzeros(Sys.nStates)
    spvc = Sys.Spins
    nE = int(Sys.nElectrons)
    for n in nuc:
        Qa = np.asarray(Sys.Q)
        Q = Qa[3 * (n - 1):3 * n, :] if Sys.fullQ else np.diag(Qa[n - 1, :])
        if not np.any(Q):
            continue
        ang = Sys.QFrame[n - 1, :]
        if np.any(ang):
            Q = rotate(Q, ang, F)
        I = [F.sop(spvc, np.array([[nE + n, k]], float), 'sparse') for k in (1, 2, 3)]
        for k in range(3):
            for q in range(3):
                H = H + I[k] @ (Q[k, q] * I[q])
    return finish(herm(H), useSparse)

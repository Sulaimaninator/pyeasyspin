"""Hyperfine interaction Hamiltonian."""
import numpy as np
from .._registry import F
from .._mat import a2, vec
from .._ham import spzeros, finish, idxlist, rotate, herm


def ham_hf(System=None, elSpins=None, nucSpins=None, opt=None):
    """Hyperfine Hamiltonian (MHz) for the given 1-based electron and nuclear indices."""
    if opt is None:
        opt = ''
    if not isinstance(opt, str):
        raise ValueError("Fourth input must be a string, 'sparse'.")
    useSparse = opt == 'sparse'
    Sys, err = F.validatespinsys(System)
    if err:
        raise ValueError(err)
    SpinVec = Sys.Spins
    nE, nN = int(Sys.nElectrons), int(Sys.nNuclei)
    H = spzeros(Sys.nStates)
    if nN == 0:
        return finish(H, useSparse)
    eS = idxlist(elSpins, range(1, nE + 1))
    if elSpins is not None and any(e < 1 or e > nE for e in eS):
        raise ValueError('Electron spins (2nd argument) contains out-of-range values!')
    nS = idxlist(nucSpins, range(1, nN + 1))
    if nucSpins is not None and any(n < 1 or n > nN for n in nS):
        raise ValueError('Nuclear spins (3rd argument) contains out-of-range values!')
    fullA = a2(System['A'] if 'A' in System else Sys.A).shape[0] > nN
    Aall = np.asarray(Sys.A)
    Ivec = vec(Sys.I)
    for e in eS:
        eidx = slice((e - 1) * 3, e * 3)
        for n in nS:
            if Ivec[n - 1] == 0:
                continue
            A = Aall[(n - 1) * 3:n * 3, eidx] if fullA else np.diag(Aall[n - 1, eidx])
            if not np.any(A):
                continue
            ang = Sys.AFrame[n - 1, eidx]
            if np.any(ang):
                A = rotate(A, ang, F)
            for c1 in range(1, 4):
                for c2 in range(1, 4):
                    H = H + A[c1 - 1, c2 - 1] * F.sop(SpinVec, np.array([[e, c1], [nE + n, c2]], float), 'sparse')
    return finish(herm(H), useSparse)

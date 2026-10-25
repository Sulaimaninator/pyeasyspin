"""Crystal-field Hamiltonian for orbital angular momenta."""
import numpy as np
from .._registry import F
from .._mat import vec, isempty, numel, a2
from .._ham import spzeros, finish, idxlist, herm, add


def ham_cf(SpinSystem=None, idxL=None, opt=None):
    """Crystal-field Hamiltonian (MHz) from Stevens coefficients Sys.CF1..CF12."""
    if opt is None:
        opt = ''
    if not isinstance(opt, str):
        raise ValueError("Third input must be a string, 'sparse'.")
    useSparse = opt == 'sparse'
    Sys, err = F.validatespinsys(SpinSystem)
    if err:
        raise ValueError(err)
    idx = idxlist(idxL, range(1, int(Sys.nL) + 1))
    if any(i > int(Sys.nL) or i < 1 for i in idx):
        raise ValueError('OAM index/indices (2nd argument) out of range!')
    H = spzeros(Sys.nStates)
    if int(Sys.nL) == 0:
        return finish(H, useSparse)
    offset = int(Sys.nElectrons) + int(Sys.nNuclei)
    L = vec(Sys.L)
    for iL in idx:
        if L[iL - 1] < 1:
            continue
        for k in range(1, 13):
            fn = f'CF{k}'
            if fn not in Sys:
                continue
            CFk = Sys[fn]
            if isempty(CFk):
                continue
            CFk = a2(CFk)
            if CFk.size != (2 * k + 1) * int(Sys.nL):
                raise ValueError(f'Sys.{fn} has wrong size. It should contain {2 * k + 1} elements for each L.')
            if np.all(CFk == 0):
                continue
            q = list(range(k, -k - 1, -1))
            for iq in np.nonzero(CFk[iL - 1, :] != 0)[0]:
                H = add(H, CFk[iL - 1, iq] * np.asarray(F.stev(Sys.Spins, np.array([k, q[iq], iL + offset], float))))
    return finish(herm(H), useSparse)

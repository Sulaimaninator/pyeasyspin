"""Electron-electron spin interaction Hamiltonian."""
import numpy as np
from .._registry import F
from .._mat import vec, isempty
from .._ham import spzeros, finish, herm, nchoosek2, rotate


def ham_ee(System=None, Spins=None, opt=None):
    """Electron-electron coupling Hamiltonian (MHz) from Sys.ee/eeFrame (and Sys.ee2)."""
    if opt is None:
        opt = ''
    if not isinstance(opt, str):
        raise ValueError("Third input must be a string, 'sparse'.")
    useSparse = opt == 'sparse'
    System, err = F.validatespinsys(System)
    if err:
        raise ValueError(err)
    sys_ = F.spinvec(System)
    n = int(np.prod(2 * vec(sys_) + 1))
    H = spzeros(n)
    nE = int(System.nElectrons)
    if nE == 1:
        return H
    if not np.any(np.asarray(System.ee)) and not np.any(np.asarray(System.ee2)):
        return H
    Sp = list(range(1, nE + 1)) if Spins is None or isempty(Spins) else [int(s) for s in vec(Spins)]
    if len(Sp) < 2:
        raise ValueError('Spins (2nd argument) must contain at least 2 values!')
    if any(s < 1 or s > nE for s in Sp):
        raise ValueError(f'Spins (2nd argument) must contain values between 1 and {nE}!')
    if len(set(Sp)) != len(Sp):
        raise ValueError('Spins (2nd argument) contains double entries!')
    Sp = sorted(Sp)
    Pairs = [(Sp[a - 1], Sp[b - 1]) for a, b in nchoosek2(len(Sp))]
    allPairs = nchoosek2(nE)
    ee = np.asarray(System.ee)
    eeFrame = np.asarray(System.eeFrame) if 'eeFrame' in System else np.zeros((0, 0))
    ee2 = vec(System.ee2)
    for p1, p2 in Pairs:
        ic = allPairs.index((p1, p2))
        J = ee[3 * ic:3 * ic + 3, :] if System.fullee else np.diag(ee[ic, :])
        if eeFrame.size and np.any(eeFrame[ic, :]):
            J = rotate(J, eeFrame[ic, :], F)
        for c1 in range(1, 4):
            so1 = F.sop(sys_, np.array([[p1, c1]], float), 'sparse')
            for c2 in range(1, 4):
                so2 = F.sop(sys_, np.array([[p2, c2]], float), 'sparse')
                H = H + so1 @ (J[c1 - 1, c2 - 1] * so2)
        if ee2[ic] == 0:
            continue
        H2 = 0
        for c in range(1, 4):
            H2 = H2 + F.sop(sys_, np.array([[p1, c], [p2, c]], float), 'sparse')
        H = H + ee2[ic] * (H2 @ H2)
    return finish(herm(H), useSparse)

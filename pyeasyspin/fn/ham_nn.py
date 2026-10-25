"""Nuclear-nuclear spin interaction Hamiltonian."""
import numpy as np
from .._registry import F
from .._mat import vec, isempty
from .._ham import spzeros, finish, herm, nchoosek2, rotate


def ham_nn(System=None, nucSpins=None, opt=None):
    """Nuclear-nuclear coupling Hamiltonian (MHz) from Sys.nn/nnFrame."""
    if opt is None:
        opt = ''
    if not isinstance(opt, str):
        raise ValueError("Third input must be a string, 'sparse'.")
    sparseResult = opt == 'sparse'
    System, err = F.validatespinsys(System)
    if err:
        raise ValueError(err)
    sys_ = F.spinvec(System)
    n = int(np.prod(2 * vec(sys_) + 1))
    H = spzeros(n)
    nN = int(System.nNuclei)
    if nN < 2 or not np.any(np.asarray(System.nn)):
        return finish(H, sparseResult)
    Sp = list(range(1, nN + 1)) if nucSpins is None or isempty(nucSpins) else [int(s) for s in vec(nucSpins)]
    if len(Sp) < 2:
        raise ValueError('Spins (2nd argument) must contain at least 2 values!')
    if any(s < 1 or s > nN for s in Sp):
        raise ValueError(f'Spins (2nd argument) must contain values between 1 and {nN}!')
    if len(set(Sp)) != len(Sp):
        raise ValueError('Spins (2nd argument) contains double entries!')
    Sp = sorted(Sp)
    Pairs = [(Sp[a - 1], Sp[b - 1]) for a, b in nchoosek2(len(Sp))]
    allPairs = nchoosek2(nN)
    nn = np.asarray(System.nn)
    nE = int(System.nElectrons)
    for p1, p2 in Pairs:
        ic = allPairs.index((p1, p2))
        J = nn[3 * ic:3 * ic + 3, :] if System.fullnn else np.diag(nn[ic, :])
        if not np.any(J):
            continue
        ang = np.asarray(System.nnFrame)[ic, :]
        if np.any(ang):
            J = rotate(J, ang, F)
        for c1 in range(1, 4):
            so1 = F.sop(sys_, np.array([[nE + p1, c1]], float), 'sparse')
            for c2 in range(1, 4):
                so2 = F.sop(sys_, np.array([[nE + p2, c2]], float), 'sparse')
                H = H + so1 @ (J[c1 - 1, c2 - 1] * so2)
    return finish(herm(H), sparseResult)

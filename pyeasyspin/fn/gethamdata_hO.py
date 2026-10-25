"""Diagonalize the spin Hamiltonian with higher-order Zeeman terms; dE/dB by finite difference."""
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from scipy.linalg import block_diag
from .._registry import F, nout
from .._mat import vec


@nout(4)
def gethamdata_hO(B=None, n0=None, Sys=None, sparse=None, idxT=None, nLevels=None, nargout=4):
    """(V, E, dEdB, dE) at field B (mT) along n0; dEdB from a 0.01 mT forward difference."""
    dB = 0.01
    n0 = vec(n0)
    nLevels = int(nLevels)
    H2 = E2 = None
    if sparse:
        H = F.ham(Sys, B * n0, 'sparse')
        E, V = spla.eigsh(H, k=nLevels, which='LM')
        idx = np.argsort(E, kind='stable')
        E, V = E[idx], V[:, idx]
        if nargout > 2:
            H2 = F.ham(Sys, (B + dB) * n0, 'sparse')
            E2 = np.sort(spla.eigsh(H2, k=nLevels, which='LM', return_eigenvectors=False))
    else:
        H = np.asarray(F.ham(Sys, B * n0))
        E, V = np.linalg.eig(H)
        if np.sum(np.abs(np.imag(E))) > 1e-6:
            raise ValueError('Imaginary energies obtained! Please report!')
        idx = np.argsort(np.real(E), kind='stable')
        E, V = np.real(E)[idx], V[:, idx]
        if nLevels < E.size:
            E, V = E[:nLevels], V[:, :nLevels]
        if nargout > 2:
            H2 = np.asarray(F.ham(Sys, (B + dB) * n0))
            E2 = np.linalg.eigvals(H2)
            if np.sum(np.abs(np.imag(E2))) > 1e-6:
                raise ValueError('Imaginary energies obtained! Please report!')
            E2 = np.sort(np.real(E2))[:nLevels]
    dE = None
    if B == 0:
        if H2 is None:
            H2 = F.ham(Sys, (B + dB) * n0)
        G = H2 - H
        G = G.toarray() if sps.issparse(G) else np.asarray(G)
        d = np.abs(np.diff(E))
        tol = 1e3 * np.finfo(float).eps * (d.max() if d.size else 0)
        blk = np.cumsum(np.concatenate([[1], d > tol]))
        GG = V.conj().T @ G @ V
        GG = (GG + GG.conj().T) / 2
        blocks = [np.linalg.eig(GG[np.ix_(ix, ix)])[1]
                  for ix in (np.nonzero(blk == k)[0] for k in range(1, int(blk.max()) + 1))]
        V = V @ block_diag(*blocks)
        dE = d
    dEdB = None
    if nargout > 2:
        dEdB = (E2.reshape(-1, 1) - E.reshape(1, -1)) / dB
    if nargout > 3:
        dEm = -(E.reshape(-1, 1) - E.reshape(1, -1))
        dE = vec(dEm)[np.asarray(vec(idxT), dtype=int) - 1]
    return V, E, dEdB, dE

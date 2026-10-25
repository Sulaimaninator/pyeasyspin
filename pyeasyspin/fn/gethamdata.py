"""Diagonalize H0 - B*muzL (used by resfields and resfreqs_matrix)."""
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from scipy.linalg import block_diag
from .._registry import nout
from .._mat import vec


@nout(4)
def gethamdata(B=None, H0=None, muzL=None, idxT=None, nLevels=None, nargout=4):
    """(V, E, dEdB, dE): eigenvectors, sorted energies, dE/dB and transition energies dE(idxT).
    At B=0, degenerate eigenvectors are rotated to diagonalize the Zeeman operator."""
    H = H0 - B * muzL
    nLevels = int(nLevels)
    if sps.issparse(H):
        E, V = spla.eigsh(H, k=nLevels, which='LM')
    else:
        E, V = np.linalg.eigh(np.asarray(H))
    idx = np.argsort(E, kind='stable')
    if np.any(np.diff(E) < 0):
        E, V = E[idx], V[:, idx]
    if nLevels < E.size:
        E, V = E[:nLevels], V[:, :nLevels]
    mz = muzL.toarray() if sps.issparse(muzL) else np.asarray(muzL)
    dE = None
    if B == 0:
        d = np.abs(np.diff(E))
        tol = 1e3 * np.finfo(float).eps * (d.max() if d.size else 0)
        blk = np.cumsum(np.concatenate([[1], d > tol]))
        GG = V.conj().T @ (-mz) @ V
        GG = (GG + GG.conj().T) / 2
        blocks = []
        for k in range(1, int(blk.max()) + 1):
            ix = np.nonzero(blk == k)[0]
            blocks.append(np.linalg.eigh(GG[np.ix_(ix, ix)])[1])
        V = V @ block_diag(*blocks)
        dE = d
    dEdB = None
    if nargout > 2:
        dEdB = np.real(np.diag(V.conj().T @ (-mz) @ V))
    if nargout > 3:
        dEm = -(E.reshape(-1, 1) - E.reshape(1, -1))
        dE = vec(dEm)[np.asarray(vec(idxT), dtype=int) - 1] if np.asarray(idxT).dtype != bool \
            else vec(dEm)[vec(idxT)]
    return V, E, dEdB, dE

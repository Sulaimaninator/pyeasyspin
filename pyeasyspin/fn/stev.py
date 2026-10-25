"""Extended Stevens spin operator matrices"""
import numpy as np
import scipy.sparse as sps
from .._registry import F

_F = np.zeros((13, 13))
_F[12, :13] = [1916006400, 958003200, 958003200, 31933440, 3991680, 1995840, 31680, 15840, 1584, 264, 24, 12, 1]
_F[11, :12] = [319334400, 79833600, 79833600, 13305600, 2661120, 23760, 7920, 1320, 1320, 22, 22, 1]
_F[10, :11] = [14515200, 7257600, 1209600, 604800, 86400, 2880, 360, 180, 20, 10, 1]
_F[9, :10] = [1451520, 725700, 725700, 60480, 60480, 864, 288, 18, 18, 1]
_F[8, :9] = [80640, 40320, 40320, 6720, 672, 336, 16, 8, 1]
_F[7, :8] = [40320, 5040, 1680, 168, 168, 14, 14, 1]
_F[6, :7] = [2880, 1440, 360, 60, 12, 6, 1]
_F[5, :6] = [480, 240, 240, 10, 10, 1]
_F[4, :5] = [48, 24, 8, 4, 1]
_F[3, :4] = [24, 6, 6, 1]
_F[2, :3] = [4, 2, 1]
_F[1, :2] = [2, 1]
_F[0, 0] = 1
_kmax = 12


def stev(Spins=None, Comps=None, Sparse=None):
    """stev(Spins, [k q] or [k q iSpin][, 'sparse']) -> Stevens operator matrix (iSpin 1-based)."""
    if Comps is None:
        raise ValueError('Wrong number of input arguments!')
    if Sparse is None:
        useSparse = False
    else:
        if not isinstance(Sparse, str):
            raise ValueError("Third input argument must be a character array ('sparse').")
        useSparse = Sparse == 'sparse'
    if isinstance(Spins, dict):
        Spins = F.spinvec(Spins)
    Spins = np.asarray(Spins)
    if not np.issubdtype(Spins.dtype, np.number) or np.iscomplexobj(Spins) or np.any((2 * Spins) % 1) \
            or np.any(Spins < 0):
        raise ValueError('S must contain positive multiples of 1/2.')
    Spins = Spins.astype(float).ravel(order='F')
    C = np.asarray(Comps)
    if not np.issubdtype(C.dtype, np.number) or np.iscomplexobj(C) or (C.ndim == 2 and C.shape[0] != 1) \
            or C.ndim > 2:
        raise ValueError('Second input must be a 1x2 or 1x3 array.')
    C = list(C.astype(float).ravel())
    if len(C) == 2:
        if Spins.size > 1:
            raise ValueError('For a system with more than one spin, provide the desired spin index in iSpins.')
        C.append(1.0)
    k, q, iSpin = C[0], C[1], C[2]
    if iSpin < 0 or iSpin > Spins.size:
        raise ValueError(f'iSpin = {iSpin:g} is out of range. It should be between 1 and {Spins.size}')
    S = Spins[int(iSpin) - 1]
    if k % 1 or k < 0:
        raise ValueError('k must be a non-negative integer.')
    if k > _kmax:
        raise ValueError(f'k too large. Maximum supported k is {_kmax}.')
    if k > 2 * S:
        raise ValueError(f'Stevens operator with k={k:g} given. k must not be larger than 2*S '
                         f'({2 * S:g} with S={S:g}).')
    if q % 1 or abs(q) > k:
        raise ValueError(f'q must be an integer between -k and k ({-k:g} and {k:g}).')
    k, q = int(k), int(q)
    Jp = sps.csr_matrix(F.sop(Spins, np.array([[iSpin, 4.]]), 'sparse'))
    Jm = sps.csr_matrix(F.sop(Spins, np.array([[iSpin, 5.]]), 'sparse'))
    T = sps.identity(Jp.shape[0], format='csr')
    for _ in range(k):
        T = T @ Jp
    for _ in range(k - 1, abs(q) - 1, -1):
        T = Jm @ T - T @ Jm
    if k % 2:
        alpha = 1.0
    else:
        alpha = 0.5 if q % 2 else 1.0
    c = alpha / _F[k, abs(q)]
    c = (-1) ** (k - q) * c
    if q >= 0:
        Op = c / 2 * (T + T.conj().T)
    else:
        Op = c / 2j * (T - T.conj().T)
    Op = Op.toarray()
    # MATLAB: Op(abs(Op(find(Op)))<1e-14) = 0 -- the logical mask over the nonzeros is applied to the
    # leading column-major linear indices of Op (EasySpin quirk, reproduced literally)
    flat = Op.ravel(order='F')
    nzv = flat[np.flatnonzero(flat)]
    pos = np.flatnonzero(np.abs(nzv) < 1e-14)
    if pos.size:
        flat = flat.copy()
        flat[pos] = 0
        Op = flat.reshape(Op.shape, order='F')
    if not np.any(Op.imag):
        Op = Op.real if q >= 0 else Op
    if useSparse:
        return sps.csr_matrix(Op)
    return Op

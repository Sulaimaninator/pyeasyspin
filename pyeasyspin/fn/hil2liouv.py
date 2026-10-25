"""takes Hilbert-space operators and transforms them to Liouville space"""
import scipy.sparse as sp
from .._registry import F


def _kronkron(A):
    n = max(A.shape)
    return F.eyekron(n, A) - F.kroneye(A.T, n)


def hil2liouv(H=None):
    """hil2liouv(H) -> commutation superoperator kron(I, H) - kron(H.', I) (sparse); H may be a
    list (cell) of sparse matrices, giving a list."""
    if not isinstance(H, list):
        if not sp.issparse(H):
            raise ValueError('Input matrix must be in sparse format.')
        return _kronkron(H)
    HL = []
    for Hk in H:
        if not sp.issparse(Hk):
            raise ValueError('Input matrix must be in sparse format.')
        HL.append(_kronkron(Hk))
    return HL

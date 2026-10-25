"""Index array that reorders the Liouville-space basis by coherence order p and m-sum q."""
import numpy as np
from .._registry import nout
from .._mat import vec


@nout(3)
def pqorder(Spins=None):
    """(idx, pq, mm): idx = 1-based permutation (N^2 x 1) that sorts the Liouville basis rows of
    pq = [m1-m2, m1+m2] (per spin, 2 columns each) with MATLAB sortrows; mm = [m1, m2] per spin."""
    Spins = vec(np.asarray(Spins, dtype=float))
    nSpins = Spins.size
    nStates = np.rint(2 * Spins + 1).astype(int)
    N = int(np.prod(nStates))
    n1, n2 = N, 1
    m = np.zeros((N, nSpins))
    for iSpin in range(nSpins):
        n1 //= nStates[iSpin]
        mrow = Spins[iSpin] - np.arange(nStates[iSpin])          # Spins(iSpin):-1:-Spins(iSpin)
        m_ = np.tile(mrow.reshape(1, -1), (n1, n2))
        m[:, iSpin] = vec(m_)
        n2 *= nStates[iSpin]
    mm = np.zeros((N * N, 2 * nSpins))
    pq = np.zeros((N * N, 2 * nSpins))
    for iSpin in range(nSpins):
        m1 = np.tile(m[:, [iSpin]], (1, N))
        m2 = m1.T
        m1, m2 = vec(m1), vec(m2)
        mm[:, 2 * iSpin] = m1
        mm[:, 2 * iSpin + 1] = m2
        pq[:, 2 * iSpin] = m1 - m2
        pq[:, 2 * iSpin + 1] = m1 + m2
    # sortrows: stable lexicographic sort, first column primary
    idx = np.lexsort(pq.T[::-1]) if pq.size else np.zeros(0, dtype=int)
    return (idx + 1).astype(float).reshape(-1, 1), pq, mm

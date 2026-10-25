"""function takes as input the orientational basis size, the isotropic and anisotropic spin
Liouvillian components, and the 3j symbol tables, and returns the Hamiltonian superoperator
in the K-symmetrized (LMKjK) orientational basis times the spin basis"""
import numpy as np
import scipy.sparse as sp


def _dense(Q):
    return Q.toarray() if sp.issparse(Q) else np.atleast_2d(np.asarray(Q))


def _find(A):
    """MATLAB [row,col,val] = find(A): nonzeros in column-major order, 1-based indices."""
    c, r = np.nonzero(A.T)
    return r + 1, c + 1, A[r, c]


def _sqrtjk(jK):
    return 1.0 if jK >= 0 else 1j * np.sqrt(-jK)


def _isempty(Q):
    return Q is None or (isinstance(Q, list) and len(Q) == 0) or (not isinstance(Q, list) and np.size(Q) == 0)


def _cellget(Q, n):
    """Q{i,j} (1-based) of an n x n cell given as a column-major list, densified."""
    return lambda i, j: Q[(i - 1) + (j - 1) * n]


def _assemble(bra, ket, el, N):
    """sparse(bra, ket, el, N, N) (duplicates summed); no elements gives an all-zero matrix."""
    if not el:
        return sp.csc_matrix((N, N))
    bra = np.concatenate(bra) - 1
    ket = np.concatenate(ket) - 1
    el = np.concatenate(el)
    return sp.csc_matrix((el, (bra, ket)), shape=(N, N))


def liouvhamiltonian(basis=None, Q0=None, Q1=None, Q2=None, jjj0=None, jjj1=None, jjj2=None):
    """liouvhamiltonian(basis, Q0, Q1, Q2, jjj0, jjj1, jjj2) -> sparse H (rank 0, 1, 2 parts;
    Q1 3x3 and Q2 5x5 cells as column-major lists, Q1 may be empty)."""
    L = np.asarray(basis['L'], dtype=float).ravel(order='F').astype(int)
    M = np.asarray(basis['M'], dtype=float).ravel(order='F').astype(int)
    K = np.asarray(basis['K'], dtype=float).ravel(order='F').astype(int)
    jK = np.asarray(basis['jK'], dtype=float).ravel(order='F').astype(int)
    Q0d = _dense(Q0)
    nSpinBasis = max(Q0d.shape)
    nOriBasis = L.size
    nTotalBasis = nOriBasis * nSpinBasis
    jjj0 = np.asarray(jjj0)

    # rank 0
    row, col, val = _find(Q0d)
    bra, ket, el = [], [], []
    for iBasis in range(nOriBasis):
        L_, M_, K_ = L[iBasis], M[iBasis], K[iBasis]
        idx0 = iBasis * nSpinBasis
        jjjM = jjj0[L_ ** 2 + L_ + M_, L_ ** 2 + L_ - M_]
        jjjK = jjj0[L_ ** 2 + L_ + K_, L_ ** 2 + L_ - K_]
        el.append((-1.0) ** (K_ - M_) * (2 * L_ + 1) * jjjM * jjjK * val)
        bra.append(idx0 + row)
        ket.append(idx0 + col)
    H0 = _assemble(bra, ket, el, nTotalBasis)

    # rank 1
    H1 = None
    if not _isempty(Q1):
        jjj1 = np.asarray(jjj1)
        Q1d = [_dense(q) for q in Q1]
        Q1c = _cellget(Q1d, 3)
        bra, ket, el = [], [], []
        for iBasis in range(nOriBasis):
            L1, M1, K1, jK1 = L[iBasis], M[iBasis], K[iBasis], jK[iBasis]
            idx1 = iBasis * nSpinBasis
            K1zero = K1 == 0
            for jBasis in range(nOriBasis):
                L2 = L[jBasis]
                if abs(L1 - L2) > 1:
                    continue
                M2 = M[jBasis]
                if abs(M1 - M2) > 1:
                    continue
                K2 = K[jBasis]
                if abs(K1 - K2) > 1:
                    continue
                jK2 = jK[jBasis]
                idx2 = jBasis * nSpinBasis
                NL = np.sqrt((2 * L1 + 1) * (2 * L2 + 1))
                # rank-1 fix (DEVIATIONS.md): 3j(L1 1 L2; -M1, M1-M2, M2) etc. as in rank 2
                # (EasySpin: -M2, and the K2 signs of Ka/Kb swapped)
                jjjM = jjj1[L1 ** 2 + L1 - M1, L2 ** 2 + L2 + M2]
                jjjKa = jjj1[L1 ** 2 + L1 - K1, L2 ** 2 + L2 + K2]
                K2zero = K2 == 0
                idxM = 2 - (M1 - M2)
                idxKa = 2 - (K1 - K2)
                idxKb = 2 - (-K1 + K2)
                prefactor = (1 / (2 * np.sqrt((1 + K1zero) * (1 + K2zero)))) \
                    * np.conj(_sqrtjk(jK1)) * _sqrtjk(jK2) * NL * jjjM
                # the two terms reached through the 3j column-sign symmetry carry
                # (-1)^(L1+j+L2): -1 for j = 1 (EasySpin used the rank-2 signs)
                spinblock = prefactor * jjjKa * ((-1.0) ** (K1 - M1) * Q1c(idxM, idxKa)
                                                 - jK1 * jK2 * (-1.0) ** (K2 - M1) * Q1c(idxM, idxKb))
                if K1 + K2 <= 1:
                    jjjKb = jjj1[L1 ** 2 + L1 - K1, L2 ** 2 + L2 - K2]
                    idxKc = 2 - (-K1 - K2)
                    idxKd = 2 - (K1 + K2)
                    spinblock = spinblock + prefactor * jjjKb * (-jK1 * (-1.0) ** (L2 - M1) * Q1c(idxM, idxKc)
                                                                 + jK2 * (-1.0) ** (L2 + K1 + K2 - M1) * Q1c(idxM, idxKd))
                r, c, v = _find(spinblock)
                bra.append(r + idx1)
                ket.append(c + idx2)
                el.append(v)
        # all pairs are computed above; EasySpin also added triu(H1,1).' (double counting)
        H1 = _assemble(bra, ket, el, nTotalBasis)

    # rank 2
    jjj2 = np.asarray(jjj2)
    Q2d = [_dense(q) for q in Q2]
    Q2c = _cellget(Q2d, 5)
    bra, ket, el = [], [], []
    for iBasis in range(nOriBasis):
        L1, M1, K1, jK1 = L[iBasis], M[iBasis], K[iBasis], jK[iBasis]
        idx1 = iBasis * nSpinBasis
        K1zero = K1 == 0
        for jBasis in range(iBasis, nOriBasis):
            L2 = L[jBasis]
            if abs(L1 - L2) > 2:
                break
            M2 = M[jBasis]
            if abs(M1 - M2) > 2:
                continue
            K2 = K[jBasis]
            if abs(K1 - K2) > 2:
                continue
            jK2 = jK[jBasis]
            idx2 = jBasis * nSpinBasis
            NL = np.sqrt((2 * L1 + 1) * (2 * L2 + 1))
            jjjM = jjj2[L1 ** 2 + L1 - M1, L2 ** 2 + L2 + M2]
            jjjKa = jjj2[L1 ** 2 + L1 - K1, L2 ** 2 + L2 + K2]
            K2zero = K2 == 0
            idxM = 3 - (M1 - M2)
            idxKa = 3 - (K1 - K2)
            idxKb = 3 - (-K1 + K2)
            prefactor = (1 / (2 * np.sqrt((1 + K1zero) * (1 + K2zero)))) \
                * np.conj(_sqrtjk(jK1)) * _sqrtjk(jK2) * NL * jjjM
            spinblock = prefactor * jjjKa * ((-1.0) ** (K1 - M1) * Q2c(idxM, idxKa)
                                             + jK1 * jK2 * (-1.0) ** (K2 - M1) * Q2c(idxM, idxKb))
            if K1 + K2 <= 2:
                jjjKb = jjj2[L1 ** 2 + L1 - K1, L2 ** 2 + L2 - K2]
                idxKc = 3 - (-K1 - K2)
                idxKd = 3 - (K1 + K2)
                spinblock = spinblock + prefactor * jjjKb * (jK1 * (-1.0) ** (L2 - M1) * Q2c(idxM, idxKc)
                                                             + jK2 * (-1.0) ** (L2 + K1 + K2 - M1) * Q2c(idxM, idxKd))
            r, c, v = _find(spinblock)
            bra.append(r + idx1)
            ket.append(c + idx2)
            el.append(v)
    H2 = _assemble(bra, ket, el, nTotalBasis)
    H2 = sp.triu(H2)
    H2 = H2 + sp.triu(H2, 1).T
    H = H0 + H2 if H1 is None else H0 + H1 + H2
    return sp.csc_matrix(H)

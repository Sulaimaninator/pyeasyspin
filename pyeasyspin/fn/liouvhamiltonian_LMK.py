"""function takes as input the orientational basis size, the isotropic and anisotropic spin
Liouvillian components, and the 3j symbol tables, and returns the Hamiltonian superoperator
in the LMK orientational basis (no K symmetrization) times the spin basis"""
import numpy as np
import scipy.sparse as sp
from .liouvhamiltonian import _dense, _find, _isempty, _cellget, _assemble


def liouvhamiltonian_LMK(basis=None, Q0=None, Q1=None, Q2=None, jjj0=None, jjj1=None, jjj2=None):
    """liouvhamiltonian_LMK(basis (LMK), Q0, Q1, Q2, jjj0, jjj1, jjj2) -> sparse H."""
    if 'jK' in basis and np.size(basis['jK']) > 0:
        raise ValueError('This function expects an LMK basis, without jK.')
    L = np.asarray(basis['L'], dtype=float).ravel(order='F').astype(int)
    M = np.asarray(basis['M'], dtype=float).ravel(order='F').astype(int)
    K = np.asarray(basis['K'], dtype=float).ravel(order='F').astype(int)
    Q0d = _dense(Q0)
    nSpinBasis = max(Q0d.shape)
    nOriBasis = L.size
    nTotalBasis = nOriBasis * nSpinBasis
    jjj0 = np.asarray(jjj0)

    row, col, Qval = _find(Q0d)
    bra, ket, el = [], [], []
    for b1 in range(nOriBasis):
        L_, M_, K_ = L[b1], M[b1], K[b1]
        idx0 = b1 * nSpinBasis
        jjjM = jjj0[L_ ** 2 + L_ + M_, L_ ** 2 + L_ - M_]
        jjjK = jjj0[L_ ** 2 + L_ + K_, L_ ** 2 + L_ - K_]
        el.append((-1.0) ** (K_ - M_) * (2 * L_ + 1) * jjjM * jjjK * Qval)
        bra.append(idx0 + row)
        ket.append(idx0 + col)
    H0 = _assemble(bra, ket, el, nTotalBasis)

    H1 = None
    if not _isempty(Q1):
        jjj1 = np.asarray(jjj1)
        Q1c = _cellget([_dense(q) for q in Q1], 3)
        bra, ket, el = [], [], []
        for b1 in range(nOriBasis):
            L1, M1, K1 = L[b1], M[b1], K[b1]
            idx1 = b1 * nSpinBasis
            # all pairs (rank-1 fix, DEVIATIONS.md): EasySpin loops b2 >= b1 and mirrors with .'
            for b2 in range(nOriBasis):
                L2 = L[b2]
                if abs(L1 - L2) > 1:
                    continue
                M2 = M[b2]
                if abs(M1 - M2) > 1:
                    continue
                K2 = K[b2]
                if abs(K1 - K2) > 1:
                    continue
                idx2 = b2 * nSpinBasis
                NL = np.sqrt((2 * L1 + 1) * (2 * L2 + 1))
                # (L1 1 L2; -M1, M1-M2, M2): +M2/+K2 as in rank 2 (EasySpin: -M2/-K2)
                jjjM = jjj1[L1 ** 2 + L1 - M1, L2 ** 2 + L2 + M2]
                jjjK = jjj1[L1 ** 2 + L1 - K1, L2 ** 2 + L2 + K2]
                idxM = 2 - (M1 - M2)
                idxK = 2 - (K1 - K2)
                spinblock = (-1.0) ** (K1 - M1) * NL * jjjM * jjjK * Q1c(idxM, idxK)
                r, c, v = _find(spinblock)
                bra.append(r + idx1)
                ket.append(c + idx2)
                el.append(v)
        H1 = _assemble(bra, ket, el, nTotalBasis)

    jjj2 = np.asarray(jjj2)
    Q2c = _cellget([_dense(q) for q in Q2], 5)
    bra, ket, el = [], [], []
    for b1 in range(nOriBasis):
        L1, M1, K1 = L[b1], M[b1], K[b1]
        idx1 = b1 * nSpinBasis
        for b2 in range(b1, nOriBasis):
            L2 = L[b2]
            if abs(L1 - L2) > 2:
                continue
            M2 = M[b2]
            if abs(M1 - M2) > 2:
                continue
            K2 = K[b2]
            if abs(K1 - K2) > 2:
                continue
            idx2 = b2 * nSpinBasis
            NL = np.sqrt((2 * L1 + 1) * (2 * L2 + 1))
            jjjM = jjj2[L1 ** 2 + L1 - M1, L2 ** 2 + L2 + M2]
            jjjK = jjj2[L1 ** 2 + L1 - K1, L2 ** 2 + L2 + K2]
            idxM = 3 - (M1 - M2)
            idxK = 3 - (K1 - K2)
            spinblock = (-1.0) ** (K1 - M1) * NL * jjjM * jjjK * Q2c(idxM, idxK)
            r, c, v = _find(spinblock)
            bra.append(r + idx1)
            ket.append(c + idx2)
            el.append(v)
    H2 = _assemble(bra, ket, el, nTotalBasis)
    H2 = sp.triu(H2)
    H2 = H2 + sp.triu(H2, 1).conj().T
    H = H0 + H2 if H1 is None else H0 + H1 + H2
    return sp.csc_matrix(H)

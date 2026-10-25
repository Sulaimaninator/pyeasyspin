"""transformation matrix from the LMK basis to the K-symmetrized LMKjK basis"""
import cmath
import numpy as np
import scipy.sparse as sp


def ksymmetrizer(basis=None):
    """ksymmetrizer(basis (LMK, no jK)) -> sparse T (nLMK x nLMKjK) with columns the K-symmetrized
    functions; basis.jKmin == 1 drops the jK = -1 functions."""
    DebugMode = True
    if 'jK' in basis:
        raise ValueError('This function expects an LMK basis, without jK.')
    negjKtrim = 'jKmin' in basis and np.all(np.asarray(basis['jKmin']) == 1)
    L = np.asarray(basis['L'], dtype=float).ravel(order='F')
    M = np.asarray(basis['M'], dtype=float).ravel(order='F')
    K = np.asarray(basis['K'], dtype=float).ravel(order='F')
    nBasis1 = L.size
    L_, M_, K_ = L.copy(), M.copy(), np.abs(K)
    jK_ = np.sign(K)
    idxKzero = K_ == 0
    jK_[idxKzero] = (-1.0) ** L[idxKzero]
    if negjKtrim:
        keep = ~(jK_ < 0)
        L_, M_, K_, jK_ = L_[keep], M_[keep], K_[keep], jK_[keep]
    nBasis2 = L_.size
    bra, ket, vals = [], [], []
    for b1 in range(nBasis1):
        L1, M1, K1 = L[b1], M[b1], K[b1]
        for b2 in range(nBasis2):
            L2 = L_[b2]
            if L1 != L2:
                continue
            M2 = M_[b2]
            if M1 != M2:
                continue
            K2 = abs(K_[b2])
            jK = jK_[b2]
            sj = 1.0 if jK >= 0 else cmath.sqrt(jK)
            v = sj / np.sqrt(2 * (1 + (K2 == 0))) * ((K1 == K2) + jK * (-1.0) ** (L2 + K2) * (K1 == -K2))
            if v == 0:
                continue
            bra.append(b1)
            ket.append(b2)
            vals.append(v)
    T = sp.csc_matrix((np.array(vals), (bra, ket)), shape=(nBasis1, nBasis2))
    if DebugMode:
        err = (T.conj().T @ T - sp.identity(nBasis2)).toarray()
        if np.max(np.abs(err)) > 1e-10:
            raise ValueError('Transformation matrix is not unitary.')
    return T

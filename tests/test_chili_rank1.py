"""Corrected rank-1 (antisymmetric tensor) part of the chili Hamiltonian superoperator.

EasySpin 6.0.12's rank-1 code in liouvhamiltonian_LMK / liouvhamiltonian is wrong
(docs/DEVIATIONS.md). The reference implementations
below apply the rank-2 formula to rank 1:

    <L1 M1 K1| D^j_mk |L2 M2 K2> = (-1)^(K1-M1) sqrt((2L1+1)(2L2+1))
                                   (L1 j L2; -M1 m M2) (L1 j L2; -K1 k K2),

with the jjj1 lookups jjj1[L1^2+L1-M1, L2^2+L2+M2] (EasySpin: -M2) and likewise for K, and in
the K-symmetrized basis |L M K jK> ~ |L M K> + jK (-1)^(L+K) |L M -K> the two terms that pick up
the 3j column-sign parity (-1)^(L1+j+L2) change sign for odd j. Every pair of basis functions is
computed (no triangle mirroring), so no symmetry of the block is assumed.

This fix is applied in pyeasyspin/fn/liouvhamiltonian[_LMK].py;
the reference implementations stay here as independent checks. Checked against a brute-force
SO(3) quadrature (from test_chili_basis.py) and the LMK -> K-symmetrized transformation; the rank-2
path is unchanged and is checked the same way. magint's rank-1 symmetry flags: last test.
"""
import numpy as np
import scipy.sparse as sp

from pyeasyspin._registry import F
from pyeasyspin.fn.liouvhamiltonian import _dense, _find, _sqrtjk, _cellget, _assemble
from test_chili_basis import _lmk_basis, _ks_basis, _bruteforce_H, _physical_q, _GASYM, _rel


def rank1_lmk_fixed(basis, Q1, jjj1):
    L, M, K = (np.asarray(basis[x], dtype=float).ravel(order='F').astype(int) for x in 'LMK')
    Q1d = [_dense(q) for q in Q1]
    nSpinBasis = Q1d[0].shape[0]
    N = L.size * nSpinBasis
    Q1c = _cellget(Q1d, 3)
    bra, ket, el = [], [], []
    for b1 in range(L.size):
        L1, M1, K1 = L[b1], M[b1], K[b1]
        for b2 in range(L.size):
            L2, M2, K2 = L[b2], M[b2], K[b2]
            if abs(L1 - L2) > 1 or abs(M1 - M2) > 1 or abs(K1 - K2) > 1:
                continue
            NL = np.sqrt((2 * L1 + 1) * (2 * L2 + 1))
            jjjM = jjj1[L1 ** 2 + L1 - M1, L2 ** 2 + L2 + M2]
            jjjK = jjj1[L1 ** 2 + L1 - K1, L2 ** 2 + L2 + K2]
            blk = (-1.0) ** (K1 - M1) * NL * jjjM * jjjK * Q1c(2 - (M1 - M2), 2 - (K1 - K2))
            r, c, v = _find(blk)
            bra.append(r + b1 * nSpinBasis)
            ket.append(c + b2 * nSpinBasis)
            el.append(v)
    return _assemble(bra, ket, el, N)


def rank1_ksym_fixed(basis, Q1, jjj1):
    L, M, K, jK = (np.asarray(basis[x], dtype=float).ravel(order='F').astype(int) for x in ('L', 'M', 'K', 'jK'))
    Q1d = [_dense(q) for q in Q1]
    nSpinBasis = Q1d[0].shape[0]
    N = L.size * nSpinBasis
    Q1c = _cellget(Q1d, 3)
    bra, ket, el = [], [], []
    for b1 in range(L.size):
        L1, M1, K1, jK1 = L[b1], M[b1], K[b1], jK[b1]
        for b2 in range(L.size):
            L2, M2, K2, jK2 = L[b2], M[b2], K[b2], jK[b2]
            if abs(L1 - L2) > 1 or abs(M1 - M2) > 1 or abs(K1 - K2) > 1:
                continue
            NL = np.sqrt((2 * L1 + 1) * (2 * L2 + 1))
            jjjM = jjj1[L1 ** 2 + L1 - M1, L2 ** 2 + L2 + M2]
            jjjKa = jjj1[L1 ** 2 + L1 - K1, L2 ** 2 + L2 + K2]
            idxM = 2 - (M1 - M2)
            pre = (1 / (2 * np.sqrt((1 + (K1 == 0)) * (1 + (K2 == 0))))) * np.conj(_sqrtjk(jK1)) * _sqrtjk(jK2) * NL * jjjM
            blk = pre * jjjKa * ((-1.0) ** (K1 - M1) * Q1c(idxM, 2 - (K1 - K2))
                                 - jK1 * jK2 * (-1.0) ** (K2 - M1) * Q1c(idxM, 2 - (-K1 + K2)))
            if K1 + K2 <= 1:
                jjjKb = jjj1[L1 ** 2 + L1 - K1, L2 ** 2 + L2 - K2]
                blk = blk + pre * jjjKb * (-jK1 * (-1.0) ** (L2 - M1) * Q1c(idxM, 2 - (-K1 - K2))
                                           + jK2 * (-1.0) ** (L2 + K1 + K2 - M1) * Q1c(idxM, 2 - (K1 + K2)))
            r, c, v = _find(blk)
            bra.append(r + b1 * nSpinBasis)
            ket.append(c + b2 * nSpinBasis)
            el.append(v)
    return _assemble(bra, ket, el, N)


def _setup(Lmax=2):
    B = _lmk_basis(Lmax, Lmax, Lmax)
    j0, j1, j2 = F.jjjsymbol(Lmax, Lmax - 1 if Lmax % 2 == 0 else Lmax, True)
    Q1, Q2 = _physical_q(_GASYM)
    return B, j0, j1, j2, Q1, Q2


def test_rank1_lmk_fixed_matches_bruteforce():
    B, j0, j1, j2, Q1, _ = _setup(2)
    H1 = rank1_lmk_fixed(B, Q1, j1)
    R = _bruteforce_H(B, 1, Q1, 4)
    assert _rel(H1, R) < 1e-12
    assert _rel(H1, H1.conj().T) < 1e-13      # Hermitian, as a Hamiltonian superoperator must be


def test_rank1_ksym_fixed_matches_transformed_lmk():
    for Lmax in (2, 3):
        B, j0, j1, j2, Q1, Q2 = _setup(Lmax)
        T = F.ksymmetrizer(B)
        TT = sp.kron(T, sp.identity(4))
        BK = _ks_basis(B)
        Hk = rank1_ksym_fixed(BK, Q1, j1)
        assert _rel(TT.conj().T @ rank1_lmk_fixed(B, Q1, j1) @ TT, Hk) < 1e-12
        # rank 2 for comparison: EasySpin's (unchanged) code passes the same check
        Z = sp.csc_matrix((4, 4))
        H2l = F.liouvhamiltonian_LMK(B, Z, [], Q2, j0, j1, j2)
        H2k = F.liouvhamiltonian(BK, Z, [], Q2, j0, j1, j2)
        assert _rel(TT.conj().T @ H2l @ TT, H2k) < 1e-12


def test_liouvhamiltonian_rank1_direct():
    """The patched F.liouvhamiltonian_LMK / F.liouvhamiltonian rank-1 parts (Q0 = Q2 = 0) equal the
    brute-force SO(3) integral, the reference implementations above, and each other through the
    LMK -> K-symmetrized transformation; the LMK block is Hermitian."""
    Z = sp.csc_matrix((4, 4))
    Z2 = [Z] * 25
    B, j0, j1, j2, Q1, _ = _setup(2)
    H = F.liouvhamiltonian_LMK(B, Z, Q1, Z2, j0, j1, j2)
    assert _rel(H, _bruteforce_H(B, 1, Q1, 4)) < 1e-12
    assert _rel(H, H.conj().T) < 1e-13
    for Lmax in (2, 3):
        B, j0, j1, j2, Q1, _ = _setup(Lmax)
        BK = _ks_basis(B)
        TT = sp.kron(F.ksymmetrizer(B), sp.identity(4))
        Hl = F.liouvhamiltonian_LMK(B, Z, Q1, Z2, j0, j1, j2)
        Hk = F.liouvhamiltonian(BK, Z, Q1, Z2, j0, j1, j2)
        assert _rel(Hl, rank1_lmk_fixed(B, Q1, j1)) < 1e-14
        assert _rel(Hk, rank1_ksym_fixed(BK, Q1, j1)) < 1e-14
        assert _rel(TT.conj().T @ Hl @ TT, Hk) < 1e-12


def test_liouvhamiltonian_all_ranks_vs_bruteforce():
    """Full Hamiltonian (ranks 1 and 2 together, physical antisymmetric g) in the LMK basis vs the
    brute-force integral of both ranks."""
    B, j0, j1, j2, Q1, Q2 = _setup(2)
    Z = sp.csc_matrix((4, 4))
    H = F.liouvhamiltonian_LMK(B, Z, Q1, Q2, j0, j1, j2)
    R = _bruteforce_H(B, 1, Q1, 4) + _bruteforce_H(B, 2, Q2, 4)
    assert _rel(H, R) < 1e-12


def _antisym_setup():
    from pyeasyspin._struct import Struct
    g = np.diag([2.01, 2.006, 2.002]) + np.array([[0, 0.02, -0.01], [-0.02, 0, 0.015], [0.01, -0.015, 0]])
    Exp = Struct(mwFreq=9.5, Range=np.array([[334.0, 341.0]]), nPoints=80.0, Harmonic=0)
    return Struct, g, Exp


def test_chili_antisymmetric_g_rotation_invariance():
    """chili level: rotating a g tensor that has an antisymmetric part (isotropic diffusion, no
    potential, full M/K basis) leaves the spectrum unchanged (4.7e-12 measured; EasySpin's
    rank-1 code changed it by 0.47 of peak)."""
    Struct, g, Exp = _antisym_setup()
    R = F.erot([0.3, 0.8, 1.2])
    Opt = Struct(LiouvMethod='general', LLMK=np.array([[6.0, 5.0, 6.0, 6.0]]), jKmin=-1.0, evenK=False)

    def spec(gm):
        return np.ravel(F.chili(Struct(g=gm, tcorr=1e-9), Exp, Opt)[1])

    assert _rel(spec(R @ g @ R.T), spec(g)) < 1e-8


def test_magint_rank1_flags_default_basis_converged():
    """magint's Symmetry flags now account for rank-1 terms. For g = diag + antisymmetric part,
    EasySpin's flags (F2 only) give nobetatilts = tensorsCollinear = true, so chili's default basis
    drops odd K and jK = -1 (= Opt.evenK = true, Opt.jKmin = +1 here) and the spectrum is 12.5 %
    of peak off. With the fix the default basis (LLMK [14 7 2 6]) matches a large basis
    ([20 15 4 10]: 2.8e-11; [16 11 4 8] here)."""
    Struct, g, Exp = _antisym_setup()
    Sys = Struct(g=g, tcorr=3e-9)
    Sym = F.magint(*_magint_args(Sys))[3]
    assert not Sym['nobetatilts'] and not Sym['tensorsCollinear']
    y_def = F.chili(Sys, Exp, Struct(LiouvMethod='general'))[1]
    y_old = F.chili(Sys, Exp, Struct(LiouvMethod='general', evenK=True, jKmin=1.0))[1]
    y_big = F.chili(Sys, Exp, Struct(LiouvMethod='general', LLMK=np.array([[16.0, 11.0, 4.0, 8.0]])))[1]
    assert _rel(y_def, y_big) < 1e-8
    assert _rel(y_old, y_big) > 0.05
    # a symmetric g keeps EasySpin's flags (and its smaller default basis)
    Sym0 = F.magint(*_magint_args(Struct(g=np.diag([2.01, 2.006, 2.002]), tcorr=3e-9)))[3]
    assert Sym0['nobetatilts'] and Sym0['tensorsCollinear']


def _magint_args(Sys):
    S, err = F.validatespinsys(Sys)
    S.DiffFrame = np.zeros((1, 3))
    ops = [F.sop(S.Spins, np.array([[1, k + 1.0]]), 'sparse') for k in range(3)]
    return S, ops, 340.0, True, False, False

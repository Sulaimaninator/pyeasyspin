"""Checks of chili's basis and Liouvillian functions on paths that EasySpin's own tests do not reach.

- ksymmetrizer, diffsuperop (LMK basis), diffsuperop_LMK (methods 1 and 2) and
  liouvhamiltonian_LMK are checked against the K-symmetrized functions (validated against
  EasySpin) through the basis change T from ksymmetrizer.
- chili_lm: Meirovitch symmetry (Opt.MpSymm) must give exactly the submatrix of the full
  Liouvillian on the kept basis functions; complex tensor components (jK1 ~= jK2 imaginary
  branches) are checked by the invariance of the Liouvillian spectrum under a rotation of all
  tensors about the molecular z axis (axial diffusion, no potential); chili_basisbuild must
  enumerate the same basis as chili_lm.
- _integral/_integral2 (MATLAB integral/integral2 reimplementations) against closed forms.
"""
import numpy as np
import scipy.sparse as sp

from pyeasyspin._registry import F  # noqa: E402
from pyeasyspin._struct import Struct  # noqa: E402
from pyeasyspin.fn.chili_eqpopvec import _integral, _integral2  # noqa: E402


# ------------------------------------------------------------------ LMK vs K-symmetrized
def _lmk_basis(Lmax, Mmax, Kmax):
    L, M, K = [], [], []
    for l in range(Lmax + 1):
        for m in range(-min(l, Mmax), min(l, Mmax) + 1):
            for k in range(-min(l, Kmax), min(l, Kmax) + 1):
                L.append(l); M.append(m); K.append(k)
    c = lambda v: np.array(v, dtype=float).reshape(-1, 1)
    return Struct(L=c(L), M=c(M), K=c(K))


def _ks_basis(B):
    """The K-symmetrized basis that ksymmetrizer's columns correspond to (same order)."""
    L = B['L'].ravel(); M = B['M'].ravel(); K = B['K'].ravel()
    jK = np.sign(K)
    jK[K == 0] = (-1.0) ** L[K == 0]
    c = lambda v: np.asarray(v, dtype=float).reshape(-1, 1)
    return Struct(L=c(L), M=c(M), K=c(np.abs(K)), jK=c(jK))


def _rel(A, B):
    A = A.toarray() if sp.issparse(A) else np.asarray(A)
    B = B.toarray() if sp.issparse(B) else np.asarray(B)
    return np.max(np.abs(A - B)) / np.max(np.abs(B))


def test_ksymmetrizer_unitary():
    B = _lmk_basis(4, 3, 4)
    T = F.ksymmetrizer(B)
    n = T.shape[1]
    assert T.shape == (B['L'].size, B['L'].size)
    assert np.max(np.abs((T.conj().T @ T - sp.identity(n)).toarray())) < 1e-14


def _pot(L, M, K, lam):
    P = Struct(L=np.array(L, float), M=np.array(M, float), K=np.array(K, float))
    P['lambda'] = np.array(lam, float)
    return P


def test_diffsuperop_lmk_vs_ksym():
    B = _lmk_basis(4, 2, 4)
    T = F.ksymmetrizer(B)
    BK = _ks_basis(B)
    R = np.array([1.0e7, 2.5e7, 4.0e7])
    for pot in (_pot([2, 2, 4], [0, 0, 0], [0, 2, 0], [1.0, 0.5, 0.3]),
                _pot([2, 2], [0, 1], [0, 1], [0.8, 0.4])):
        X = F.chili_xlmk(pot, R)
        G_lmk = F.diffsuperop(B, R, X, pot)
        G_ks = F.diffsuperop(BK, R, X, pot)
        assert _rel(T.conj().T @ G_lmk @ T, G_ks) < 1e-12
        G1 = F.diffsuperop_LMK(B, R, X, pot, 1)
        G2 = F.diffsuperop_LMK(B, R, X, pot, 2)
        if np.all(pot['M'] == 0):     # method 1 skips M1 ~= M2 (EasySpin: M=0 potentials only)
            assert _rel(G1, G_lmk) < 1e-12
        assert _rel(G2, G_lmk) < 1e-12
    # no potential: axial (diagonal shortcut) and rhombic
    nopot = _pot([], [], [], [])
    for Rv in (np.array([2e7, 2e7, 5e7]), R):
        G_lmk = F.diffsuperop(B, Rv, [], nopot)
        assert _rel(T.conj().T @ G_lmk @ T, F.diffsuperop(BK, Rv, [], nopot)) < 1e-12
        assert _rel(F.diffsuperop_LMK(B, Rv, None, nopot, 1), G_lmk) < 1e-12


def _q_matrices():
    """Q0, Q2 (5x5 cell, column-major list) of an S=1/2 rhombic-g Zeeman Liouvillian (as rbos)."""
    Sx = sp.csc_matrix([[0, .5], [.5, 0]], dtype=complex)
    Sy = sp.csc_matrix([[0, -.5j], [.5j, 0]])
    Sz = sp.csc_matrix([[.5, 0], [0, -.5]], dtype=complex)
    T0, _, T2 = F.istotensor([0, 0, 0.35], [Sx, Sy, Sz])
    T0L = F.hil2liouv(sp.csc_matrix(T0))
    T2L = F.hil2liouv([sp.csc_matrix(t) for t in T2])
    g = np.diag([2.01, 2.006, 2.002])
    R = F.erot([0.3, 0.7, 1.1])
    F0, _, F2 = F.tensor_cart2sph(R @ g @ R.T * F.bmagn()[0] / F.planck())
    F2 = np.ravel(F2)
    Q0 = np.conj(F0) * T0L
    Q2 = [np.conj(F2[mq]) * T2L[mp] for mq in range(5) for mp in range(5)]   # Q2{mp,mq}, column-major
    return sp.csc_matrix(Q0), [sp.csc_matrix(q) for q in Q2]


def test_liouvhamiltonian_lmk_vs_ksym():
    Lmax = 4
    B = _lmk_basis(Lmax, 2, 4)
    T = F.ksymmetrizer(B)
    BK = _ks_basis(B)
    jjj0, jjj1, jjj2 = F.jjjsymbol(Lmax, Lmax, False)
    Q0, Q2 = _q_matrices()
    H_lmk = F.liouvhamiltonian_LMK(B, Q0, [], Q2, jjj0, jjj1, jjj2)
    H_ks = F.liouvhamiltonian(BK, Q0, [], Q2, jjj0, jjj1, jjj2)
    TT = sp.kron(T, sp.identity(4))
    assert _rel(TT.conj().T @ H_lmk @ TT, H_ks) < 1e-12


# ------------------------------------------------------------------ chili_lm
def _lm_inputs(nNuc, Lemax, Lomax, Kmax, Mmax, jKmin=-1, pSmin=-1, deltaK=1, Meir=0, gamma=0.0,
               Exchange=0.0, R=(2e7, 2e7, 6e7)):
    k = np.arange(-2, 3)
    ph = np.exp(1j * k * gamma)
    g = np.array([2.009, 2.006, 2.0022])
    F0, _, F2 = F.tensor_cart2sph(np.diag(g) * F.bmagn()[0] / F.planck())
    B0 = 0.34
    S = Struct(nNuclei=float(nNuc), EZ0=F0 * B0 * 2 * np.pi, d2psi=np.eye(5), DirTilt=False,
               EZ2=(np.ravel(F2) * B0 * 2 * np.pi * ph).reshape(5, 1))
    bo = [Lemax, Lomax, Kmax, Mmax, jKmin, pSmin, deltaK, Meir]
    if nNuc >= 1:
        A = F.erot([0.2, 0.9, 0.4]) @ np.diag([20e6, 25e6, 95e6]) @ F.erot([0.2, 0.9, 0.4]).T
        A0, _, A2 = F.tensor_cart2sph(A)
        S.update(I=1.0, NZ0=-2 * np.pi * 1.5e6, HF0=np.real(A0) * 2 * np.pi,
                 HF2=(np.ravel(A2) * 2 * np.pi * ph).reshape(5, 1))
        bo.append(2)
    if nNuc >= 2:
        A = np.diag([5e6, 6e6, 14e6])
        A0, _, A2 = F.tensor_cart2sph(A)
        S.update(Ib=0.5, NZ0b=-2 * np.pi * 14e6, HF0b=float(np.real(A0)) * 2 * np.pi,
                 HF2b=(np.ravel(A2).astype(complex) * 2 * np.pi * ph).reshape(5, 1))
        S['HF2'] = np.hstack([S['HF2'], S['HF2b']])
        bo.append(1)
    D = Struct(Exchange=Exchange, xlk=np.zeros((0, 0)), maxL=-1.0, Diff=np.array([R]))
    return S, np.array([bo], dtype=float), D


def _lm_matrix(S, bo, D):
    r, c, v, n = F.chili_lm(S, bo, D, 1e5)
    n = int(n)
    return sp.csc_matrix((v.ravel(), (r.ravel() - 1, c.ravel() - 1)), shape=(n, n))


def _basis_for(S, bo, Meir):
    b = bo.ravel()
    I = [] if S['nNuclei'] == 0 else ([S['I']] if S['nNuclei'] == 1 else [S['I'], S['Ib']])
    pImax = np.array(b[8:8 + len(I)])
    Bas = Struct(DirTilt=False, evenLmax=b[0], oddLmax=b[1], jKmin=b[4], Kmax=b[2], evenK=b[6] == 2,
                 Mmax=b[3], pSmin=b[5], pImax=pImax, MpSymm=bool(Meir))
    return F.chili_basisbuild(Bas, Struct(I=np.array(I)))


def test_chili_lm_meirovitch_submatrix_and_basis():
    for nNuc, size in ((0, (6, 3, 4, 3)), (1, (4, 3, 2, 2)), (2, (2, 1, 2, 1))):
        full = _lm_inputs(nNuc, *size)
        Lf = _lm_matrix(*full)
        Bf = _basis_for(full[0], full[1], 0)
        assert Bf['L'].size == Lf.shape[0]
        S, bo, D = full
        bo2 = bo.copy()
        bo2[0, 7] = 1
        Ls = _lm_matrix(S, bo2, D)
        Bs = _basis_for(S, bo2, 1)
        assert Bs['L'].size == Ls.shape[0]
        pI = Bf['pI1'].ravel() if nNuc >= 1 else 0
        pIb = Bf['pI2'].ravel() if nNuc >= 2 else 0
        keep = np.flatnonzero((pI + pIb + Bf['pS'].ravel() - 1) == Bf['M'].ravel())
        assert keep.size == Ls.shape[0]
        assert _rel(Lf[keep][:, keep], Ls) == 0.0


def _spec(Lm):
    return np.linalg.eigvals(Lm.toarray())


def test_chili_lm_complex_tensors_rotation_invariance():
    for nNuc, size, ex in ((0, (6, 3, 4, 2), 0.0), (1, (2, 1, 2, 1), 3e6), (2, (2, 1, 2, 0), 3e6)):
        L0 = _lm_matrix(*_lm_inputs(nNuc, *size, Exchange=ex))
        S, bo, D = _lm_inputs(nNuc, *size, gamma=0.7, Exchange=ex)
        assert np.iscomplexobj(S['EZ2']) and np.any(np.imag(S['EZ2']))
        L1 = _lm_matrix(S, bo, D)
        assert _rel(L1, L0) > 1e-6            # the matrices do differ ...
        e0, e1 = _spec(L0), _spec(L1)         # ... but the spectra agree
        d = np.abs(e0[:, None] - e1[None, :]).min(axis=1)
        assert d.max() / np.abs(e0).max() < 1e-9


# ------------------------------------------------------------------ quadratures
def test_integral_closed_forms():
    assert abs(_integral(np.exp, 0, 1) - (np.e - 1)) < 1e-12
    assert abs(_integral(lambda x: 1 / np.sqrt(x), 0, 1) - 2) < 1e-6      # endpoint singularity
    assert abs(_integral(np.sin, np.pi, 0) + 2) < 1e-12                  # reversed limits
    v = _integral2(lambda x, y: np.exp(-x) * np.cos(y) ** 2, 0, 2, 0, np.pi)
    assert abs(v - (1 - np.exp(-2)) * np.pi / 2) < 1e-6           # RelTol 1e-6
    v = _integral2(lambda x, y: x * y, 0, 1, 0, lambda x: x)               # triangle: 1/8
    assert abs(v - 1 / 8) < 1e-12


# ------------------------------------------------------------------ brute-force SO(3) reference
def _bruteforce_H(B, rank, Q, nspin, Na=12):
    """<L1M1K1| sum_mk D^rank_mk(Omega) Q{rank+1-m, rank+1-k} |L2M2K2>, basis functions
    sqrt((2L+1)/8pi^2) D^L_MK, integrated on an exact SO(3) product grid."""
    import pytest  # noqa: F401
    L, M, K = (B[x].ravel().astype(int) for x in 'LMK')
    a = 2 * np.pi * np.arange(Na) / Na
    xb, wb = np.polynomial.legendre.leggauss(Na)
    A_, Bt, G_ = np.meshgrid(a, np.arccos(xb), a, indexing='ij')
    W = wb[None, :, None] * (2 * np.pi / Na) ** 2 * np.ones((Na, 1, Na))
    Db = [F.wignerd([L[i], M[i], K[i]], A_, Bt, G_) for i in range(L.size)]
    ms = range(-rank, rank + 1)
    Do = {(m, k): F.wignerd([rank, m, k], A_, Bt, G_) for m in ms for k in ms}
    nc = 2 * rank + 1
    Qd = [q.toarray() for q in Q]
    H = np.zeros((L.size * nspin,) * 2, complex)
    for i in range(L.size):
        for j in range(L.size):
            blk = 0
            for (m, k), d in Do.items():
                v = np.sum(W * np.conj(Db[i]) * d * Db[j]) * np.sqrt((2 * L[i] + 1) * (2 * L[j] + 1)) / (8 * np.pi ** 2)
                if abs(v) > 1e-13:
                    blk = blk + v * Qd[(rank - m) + (rank - k) * nc]
            H[i * nspin:(i + 1) * nspin, j * nspin:(j + 1) * nspin] = blk
    return H


def _physical_q(g):
    Sx = sp.csc_matrix([[0, .5], [.5, 0]], dtype=complex)
    Sy = sp.csc_matrix([[0, -.5j], [.5j, 0]])
    Sz = sp.csc_matrix([[.5, 0], [0, -.5]], dtype=complex)
    _, T1, T2 = F.istotensor([0, 0, 0.35], [Sx, Sy, Sz])
    T1L = F.hil2liouv([sp.csc_matrix(x) for x in T1])
    T2L = F.hil2liouv([sp.csc_matrix(x) for x in T2])
    _, f1, f2 = F.tensor_cart2sph(g * F.bmagn()[0] / F.planck())
    f1, f2 = np.ravel(f1), np.ravel(f2)
    Q1 = [sp.csc_matrix(-np.conj(f1[mq]) * T1L[mp]) for mq in range(3) for mp in range(3)]   # as rbos
    Q2 = [sp.csc_matrix(np.conj(f2[mq]) * T2L[mp]) for mq in range(5) for mp in range(5)]
    return Q1, Q2


_GASYM = np.array([[2.01, 0.003, 0.001], [-0.002, 2.006, 0.0], [0.0005, 0.001, 2.002]])


def test_liouvhamiltonian_rank2_vs_bruteforce():
    B = _lmk_basis(2, 2, 2)
    j0, j1, j2 = F.jjjsymbol(2, 2, True)
    _, Q2 = _physical_q(_GASYM)
    H = F.liouvhamiltonian_LMK(B, sp.csc_matrix((4, 4)), [], Q2, j0, j1, j2)
    R = _bruteforce_H(B, 2, Q2, 4)
    assert _rel(H, R) < 1e-13


def test_liouvhamiltonian_rank1_vs_bruteforce():
    """Rank-1 (antisymmetric tensor) part. EasySpin 6.0.12 gets it wrong (jjj1 lookups with
    -M2/-K2, triangle mirrored as if complex symmetric); the port fixes it
    (docs/DEVIATIONS.md). Before the fix this was > 0.5."""
    B = _lmk_basis(2, 2, 2)
    j0, j1, j2 = F.jjjsymbol(2, 2, True)
    Q1, _ = _physical_q(_GASYM)
    Z = [sp.csc_matrix((4, 4))] * 25
    H = F.liouvhamiltonian_LMK(B, sp.csc_matrix((4, 4)), Q1, Z, j0, j1, j2)
    R = _bruteforce_H(B, 1, Q1, 4)
    assert _rel(H, R) < 1e-12

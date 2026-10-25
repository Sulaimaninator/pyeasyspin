"""the irreducible spherical tensor components and, if using the general Liouvillian method, the
spin operator tensors of all magnetic interactions (for chili and cardamom)"""
import numpy as np
from .._registry import F, nout
from .._struct import Struct


def _mm(A, B):
    """3x3 matrix product summed in index order without fused multiply-add, which reproduces
    MATLAB's rounding here (numpy's BLAS matmul can differ in the last bit; an antisymmetric
    roundoff part of a rotated tensor then shows up in F1)."""
    A = np.asarray(A)
    B = np.asarray(B)
    C = A[:, 0:1] * B[0:1, :]
    for k in range(1, A.shape[1]):
        C = C + A[:, k:k + 1] * B[k:k + 1, :]
    return C


def _rows(A, idx):
    A = np.atleast_2d(np.asarray(A, dtype=float))
    return A[idx, :]


def _setidx(S, name, idx, value, dtype=float):
    """MATLAB S.name(idx...) = value (idx 0-based tuple; ':' given as slice(None)) with growth."""
    v = np.asarray(value)
    if name in S and np.size(S[name]) > 0:
        A = np.asarray(S[name])
        A = A.astype(np.result_type(A.dtype, v.dtype, dtype))
    else:
        A = np.zeros((0, 0), dtype=np.result_type(v.dtype, dtype))
    nd = len(idx)
    if nd == 1:                                   # linear index: row unless an existing column
        k = idx[0]
        if A.ndim < 2:
            A = A.reshape(1, -1)
        if A.size == 0:
            A = np.zeros((1, 0), A.dtype)
        if A.shape[0] == 1 or A.size == 0:
            if A.shape[1] <= k:
                A = np.hstack([A, np.zeros((1, k + 1 - A.shape[1]), A.dtype)])
            A[0, k] = v
        elif A.shape[1] == 1:
            if A.shape[0] <= k:
                A = np.vstack([A, np.zeros((k + 1 - A.shape[0], 1), A.dtype)])
            A[k, 0] = v
        else:
            A.flat[np.ravel_multi_index(np.unravel_index(k, A.shape, order='F'), A.shape)] = v
        S[name] = A
        return
    # multi-index: grow every dimension as needed
    need = []
    vv = v.ravel(order='F')
    for d, ix in enumerate(idx):
        if isinstance(ix, slice):
            need.append(max(A.shape[d] if d < A.ndim else 1, vv.size) if A.size == 0 else A.shape[d])
        else:
            need.append(ix + 1)
    while A.ndim < nd:
        A = A.reshape(A.shape + (1,))
    if A.size == 0:
        B = np.zeros(tuple(need), A.dtype)
    else:
        shape = tuple(max(a, n) for a, n in zip(A.shape, need))
        B = np.zeros(shape, A.dtype)
        B[tuple(slice(0, s) for s in A.shape)] = A
    sl = tuple(ix if isinstance(ix, slice) else ix for ix in idx)
    B[sl] = vv.reshape(B[sl].shape, order='F') if vv.size > 1 else vv[0]
    while B.ndim > 2 and B.shape[-1] == 1:        # MATLAB drops trailing singleton dimensions
        B = B[..., 0]
    S[name] = B


def magint(System=None, SpinOps=None, CenterField=None, includeNuclearZeeman=None,
           includeNuclearQuadrupole=None, explicitFieldSweep=None):
    """magint(System, SpinOps, CenterField, includeNuclearZeeman, includeNuclearQuadrupole,
    explicitFieldSweep) -> (T, F, System, Symmetry, isFieldDep). T.T0/T1/T2 and F.F0/F1/F2 are the
    spin-operator and spatial ISTO components per interaction (cells as column-major lists)."""
    System = Struct(System)
    generalLiouvillian = not (SpinOps is None or np.size(SpinOps) == 0 if not isinstance(SpinOps, list)
                              else len(SpinOps) == 0)
    R_M2Diff = F.erot(System['DiffFrame'])
    nElSpins = int(System['nElectrons'])
    System['I'] = F.nucspin(System['Nucs'])
    nNucSpins = int(System['nNuclei'])
    nSpins = len(SpinOps) // 3 if generalLiouvillian else 0

    def ops(i):
        return [SpinOps[i + k * nSpins] for k in range(3)]

    Sv = np.asarray(System['S'], dtype=float).ravel(order='F')
    nElZeeman = nElSpins
    nHyperfine = nElSpins * nNucSpins
    nZFS = int(np.sum(Sv > 1 / 2))
    nElPairs = nElSpins * (nElSpins - 1) // 2
    nInteractions = nElZeeman + nHyperfine + nZFS + nElPairs
    if includeNuclearZeeman:
        nInteractions += nNucSpins
    if includeNuclearQuadrupole:
        nInteractions += nNucSpins
    empty = np.zeros((0, 0))
    T0 = [empty] * nInteractions
    T1 = [[empty] * 3 for _ in range(nInteractions)]
    T2 = [[empty] * 5 for _ in range(nInteractions)]
    F0 = np.zeros((nInteractions, 1), dtype=complex)
    F1 = np.zeros((nInteractions, 3), dtype=complex)
    F2 = np.zeros((nInteractions, 5), dtype=complex)
    isFieldDep = np.zeros((nInteractions, 1))
    iInt = 0

    def put(i, t0, t1, t2):
        F0[i, 0] = complex(np.ravel(t0)[0])
        F1[i, :] = np.ravel(t1, order='F')
        F2[i, :] = np.ravel(t2, order='F')

    def setT(i, res):
        T0[i] = res[0]
        T1[i] = list(res[1])
        T2[i] = list(res[2])

    B0 = [0, 0, 1] if explicitFieldSweep else [0, 0, CenterField / 1e3]
    gm = np.atleast_2d(np.asarray(System['g'], dtype=float))
    for iElSpin in range(nElSpins):
        if not System['fullg']:
            g = np.diag(gm[iElSpin, :])
        else:
            g = gm[3 * iElSpin:3 * iElSpin + 3, :]
        if 'gFrame' in System:
            R_g2M = F.erot(_rows(System['gFrame'], iElSpin)).T
            g = _mm(_mm(R_g2M, g), R_g2M.T)
        if not generalLiouvillian:
            g = _mm(_mm(R_M2Diff, g), R_M2Diff.T)
        if generalLiouvillian:
            setT(iInt, F.istotensor(B0, ops(iElSpin)))
        put(iInt, *F.tensor_cart2sph(g * F.bmagn()[0] / F.planck()))
        isFieldDep[iInt] = True
        if not generalLiouvillian:
            if np.any(np.abs(F1[iInt, :] / F0[iInt, 0]) > 1e-9):
                raise ValueError('g tensor must be symmetric for this method.')
            _setidx(System, 'g_axial', (iElSpin,), F2[iInt, 0] == 0, bool)
            _setidx(System, 'EZ0', (iElSpin,), _creal(F0[iInt, 0] * B0[2] * 2 * np.pi))
            _setidx(System, 'EZ2', (slice(None), iElSpin), _creal(F2[iInt, :].T * B0[2] * 2 * np.pi))
        iInt += 1

    for iElSpin in range(nElSpins):
        eidx = slice(3 * iElSpin, 3 * iElSpin + 3)
        for iNucSpin in range(nNucSpins):
            Am = np.atleast_2d(np.asarray(System['A'], dtype=float))
            if System['fullA']:
                A_ = Am[3 * iNucSpin:3 * iNucSpin + 3, eidx] * 1e6
            else:
                A_ = np.diag(Am[iNucSpin, eidx]) * 1e6
            AFr = np.atleast_2d(np.asarray(System['AFrame'], dtype=float))[iNucSpin, eidx]
            if np.any(AFr):
                R_M2A = F.erot(AFr)
                R_A2M = R_M2A.T
                A_ = _mm(_mm(R_A2M, A_), R_A2M.T)
            if not generalLiouvillian:
                A_ = _mm(_mm(R_M2Diff, A_), R_M2Diff.T)
            if generalLiouvillian:
                setT(iInt, F.istotensor(ops(iElSpin), ops(nElSpins + iNucSpin)))
            put(iInt, *F.tensor_cart2sph(A_))
            isFieldDep[iInt] = False
            if not generalLiouvillian:
                if np.any(np.abs(F1[iInt, :]) > 1e-6):
                    raise ValueError('Hyperfine tensors must be symmetric for this method.')
                _setidx(System, 'A_axial', (iNucSpin, iElSpin), F2[iInt, 0] == 0, bool)
                _setidx(System, 'HF0', (iNucSpin, iElSpin), _creal(F0[iInt, 0] * 2 * np.pi))
                _setidx(System, 'HF2', (slice(None), iNucSpin, iElSpin), _creal(F2[iInt, :].T * 2 * np.pi))
            iInt += 1

    if nZFS > 0:
        Dm = np.atleast_2d(np.asarray(System['D'], dtype=float))
        DFr = np.atleast_2d(np.asarray(System['DFrame'], dtype=float))
        for iSpin in range(nElSpins):
            if Sv[iSpin] < 1:
                continue
            if System['fullD']:
                D_ = Dm[3 * iSpin:3 * iSpin + 3, :]
            else:
                D_ = np.diag(Dm[iSpin, :])
            D_ = D_ * 1e6
            if np.any(DFr[iSpin, :]):
                R_M2D = F.erot(DFr[iSpin, :])
                R_D2M = R_M2D.T
                D_ = _mm(_mm(R_D2M, D_), R_D2M.T)
            if generalLiouvillian:
                S_ = ops(iSpin)
                setT(iInt, F.istotensor(S_, S_))
            put(iInt, *F.tensor_cart2sph(D_))
            isFieldDep[iInt] = False
            iInt += 1

    if nElSpins > 1:
        iCoupling = 0
        eem = np.atleast_2d(np.asarray(System['ee'], dtype=float))
        eeFr = np.atleast_2d(np.asarray(System['eeFrame'], dtype=float))
        for iEl1 in range(nElSpins):
            for iEl2 in range(iEl1 + 1, nElSpins):
                if System['fullee']:
                    J_ = eem[3 * iCoupling:3 * iCoupling + 3, :]
                else:
                    J_ = np.diag(eem[iCoupling, :])
                if np.any(eeFr[iCoupling, :]):
                    R_M2ee = F.erot(eeFr[iCoupling, :])
                    R_ee2M = R_M2ee.T
                    J_ = _mm(_mm(R_ee2M, np.diag(eem[iCoupling, :])), R_ee2M.T)
                if generalLiouvillian:
                    setT(iInt, F.istotensor(ops(iEl1), ops(iEl2)))
                put(iInt, *F.tensor_cart2sph(J_ * 1e6))
                isFieldDep[iInt] = False
                iInt += 1
                iCoupling += 1

    gn = np.asarray(System['gn'], dtype=float).ravel(order='F') if nNucSpins else np.zeros(0)
    if generalLiouvillian:
        if includeNuclearZeeman:
            for iNucSpin in range(nNucSpins):
                I_ = ops(nElSpins + iNucSpin)
                gn_ = gn[iNucSpin]
                setT(iInt, F.istotensor(B0, I_))
                put(iInt, *F.tensor_cart2sph(-gn_ * F.nmagn()[0] / F.planck()))
                isFieldDep[iInt] = True
                iInt += 1
    else:
        if includeNuclearZeeman:
            gn0 = np.zeros(nNucSpins)
            for iNucSpin in range(nNucSpins):
                gn0[iNucSpin] = np.real(F.tensor_cart2sph(gn[iNucSpin])[0])
                _setidx(System, 'NZ0', (iNucSpin,), F.nmagn()[0] * B0[2] * gn0[iNucSpin] / F.planck() * 2 * np.pi)
        else:
            System['NZ0'] = np.zeros((1, nNucSpins))
        if int(System['nNuclei']) >= 2:
            Iv = np.asarray(System['I'], dtype=float).ravel(order='F')
            System['Ib'] = float(Iv[1])
            System['NZ0b'] = float(np.asarray(System['NZ0']).ravel(order='F')[1])
            System['HF0b'] = np.asarray(System['HF0']).ravel(order='F')[1]
            HF2 = np.asarray(System['HF2'])
            System['HF2b'] = HF2.reshape(5, -1, order='F')[:, 1:2]

    if includeNuclearQuadrupole:
        Qm = np.atleast_2d(np.asarray(System['Q'], dtype=float))
        QFr = np.atleast_2d(np.asarray(System['QFrame'], dtype=float))
        for iNucSpin in range(nNucSpins):
            Q_ = Qm[iNucSpin, :] * 1e6
            if np.any(QFr[iNucSpin, :]):
                R_M2Q = F.erot(QFr[iNucSpin, :])
                R_Q2M = R_M2Q.T
                Q_ = _mm(_mm(R_Q2M, np.diag(Q_)), R_Q2M.T)
            if generalLiouvillian:
                I_ = ops(nElSpins + iNucSpin)
                setT(iInt, F.istotensor(I_, I_))
            put(iInt, *F.tensor_cart2sph(Q_))
            isFieldDep[iInt] = False
            iInt += 1

    F0o, F1o, F2o = _creal(F0), _creal(F1), _creal(F2)
    Symmetry = Struct()
    # Rank-1 (antisymmetric) terms couple K to K+-1 and both jK sectors, so a nonzero F1 rules out
    # the even-K and jK=+1 basis restrictions that chili derives from these flags. EasySpin looks
    # at F2 only (corrected, docs/DEVIATIONS.md). Roundoff-level F1 (1e-17 of the tensors, from
    # rotated symmetric tensors) is ignored.
    scaleF = max(float(np.max(np.abs(F0o), initial=0)), float(np.max(np.abs(F2o), initial=0)))
    rank1 = float(np.max(np.abs(F1o), initial=0)) > 1e-10 * scaleF
    Symmetry['nobetatilts'] = F.allclose(F2o[:, [1, 3]], 0) and not rank1
    Symmetry['tensorsCollinear'] = bool(not np.iscomplexobj(F2o)) and not rank1
    Symmetry['axialSystem'] = F.allclose(F2o[:, 0:1], 0)
    T = Struct()
    T['T0'] = T0
    T['T1'] = [T1[i][k] for k in range(3) for i in range(nInteractions)]
    T['T2'] = [T2[i][k] for k in range(5) for i in range(nInteractions)]
    Fo = Struct()
    Fo['F0'] = F0o
    Fo['F1'] = F1o
    Fo['F2'] = F2o
    return T, Fo, System, Symmetry, isFieldDep


def _creal(x):
    """MATLAB keeps an array real when no element has a nonzero imaginary part."""
    a = np.asarray(x)
    if np.iscomplexobj(a) and not np.any(a.imag):
        return a.real.copy() if a.ndim else float(a.real)
    return a


magint = nout(5)(magint)

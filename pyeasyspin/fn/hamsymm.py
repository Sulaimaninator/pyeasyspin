"""Determine the symmetry (point group and symmetry frame) of a spin Hamiltonian."""
import numpy as np
import scipy.sparse as sps
from .._registry import F, nout
from .._mat import a2, vec, isempty

_GROUPS = ['Ci', 'C2h', 'D2h', 'C4h', 'D4h', 'S6', 'D3d', 'C6h', 'D6h', 'Th', 'Oh', 'Dinfh', 'O3']


def _eig(H):
    H = H.toarray() if sps.issparse(H) else np.asarray(H)
    return np.linalg.eigvalsh(H) if np.array_equal(H, H.conj().T) else np.sort(np.linalg.eigvals(H).real)


def _eqeig(a, b):
    a, b = np.sort(np.real(a)), np.sort(np.real(b))
    return np.linalg.norm(a - b) < 1e-12 * np.linalg.norm(a)


def _uniquerows_first(M):
    seen, keep = set(), []
    for i, r in enumerate(M):
        t = tuple(r.tolist())
        if t not in seen:
            seen.add(t)
            keep.append(i)
    return keep


def _eigs(Sys, hz, debug):
    pa = [np.zeros((1, 3))]
    for f in ('gFrame', 'eeFrame', 'DFrame'):
        if f in Sys:
            pa.append(a2(Sys[f]))
    if 'AFrame' in Sys:
        AF = a2(Sys.AFrame)
        pa.append(np.reshape(vec(AF.T), (3, -1), order='F').T if AF.size else np.zeros((0, 3)))
    for f in ('QFrame', 'nnFrame', 'sigmaFrame'):
        if f in Sys:
            pa.append(a2(Sys[f]))
    pa = np.vstack([p for p in pa if p.size])
    pa = pa[_uniquerows_first(pa)]
    Rz = np.eye(3)
    Rx, Ry = Rz[[1, 2, 0], :], Rz[[2, 0, 1], :]
    Rots = []
    for k in range(pa.shape[0]):
        R = np.asarray(F.erot(pa[k, :])).T
        Rots += [R @ Rz, R @ Rx, R @ Ry]
    keep = _uniquerows_first(np.array([vec(R) for R in Rots]))
    Rots = [Rots[i] for i in keep]
    if not hz:
        H0, mux, muy, muz = (np.asarray(x.toarray() if sps.issparse(x) else x) for x in F.ham(Sys, nargout=4))
    q, qq = 7.3234, 6.3456
    the = np.pi / 180 * np.array([q, q, q, q / 2, q / 2, q, 180 - q, 180 - q, 180 - q, 90 - q, q, q])
    phi = np.pi / 180 * (np.array([0, 120, 90, 90, 0, -90, 0, -90, 180, 90, -2 * qq, 180]) + qq)
    FieldVecs = 350 * np.asarray(F.ang2vec(phi, the))
    if debug:
        print('===========================================================')
        print('initial symmetry: C1')
    best, RMatrix = 0, Rz
    for iF, R in enumerate(Rots):
        if debug:
            print(f'Frame {iF + 1} ------------------------')
        B = R @ FieldVecs

        def e(j, perm=None):
            b = B[:, j - 1]
            if perm is not None:
                b = b[perm]
            if hz:
                return _eig(F.ham(Sys, b))
            return _eig(H0 - b[0] * mux - b[1] * muy - b[2] * muz)
        eA, eB, eC = e(1), e(2), e(3)
        C3, C4 = _eqeig(eB, eA), _eqeig(eC, eA)
        if debug:
            print('C3z axis found' if C3 else 'No C3z axis found')
            print('C4z axis found' if C4 else 'No C4z axis found')
        if not C3 and not C4:
            if not _eqeig(eA, e(12)):
                pg = 1
            else:
                pg = 3 if _eqeig(eA, e(11)) else 2
        elif C3 and not C4:
            if _eqeig(eA, e(7)):
                if _eqeig(eC, e(6)):
                    pg = 9 if _eqeig(eC, e(8)) else 8
                else:
                    pg = 10
            else:
                C2x = _eqeig(eC, e(8))
                C2y = False if C2x else _eqeig(eA, e(9))
                pg = 7 if (C2x or C2y) else 6
        elif not C3 and C4:
            if _eqeig(eC, e(8)):
                pg = 11 if _eqeig(e(10, [1, 2, 0]), e(10)) else 5
            else:
                pg = 4
        else:
            pg = 13 if _eqeig(eC, e(4)) else 12
        if debug:
            print(f'Point group {_GROUPS[pg - 1]}' if pg > 0 else 'No point group identified.')
        if pg > best:
            best, RMatrix = pg, R
            if debug:
                print('Highest symmetry up to now.')
    if best == 0:
        raise ValueError('No point group found! Save input spin system and report bug!')
    Group = _GROUPS[best - 1]
    if debug:
        print('===========================================================')
        print(f'Symmetry = {Group}')
    return Group, RMatrix


def _tensorsym(pv, ang):
    pv = vec(pv)
    neq = int(np.sum(np.diff(np.sort(pv)) == 0))
    if neq == 0:
        sym, z = 2, 3
    elif neq == 2:
        sym, z = 0, 3
    else:
        sym = 1
        z = 1 if pv[1] == pv[2] else 2 if pv[0] == pv[2] else 3
    idx = [2, 3, 1, 2, 3]
    fr = np.asarray(F.erot(ang)).T
    return sym, fr[:, [idx[z - 1 + i] - 1 for i in range(3)]]


def _combine(S1, R1, S2, R2):
    O3, Dinfh, D2h, C2h, Ci = 0, 1, 2, 3, 4
    par, perp = 0.2 * np.pi / 180, (90 - 0.2) * np.pi / 180
    if S1 < S2:
        S1, S2, R1, R2 = S2, S1, R2, R1
    if S1 == Ci or S2 == O3:
        return S1, R1
    if S1 == O3:
        return S2, R2
    ang = lambda x: np.real(np.arccos(np.clip(np.abs(x), -np.inf, np.inf).astype(complex)))
    idx = [2, 3, 1, 2, 3]
    cyc = lambda R, z: R[:, [idx[z - 1 + i] - 1 for i in range(3)]]
    if S1 == Dinfh and S2 == Dinfh:
        z1, z2 = R1[:, 2], R2[:, 2]
        a = ang(z1 @ z2)
        if a < par:
            return Dinfh, R1
        nz = np.cross(z1, z2); nz = nz / np.linalg.norm(nz)
        return (D2h if a > perp else C2h), np.column_stack([np.cross(z1, nz), z1, nz])
    if S1 == D2h and S2 == Dinfh:
        a = ang(R2[:, 2] @ R1)
        if np.any(a < par):
            return D2h, R1
        if np.any(a > perp):
            return C2h, cyc(R1, int(np.argmax(a)) + 1)
        return Ci, R1
    if S1 == C2h and S2 == Dinfh:
        a = ang(R1[:, 2] @ R2[:, 2])
        return (C2h if (a < par or a > perp) else Ci), R1
    if S1 == D2h and S2 == D2h:
        allA = ang(R2.T @ R1)
        col = allA.min(axis=0) < par
        if np.all(col):
            return D2h, R1
        if np.any(col):
            return C2h, cyc(R1, int(np.argmin(allA.min(axis=0))) + 1)
        return Ci, R1
    if S1 == C2h and S2 == D2h:
        a = ang(R1[:, 2] @ R2)
        return (C2h if np.any(a < par) else Ci), R1
    raise ValueError(f'Cannot compute symmetry combination: {S1} {S2}')


def _geom(Sys, debug):
    if debug:
        print('===========================================================')
    Sys = F.validatespinsys(Sys)[0]
    nE, nN = int(Sys.nElectrons), int(Sys.nNuclei)
    S, I = vec(Sys.S), vec(Sys.I) if np.size(Sys.I) else np.zeros(0)
    items = []
    g, gF = a2(Sys.g), a2(Sys.gFrame)
    for i in range(nE):
        items.append(_tensorsym(g[i, :], gF[i, :]) + (f'g{i + 1}',))
    D, DF = a2(Sys.D), a2(Sys.DFrame)
    for i in range(nE):
        if S[i] > 0.5:
            items.append(_tensorsym(D[i, :], DF[i, :]) + (f'D{i + 1}',))
    for c in range(nE * (nE - 1) // 2):
        items.append(_tensorsym(a2(Sys.ee)[c, :], a2(Sys.eeFrame)[c, :]) + (f'ee{c + 1}',))
    for n in range(nN):
        items.append(_tensorsym(a2(Sys.sigma)[n, :], a2(Sys.sigmaFrame)[n, :]) + (f'sigma{nE}',))
    for n in range(nN):
        for e in range(nE):
            items.append(_tensorsym(a2(Sys.A)[n, 3 * e:3 * e + 3], a2(Sys.AFrame)[n, 3 * e:3 * e + 3])
                         + (f'A{e + 1}{n + 1}',))
    for n in range(nN):
        if I[n] > 0.5:
            items.append(_tensorsym(a2(Sys.Q)[n, :], a2(Sys.QFrame)[n, :]) + (f'Q{n + 1}',))
    for c in range(nN * (nN - 1) // 2):
        items.append(_tensorsym(a2(Sys.nn)[c, :], a2(Sys.nnFrame)[c, :]) + (f'nn{c + 1}',))
    if debug:
        print(f'  {len(items)} tensor(s)')
    items = [it for it in items if it[0] != 0]
    if debug:
        print(f'  {len(items)} anisotropic tensor(s)')
    names = ['O3', 'Dinfh', 'D2h', 'C2h', 'Ci']
    Grp, fr = 0, np.eye(3)
    if debug:
        print('  O3 as starting symmetry')
    for s, ax, nm in items:
        Grp, fr = _combine(Grp, fr, s, ax)
        if debug:
            print(f'   + {names[s]} ({nm}) = {names[Grp]}')
    return names[Grp], fr


def _isisotropic(Sys):
    def isiso(T):
        return T in Sys and np.size(Sys[T]) > 0 and bool(np.any(np.diff(a2(Sys[T]), axis=1)))
    iso = all(isiso(T) for T in ('g', 'D', 'ee', 'Q', 'sigma', 'nn'))
    if iso and 'A' in Sys:
        A = a2(Sys.A)
        for e in range(int(Sys.nElectrons)):
            d = np.any(np.diff(A[:, 3 * e:3 * e + 3], axis=1), axis=0)
            if d.size != 1:
                raise ValueError('Operands to the || and && operators must be convertible to logical scalar values.')
            iso = iso and bool(d)
    return iso


@nout(2)
def hamsymm(Sys=None, *varargin):
    """(PGroup, RMatrix): point group of the spin Hamiltonian (Ci ... O3) and the symmetry frame.
    A trailing 'debug' option prints the analysis."""
    debug = bool(varargin) and isinstance(varargin[-1], str) and 'debug' in varargin[-1]
    Sys, err = F.validatespinsys(Sys)
    if err:
        raise ValueError(err)
    hz = any(f[:3] == 'Ham' and np.any(np.asarray(Sys[f]) != 0) for f in Sys)
    cf = any(f[:2] == 'CF' and np.any(np.asarray(Sys[f]) != 0) for f in Sys)
    high = (not isempty(Sys.B)) or hz or cf
    if debug:
        print('High-order terms present!' if high else 'No high-order terms present!')
    full = any(bool(Sys[f]) for f in ('fullg', 'fullA', 'fullD', 'fullee', 'fullQ', 'fullsigma', 'fullnn'))
    if not full and not high:
        if debug:
            print('Check whether isotropic...')
        if _isisotropic(Sys):
            return 'O3', np.eye(3)
    qm = high or full
    if debug:
        print('Quantum mechanical symmetry analysis...' if qm else 'Geometric symmetry analysis...')
    PGroup, RMatrix = _eigs(Sys, hz, debug) if qm else _geom(Sys, debug)
    if not isempty(Sys.tdm):
        PGroup = 'C1'
    return PGroup, RMatrix

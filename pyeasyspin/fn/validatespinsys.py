"""Validate a spin system structure and supplement all fields to their full form."""
import re
import numpy as np
from .._registry import F, nout
from .._struct import Struct
from .._mat import a2, issize, isempty, vec, numel, size

_CORRECT = (['S', 'Nucs', 'Abund', 'n', 'g', 'g_', 'gFrame', 'gStrain', 'D', 'D_', 'DFrame', 'DStrain',
             'ee', 'J', 'dip', 'dvec', 'ee2', 'eeFrame', 'A', 'A_', 'AFrame', 'AStrain', 'Q', 'QFrame',
             'HStrain', 'L', 'soc', 'gL', 'initState', 'tdm', 'lw', 'lwpp', 'lwEndor',
             'tcorr', 'logtcorr', 'Diff', 'logDiff']
            + [f'B{k}' for k in range(1, 13)] + [f'CF{k}' for k in range(1, 13)])


def _sizecheck(Sys, name, siz):
    return '' if issize(a2(Sys[name]), siz) else f'Spin system field {name} has wrong size for the given spins.'


def _pa(Sys, pa, frame):
    return f'Obsolete field Sys.{pa}. Use Sys.{frame} instead.' if pa in Sys else ''


def _any(x):
    return bool(np.any(np.asarray(x if x is not None else 0) != 0))


@nout(2)
def validatespinsys(Sys=None):
    """(FullSys, err): spin system with all fields in full form (g as nElectrons x 3, A as
    nNuclei x 3*nElectrons, ...) and processed=True; on failure FullSys=None and err is the message."""
    err = ''
    if not isinstance(Sys, dict):
        return None, 'Spin system must be a structure.'
    reprocessing = False
    if 'processed' in Sys:
        if Sys['processed']:
            return Sys, err
        reprocessing = True
    Sys = Struct(Sys)

    lower = [c.lower() for c in _CORRECT]
    for giv in list(Sys.keys()):
        if giv == 'ZeemanFreq':
            return None, 'Field Sys.ZeemanFreq can only be used in conjunction with the function spidyan.'
        if giv.lower() in lower:
            corr = _CORRECT[lower.index(giv.lower())]
            if giv != corr:
                raise ValueError(f'Fix capitalization: Sys.{giv} should be Sys.{corr}')
    hamfields = [f for f in Sys if f[:3].lower() == 'ham']
    if hamfields:   # MATLAB loops over a column vector: only the first Ham field is checked
        field = hamfields[0]
        if len(field) != 6:
            try:
                s = float(field[3]) + float(field[4])
            except (ValueError, IndexError):
                s = float('nan')
            if s < 10:
                return None, f'Wrong length of Sys.{field} entry, should be Hamxyz (with x,y,z integer numbers)'
            elif len(field) != 7:
                return None, f'Wrong length of Sys.{field} entry, should be Hamxyz (with x,y,z integer numbers)'
        if not field.startswith('Ham'):
            return None, f'Fix capitalization: Sys.{field} should be Sys.Ham{field[3:]}'
    anyHam = any(f.startswith('Ham') for f in Sys)

    if 'S' not in Sys:
        Sys.S = 0.5
    S = vec(Sys.S) if not isempty(Sys.S) else np.zeros(0)
    if S.size == 0 or np.iscomplexobj(S) or np.any(np.mod(np.real(S), 0.5)) or np.any(S < 0):
        return None, 'Electron spins in S must be positive integer multiples of 1/2.'
    if np.any(S == 0):
        return None, 'Systems with zero spin are not allowed.'
    nE = S.size
    Sys.nElectrons = nE

    # g
    if 'g' in Sys:
        if 'g_' in Sys:
            return None, 'Sys.g and Sys.g_ are given. Remove one of them.'
        g = a2(Sys.g)
        Sys.fullg = issize(g, [3 * nE, 3])
        if Sys.fullg:
            pass
        elif g.size == nE:
            Sys.g = vec(g).reshape(-1, 1) * np.ones((1, 3))
        elif issize(g, [nE, 2]):
            Sys.g = g[:, [0, 0, 1]]
        elif issize(g, [nE, 3]):
            Sys.g = g
        else:
            return None, 'Sys.g has wrong size.'
    elif 'g_' in Sys:
        Sys.fullg = False
        g_ = a2(Sys.g_)
        if g_.size == nE or issize(g_, [nE, 2]):
            t = np.zeros((nE, 3), dtype=g_.dtype)
            t[:g_.shape[0], :g_.shape[1]] = g_
            g_ = t
        elif issize(g_, [nE, 3]):
            pass
        else:
            return None, 'Sys.g_ has wrong size.'
        Sys.g_ = g_
        gc = np.zeros((nE, 3))
        for i in range(nE):
            gs = g_[i, :]
            gc[i, :] = gs[0] + gs[1] * np.array([-1, -1, 2]) + gs[2] * np.array([-1, 1, 0])
        Sys.g = gc
    else:
        Sys.fullg = False
        Sys.g = np.zeros((nE, 3)) if anyHam else F.gfree()[0] * np.ones((nE, 3))
    err = _pa(Sys, 'gpa', 'gFrame')
    if err:
        return None, err
    if 'gFrame' not in Sys or isempty(Sys.gFrame):
        Sys.gFrame = np.zeros((nE, 3))
    err = _sizecheck(Sys, 'gFrame', [nE, 3])
    if err:
        return None, err
    Sys.gFrame = a2(Sys.gFrame)

    # D
    if 'D' in Sys and 'D_' in Sys:
        return None, 'Zero-field splitting given in both Sys.D and Sys.D_. Please remove one of them.'
    if 'D' in Sys:
        D = a2(Sys.D)
        Sys.fullD = issize(D, [3 * nE, 3])
        if Sys.fullD:
            Sys.D = D
        elif D.size == nE:
            Sys.D = vec(D).reshape(-1, 1) * np.array([[-1 / 3, -1 / 3, 2 / 3]])
        elif issize(D, [nE, 2]):
            Sys.D = D[:, [0]] * np.array([[-1 / 3, -1 / 3, 2 / 3]]) + D[:, [1]] * np.array([[1, -1, 0]])
        elif issize(D, [nE, 3]):
            Sys.D = D
        else:
            return None, 'Sys.D has wrong size.'
    elif 'D_' in Sys:
        Sys.fullD = False
        D_ = a2(Sys.D_)
        if issize(D_, [nE, 2]):
            Dv = D_[:, [0]]
            Ev = Dv * D_[:, [1]]
            Sys.D = Dv * np.array([[-1 / 3, -1 / 3, 2 / 3]]) + Ev * np.array([[1, -1, 0]])
        else:
            return None, 'Sys.D_ has wrong size.'
    else:
        Sys.D = np.zeros((nE, 3))
        Sys.fullD = False
    err = _pa(Sys, 'Dpa', 'DFrame')
    if err:
        return None, err
    if 'DFrame' not in Sys or isempty(Sys.DFrame):
        Sys.DFrame = np.zeros((nE, 3))
    err = _sizecheck(Sys, 'DFrame', [nE, 3])
    if err:
        return None, err
    Sys.DFrame = a2(Sys.DFrame)
    if 'aF' in Sys or 'aFFrame' in Sys:
        raise ValueError('Sys.aF and Sys.aFFrame are no longer supported. Use Sys.B4 and Sys.B4Frame instead.')
    if 'BFrame' in Sys and not reprocessing:
        return None, 'Sys.BFrame is not supported. Use Sys.B1Frame, Sys.B2Fame, etc instead.'

    # Stevens operator coefficients B1..B12
    Bcell, BFcell = None, None
    D_present = _any(Sys.D)
    for k in range(1, 13):
        fn = f'B{k}'
        if fn not in Sys:
            continue
        Bk = a2(Sys[fn])
        if k == 2 and _any(Bk) and D_present:
            return None, 'Cannot use Sys.D and Sys.B2 simultaneously. Remove one of them.'
        if Bk.shape[0] != nE:
            return None, f'Field Sys.{fn} has to have {nE} rows, since there are {nE} electron spins.'
        if Bk.shape[1] == 1:
            Bk = np.hstack([np.zeros((nE, k)), Bk, np.zeros((nE, k))])
        elif Bk.shape[1] == 2 * k + 1:
            pass
        else:
            return None, f'Field Sys.{fn} has {Bk.shape[1]} columns, but {2 * k + 1} are required.'
        if np.iscomplexobj(Bk) and np.any(np.imag(Bk)):
            return None, f'Field Sys.{fn} contains complex numbers. Only real ones are.'
        Sys[fn] = Bk
        if Bcell is None:
            Bcell = []
        Bcell += [np.zeros((0, 0))] * (k - len(Bcell))
        Bcell[k - 1] = Bk
        fnF = f'B{k}Frame'
        if BFcell is None:
            BFcell = list(Sys['BFrame']) if isinstance(Sys.get('BFrame'), list) else []
        BFcell += [np.zeros((0, 0))] * max(0, k - len(BFcell))
        BFcell[k - 1] = a2(Sys[fnF]) if (fnF in Sys and _any(Sys[fnF])) else np.zeros((nE, 3))
    Sys.B = Bcell if Bcell is not None else np.zeros((0, 0))
    if BFcell is not None:
        Sys.BFrame = BFcell

    # electron-electron couplings
    if 'fullee' not in Sys:
        Sys.fullee = False
    if nE > 1 and not reprocessing:
        eeMatrix = 'ee' in Sys
        JdD = any(f in Sys and not isempty(Sys[f]) for f in ('J', 'dvec', 'dip'))
        if not eeMatrix and not JdD:
            return None, 'Spin system contains 2 or more electron spins, but coupling terms are missing (ee; or J, dip, dvec)!'
        if eeMatrix and JdD:
            return None, 'Both Sys.ee and (Sys.J,Sys.dip,Sys.dvec) are given - use only one or the other!'
        nP = nE * (nE - 1) // 2
        if eeMatrix:
            ee = a2(Sys.ee)
            if ee.size == nP:
                ee = vec(ee).reshape(-1, 1) * np.ones((1, 3))
            Sys.ee = ee
            Sys.fullee = issize(ee, [3 * nP, 3])
            if not Sys.fullee:
                err = _sizecheck(Sys, 'ee', [nP, 3])
                if err:
                    return None, err
        else:
            if 'J' not in Sys:
                Sys.J = np.zeros((1, nP))
            Sys.J = vec(Sys.J).reshape(-1, 1)
            err = _sizecheck(Sys, 'J', [nP, 1])
            if err:
                return None, err
            if 'dvec' not in Sys:
                Sys.dvec = np.zeros((nP, 3))
            err = _sizecheck(Sys, 'dvec', [nP, 3])
            if err:
                return None, err
            Sys.dvec = a2(Sys.dvec)
            if 'dip' not in Sys:
                Sys.dip = np.zeros((nP, 3))
            dip = a2(Sys.dip)
            if dip.size == nP:
                dip = vec(dip).reshape(-1, 1)
            if dip.shape[0] != nP:
                return None, f'Sys.dip must contain {nP} rows, since there are {nP} unique pairs of electron spins.'
            if 'eeD' in Sys:
                return None, 'Sys.eeD is obsolete. Use Sys.dip instead.'
            if dip.shape[1] == 1:
                dip = dip * np.array([[1, 1, -2]])
            elif dip.shape[1] == 2:
                dip = dip[:, [0]] * np.array([[1, 1, -2]]) + dip[:, [1]] * np.array([[1, -1, 0]])
            elif dip.shape[1] == 3:
                dip = dip - np.mean(dip, axis=1, keepdims=True)
            else:
                return None, 'Sys.dip must contain 1, 2, or 3 columns.'
            Sys.dip = dip
            fullee = _any(Sys.dvec)
            Sys.fullee = fullee
            if fullee:
                ee = np.zeros((3 * nP, 3))
                for p in range(nP):
                    J = Sys.J[p, 0]
                    d = Sys.dvec[p, :]
                    ee[3 * p:3 * p + 3, :] = (J * np.eye(3) + np.array([[0, d[2], -d[1]], [-d[2], 0, d[0]],
                                                                        [d[1], -d[0], 0]]) + np.diag(dip[p, :]))
            else:
                ee = Sys.J[:, [0]] + dip
            Sys.ee = ee
        err = _pa(Sys, 'eepa', 'eeFrame')
        if err:
            return None, err
        if 'eeFrame' not in Sys:
            Sys.eeFrame = np.zeros((nP, 3))
        err = _sizecheck(Sys, 'eeFrame', [nP, 3])
        if err:
            return None, err
        Sys.eeFrame = a2(Sys.eeFrame)
    if nE > 1:
        nP = nE * (nE - 1) // 2
        Sys.ee2 = np.zeros((nP, 1)) if 'ee2' not in Sys else vec(Sys.ee2).reshape(-1, 1)
        err = _sizecheck(Sys, 'ee2', [nP, 1])
        if err:
            return None, err

    # nuclei
    if 'Nucs' not in Sys:
        Sys.Nucs = ''
    if isempty(Sys.Nucs):
        if any(f in Sys and not isempty(Sys[f]) for f in ('A', 'AFrame', 'Q', 'QFrame')):
            return None, 'The system contains A and/or Q fields, but no nucleus is specified!'
    else:
        Sys.Nucs = F.nucstring2list(Sys.Nucs)
    nd = F.nucdata(Sys.Nucs)
    I, gn = nd[0], nd[1]
    I = np.zeros(0) if isempty(I) else np.atleast_1d(np.asarray(I, float))
    gn = np.zeros(0) if isempty(gn) else np.atleast_1d(np.asarray(gn, float))
    Sys.I, Sys.gn = I, gn
    nN = I.size
    Sys.nNuclei = nN
    if 'n' in Sys:
        if isempty(Sys.n):
            Sys.n = np.ones((1, nN))   # 1x0 when there are no nuclei (as MATLAB ones(1,0))
        err = _sizecheck(Sys, 'n', [1, nN])
        if err:
            return None, err
    else:
        Sys.n = np.ones(nN)
    if 'gnscale' in Sys:
        if numel(Sys.gnscale) < nN:
            return None, 'Incorrect number of elements in Sys.gnscale.'
    else:
        Sys.gnscale = np.ones(nN) if nN > 0 else np.zeros((0, 0))
    if 'sigma' in Sys:
        sg = a2(Sys.sigma)
        Sys.fullsigma = issize(sg, [3 * nN, 3])
        if Sys.fullsigma:
            Sys.sigma = sg
        elif sg.size == nN:
            Sys.sigma = vec(sg).reshape(-1, 1) * np.ones((1, 3))
        elif issize(sg, [nN, 2]):
            Sys.sigma = sg[:, [0, 0, 1]]
        elif issize(sg, [nN, 3]):
            Sys.sigma = sg
        else:
            return None, 'Sys.sigma has wrong size.'
    else:
        Sys.fullsigma = False
        Sys.sigma = np.ones((nN, 3))
    if 'sigmaFrame' not in Sys or isempty(Sys.sigmaFrame):
        Sys.sigmaFrame = np.zeros((nN, 3))
    err = _sizecheck(Sys, 'sigmaFrame', [nN, 3])
    if err:
        return None, err

    Sys.fullA = False
    if nN > 0:
        if 'A' not in Sys and 'A_' not in Sys:
            return None, f'No hyperfine tensors A given for the {nE} electron spins and {nN} nuclei in the system!'
        if 'A' in Sys and 'A_' in Sys:
            return None, 'Hyperfine data given in both Sys.A and Sys.A_. Please remove one of them.'
        if 'A_' in Sys:
            A_ = a2(Sys.A_)
            if issize(A_, [1, nN]):
                A_ = np.hstack([A_.T, np.zeros((nN, 2))])
            elif issize(A_, [nN, nE]):
                t = np.zeros((nN, max(3 * nE - 1, nE)), dtype=A_.dtype)
                t[:, :nE] = A_
                old = A_.copy()
                for e in range(nE):
                    t[:, 3 * e] = old[:, e]
                    if 3 * e + 2 < t.shape[1]:
                        t[:, 3 * e + 2] = 0
                    else:
                        t = np.hstack([t, np.zeros((nN, 3 * e + 3 - t.shape[1]))])
                A_ = t
            elif issize(A_, [nN, 2 * nE]):
                old = A_.copy()
                t = np.zeros((nN, max(3 * nE, 2 * nE)), dtype=A_.dtype)
                t[:, :2 * nE] = old
                for e in range(nE):
                    t[:, 3 * e] = old[:, 2 * e]
                    t[:, 3 * e + 1] = old[:, 2 * e + 1]
                    t[:, 3 * e + 2] = 0
                A_ = t
            elif issize(A_, [nN, 3 * nE]):
                pass
            else:
                return None, 'Wrong size of the A_ hyperfine array in the spin system.'
            Sys.A_ = A_
            Ac = np.zeros((nN, 3 * nE))
            for e in range(nE):
                for n in range(nN):
                    s = A_[n, 3 * e:3 * e + 3]
                    Ac[n, 3 * e:3 * e + 3] = s[0] + np.array([-1, -1, 2]) * s[1] + np.array([-1, 1, 0]) * s[2]
            Sys.A = Ac
        else:
            if isinstance(Sys.A, str) or isinstance(Sys.A, (list, dict)) and not all(
                    isinstance(v, (int, float, complex, np.number)) for v in (Sys.A if isinstance(Sys.A, list) else [None])):
                return None, 'Sys.A must be a numeric array.'
            A = a2(Sys.A)
            if issize(A, [3 * nN, 3 * nE]):
                Sys.fullA = True
                Sys.A = A
            elif issize(A, [1, nN]):
                if nE == 1:
                    Sys.A = vec(A).reshape(-1, 1) * np.ones((1, 3))
                else:
                    return None, 'Size of Sys.A matrix is inconsistent with number of electrons and nuclei.'
            elif issize(A, [nN, nE]):
                Sys.A = np.kron(A, np.ones((1, 3)))
            elif issize(A, [nN, 2 * nE]):
                idx = [0, 0, 1]
                for k in range(2, nE + 1):
                    idx += [2 * k - 2, 2 * k - 2, 2 * k - 1]
                Sys.A = A[:, idx]
            elif issize(A, [nN, 3 * nE]):
                Sys.A = A
            else:
                return None, (f'Size of Sys.A ({A.shape[0]}x{A.shape[1]}) is inconsistent with number of nuclei '
                              f'({nN}) and electrons ({nE}).')
        err = _pa(Sys, 'Apa', 'AFrame')
        if err:
            return None, err
        if 'AFrame' not in Sys or isempty(Sys.AFrame):
            Sys.AFrame = np.zeros((nN, 3 * nE))
        err = _sizecheck(Sys, 'AFrame', [nN, 3 * nE])
        if err:
            return None, err
        Sys.AFrame = a2(Sys.AFrame)

    Sys.fullQ = False
    if nN > 0:
        if 'Q' not in Sys:
            Sys.Q = np.zeros((nN, 3))
            Sys.fullQ = False
        else:
            Q = a2(Sys.Q)
            Sys.fullQ = issize(Q, [3 * nN, 3])
            if not Sys.fullQ:
                if Q.size == nN:
                    Q = np.hstack([vec(Q).reshape(-1, 1), np.zeros((nN, 1))])
                if issize(Q, [nN, 2]):
                    Q2 = np.zeros((nN, 3))
                    for i in range(nN):
                        Ii = I[i]
                        if Ii >= 1:
                            Q2[i, :] = Q[i, 0] / (4 * Ii * (2 * Ii - 1)) * np.array([-1 + Q[i, 1], -1 - Q[i, 1], 2])
                    Q = Q2
                Sys.Q = Q
                err = _sizecheck(Sys, 'Q', [nN, 3])
                if err:
                    return None, err
            else:
                Sys.Q = Q
        if Sys.fullQ:
            for i in range(nN):
                Q_ = Sys.Q[3 * i:3 * i + 3, :]
                if np.linalg.norm(Q_ - Q_.T, 2) / np.linalg.norm(Q_, 2) > 1e-5:
                    return None, 'Sys.Q contains asymmetric full Q matrix. Only symmetric Q matrices are allowed.'
        err = _pa(Sys, 'Qpa', 'QFrame')
        if err:
            return None, err
        if 'QFrame' not in Sys or isempty(Sys.QFrame):
            Sys.QFrame = np.zeros((nN, 3))
        err = _sizecheck(Sys, 'QFrame', [nN, 3])
        if err:
            return None, err
        Sys.QFrame = a2(Sys.QFrame)

    Sys.fullnn = False
    hasnn = 'nn' in Sys and not isempty(Sys.nn) and _any(Sys.nn)
    if nN < 2:
        if hasnn:
            return None, 'Nuclear-nuclear couplings specified in Sys.nn, but fewer than two nuclei given.'
    else:
        nNP = nN * (nN - 1) // 2
        if hasnn:
            nn = a2(Sys.nn)
            if nn.size == nNP:
                nn = vec(nn).reshape(-1, 1) * np.ones((1, 3))
            Sys.nn = nn
            Sys.fullnn = issize(nn, [3 * nNP, 3])
            if not Sys.fullnn:
                err = _sizecheck(Sys, 'nn', [nNP, 3])
                if err:
                    return None, err
        else:
            Sys.fullnn = False
            Sys.nn = np.zeros((nNP, 3))
            Sys.nnFrame = np.zeros((nNP, 3))
        if 'nnFrame' not in Sys:
            Sys.nnFrame = np.zeros((nNP, 3))
        err = _sizecheck(Sys, 'nnFrame', [nNP, 3])
        if err:
            return None, err
        Sys.nnFrame = a2(Sys.nnFrame)

    rmv = I == 0
    if np.any(rmv):
        keep = ~rmv
        Sys.Nucs = [n for n, r in zip(Sys.Nucs, rmv) if not r]
        Sys.I = I[keep]
        Sys.gn = gn[keep]
        Sys.nNuclei = Sys.gn.size
        Sys.gnscale = vec(Sys.gnscale)[np.concatenate([keep, np.ones(numel(Sys.gnscale) - nN, bool)])]
        Sys.n = vec(Sys.n)[keep]
        if Sys.fullA:
            Sys.A = Sys.A[np.repeat(keep, 3), :]
        else:
            Sys.A = Sys.A[keep, :]
            Sys.AFrame = Sys.AFrame[keep, :]
        if Sys.fullQ:
            Sys.Q = Sys.Q[np.repeat(keep, 3), :]
        else:
            Sys.Q = Sys.Q[keep, :]
            Sys.QFrame = Sys.QFrame[keep, :]

    # line widths
    for fld, label, msgneg in (('lw', 'System.lw', 'System.lw cannot be negative.'),
                               ('lwEndor', 'System.lwEndor', 'System.lwEndor cannot be negative.'),
                               ('lwpp', 'System.lwpp', 'Linewidths cannot be negative.')):
        if fld not in Sys:
            Sys[fld] = np.array([[0.0, 0.0]])
        v = a2(Sys[fld])
        if v.size == 1:
            v = np.array([[v.item(), 0.0]])
        Sys[fld] = v
        if (v.size != 2) if fld == 'lwEndor' else (not issize(v, [1, 2])):
            return None, f'{label} has wrong size.'
        if np.any(v < 0):
            return None, msgneg
    if _any(Sys.lwpp):
        if _any(Sys.lw):
            return None, 'System.lw and Sys.lwpp cannot be used simultaneously.'
        Sys.lw = Sys.lwpp * np.array([[np.sqrt(2 * np.log(2)), np.sqrt(3)]])
        Sys.lwpp = np.array([[0.0, 0.0]])

    # strains
    if 'gStrain' not in Sys or isempty(Sys.gStrain):
        Sys.gStrain = np.zeros((nE, 3))
    gs = a2(Sys.gStrain)
    n1, n2 = gs.shape
    if n1 != nE:
        return None, f'Sys.gStrain must have {nE} rows, one per electron spin!'
    if n2 == 1:
        gs = gs[:, [0, 0, 0]]
    elif n2 == 2:
        gs = gs[:, [0, 0, 1]]
    elif n2 != 3:
        err = 'Sys.gStrain must have 1, 2, or 3 columns!'
    Sys.gStrain = gs
    if np.any(gs < 0):
        return None, 'Sys.gStrain must contain nonnegative values!'
    if 'DStrain' not in Sys or isempty(Sys.DStrain):
        Sys.DStrain = np.zeros((nE, 3))
    if 'DStrainCorr' not in Sys:
        Sys.DStrainCorr = np.zeros(nE)
    ds = a2(Sys.DStrain)
    n1, n2 = ds.shape
    if n1 != nE:
        return None, f'Sys.DStrain must have {nE} rows, one per electron spin!'
    if n2 == 1:
        ds = np.hstack([ds, np.zeros((nE, 2))])
    elif n2 == 2:
        ds = np.hstack([ds, np.zeros((nE, 1))])
    elif n2 != 3:
        err = 'Sys.DStrain must have 1, 2, or 3 columns!'
    Sys.DStrain = ds
    if numel(Sys.DStrainCorr) != nE:
        err = f'Sys.DStrainCorr must contain {nE} elements, since you have {nE} electron spins'
    if np.any(ds < 0):
        return None, 'Sys.DStrain must contain nonnegative values!'
    dc = np.asarray(Sys.DStrainCorr, float)
    if np.any(dc < -1) or np.any(dc > 1):
        return None, 'D-E strain correlation coefficient in Sys.DStrainCorr must be between -1 and +1.'
    for fld in ('HStrain', 'AStrain'):
        if fld not in Sys or isempty(Sys[fld]):
            Sys[fld] = np.zeros(3)
            continue
        p = vec(Sys[fld]).astype(float) if np.size(Sys[fld]) <= 2 else Sys[fld]
        if np.size(p) == 1:
            p = p[[0, 0, 0]]
        elif np.size(p) == 2:
            p = p[[0, 0, 1]]
        if np.size(p) != 3:
            return None, f'Sys.{fld} must have 3 elements!'
        if np.any(np.asarray(p) < 0):
            return None, f'Sys.{fld} must contain nonnegative values!'
        Sys[fld] = p

    # initial state
    if 'Pop' in Sys:
        raise ValueError('Sys.Pop is obsolete. Use Sys.initState to specify a non-equilibrium state for the spin system.')
    if 'initState' not in Sys:
        Sys.initState = np.zeros((0, 0))
    if not isempty(Sys.initState):
        st = Sys.initState
        if isinstance(st, list):
            if len(st) != 2:
                raise ValueError('Sys.initState must be a cell array with two elements, {popvector,basis}.')
            init, basis = a2(st[0]), st[1]
            if not isinstance(basis, str) or basis not in ('uncoupled', 'coupled', 'eigen', 'zerofield', 'xyz'):
                err = ("The basis specified in Sys.initState must be either 'zerofield',  'xyz', 'eigen', "
                       "'coupled' or 'uncoupled'.")
            s1, s2 = init.shape
            isvector = min(s1, s2) == 1
            if s1 != s2 and not isvector:
                err = 'A density matrix or a population vector must be specified within Sys.initState.'
            if basis == 'coupled':
                if nE != 2:
                    err = 'Sys.initState in the coupled basis is only available for systems with two electron spins.'
                else:
                    C2U = np.asarray(F.cgmatrix(S[0], S[1])[0]).conj().T
                    nES = C2U.shape[0]
                    if max(s1, s2) != nES:
                        nStates = F.hsdim(np.concatenate([S, Sys.I]))
                        C2U = np.kron(C2U, np.eye(int(nStates / nES)))
                    init = C2U @ (init if s1 == s2 else np.diag(vec(init))) @ C2U.conj().T
                    basis = 'uncoupled'
            elif basis == 'xyz':
                if np.any(S != 1):
                    err = "Sys.initState population input with 'xyz' basis only allowed for triplet states (Sys.S = 1)."
                if init.size != 3:
                    err = "Sys.initState population input with 'xyz' basis requires three population values [px py pz]."
                if err:
                    return None, err
                Z = np.array([[1, 1, 0], [0, 0, np.sqrt(2)], [-1, 1, 0]]) / np.sqrt(2)
                if _any(Sys.DFrame):
                    Dw = F.wignerd(1, Sys.DFrame[0, 0], Sys.DFrame[0, 1], Sys.DFrame[0, 2])
                    Z = np.asarray(Dw) @ Z
                init = Z @ np.diag(vec(init)) @ Z.conj().T
                basis = 'uncoupled'
            elif basis == 'eigen':
                if isvector:
                    init = np.diag(vec(init))
        elif isinstance(st, str):
            if st == 'singlet':
                if nE != 2 or S[0] != S[1]:
                    err = "Sys.initState = 'singlet' is only available for systems of two identical electron spins."
                else:
                    Sv = np.asarray(F.cgmatrix(S[0], S[1], 0)[0]).T
                    Sv = Sv.reshape(-1, 1) if Sv.ndim == 1 else Sv
                    init = Sv @ Sv.conj().T
                    basis = 'uncoupled'
            elif st == 'T0':
                if np.any(S != 1):
                    err = "Sys.initState = 'T0' is only available for triplet states (Sys.S = 1)."
                else:
                    init = np.diag([0.0, 1.0, 0.0])
                    basis = 'eigen'
            else:
                err = 'String input for initial state not yet supported for selected spin system and initial state.'
        elif isinstance(st, (np.ndarray, int, float, complex)) or hasattr(st, 'toarray'):
            init = a2(st)
            if init.shape[0] != init.shape[1]:
                err = ("Sys.initState called with a population vector requires a basis specification in the format "
                       "Sys.initState = {popvec,'basis'}.")
            basis = 'uncoupled'
        else:
            err = 'Invalid input for Sys.initState. Check documentation for details.'
        if err:
            return None, err
        Sys.initState = [init, basis]

    if 'tdm' not in Sys:
        Sys.tdm = np.zeros((0, 0))
    if not isempty(Sys.tdm):
        if isinstance(Sys.tdm, str):
            Sys.tdm = F.letter2vec(Sys.tdm)
        elif np.size(Sys.tdm) == 3:
            t = vec(Sys.tdm).reshape(-1, 1).astype(float)
            Sys.tdm = t / np.linalg.norm(t)
        elif np.size(Sys.tdm) == 2:
            t = vec(Sys.tdm)
            Sys.tdm = F.ang2vec(t[0], t[1])
        else:
            raise ValueError('tdm (first input) must be a letter designating a direction, a 3-vector, or an array with two angles.')

    for f in ('Diff', 'logDiff', 'tcorr', 'logtcorr'):
        if f in Sys and numel(Sys[f]) not in (0, 1, 2, 3):
            return None, f'Sys.{f} must have 1, 2, or 3 elements.'
    if 'DiffFrame' in Sys:
        err = _sizecheck(Sys, 'DiffFrame', [1, 3])
        if err:
            return None, err

    # higher-order Zeeman terms Ham<lB><lS><l>
    Sys.MO_present = False
    if anyHam:
        present = {}
        for lB in range(9):
            for lS in range(9):
                for l in range(abs(lB - lS), lB + lS + 1):
                    nm = f'Ham{lB}{lS}{l}'
                    if nm in Sys:
                        present[nm] = (lB, lS, l)
        if present:
            Sys.MO_present = True
            if any(v[0] == 0 for v in present.values()):
                if D_present and 'Ham022' in Sys:
                    return None, 'Cannot use Sys.D and Sys.Ham022 simultaneously. Remove one of them.'
                if any(v[0] == 0 and v[2] == v[1] and f'B{v[1]}' in Sys for v in present.values()):
                    err = ('Cannot use higher order operators and corresponding general parameters simultaneously. '
                           'Remove one of them.')
            if any(v[0] == 1 and v[1] == 1 for v in present.values()) and _any(Sys.g):
                err = 'Cannot use Sys.g and and Sys.Ham112 or Sys.Ham110 simultaneously. Remove one of them.'
            # MATLAB processes fields in column-major order of the (lB,lS,l-lmin) cell array
            order = sorted(present.items(), key=lambda kv: (kv[1][2] - abs(kv[1][0] - kv[1][1]), kv[1][1], kv[1][0]))
            for nm, (lB, lS, l) in order:
                v = a2(Sys[nm])
                if issize(v, [nE, 1]):
                    Sys[nm] = np.hstack([np.zeros((nE, l)), v, np.zeros((nE, l))])
                elif not issize(v, [nE, 2 * l + 1]):
                    if not issize(v, [2 * l + 1, 1]) or nE != 1:
                        return None, f'Sys.{nm} has wrong size!'
                    Sys[nm] = v.T

    # orbital angular momenta
    if 'L' in Sys and not isempty(Sys.L):
        L = vec(Sys.L)
        if np.iscomplexobj(L) or np.any(np.mod(np.real(L), 1)) or np.any(S < 0):
            return None, 'Orbital angular momentum in Sys.L must be nonnegative integers.'
        if L.size != nE:
            return None, 'Sys.L and Sys.S must have the same number of elements.'
        if 'soc' not in Sys:
            return None, 'Sys.L is given, but no spin-orbit coupling is defined in Sys.soc.'
        if isempty(Sys.soc) or (np.iscomplexobj(Sys.soc) and np.any(np.imag(Sys.soc))):
            return None, 'Spin-orbit coupling in soc must be real numbers.'
        soc = a2(Sys.soc)
        if soc.shape[0] != nE:
            if issize(soc, [1, nE]):
                soc = soc.T
            else:
                return None, 'Number of spin-orbit couplings in Sys.soc must match number of spins in Sys.S.'
        Sys.soc = soc
        if 'gL' not in Sys:
            Sys.gL = np.ones((nE, 1))
        else:
            if numel(Sys.gL) != nE:
                return None, 'Number of orbital g factors must match number of orbital angular momenta!'
            if isempty(Sys.gL) or (np.iscomplexobj(Sys.gL) and np.any(np.imag(Sys.gL))):
                return None, 'Orbital g factors in Sys.gL must be real numbers.'
        for k in range(1, 13):
            fn = f'CF{k}'
            if fn not in Sys:
                continue
            CFk = a2(Sys[fn])
            if CFk.shape[0] != nE:
                err = (f'Field Sys.{fn} has to have {nE} rows, since there are {nE} orbital angular momenta.')
            if CFk.shape[1] == 1:
                CFk = np.hstack([np.zeros((nE, k)), vec(CFk).reshape(-1, 1), np.zeros((nE, k))])
            elif CFk.shape[1] != 2 * k + 1:
                err = f'Field Sys.{fn} has {CFk.shape[1]} instead of {2 * k + 1} coloumns.'
            Sys[fn] = CFk
    else:
        if 'gL' in Sys and not isempty(Sys.gL):
            return None, 'Sys.gL is given, but Sys.L is missing. Specify Sys.L.'
        if 'soc' in Sys and not isempty(Sys.soc):
            return None, 'Sys.soc is given, but Sys.L is missing. Specify Sys.L.'
        for k in range(1, 13):
            fn = f'CF{k}'
            if fn in Sys and not isempty(Sys[fn]):
                return None, f'Sys.{fn} is given, but Sys.L is missing. Specify Sys.L.'
        Sys.L = np.zeros((0, 0))
        Sys.gL = np.zeros((0, 0))
    Sys.nL = numel(Sys.L)
    Sys.Spins = np.concatenate([S, vec(Sys.I) if numel(Sys.I) else np.zeros(0),
                                vec(Sys.L) if numel(Sys.L) else np.zeros(0)]).astype(float)
    Sys.nStates = F.hsdim(Sys.Spins)
    Sys.processed = True
    return Sys, err

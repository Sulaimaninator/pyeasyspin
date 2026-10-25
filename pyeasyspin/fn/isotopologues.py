"""Generate the list of isotopologues of a spin system or nuclear list."""
import math
import re
import numpy as np
from .._registry import F
from .._struct import Struct
from .._mat import a2, vec, isempty, numel, issize


def _multisetlist(n, k):
    nSets = math.comb(n + k - 1, k - 1)
    kvec = np.zeros((nSets, k))
    v = np.zeros(k, dtype=int)
    v[0] = n
    idx, j = 0, 1
    while j > 0:
        kvec[idx, :] = v
        idx += 1
        if j < k and v[j - 1] > 0:
            v[j - 1] -= 1
            v[j] += 1
            j += 1
        else:
            s = v[j - 1]
            v[j - 1] = 0
            if s == n:
                break
            while v[j - 1] == 0:
                j -= 1
            v[j - 1] -= 1
            v[j] = s + 1
            j += 1
    mult = np.zeros(nSets)
    for i in range(nSets):
        k_ = []
        for c in range(k):
            k_ += list(range(1, int(kvec[i, c]) + 1))
        mult[i] = np.prod(np.arange(1, n + 1) / np.sort(np.array(k_, float)))
    return kvec, mult


def _traverse(nIso, Abund, thr):
    if thr < 0:
        raise ValueError('Abundance threshold cannot be negative.')
    nN = len(Abund)
    iIso = [0] * nN
    ab = [0.0] * nN
    idxl, abl = [], []
    i = 1
    while i >= 1:
        if i > nN:
            if ab[nN - 1] > thr:
                idxl.append(list(iIso))
                abl.append(ab[nN - 1])
            i -= 1
        else:
            if iIso[i - 1] < nIso[i - 1]:
                iIso[i - 1] += 1
                prev = ab[i - 2] if i > 1 else 1.0
                ab[i - 1] = prev * Abund[i - 1][iIso[i - 1] - 1]
                i += 1
            else:
                iIso[i - 1] = 0
                i -= 1
    return idxl, np.array(abl)


def _structarray(items):
    """MATLAB struct-array semantics: every element gets every field ([] if unset)."""
    keys = []
    for it in items:
        for k in it:
            if k not in keys:
                keys.append(k)
    out = [Struct((k, it.get(k, np.zeros((0, 0)))) for k in keys) for it in items]
    return out[0] if len(out) == 1 else out


def isotopologues(*varargin, nargout=1):
    """Isotopologues of a spin system Sys (with optional relative abundance threshold), or of a
    nuclear list string (with optional nEquiv, Abundances, threshold). Returns a Struct (one
    isotopologue) or a list of Structs with fields Nucs, n, weight (and scaled A, A_, Q, ...)."""
    SysInput = isinstance(varargin[0], dict)
    thr_default = 1e-4
    if SysInput:
        Sys = Struct(varargin[0])
        rel = varargin[1] if len(varargin) == 2 else thr_default
        if len(varargin) > 2:
            raise ValueError('At most two inputs possible when a spin system is given.')
        NucList = Sys.get('Nucs', '')
        nEquiv = Sys.get('n', None)
        Abundances = Sys.get('Abund', None)
    else:
        Sys = None
        NucList = varargin[0]
        nEquiv = varargin[1] if len(varargin) >= 2 else None
        Abundances = varargin[2] if len(varargin) >= 3 else None
        rel = varargin[3] if len(varargin) == 4 else thr_default
        if len(varargin) > 4:
            raise ValueError('At most four inputs are possible.')
    if not isinstance(NucList, str):
        if SysInput:
            raise ValueError("List of nuclei in Sys.Nucs must be a string (e.g. 'Cu,Cl,H,H').")
        raise ValueError("List of nuclei must be a string (e.g. 'Cu,Cl,H,H').")
    if NucList == '':
        if nargout == 0:
            print('Abs.Abund   Rel.Abund   Composition')
            print('1.000       1.000       (no nuclei)')
            return None
        if SysInput:
            if 'weight' not in Sys:
                Sys.weight = 1.0
            return Sys
        return Struct(Nucs='', weight=1.0, n=np.zeros((0, 0)))
    IL = F.nucdata()
    if isinstance(IL, tuple):
        IL = IL[0]
    custom = '(' in NucList
    NucList = F.nucstring2list(NucList, 'm')
    nNucs = len(NucList)
    if custom and isempty(Abundances):
        raise ValueError('For custom isotope mixtures, abundances must be given!')
    if not isinstance(Abundances, list):
        if nNucs == 1:
            Abundances = [Abundances]
        if isempty(Abundances):
            Abundances = [None] * nNucs
    elif len(Abundances) != nNucs:
        raise ValueError('Abundances must be a cell array with one entry (array) per nucleus.')
    if len(Abundances) < nNucs:
        raise ValueError(f'{nNucs} nuclei, but only {len(Abundances)} abundances given. Please specify all abundances!')
    nEquiv = np.ones(nNucs, int) if isempty(nEquiv) else vec(nEquiv).astype(int)
    if nEquiv.size != nNucs:
        if SysInput:
            raise ValueError(f'{nNucs} nuclei given in Sys.Nucs, but {nEquiv.size} numbers in Sys.n.')
        raise ValueError(f'{nNucs} nuclei given, but {nEquiv.size} numbers in nEquiv.')

    if SysInput:
        nE = numel(Sys.S) if 'S' in Sys else 1
        if 'A' in Sys:
            A0 = a2(Sys.A)
            Aiso = A0.size == nNucs if nE == 1 else issize(A0, [nNucs, nE])
            if Aiso:
                Sys.A = np.reshape(vec(A0), (nNucs, nE), order='F')
            A0 = a2(Sys.A)
            Aax, Arh = issize(A0, [nNucs, 2 * nE]), issize(A0, [nNucs, 3 * nE])
            Afull, Aexch = issize(A0, [3 * nNucs, 3 * nE]), A0.shape[0] == nNucs
        if 'A_' in Sys:
            B0 = a2(Sys.A_)
            A_iso = B0.size == nNucs if nE == 1 else issize(B0, [nNucs, nE])
            A_ax, A_rh = issize(B0, [nNucs, 2 * nE]), issize(B0, [nNucs, 3 * nE])
        if 'Q' in Sys:
            Qax = numel(Sys.Q) == nNucs
            if Qax:
                Sys.Q = vec(Sys.Q).reshape(-1, 1)
            Q0 = a2(Sys.Q)
            Qrh, Qpv, Qfull = issize(Q0, [nNucs, 2]), issize(Q0, [nNucs, 3]), issize(Q0, [3 * nNucs, 3])

    Groups = []
    for iN, N in enumerate(NucList):
        if N == '':
            raise ValueError('Missing nucleus in list.')
        if N[0] == '(':
            i = N.index(')')
            El = N[i + 1:]
            Mass = [int(x) for x in re.findall(r'\d+', N[1:i])]
            isoAb = list(vec(Abundances[iN]).astype(float))
        elif re.match(r'\s*[+-]?\d', N) is None:
            El = N
            iidx = [k for k, e in enumerate(IL.Element) if e == El]
            if not iidx:
                raise ValueError(f"Could not find element '{El}' in isotope list.")
            Mass = [int(IL.Nucleons[k, 0]) for k in iidx]
            isoAb = [IL.Abundances[k, 0] / 100 for k in iidx]
            keep = [a != 0 for a in isoAb]
            Mass = [m for m, kp in zip(Mass, keep) if kp]
            isoAb = [a for a, kp in zip(isoAb, keep) if kp]
            if not isempty(Abundances[iN]):
                raise ValueError(f'Nucleus {iN + 1} is a natural-abundance mixture; provide [] in Abundances{{{iN + 1}}}.')
        else:
            m = int(re.match(r'\s*([+-]?\d+)', N).group(1))
            Mass = [m]
            El = N[int(math.floor(math.log10(m) + 1)):]
            isoAb = [1.0]
        G = Struct(Element=El, n=int(nEquiv[iN]))
        G.Isotopes = [f'{Mass[k]}{El}' for k in range(len(isoAb))]
        nd = F.nucdata(G.Isotopes)
        I, gn, qm = list(np.atleast_1d(nd[0])), list(np.atleast_1d(nd[1])), list(np.atleast_1d(nd[2]))
        idx0 = [k for k, v in enumerate(I) if v == 0]
        if idx0:
            isoAb[idx0[0]] = sum(isoAb[k] for k in idx0)
            rm = set(idx0[1:])
            isoAb = [a for k, a in enumerate(isoAb) if k not in rm]
            G.Isotopes = [a for k, a in enumerate(G.Isotopes) if k not in rm]
            I = [a for k, a in enumerate(I) if k not in rm]
            if SysInput:
                gn = [a for k, a in enumerate(gn) if k not in rm]
                qm = [a for k, a in enumerate(qm) if k not in rm]
        G.I = np.array(I, float)
        G.Abundances = np.array(isoAb, float)
        if SysInput:
            ab_ = G.Abundances.copy()
            ab_[G.I < 0.5] = -1
            gnref = gn[int(np.argmax(ab_))] if ab_.max() > 0 else None
            ab_[G.I < 1] = -1
            qmref = qm[int(np.argmax(ab_))] if ab_.max() > 0 else None
            if gnref is None or gnref == 0:
                gnref = 1
            if 'A' in Sys:
                A0 = a2(Sys.A)
                if Aiso or Aax or Arh or Aexch:
                    G.A = [A0[iN, :] / gnref * g for g in gn]
                elif Afull:
                    G.A = [A0[3 * iN:3 * iN + 3, :] / gnref * g for g in gn]
                else:
                    raise ValueError(f'Sys.A size ({A0.shape[0]}x{A0.shape[1]}) is inconsistent with number of '
                                     f'spins ({nE} electrons, {nNucs} nuclei).')
            if 'A_' in Sys:
                B0 = a2(Sys.A_)
                if A_iso:
                    G.A_ = [vec(B0)[iN] / gnref * g for g in gn]
                elif A_ax or A_rh:
                    G.A_ = [B0[iN, :] / gnref * g for g in gn]
                else:
                    raise ValueError(f'Sys.A_ size ({B0.shape[0]}x{B0.shape[1]}) is inconsistent with number of '
                                     f'spins ({nE} electrons, {nNucs} nuclei).')
            if qmref is None or qmref == 0:
                qmref = 1
            if 'Q' in Sys:
                Q0 = a2(Sys.Q)
                if Qax:
                    G.Q = [vec(Q0)[iN] / qmref * q for q in qm]
                elif Qrh or Qpv:
                    G.Q = [Q0[iN, :] / qmref * qm[k] for k in range(len(gn))]
                elif Qfull:
                    G.Q = [Q0[3 * iN:3 * iN + 3, :] / qmref * q for q in qm]
                else:
                    raise ValueError(f'Sys.Q size ({Q0.shape[0]}x{Q0.shape[1]}) is inconsistent with number of '
                                     f'nuclear spins ({nNucs}).')
            if 'AFrame' in Sys:
                if issize(a2(Sys.AFrame), [nNucs, nE * 3]):
                    G.AFrame = a2(Sys.AFrame)[iN, :]
                else:
                    raise ValueError('Sys.AFrame has the wrong size.')
            if 'QFrame' in Sys:
                G.QFrame = a2(Sys.QFrame)[iN, :]
        Groups.append(G)

    Nucs, Abund, nEqL = [], [], []
    for iN in range(nNucs):
        G = Groups[iN]
        nIso = len(G.Isotopes)
        kvec, mult = _multisetlist(int(nEquiv[iN]), nIso)
        if np.any(np.isinf(mult)):
            raise ValueError(f'Multiplicity overflow. - Cannot handle nEquiv={nEquiv[iN]} for nucleus #{iN + 1}.')
        Nl, Al, El = [], [], []
        for im in range(mult.size):
            abu = 1.0
            for ii in range(nIso):
                abu *= G.Abundances[ii] ** kvec[im, ii]
            if abu == 0:
                raise ValueError(f'Abundance underflow. - Cannot handle nEquiv={nEquiv[iN]} for nucleus #{iN + 1}.')
            Nl.append(list(G.Isotopes))
            Al.append(abu * mult[im])
            El.append(kvec[im, :].copy())
        Nucs.append(Nl); Abund.append(Al); nEqL.append(El)
        if SysInput:
            G.kvec = kvec
    maxAb = np.prod([max(a) for a in Abund])
    nIsos = [len(n) for n in Nucs]
    IsoIdx, IsoAb = _traverse(nIsos, Abund, rel * maxAb)
    nIsotopologues = IsoAb.size

    out = []
    for k in range(nIsotopologues):
        iso = Struct(Sys) if SysInput else Struct()
        Nucs_, n = [], []
        A, A_, Q, AFr, QFr = [], [], [], [], []
        for iN in range(nNucs):
            idx = IsoIdx[k][iN] - 1
            gr = Groups[iN]
            for ii in np.nonzero(nEqL[iN][idx] != 0)[0]:
                if gr.I[ii] == 0:
                    continue
                Nucs_.append(Nucs[iN][idx][ii])
                n.append(nEqL[iN][idx][ii])
                if SysInput:
                    if 'A' in Sys and (Aiso or Aax or Arh or Aexch or Afull):
                        A.append(a2(gr.A[ii]))
                    if 'A_' in Sys:
                        if A_iso:
                            A_.append(gr.A_[ii])
                        elif A_ax or A_rh:
                            A_.append(a2(gr.A_[ii]))
                    if 'Q' in Sys and (Qax or Qrh or Qpv or Qfull):
                        Q.append(a2(gr.Q[ii]))
                    if 'AFrame' in Sys:
                        AFr.append(a2(gr.AFrame))
                    if 'QFrame' in Sys:
                        QFr.append(a2(gr.QFrame))
        iso.Nucs = F.nuclist2string(Nucs_)
        iso.n = np.array(n, float).reshape(1, -1) if n else np.zeros((0, 0))
        w = iso.get('weight', None)
        iso.weight = w * IsoAb[k] if (w is not None and not isempty(w)) else IsoAb[k]
        if A:
            iso.A = np.vstack(A)
        if A_:
            iso.A_ = np.array(A_, float).reshape(1, -1) if A_iso else np.vstack(A_)
        if Q:
            iso.Q = np.vstack(Q)
        if AFr:
            iso.AFrame = np.vstack(AFr)
        if QFr:
            iso.QFrame = np.vstack(QFr)
        out.append(iso)
    res = _structarray(out)
    lst = res if isinstance(res, list) else [res]
    for iso in lst:
        if iso.Nucs == '':
            for f in ('A', 'A_', 'Q', 'AFrame', 'QFrame'):
                if f in iso:
                    iso[f] = np.zeros((0, 0))
    if nargout == 0:
        L = max(len(i.Nucs) for i in lst)
        print('Sys.Abund   Sys.Nucs' + ' ' * (L - 4) + 'Sys.n')
        for i in lst:
            print(f'  {i.weight:0.6f}   {i.Nucs}{" " * (L - len(i.Nucs))}    ' +
                  ''.join(f'{int(v):2d} ' for v in vec(i.n)))
        print(f'{nIsotopologues} of {int(np.prod(nIsos))} isotopologues with abundance above threshold ({rel:0.2e})')
        return None
    return res

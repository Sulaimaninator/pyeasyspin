"""Read EPR properties from main ORCA output file"""
import re
import numpy as np
import scipy.linalg
from .._registry import F, nout
from .._struct import Struct
from .._io import sscanf_f, textscan_lines

_EMPTY = lambda: np.zeros((0, 0))
_TRIM = ' \t\n\r\v\f\0'
_INT = re.compile(r'\s*([+-]?\d+)')


def _c(x):
    """Value of a constant function that returns (value, sigma)."""
    return x[0] if isinstance(x, tuple) else x


def _f(s):
    """sscanf(s,'%f ...') as a list of floats (the format repeats until the first mismatch)."""
    return sscanf_f(s)[0]


def _d1(s):
    """sscanf(s,'%d',1)."""
    m = _INT.match(s)
    return float(m.group(1)) if m else np.zeros((0, 0))


def _skip_f(s):
    """sscanf(s,'%*s %f %f %f'): skip the first token, then floats."""
    t = s.lstrip(' \t')
    i = 0
    while i < len(t) and not t[i].isspace():
        i += 1
    return _f(t[i:])


def _tok(s):
    """sscanf(s,'%s',1): first whitespace-delimited token."""
    p = s.split()
    return p[0] if p else ''


def readlines(path):
    """textscan(fh,'%s','Whitespace','','Delimiter','\\n'){1}, decoding UTF-16/UTF-8 files with a
    byte-order mark as MATLAB's fopen does (ORCA 5 outputs redirected in PowerShell are UTF-16)."""
    with open(path, 'rb') as fh:
        b = fh.read()
    if b.startswith(b'\xff\xfe') or b.startswith(b'\xfe\xff'):
        t = b.decode('utf-16')
    elif b.startswith(b'\xef\xbb\xbf'):
        t = b[3:].decode('utf-8', errors='replace')
    else:
        return textscan_lines(path)
    t = t.replace('\r\n', '\n').replace('\r', '\n')
    lines = t.split('\n')
    if lines and lines[-1] == '':
        lines.pop()
    return lines


def readmatrix(L, startidx=1):
    return np.array([_f(L[r][startidx - 1:]) for r in range(3)], dtype=float)


def findheader(header, L, krange):
    for k in krange:
        if L[k].startswith(header):
            return k
    return None


def eigsym(A):
    """MATLAB [V,D] = eig(A) for a real matrix: the symmetric solver (ascending eigenvalues) if A
    is exactly symmetric, otherwise the general one."""
    A = np.asarray(A, dtype=float)
    if np.array_equal(A, A.T):
        d, V = scipy.linalg.eigh(A, driver='ev')
    else:
        d, V = np.linalg.eig(A)
        if np.iscomplexobj(d) and not np.any(np.imag(d)):
            d, V = np.real(d), np.real(V)
    return V, d


def sqrtm_eig(A):
    """MATLAB A^(1/2) (non-integer mpower: V*D.^p/V)."""
    V, d = eigsym(A)
    return np.real_if_close(np.linalg.solve(V.T, (V * np.sqrt(d + 0j)).T).T)


def setrows(M, idx, vals):
    """MATLAB M(idx,1:n) = vals (idx 0-based) with zero-padding growth; M may be None/[]."""
    vals = np.asarray(vals).ravel()
    n = vals.size
    if M is None or np.size(M) == 0:
        M = np.zeros((0, n), dtype=vals.dtype)
    M = np.asarray(M)
    r, c = M.shape
    if idx >= r or n > c:
        G = np.zeros((max(r, idx + 1), max(c, n)), dtype=np.result_type(M, vals))
        G[:r, :c] = M
        M = G
    else:
        M = M.astype(np.result_type(M, vals), copy=True)
    M[idx, :n] = vals
    return M


def setelem(v, idx, val, empty=None):
    """MATLAB v(idx) = val on a row vector (cell when val is not numeric), idx 0-based."""
    if isinstance(val, str) or empty == 'cell':
        v = list(v) if v is not None and not (isinstance(v, np.ndarray) and v.size == 0) else []
        while len(v) <= idx:
            v.append(np.zeros((0, 0)))
        v[idx] = val
        return v
    v = np.zeros((1, 0)) if v is None or np.size(v) == 0 else np.asarray(v, dtype=float).reshape(1, -1)
    if v.shape[1] <= idx:
        v = np.concatenate([v, np.zeros((1, idx + 1 - v.shape[1]))], axis=1)
    v[0, idx] = val
    return v


def structarray(items, order):
    """Make a MATLAB struct array: every element gets every field ([] where unset), in the
    order fields were created. One element comes back as a Struct (MATLAB's 1x1 struct rule)."""
    out = []
    for it in items:
        s = Struct()
        for k in order:
            s[k] = it[k] if k in it else _EMPTY()
        out.append(s)
    return out[0] if len(out) == 1 else out


@nout(2)
def orca2easyspin_maintxt(mainfile=None):
    """(Sys, info) from the main ORCA output file: S, g/gFrame, D/DFrame, Nucs/NucsIdx,
    A/AFrame, Q/QFrame, xyz and the raw parsed data (Sys.data). Several structures (scans)
    give a list of Sys."""
    L = readlines(mainfile)
    L = [x for x in L if len(x.strip(_TRIM)) > 1]      # strtrim(x) longer than 1 char
    nLines = len(L)
    if '* O   R   C   A *' not in L[1]:
        raise ValueError('This is not an ORCA output file.')
    info = Struct()
    vl = [x for x in L if 'Program Version' in x]
    m = re.search(r'\d+\.\d+\.\d+', vl[0])
    info.OrcaVersion = m.group(0) if m else ''
    k = 0
    while L[k][0] != '|':
        k += 1
    startInput = k
    while L[k][0] == '|':
        k += 1
    endInput = k - 2
    inputFile = [re.sub(r'^\|\s*\d+>\s+', '', x, count=1) for x in L[startInput:endInput + 1]]
    info.InputFile = inputFile

    RunType = [('* Single Point Calculation *', ''),
               ('* Multiple XYZ Scan Calculation *', 'MULTIPLE XYZ STEP'),
               ('* Parameter Scan Calculation *', 'TRAJECTORY STEP'),
               ('*    Relaxed Surface Scan    *', 'RELAXED SURFACE SCAN STEP')]
    rtype = 0
    while k < nLines and rtype == 0:
        for iType in range(4):
            if RunType[iType][0] in L[k]:
                rtype = iType + 1
                break
        k += 1
    if k >= nLines:
        raise ValueError('Could not determine type of run (single points, relaxed scan, etc.) from file.')
    multipleStructures = rtype != 1
    if multipleStructures:
        startIdx = [i for i, x in enumerate(L) if RunType[rtype - 1][1] in x]
    else:
        startIdx = [k]
    nStructures = len(startIdx)

    data = []
    nAtoms = 0
    for iStructure in range(nStructures):
        if iStructure < nStructures - 1:
            krange = range(startIdx[iStructure], startIdx[iStructure + 1])
        else:
            krange = range(startIdx[iStructure], nLines)
        D = Struct()
        k = findheader('CARTESIAN COORDINATES (ANGSTROEM)', L, krange)
        k = k + 2
        xyz, Element, NucId = [], [], []
        while L[k][0] != '-':
            xyz.append(_skip_f(L[k]))
            Element.append(_tok(L[k]))
            NucId.append(F.elementsymbol2no(Element[-1]))
            k += 1
        xyz = np.array(xyz, dtype=float)
        nAtoms = xyz.shape[0]
        D.nAtoms = float(nAtoms)
        D.NucId = np.array(NucId, dtype=float).reshape(1, -1)
        D.Element = Element
        D.xyz = xyz

        def lastmatch(pattern):
            kk = None
            for kk in krange:
                if re.match(pattern, L[kk]):
                    break
            return kk

        kk = lastmatch(r'\s*Total Charge')
        mm = re.search(r'\d+$', L[kk])
        D.Charge = float(mm.group(0)) if mm else np.nan
        kk = lastmatch(r'\s*Multiplicity')
        mm = re.search(r'\d+$', L[kk])
        Multiplicity = float(mm.group(0)) if mm else np.nan
        D.Multiplicity = Multiplicity
        D.S = (Multiplicity - 1) / 2

        titles = ('MULLIKEN ATOMIC CHARGES AND SPIN DENSITIES', 'MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS')
        MullikenSection = False
        for kk in krange:
            if L[kk] in titles:
                MullikenSection = True
                break
        if MullikenSection:
            kk += 2
            Mulliken = np.zeros((nAtoms, 2))
            for iAtom in range(nAtoms):
                Mulliken[iAtom, :] = _f(L[kk + iAtom][8:])
        else:
            Mulliken = np.zeros((0, 2))
            print('No Mulliken analysis found for structure %d of %d.' % (iStructure + 1, nStructures))
        D.MullikenCharge = Mulliken[:, 0:1]
        D.MullikenSpin = Mulliken[:, 1:2]

        k = findheader('ELECTRONIC G-MATRIX', L, krange)
        if k is not None:
            while 'The g-matrix' not in L[k]:
                k += 1
            g_raw = readmatrix(L[k + 1:k + 4])
            g_sym = sqrtm_eig(g_raw.T @ g_raw)
            g_sym = (g_sym + g_sym.T) / 2
            V, g = eigsym(g_sym)
            gvals = np.asarray(g).reshape(1, -1)
            if np.linalg.det(V) < 0:
                V[:, 0] = -V[:, 0]
            gFrame = F.eulang(V.T)
        else:
            g_raw = g_sym = gvals = gFrame = _EMPTY()
        D.graw = g_raw
        D.g = g_sym
        D.gvals = gvals
        D.gFrame = gFrame

        k = findheader('ZERO-FIELD-SPLITTING TENSOR', L, krange)
        if k is not None:
            D_raw = readmatrix(L[k + 3:k + 6])
            D_vecs, D_vals = eigsym(D_raw)
            D_vals = np.asarray(D_vals).reshape(1, -1)
            if np.linalg.det(D_vecs) < 0:
                D_vecs[:, 0] = -D_vecs[:, 0]
            cl = F.clight()
            D_raw = D_raw * 100 * cl / 1e6
            D_vals = D_vals * 100 * F.clight() / 1e6
            DFrame = F.eulang(D_vecs.T)
        else:
            D_raw = D_vals = DFrame = _EMPTY()
        D.Draw = D_raw
        D.Dvals = D_vals
        D.DFrame = DFrame

        Araw = [_EMPTY() for _ in range(nAtoms)]
        efg = [_EMPTY() for _ in range(nAtoms)]
        Q = [_EMPTY() for _ in range(nAtoms)]
        QFrame = [_EMPTY() for _ in range(nAtoms)]
        A = [_EMPTY() for _ in range(nAtoms)]
        AFrame = [_EMPTY() for _ in range(nAtoms)]
        k = findheader('ELECTRIC AND MAGNETIC HYPERFINE STRUCTURE', L, krange)
        if k is not None:
            iAtom = None
            qrefEl = None
            kend = krange[-1]
            while k <= kend:
                if re.match(r'\s*Nucleus\s*', L[k]):
                    iAtom = int(_d1(L[k][8:]))          # 0-based (ORCA numbers atoms from 0)
                    _, qrefEl = F.referenceisotope(Element[iAtom])
                elif re.match(r'\s*(Raw HFC matrix|Total HFC matrix)', L[k]):
                    idx = k + 2 if L[k + 1][1:4] == '---' else k + 1
                    Araw[iAtom] = readmatrix(L[idx:idx + 3])
                    idx = idx + 3
                    while L[idx][0:7] != ' A(Tot)':
                        idx += 1
                    A_vals = np.array(_f(L[idx][12:]), dtype=float).reshape(1, -1)
                    R = readmatrix(L[idx + 2:idx + 5], 5)
                    A[iAtom] = A_vals
                    AFrame[iAtom] = F.eulang(R.T)
                    k = k + 5
                elif re.match(r'\s*Raw EFG matrix\s*', L[k]):
                    idx = k + 2 if L[k + 1][1:4] == '---' else k + 1
                    efg[iAtom] = readmatrix(L[idx:idx + 3])
                    if qrefEl is not None and not (isinstance(qrefEl, np.ndarray) and qrefEl.size == 0):
                        efg_ = efg[iAtom] * _c(F.hartree()) / F.echarge() / _c(F.bohrrad()) ** 2
                        R, eq = eigsym(efg_)
                        eq = np.asarray(eq)
                        o = np.argsort(np.abs(eq), kind='stable')
                        eq = eq[o]
                        R = R[:, o]
                        if np.linalg.det(R) < 0:
                            R[:, 0] = -R[:, 0]
                        Qmom = qrefEl['qm'] * F.barn()
                        I = qrefEl['I']
                        e2qQh = F.echarge() * Qmom * eq[2] / F.planck() / 1e6
                        K = e2qQh / (4 * I) / (2 * I - 1)
                        eta = (eq[0] - eq[1]) / eq[2]
                        Q[iAtom] = K * np.array([[-(1 - eta), -(1 + eta), 2]])
                        QFrame[iAtom] = F.eulang(R.T)
                    k = k + 3
                k += 1
        D.hfc = Araw
        D.efg = efg
        D.Q = Q
        D.QFrame = QFrame
        D.A = A
        D.AFrame = AFrame
        data.append(D)

    dataval = data[0] if len(data) == 1 else data
    Sys = [None] * nStructures
    order = []

    def put(s, key, val):
        if key not in order:
            order.append(key)
        s[key] = val

    for iStructure in range(nStructures - 1, -1, -1):
        d = data[iStructure]
        s = Struct()
        if d.xyz.size:
            put(s, 'xyz', d.xyz)
        put(s, 'S', d.S)
        if np.size(d.g):
            put(s, 'g', d.gvals)
        if np.size(d.gFrame):
            put(s, 'gFrame', d.gFrame)
        if np.size(d.Dvals):
            put(s, 'D', d.Dvals)
        if np.size(d.DFrame):
            put(s, 'DFrame', d.DFrame)
        idx = -1
        for iAtom in range(nAtoms):
            if np.size(d.A[iAtom]) or np.size(d.Q[iAtom]):
                idx += 1
                put(s, 'Nucs', setelem(s.get('Nucs'), idx, d.Element[iAtom], 'cell'))
                put(s, 'NucsIdx', setelem(s.get('NucsIdx'), idx, iAtom + 1))
                if np.size(d.A[iAtom]):
                    put(s, 'A', setrows(s.get('A'), idx, d.A[iAtom]))
                    put(s, 'AFrame', setrows(s.get('AFrame'), idx, d.AFrame[iAtom]))
                if np.size(d.Q[iAtom]):
                    put(s, 'Q', setrows(s.get('Q'), idx, d.Q[iAtom]))
                    put(s, 'QFrame', setrows(s.get('QFrame'), idx, d.QFrame[iAtom]))
                elif 'Q' in order:
                    s['Q'] = setrows(s.get('Q'), idx, np.zeros(3))
                    s['QFrame'] = setrows(s.get('QFrame'), idx, np.zeros(3))
        put(s, 'data', dataval)
        Sys[iStructure] = s
    return structarray(Sys, order), info

"""Read EPR properties from a text-based ORCA property file (_property.txt)"""
import re
import warnings
import numpy as np
from .._registry import F
from .._struct import Struct
from .._io import sscanf_f
from .orca2easyspin_maintxt import readlines, eigsym, sqrtm_eig, setrows, setelem, structarray

_EMPTY = lambda: np.zeros((0, 0))


def _readmatrix(lines):
    M = np.array([sscanf_f(lines[r])[0] for r in range(3)], dtype=float)
    return M[:, 1:]


def _readvalue(L):
    i = L.find(':')
    try:
        return float(L[i + 1:].strip())
    except ValueError:
        return np.nan


def _readvec(line):
    if isinstance(line, list):
        line = line[0]
    return np.array(sscanf_f(line)[0], dtype=float).reshape(1, -1)


def orca2easyspin_proptxt(propfilename=None):
    """Sys from an ORCA _property.txt file (S, xyz, g/gFrame, D/DFrame, Nucs, NucsIdx,
    A/AFrame, Q/QFrame, data). Several geometries give a list of Sys. orca2easyspin itself
    rejects these files (EasySpin 6.0.12), so this reader is reached only directly."""
    allLines = readlines(propfilename)
    if '!PROPERTIES!' not in allLines[1]:
        raise ValueError('This is not a valid text-based ORCA property file.')
    if not any('!GEOMETRIES!' in x for x in allLines):
        raise ValueError('Cannot find !GEOMETRIES! section.')
    geometrySection = [i for i, x in enumerate(allLines) if '!GEOMETRY!' in x]
    nStructures = len(geometrySection)
    data = [Struct() for _ in range(nStructures)]
    fields = []

    def put(d, k, v):
        if k not in fields:
            fields.append(k)
        d[k] = v

    nAtoms = 0
    for k, idx0 in enumerate(geometrySection):
        nAtoms = int(_readvalue(allLines[idx0 + 1]))
        put(data[k], 'nAtoms', float(nAtoms))
        idx = idx0 + 3
        xyz, Element = None, []
        for iAtom in range(nAtoms):
            idx += 1
            xyz = setrows(xyz, iAtom, _readvec(allLines[idx][21:]))
            tok = re.search(r'(\d+)\s+([a-zA-Z]+)', allLines[idx][:22])
            Element.append(tok.group(2))
        put(data[k], 'xyz', xyz if xyz is not None else _EMPTY())
        put(data[k], 'Element', Element)

    sections = [i for i, x in enumerate(allLines) if x[:1] == '$']
    for idx0 in sections:
        sectionTitle = allLines[idx0]
        mm = re.search(r'(\d+)$', allLines[idx0 + 2])
        iStructure = int(mm.group(1)) - 1 if mm else None
        d = data[iStructure]
        put(d, 'Nucs', d['Element'])
        if sectionTitle == '$ SCF_Energy':
            put(d, 'E', _readvalue(allLines[idx0 + 4]))
        elif sectionTitle == '$ Calculation_Info':
            multiplicity = _readvalue(allLines[idx0 + 4])
            put(d, 'S', (multiplicity - 1) / 2)
            put(d, 'charge', _readvalue(allLines[idx0 + 5]))
        elif sectionTitle == '$ EPRNMR_GTensor':
            g_raw = _readmatrix(allLines[idx0 + 8:idx0 + 11])
            g_sym = sqrtm_eig(g_raw.T @ g_raw)
            g_sym = (g_sym + g_sym.T) / 2
            g_vecs, g_vals = eigsym(g_sym)
            g_vals = np.asarray(g_vals).reshape(1, -1)
            if np.linalg.det(g_vecs) < 0:
                g_vecs[:, 0] = -g_vecs[:, 0]
            put(d, 'graw', g_raw)
            put(d, 'g', g_vecs)
            put(d, 'gvals', g_vals)
            put(d, 'gFrame', F.eulang(g_vecs.T))
        elif sectionTitle == '$ EPRNMR_DTensor':
            D_raw = _readmatrix(allLines[idx0 + 8:idx0 + 11])
            D_vecs, D_vals = eigsym(D_raw)
            D_vals = np.asarray(D_vals).reshape(1, -1)
            if np.linalg.det(D_vecs) < 0:
                D_vecs[:, 0] = -D_vecs[:, 0]
            D_raw = D_raw * 100 * F.clight() / 1e6
            D_vals = D_vals * 100 * F.clight() / 1e6
            put(d, 'Draw', D_raw)
            put(d, 'Dvals', D_vals)
            put(d, 'DFrame', F.eulang(D_vecs.T))
        elif sectionTitle == '$ EPRNMR_ATensor':
            mm = re.search(r'\d+$', allLines[idx0 + 4])
            nStoredNuclei = int(mm.group(0)) if mm else 0
            A = [_EMPTY() for _ in range(nAtoms)]
            Araw = [_EMPTY() for _ in range(nAtoms)]
            AFrame = [_EMPTY() for _ in range(nAtoms)]
            i = idx0 + 7
            for n in range(nStoredNuclei):
                m = re.search(r'(\d+)\W+(\w+)\W*$', allLines[i])
                if not m:
                    break
                atomno = int(m.group(1))
                A_raw = _readmatrix(allLines[i + 6:i + 9])
                A_vecs = _readmatrix(allLines[i + 11:i + 14])
                A_vals = _readvec(allLines[i + 16][8:])
                Araw[atomno] = A_raw
                A[atomno] = A_vals
                AFrame[atomno] = F.eulang(A_vecs.T) if not np.all(A_vecs == 0) else np.zeros(3)
                i = i + 18
            put(d, 'Araw', Araw)
            put(d, 'A', A)
            put(d, 'AFrame', AFrame)
        elif sectionTitle == '$ EPRNMR_QTensor':
            warnings.warn('EPRNMR_QTensor is not supported for property.txt files. Use the main output file instead.')
        elif sectionTitle == '$ EPRNMR_EFGTensor':
            mm = re.search(r'\d+$', allLines[idx0 + 4])
            nStoredNuclei = int(mm.group(0)) if mm else 0
            rho_present = 'true' in allLines[idx0 + 7]
            dd = 19 if rho_present else 18
            i = idx0 + 8
            Q = [_EMPTY() for _ in range(nAtoms)]
            Qraw = [_EMPTY() for _ in range(nAtoms)]
            QFrame = [_EMPTY() for _ in range(nAtoms)]
            for n in range(nStoredNuclei):
                m = re.search(r'(\d+)\W+(\w+)\W*$', allLines[i])
                atomno = int(m.group(1))
                Q_raw = _readmatrix(allLines[i + 6:i + 9])
                Q_vecs = _readmatrix(allLines[i + 11:i + 14])
                Q_vals = _readvec(allLines[i + 16][11:])
                Qraw[atomno] = Q_raw
                Q[atomno] = Q_vals
                QFrame[atomno] = F.eulang(Q_vecs.T)
                i = i + dd
            put(d, 'Q', Q)
            put(d, 'Qraw', Qraw)
            put(d, 'QFrame', QFrame)

    if 'S' not in fields:
        raise ValueError('Spin multiplicity is missing in property file.')
    data = structarray(data, fields)
    datal = data if isinstance(data, list) else [data]
    Sys = [None] * nStructures
    order = []

    def sput(s, k, v):
        if k not in order:
            order.append(k)
        s[k] = v

    for iStructure in range(nStructures - 1, -1, -1):
        d = datal[iStructure]
        s = Struct()
        if np.size(d['xyz']):
            sput(s, 'xyz', d['xyz'])
        sput(s, 'S', d['S'])
        if 'g' in d:
            if np.size(d['g']):
                sput(s, 'g', d['gvals'])
            if np.size(d['gFrame']):
                sput(s, 'gFrame', d['gFrame'])
        if 'Dvals' in d:
            if np.size(d['Dvals']):
                sput(s, 'D', d['Dvals'])
            if np.size(d['DFrame']):
                sput(s, 'DFrame', d['DFrame'])
        idx = -1
        for iAtom in range(nAtoms):
            Agiven = 'A' in d and isinstance(d['A'], list) and np.size(d['A'][iAtom]) > 0
            Qgiven = 'Q' in d and isinstance(d['Q'], list) and np.size(d['Q'][iAtom]) > 0
            if Agiven or Qgiven:
                idx += 1
                sput(s, 'Nucs', setelem(s.get('Nucs'), idx, d['Element'][iAtom], 'cell'))
                sput(s, 'NucsIdx', setelem(s.get('NucsIdx'), idx, iAtom + 1))
                if Agiven:
                    sput(s, 'A', setrows(s.get('A'), idx, d['A'][iAtom]))
                    sput(s, 'AFrame', setrows(s.get('AFrame'), idx, d['AFrame'][iAtom]))
                if Qgiven:
                    sput(s, 'Q', setrows(s.get('Q'), idx, d['Q'][iAtom]))
                    sput(s, 'QFrame', setrows(s.get('QFrame'), idx, d['QFrame'][iAtom]))
        sput(s, 'data', data)
        Sys[iStructure] = s
    return structarray(Sys, order)

"""Read protein structure file in Gromos87 format (.gro)."""
import numpy as np
from .._struct import Struct

CR = chr(13)


def _floats(s):
    out = []
    for tok in s.split():
        try:
            out.append(float(tok))
        except ValueError:
            break
    return out


def _int(s):
    try:
        return float(int(s.split()[0]))
    except (ValueError, IndexError):
        return np.zeros(0)


def md_readgro(GroFile=None, ResName=None, LabelName=None, AtomNames=None):
    """md_readgro(GroFile, ResName, LabelName, AtomNames) -> data struct with logical masks
    (idx_ProteinCA, idx_SpinLabel, idx_<atom> over the label atoms) and the parsed file
    (title, nAtoms, resnum, resname, atomname, atomnumber, pos, vel, box)."""
    if GroFile[-4:].upper() != '.GRO':
        raise ValueError('File extension must be .gro.')
    try:
        with open(GroFile, 'r', newline='') as fh:
            text = fh.read()
    except OSError:
        raise ValueError(f"Could not open file '{GroFile}'") from None
    allLines = text.split('\n')
    if allLines and allLines[-1] == '':
        allLines = allLines[:-1]
    allLines = [L[:-1] if L.endswith(CR) else L for L in allLines]     # textscan EndOfLine: LF or CRLF
    title = allLines[0]
    nAtoms = int(_int(allLines[1]))
    if len(allLines) != nAtoms + 3:
        raise ValueError(f"Number of lines in file '{GroFile}' appears incorrect.")
    resnum = np.zeros((1, nAtoms))
    resnames = [None] * nAtoms
    atomnames = [None] * nAtoms
    atomnumber = np.zeros((1, nAtoms))
    pos = np.zeros((nAtoms, 3))
    vel = np.zeros((nAtoms, 3))
    for k in range(nAtoms):
        L = allLines[k + 2]
        resnum[0, k] = _int(L[0:5])
        resnames[k] = L[5:10]
        atomnames[k] = L[10:15]
        atomnumber[0, k] = _int(L[15:20])
        pos[k, :] = _floats(L[20:44])[:3]
        if len(L) > 44:
            vel[k, :] = _floats(L[44:68])[:3]
    box = np.array(_floats(allLines[nAtoms + 2])).reshape(-1, 1)
    resnames = [s.strip() for s in resnames]
    atomnames = [s.strip() for s in atomnames]
    data = Struct()
    data.idx_ProteinCA = np.array([[a.lower() == 'ca' for a in atomnames]])
    idx_SpinLabel = np.array([r.lower() == ResName.lower() for r in resnames])
    data.idx_SpinLabel = idx_SpinLabel.reshape(1, -1)
    labelatoms = [a for a, s in zip(atomnames, idx_SpinLabel) if s]
    find = lambda name: np.array([a.lower() == name.lower() for a in labelatoms], dtype=bool).reshape(1, -1)
    if LabelName == 'R1':
        keys = ('ON', 'NN', 'C1', 'C2', 'C1R', 'C2R', 'C1L', 'S1L', 'SG', 'CB', 'CA', 'N')
    elif LabelName == 'TOAC':
        keys = ('ON', 'NN', 'CGS', 'CGR', 'CBS', 'CBR', 'CA', 'N')
    else:
        keys = ()
    for key in keys:
        data['idx_' + key] = find(AtomNames[key + 'name'])
    data.title = title
    data.nAtoms = float(nAtoms)
    data.resnum = resnum
    data.resname = resnames
    data.atomname = atomnames
    data.atomnumber = atomnumber
    data.pos = pos
    data.vel = vel
    data.box = box
    return data

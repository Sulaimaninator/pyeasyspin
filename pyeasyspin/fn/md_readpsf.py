"""Read a PSF (protein structure file, in CHARMM or X-PLOR format)."""
import re
import numpy as np
from .._struct import Struct

# textscan format '%10d %8s %8d %8s %8s %4s %14f%14f%8d %*f %*f': (kind, width); a row ends at the
# end of its line (missing trailing fields are empty), tokens longer than the width are split.
_ATOMFMT = (('d', 10), ('s', 8), ('d', 8), ('s', 8), ('s', 8), ('s', 4), ('f', 14), ('f', 14), ('d', 8),
            ('*f', None), ('*f', None))

_R1 = ('ON', 'NN', 'C1', 'C2', 'C1R', 'C2R', 'C1L', 'S1L', 'SG', 'CB', 'CA', 'N')
_TOAC = ('ON', 'NN', 'CGS', 'CGR', 'CBS', 'CBR', 'CA', 'N')


def _parse_atomline(line):
    toks = line.split()
    out = []
    i = 0
    for kind, w in _ATOMFMT:
        if i >= len(toks):
            out.append(np.nan if kind != 's' else '')
            continue
        t = toks[i]
        if w is not None and len(t) > w:
            toks[i] = t[w:]
            t = t[:w]
        else:
            i += 1
        if kind == 's':
            out.append(t)
        else:
            try:
                out.append(float(t))
            except ValueError:
                out.append(np.nan)
    return out


def md_readpsf(FileName=None, SegName=None, ResName=None, LabelName=None, AtomNames=None):
    """md_readpsf(FileName, SegName, ResName, LabelName, AtomNames) -> psf struct: NATOM, mass,
    idx_ProteinLabel (1-based indices of the segment's atoms), idx_ProteinCA and idx_SpinLabel
    (masks over those atoms), idx_<atom> (masks over the spin-label atoms)."""
    if not isinstance(FileName, str):
        raise ValueError('FileName must be given as a character array.')
    if FileName[-4:].lower() != '.psf':
        raise ValueError('Please give the full filename including the ".psf" extension.')
    if not isinstance(ResName, str):
        raise ValueError('resName must be given as a character array.')
    if any(not isinstance(v, str) for v in AtomNames.values()):
        raise ValueError('All fields in AtomNames must be given as character arrays.')
    need = {'R1': _R1, 'TOAC': _TOAC}.get(LabelName, ())
    if any(k + 'name' not in AtomNames for k in need):
        raise ValueError('One or more required atom names is missing. Please check documentation.')
    try:
        with open(FileName, 'r', newline='') as fh:
            lines = fh.read().split('\n')
    except OSError:
        raise ValueError(f'File "{FileName}" could not be opened.') from None
    lines = [L[:-1] if L.endswith(chr(13)) else L for L in lines]
    if not lines or lines[0][:3].lower() != 'psf':
        raise ValueError(f'File "{FileName}" does not have proper PSF format.')
    psf = Struct()
    psf.NATOM = 0.0
    reachedNTITLE = False
    reachedNATOM = False
    i = 1
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if re.search(r'.*\d.*!\w.*', line):
            parts = line.split()
            nLines = int(round(float(parts[0])))
            section = parts[1]
            if 'NTITLE' in section:
                reachedNTITLE = True
            if 'NATOM' in section:
                if not reachedNTITLE:
                    raise ValueError(f'Section ordering in "{FileName}" is not standard. See documentation for proper formatting.')
                reachedNATOM = True
                psf.NATOM = float(nLines)
                rows = [_parse_atomline(lines[i + k]) for k in range(nLines)]
                i += nLines
                segmentNames = [r[1] for r in rows]
                residueNames = [r[3] for r in rows]
                atomNames = [r[4] for r in rows]
                psf.mass = np.array([r[7] for r in rows], dtype=float).reshape(-1, 1)
                seg = SegName if SegName else segmentNames[0]
                keep = np.array([s.lower() == seg.lower() for s in segmentNames])
                if not np.any(keep):
                    raise ValueError(f"SegName '{seg}' not found.")
                segmentNames = [s for s, k in zip(segmentNames, keep) if k]
                residueNames = [s for s, k in zip(residueNames, keep) if k]
                atomNames = [s for s, k in zip(atomNames, keep) if k]
                psf.idx_ProteinLabel = (np.flatnonzero(keep) + 1).astype(float).reshape(-1, 1)
                segok = np.array([s.lower() == seg.lower() for s in segmentNames])
                psf.idx_ProteinCA = (segok & np.array([a.lower() == 'ca' for a in atomNames])).reshape(-1, 1)
                spin = segok & np.array([r[:4].lower() == ResName[:4].lower() for r in residueNames])   # strncmpi(.,.,4)
                psf.idx_SpinLabel = spin.reshape(-1, 1)
                labelatoms = [a for a, s in zip(atomNames, spin) if s]
                for key in need:
                    nm = AtomNames[key + 'name'].lower()
                    psf['idx_' + key] = np.array([a.lower() == nm for a in labelatoms], dtype=bool).reshape(-1, 1)
                SegName = seg
            if 'NBOND' in section:
                if not reachedNTITLE or not reachedNATOM:
                    raise ValueError(f'Section ordering in "{FileName}" is not standard. See documentation for proper formatting')
                break
    return psf

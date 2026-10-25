"""Import spin Hamiltonian parameters from ORCA"""
import os
import re
import numpy as np
from .._registry import F
from .._struct import Struct
from .._mat import isempty

_SEPS = '/\\' if os.name == 'nt' else '/'


def _fileparts(p):
    """MATLAB fileparts (both / and \\ are separators on Windows)."""
    i = max(p.rfind(s) for s in _SEPS)
    path, name = (p[:i], p[i + 1:]) if i >= 0 else ('', p)
    if i == 0:
        path = p[0]
    j = name.rfind('.')
    if j >= 0:
        return path, name[:j], name[j:]
    return path, name, ''


def _fullfile(*parts):
    """MATLAB fullfile: join with the platform separator, '/' -> filesep on Windows, collapse
    repeated separators."""
    parts = [p for p in parts if p]
    if not parts:
        return ''
    s = os.sep.join(parts)
    if os.name == 'nt':
        s = s.replace('/', '\\')
        lead = '\\\\' if s.startswith('\\\\') else ''
        s = lead + re.sub(r'\\+', r'\\', s[len(lead):])
    else:
        s = re.sub(r'/+', '/', s)
    return s


def _isfile(p):
    return bool(p) and os.path.exists(p)


def orca2easyspin(OrcaOutput=None, HyperfineCutoff=None):
    """Sys from an ORCA calculation: the main output file (text) or the binary .prop file.
    Hyperfine couplings with all |A| principal values <= HyperfineCutoff (MHz, default 0) are
    removed. As in EasySpin 6.0.12, _property.txt files are rejected (the proptxt reader is
    available as F.orca2easyspin_proptxt)."""
    if HyperfineCutoff is None:
        HyperfineCutoff = 0
    output_path, output_name, output_ext = _fileparts(OrcaOutput)
    isproptxt = output_ext == '.txt' and len(output_name) > 9 and output_name[-9:] == '_property'
    if isproptxt:
        raise ValueError('orca2easyspin() currently does not support reading spin Hamiltonian parameters '
                         'from _property.txt files. Provide the main output file instead.')
    if output_ext == '.prop':
        readmode = 'propbin'
        mainOutputFile = _fullfile(output_path, output_name)
        binaryPropFile = _fullfile(output_path, output_name + output_ext)
        textPropFile = ''
    elif isproptxt:
        readmode = 'proptxt'
        mainOutputFile = _fullfile(output_path, output_name[:-9])
        binaryPropFile = ''
        textPropFile = _fullfile(output_path, output_name + output_ext)
    else:
        readmode = 'mainout'
        mainOutputFile = _fullfile(output_path, output_name) + output_ext
        binaryPropFile = _fullfile(output_path, output_name) + output_ext + '.prop'
        textPropFile = _fullfile(output_path, output_name) + '_property.txt'
    if readmode == 'mainout' and not _isfile(mainOutputFile):
        raise ValueError('Cannot access ORCA output file %s.' % mainOutputFile)
    if readmode == 'propbin' and not _isfile(binaryPropFile):
        raise ValueError('Cannot access binary ORCA property file %s.' % binaryPropFile)
    if readmode == 'proptxt' and not _isfile(textPropFile):
        raise ValueError('Cannot access text-based ORCA property file %s.' % textPropFile)
    if readmode == 'proptxt':
        vOrca = _getOrcaVersion(mainOutputFile)
        if vOrca in ('5.0.0', '5.0.1', '5.0.2', '5.0.3'):
            raise ValueError('Cannot read property file for ORCA version %s. Use main output file instead.' % vOrca)
    if readmode == 'mainout':
        Sys = F.orca2easyspin_maintxt(mainOutputFile)
        if isinstance(Sys, tuple):
            Sys = Sys[0]
    elif readmode == 'propbin':
        Sys = F.orca2easyspin_propbin(binaryPropFile)
    else:
        Sys = F.orca2easyspin_proptxt(textPropFile)
    return _nucspinhftrim(Sys, HyperfineCutoff)


def _getOrcaVersion(mainOutputFile):
    OrcaVersion = ''
    if not _isfile(mainOutputFile):
        return OrcaVersion
    with open(mainOutputFile, 'r', errors='replace') as fh:
        for idx, line in enumerate(fh):
            if idx >= 50:
                break
            m = re.search(r'\d+\.\d+\.\d+', line)
            if m:
                return m.group(0)
    return OrcaVersion


def _keep(mask, n):
    """Logical index of length <= n padded with False (MATLAB allows shorter logical masks)."""
    k = np.zeros(n, dtype=bool)
    m = np.asarray(mask, dtype=bool).ravel()
    k[:min(n, m.size)] = m[:n]
    return k


def _nucspinhftrim(Sys, HyperfineCutoff):
    arr = Sys if isinstance(Sys, list) else [Sys]
    if not all(isinstance(s, dict) for s in arr) or not arr:
        return Sys
    if not ('Nucs' in arr[0] and 'A' in arr[0]):
        return Sys
    out = []
    for s in arr:
        s = Struct(s)
        A = np.asarray(s.A)
        Amax = np.max(np.abs(A), axis=1) if A.size else np.zeros(0)
        keep = Amax > abs(HyperfineCutoff)
        if len(arr) > 1:
            raise TypeError('Too many input arguments.')      # isempty(Sys.Nucs) on a struct array
        if 'Nucs' not in s or isempty(s.Nucs):
            out.append(s)
            continue
        Nucs = s.Nucs
        if isinstance(Nucs, str):
            Nucs = F.nucstring2list(Nucs)
        NucsIdx = np.asarray(s.NucsIdx).reshape(1, -1)
        s.NucsIdx = NucsIdx[:, _keep(keep, NucsIdx.shape[1])]
        kn = _keep(keep, len(Nucs))
        s.Nucs = F.nuclist2string([n for n, kk in zip(Nucs, kn) if kk])
        s.A = A[_keep(keep, A.shape[0]), :]
        for fld in ('AFrame', 'Q', 'QFrame'):
            if fld in s:
                M = np.asarray(s[fld])
                if M.ndim == 2 and M.shape[0] > 0:
                    s[fld] = M[_keep(keep, M.shape[0]), :]
                else:
                    s[fld] = M
        out.append(s)
    return out if isinstance(Sys, list) else out[0]

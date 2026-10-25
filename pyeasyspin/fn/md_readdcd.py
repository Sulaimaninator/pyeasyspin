"""Read a binary DCD file obtained from a molecular dynamics simulation (NAMD/CHARMM)"""
import struct
import numpy as np
from .._struct import Struct

# True restores EasySpin's behaviour for DCD files without the CHARMM extra-block flag
# (header.CHARMMextrablock is then never set and MATLAB stops with an error; DEVIATIONS).
EASYSPIN_EXTRABLOCK_FIELD = False


class _Reader:
    def __init__(self, raw):
        self.raw = raw
        self.pos = 0

    def read(self, n, fmt):
        size = struct.calcsize('<' + fmt)
        chunk = self.raw[self.pos:self.pos + n * size]
        k = len(chunk) // size
        self.pos += k * size if k < n else n * size
        if k == 0:
            return np.zeros(0)
        return np.array(struct.unpack('<%d%s' % (k, fmt), chunk[:k * size]), dtype=float)

    def one(self, fmt):
        v = self.read(1, fmt)
        return float(v[0]) if v.size else np.zeros(0)


def md_readdcd(*varargin):
    """md_readdcd(FileName[, idx]) -> Traj (xyz nFrames x 3 x numel(idx), dt (s), nAtoms, nFrames).
    idx = 1-based atom indices (or a logical mask) to keep; default all atoms."""
    nargin = len(varargin)
    FileName = varargin[0]
    idx = varargin[1] if nargin >= 2 else np.zeros((0, 0))
    if not isinstance(FileName, str):
        raise ValueError('Filename must be given as a character array.')
    if FileName[-4:].lower() != '.dcd':
        raise ValueError('Please give the full filename including the ".dcd" extension.')
    idx = np.asarray(idx)
    if idx.dtype == bool:
        idx = np.flatnonzero(idx.ravel(order='F')) + 1
    try:
        with open(FileName, 'rb') as fh:
            raw = fh.read()
    except OSError:
        raise ValueError(f'File "{FileName}" could not be opened.') from None
    fileSize = len(raw)
    r = _Reader(raw)
    header = Struct()
    header.blocksize1 = r.one('i')
    if header.blocksize1 != 84:
        raise ValueError(f'File "{FileName}" does not have proper format.')
    header.HDR = r.read(4, 'b')
    header.NSET = r.one('i')
    header.ISTRT = r.one('i')
    header.NSAVC = r.one('i')
    header.NSTEP = r.one('i')
    header.NULL4 = r.read(4, 'i')
    header.NFREAT = r.one('i')
    header.DELTA = r.one('f')
    header.NULL9 = r.read(9, 'i')
    header.VERSION = r.one('i')
    if not EASYSPIN_EXTRABLOCK_FIELD:
        header.CHARMMextrablock = 0
    if header.VERSION > 0:
        header.isCHARMM = 1
        cof = r.pos
        r.pos = 48
        if r.one('i') == 1:
            header.CHARMMextrablock = 1
        r.pos = 52
        if r.one('i') == 1:
            header.CHARMMextrablock = 1
        r.pos = cof
    else:
        header.isCHARMM = 0
        cof = r.pos
        r.pos = 44
        header.DELTA = r.one('d')
        r.pos = cof
    blocksize1 = r.one('i')
    if header.blocksize1 != blocksize1:
        raise ValueError('Inconsistent sizes for block 1.')
    header.blocksize2 = r.one('i')
    header.NTITLE = r.one('i')
    header.TITLE = r.read(int(80 * header.NTITLE), 'b')
    blocksize2 = r.one('i')
    if header.blocksize2 != blocksize2:
        raise ValueError('Inconsistent sizes for block 2.')
    header.blocksize3 = r.one('i')
    header.NATOM = r.one('i')
    blocksize3 = r.one('i')
    if header.blocksize3 != blocksize3:
        raise ValueError('Inconsistent sizes for block 3. Check the integrity of the DCD file.')
    headersize = r.pos
    if 'CHARMMextrablock' not in header:
        raise ValueError('Reference to non-existent field \'CHARMMextrablock\'.')
    extrablocksize = 4 * 2 + 8 * 6 if header.CHARMMextrablock else 0
    coordblocksize = (4 * 2 + 4 * int(header.NATOM)) * 3
    nSnapShots = np.floor(fileSize - headersize) / (extrablocksize + coordblocksize)
    if nSnapShots % 1:
        raise ValueError('Size inputs must be integers.')
    nSnapShots = int(nSnapShots)
    if idx.size == 0:
        idx = np.arange(1, int(header.NATOM) + 1)
    idx = idx.ravel(order='F').astype(np.int64) - 1
    nI = idx.size
    xyz = np.zeros((nSnapShots, nI * 3))
    for iFrame in range(nSnapShots):
        if header.CHARMMextrablock:
            r.one('i')
            r.read(6, 'd')
            r.one('i')
        bs = r.one('i')
        x = r.read(int(bs / 4), 'f')
        r.one('i')
        bs = r.one('i')
        y = r.read(int(bs / 4), 'f')
        r.one('i')
        bs = r.one('i')
        z = r.read(int(bs / 4), 'f')
        r.one('i')
        xyz[iFrame, 0::3] = x[idx]
        xyz[iFrame, 1::3] = y[idx]
        xyz[iFrame, 2::3] = z[idx]
    Traj = Struct()
    Traj.xyz = xyz.reshape((nSnapShots, 3, nI), order='F')
    timeunit = 48.88821          # isNAMD
    Traj.dt = header.NSAVC * header.DELTA * timeunit * 1e-15
    Traj.nAtoms = header.NATOM
    Traj.nFrames = float(nSnapShots)
    return Traj

"""Read MD trajectory file in Gromacs .TRR format"""
import os
import struct
import numpy as np
from .._registry import F
from .._struct import Struct


def _trr_frames(trajFile):
    """EasySpin's trr2matlab + readGmx2Matlab, in memory: returns (num_atoms, coordinate frames as
    float32 3*num_atoms vectors, time increment as float32). The byte layout is read exactly as
    trr2matlab does (skip 1+51 header bytes, then x/v/f sizes, natoms, step, time, 10 box values)."""
    with open(trajFile, 'rb') as fh:
        raw = fh.read()
    pt = struct.unpack('>9i', raw[:36])
    fileType = 'd' if pt[8] == 72 else 'f'
    fs = struct.calcsize('>' + fileType)
    pos = 1
    coords, coord_time = [], {}
    frame_num = 1
    num_atoms = 0
    while True:
        pos += 51
        data_present = struct.unpack('>3i', raw[pos:pos + 12]); pos += 12
        num_atoms = struct.unpack('>i', raw[pos:pos + 4])[0]; pos += 4
        pos += 8                                                     # frame_step
        frame_time = struct.unpack('>' + fileType, raw[pos:pos + fs])[0]; pos += fs
        pos += 10 * fs                                               # box_params
        n3 = num_atoms * 3
        if data_present[0]:
            coord_time[frame_num] = frame_time
            coords.append(np.array(struct.unpack('>%d%s' % (n3, fileType), raw[pos:pos + n3 * fs]), dtype=np.float32))
            pos += n3 * fs
        if data_present[1]:
            pos += n3 * fs
        if data_present[2]:
            pos += n3 * fs
        frame_num += 1
        if pos >= len(raw):           # the trailing fread(1,'*char') hits the end of the file
            break
        pos += 1
    coord_increment = np.float32(coord_time.get(2, 0.0) - coord_time.get(1, 0.0))
    return num_atoms, coords, coord_increment


def md_readtrr(trrfile=None):
    """md_readtrr(trrfile) -> Traj (xyz nFrames x 3 x nAtoms, nAtoms, nFrames, dt (s)). Coordinates
    pass through single precision as in EasySpin (which writes them to a temporary binary file);
    no temporary files are written here."""
    if not os.path.exists(trrfile):
        raise ValueError(f"File '{trrfile}' does not exist.")
    F.logmsg(1, 'Reading trajectory file %s... \n', trrfile)
    num_atoms, coords, dtstep = _trr_frames(trrfile)
    nFrames = len(coords)
    raw = np.concatenate(coords).astype(float) if coords else np.zeros(0)
    xyz = np.transpose(raw.reshape((3, num_atoms, nFrames), order='F'), (2, 0, 1))
    Traj = Struct()
    Traj.xyz = xyz
    Traj.nAtoms = float(xyz.shape[2])
    Traj.nFrames = float(xyz.shape[0])
    Traj.dt = float(dtstep) * 1e-12
    return Traj

"""Save a binary DCD file containing processed molecular dynamics trajectory data"""
import datetime
import getpass
import os
import struct
import numpy as np
from .._struct import Struct

TIMEFACTOR = 48.88821


def md_savedcd(FileName=None, Traj=None, Opt=None):
    """md_savedcd(FileName, Traj, Opt=None): write Traj.xyz (nSteps x 3 x nAtoms, Angstrom) and Traj.dt
    (s) as an X-PLOR-style DCD file (VERSION 0, NSAVC 1, DELTA = dt/(48.88821 fs) stored as float64 at
    byte 44, two REMARKS title lines), readable by md_readdcd. Opt.overwrite (default 0) allows
    replacing an existing file.

    EasySpin 6.0.12's md_savedcd cannot write anything: it uses the undefined variables trj, filename,
    fileSize and idx, deletes a file literally named 'FileName' and then runs DCD-reader code on the
    write-only handle. This port writes the file that its header block describes (DEVIATIONS)."""
    if Traj is None:
        raise ValueError('Wrong number of input arguments.')
    Opt = Struct() if Opt is None else Struct(Opt)
    if not isinstance(FileName, str):
        raise ValueError('Filename must be given as a character array.')
    if not FileName.lower().endswith('.dcd'):
        raise ValueError('Please give the full filename including the ".dcd" extension.')
    if not isinstance(Traj, dict):
        raise ValueError('Traj must be a structured array.')
    if 'overwrite' not in Opt:
        Opt.overwrite = 0
    if os.path.exists(FileName) and Opt.overwrite == 0:
        raise ValueError(f'{FileName} already exists. Choose a different filename or enable overwriting.')
    xyz = np.asarray(Traj['xyz'], dtype=float)
    if xyz.ndim == 2:
        xyz = xyz.reshape(xyz.shape[0], 3, -1, order='F')
    nframe, _, natom = xyz.shape
    dt = float(Traj['dt']) if 'dt' in Traj else TIMEFACTOR * 1e-15
    title1 = ('REMARKS FILENAME=%s CREATED BY MATLAB' % FileName).ljust(80)[:80]
    user = os.environ.get('USER') or os.environ.get('USERNAME') or ''
    title2 = ('REMARKS DATE: %s CREATED BY USER: %s' % (datetime.date.today().strftime('%m/%d/%y'), user)).ljust(80)[:80]
    h = bytearray()
    h += struct.pack('<i', 84)
    h += b'CORD'
    h += struct.pack('<4i', nframe, 0, 1, 0)                 # NSET, ISTRT, NSAVC, NSTEP
    h += struct.pack('<4i', 0, 0, 0, 0)                      # NULL4
    h += struct.pack('<i', 0)                                # NFREAT
    h += struct.pack('<d', dt / (TIMEFACTOR * 1e-15))        # DELTA (float64 at byte 44, X-PLOR)
    h += struct.pack('<8i', *([0] * 8))                      # rest of NULL9
    h += struct.pack('<i', 0)                                # VERSION
    h += struct.pack('<i', 84)
    h += struct.pack('<2i', 164, 2) + (title1 + title2).encode('latin-1') + struct.pack('<i', 164)
    h += struct.pack('<3i', 4, natom, 4)
    with open(FileName, 'wb') as fh:
        fh.write(bytes(h))
        nb = 4 * natom
        for k in range(nframe):
            for c in range(3):
                fh.write(struct.pack('<i', nb))
                fh.write(xyz[k, c, :].astype('<f4').tobytes())
                fh.write(struct.pack('<i', nb))
    return None

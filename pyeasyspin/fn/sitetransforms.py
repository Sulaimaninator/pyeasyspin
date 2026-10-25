"""Site transformation matrices (Laue class rotations) for space and point groups."""
import os
import numpy as np
from .._mat import vec, numel

_SG = None
_SQ = np.sqrt(3) / 2


def _spacegroups():
    global _SG
    if _SG is None:
        f = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'spacegroups.txt')
        No, Names, Axes = [], [], []
        with open(f) as fh:
            for line in fh:
                p = line.split('%', 1)[0].split()
                if len(p) >= 3:
                    No.append(float(p[0])); Names.append(p[1]); Axes.append(p[2])
        _SG = (No, Names, Axes)
    return _SG


def _laue(n):
    for lim, c in ((2, 1), (15, 2), (74, 3), (88, 4), (142, 5), (148, 6), (167, 7), (176, 8), (194, 9),
                   (206, 10), (230, 11)):
        if n <= lim:
            return c


_PGS = ['C1', 'Ci', 'C2', 'Cs', 'C2h', 'D2', 'C2v', 'D2h', 'C4', 'S4', 'C4h', 'D4', 'C4v', 'D2d', 'D4h', 'C3',
        'C3i', 'D3', 'C3v', 'D3d', 'C6', 'C3h', 'C6h', 'D6', 'C6v', 'D3h', 'D6h', 'T', 'Th', 'O', 'Td', 'Oh']
_PGHM = ['1', '-1', '2', 'm', '2/m', '222', 'mm2', 'mmm', '4', '-4', '4/m', '422', '4mm', '-42m', '4/mmm', '3', '-3',
         '32', '3m', '-3m', '6', '-6', '6/m', '622', '6mm', '-6m2', '6/mmm', '23', 'm-3', '432', '-43m', 'm-3m']
_LC = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 9, 10, 10, 11, 11, 11]

M = lambda *r: np.array(r, dtype=float)
E = M([1, 0, 0], [0, 1, 0], [0, 0, 1])
C2x = M([1, 0, 0], [0, -1, 0], [0, 0, -1])
C2y = M([-1, 0, 0], [0, 1, 0], [0, 0, -1])
C2z = M([-1, 0, 0], [0, -1, 0], [0, 0, 1])
C4zp = M([0, -1, 0], [1, 0, 0], [0, 0, 1])
C4zm = M([0, 1, 0], [-1, 0, 0], [0, 0, 1])
C2xy1 = M([0, 1, 0], [1, 0, 0], [0, 0, -1])
C2xy2 = M([0, -1, 0], [-1, 0, 0], [0, 0, -1])
C3zp = M([-.5, -_SQ, 0], [_SQ, -.5, 0], [0, 0, 1])
C3zm = M([-.5, _SQ, 0], [-_SQ, -.5, 0], [0, 0, 1])
C6p = M([.5, -_SQ, 0], [_SQ, .5, 0], [0, 0, 1])
C6m = M([.5, _SQ, 0], [-_SQ, .5, 0], [0, 0, 1])
D3a = M([-.5, _SQ, 0], [_SQ, .5, 0], [0, 0, -1])
D3b = M([-.5, -_SQ, 0], [-_SQ, .5, 0], [0, 0, -1])
D6a = M([.5, _SQ, 0], [_SQ, -.5, 0], [0, 0, -1])
D6b = M([.5, -_SQ, 0], [-_SQ, -.5, 0], [0, 0, -1])
CUB = [M([0, 0, 1], [1, 0, 0], [0, 1, 0]), M([0, 1, 0], [0, 0, 1], [1, 0, 0]), M([0, 0, -1], [-1, 0, 0], [0, 1, 0]),
       M([0, -1, 0], [0, 0, 1], [-1, 0, 0]), M([0, 0, 1], [-1, 0, 0], [0, -1, 0]), M([0, -1, 0], [0, 0, -1], [1, 0, 0]),
       M([0, 0, -1], [1, 0, 0], [0, -1, 0]), M([0, 1, 0], [0, 0, -1], [-1, 0, 0])]
OCT = [M([1, 0, 0], [0, 0, -1], [0, 1, 0]), M([1, 0, 0], [0, 0, 1], [0, -1, 0]), M([0, 0, 1], [0, 1, 0], [-1, 0, 0]),
       M([0, 0, -1], [0, 1, 0], [1, 0, 0]), M([0, 0, 1], [0, -1, 0], [1, 0, 0]), M([0, 0, -1], [0, -1, 0], [-1, 0, 0]),
       M([-1, 0, 0], [0, 0, 1], [0, 1, 0]), M([-1, 0, 0], [0, 0, -1], [0, -1, 0])]


def sitetransforms(ID=None, vec_=None):
    """List of site rotation matrices for a space group (number or H-M symbol) or point group
    (Schoenflies or H-M); with a 3-vector, returns the 3 x nSites array of rotated vectors."""
    transformVector = vec_ is not None
    if transformVector and numel(vec_) != 3:
        raise ValueError('Vector (2nd argument) must have 3 elements.')
    No, Names, Axes = _spacegroups()
    if isinstance(ID, str):
        if ID in Names:
            i = Names.index(ID)
            LaueClass, Axis = _laue(No[i]), Axes[i]
        else:
            if ID in _PGS:
                i = _PGS.index(ID)
            elif ID in _PGHM:
                i = _PGHM.index(ID)
            else:
                raise ValueError(f"Point or space group symmetry symbol '{ID}' is unknown.")
            LaueClass, Axis = _LC[i], 'default'
    else:
        if not isinstance(ID, (int, float, np.number)) and not (isinstance(ID, np.ndarray) and ID.dtype.kind in 'iuf'):
            raise ValueError('Space group ID must be a number or a string/character array.')
        elif numel(ID) != 1:
            raise ValueError('Space group ID must be a single number, not an array.')
        ID = float(np.asarray(ID).ravel()[0])
        if ID % 1 != 0 or ID < 1 or ID > 230:
            raise ValueError(f'Invalid space group number {ID:g}. Must be between 1 and 230.')
        LaueClass, Axis = _laue(ID), 'default'
    if Axis == 'default':
        Axis = 'b' if LaueClass == 2 else 'z'
    if LaueClass == 1:
        R = [E]
    elif LaueClass == 2:
        if Axis in ('b', '-b'):
            R = [E, C2y]
        elif Axis in ('a', '-a'):
            R = [E, C2x]
        elif Axis in ('c', '-c', 'z'):
            R = [E, C2z]
        else:
            raise ValueError('Unknown unique axis for this monoclinic space group or crystallographic point group.')
    elif LaueClass == 3:
        R = [E, C2z, C2x, C2y]
    elif LaueClass == 4:
        R = [E, C2z, C4zp, C4zm]
    elif LaueClass == 5:
        R = [E, C2z, C4zp, C4zm, C2x, C2y, C2xy1, C2xy2]
    elif LaueClass == 6:
        R = [E, C3zp, C3zm]
    elif LaueClass == 7:
        R = [E, C3zp, C3zm, C2x, D3a, D3b]
    elif LaueClass == 8:
        R = [E, C2z, C3zp, C3zm, C6p, C6m]
    elif LaueClass == 9:
        R = [E, C2z, C3zp, C3zm, C6p, C6m, C2x, C2y, D3a, D3b, D6a, D6b]
    elif LaueClass == 10:
        R = [E, C2z, C2x, C2y] + CUB
    else:
        R = [E, C2z, C4zp, C4zm, C2x, C2y, C2xy1, C2xy2] + CUB + OCT
    if transformVector:
        v = vec(vec_).astype(float)
        return np.column_stack([r @ v for r in R])
    return [r.copy() for r in R]

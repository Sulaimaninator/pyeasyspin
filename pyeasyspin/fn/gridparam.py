"""Get grid parameter for a given point group symmetry"""
import numpy as np
from .._registry import nout

_maxPhi = np.array([2, 2, 1, 2 / 3, 1 / 2, 1 / 3, 1 / 2, 1 / 2, 1 / 3, 1 / 4, 1 / 4, 1 / 6, 0, 0]) * np.pi
_nOctants = [8, 4, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1]
_closedPhi = [False] * 6 + [True] * 8
_Groups = ['C1', 'Ci', 'C2h', 'S6', 'C4h', 'C6h', 'D2h', 'Th', 'D3d', 'D4h', 'Oh', 'D6h', 'Dinfh', 'O3']


@nout(3)
def gridparam(PointGroup=None, nargout=None):
    """gridparam(PointGroup) -> (maxPhi, closedPhi, nOctants)."""
    if PointGroup not in _Groups:
        raise ValueError(f'Unsupported symmetry group {PointGroup}!')
    i = _Groups.index(PointGroup)
    out = (float(_maxPhi[i]), _closedPhi[i], _nOctants[i])
    if nargout is not None and 0 < nargout < 3:
        return out[0] if nargout == 1 else out[:nargout]
    return out

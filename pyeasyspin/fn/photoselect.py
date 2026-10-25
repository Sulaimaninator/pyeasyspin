"""Orientation-dependent photoselection weights."""
import numpy as np

from .._registry import F
from .._mat import a2, vec, numel
from .p_excitationgeometry import _kori


def photoselect(tdm=None, ori=None, k=None, alpha=None):
    """Weights (1 x nOri) for transition dipole tdm (molecular frame), orientations ori
    ([phi theta] rows: chi integrated; [phi theta chi]: explicit), light propagation k and
    polarization angle alpha (NaN = unpolarized)."""
    if isinstance(tdm, str):
        tdm_mol = vec(np.asarray(F.letter2vec(tdm), dtype=float))
    elif numel(tdm) == 3:
        t = vec(np.asarray(tdm, dtype=float))
        tdm_mol = t / np.linalg.norm(t)
    elif numel(tdm) == 2:
        t = vec(np.asarray(tdm, dtype=float))
        tdm_mol = vec(np.asarray(F.ang2vec(t[0], t[1]), dtype=float))
    else:
        raise ValueError('tdm (first input) must be a letter designating a direction, a 3-vector, or an array with '
                         'two angles.')
    ori = a2(np.asarray(ori, dtype=float))
    if ori.shape[1] == 2:
        integrateChi = True
    elif ori.shape[1] == 3:
        integrateChi = False
    else:
        raise ValueError('ori (second input) must consist of an array with two or three Euler angles on each row '
                         '([phi theta] or [phi theta chi]).')
    kOri = _kori(k, 'third input')
    a = np.asarray(alpha)
    if a.size != 1 or a.dtype.kind not in 'f':
        raise ValueError('Polarization angle alpha (4th input) must be a scalar.')
    alpha = float(a.ravel()[0])
    unpolarized = np.isnan(alpha)
    if unpolarized:
        alpha = 0.0
    p_lab, _, k_lab = (vec(np.asarray(x, dtype=float)) for x in
                       F.erot(np.array([kOri[0], kOri[1], alpha]), 'rows', nargout=3))
    weight = np.zeros((1, ori.shape[0]))
    for iOri in range(ori.shape[0]):
        if integrateChi:
            tdm_lab = np.asarray(F.erot(np.array([ori[iOri, 0], ori[iOri, 1], 0.0])), dtype=float) @ tdm_mol
            if unpolarized:
                weight_k = ((tdm_lab[0] ** 2 + tdm_lab[1] ** 2) * (k_lab[0] ** 2 + k_lab[1] ** 2) / 2
                            + tdm_lab[2] ** 2 * k_lab[2] ** 2)
                weight[0, iOri] = (1 - weight_k) / 2
            else:
                weight[0, iOri] = ((tdm_lab[0] ** 2 + tdm_lab[1] ** 2) * (p_lab[0] ** 2 + p_lab[1] ** 2) / 2
                                   + tdm_lab[2] ** 2 * p_lab[2] ** 2)
        else:
            tdm_lab = np.asarray(F.erot(ori[iOri, :]), dtype=float) @ tdm_mol
            if unpolarized:
                weight[0, iOri] = (1 - abs(tdm_lab @ k_lab) ** 2) / 2
            else:
                weight[0, iOri] = abs(tdm_lab @ p_lab) ** 2
    return weight

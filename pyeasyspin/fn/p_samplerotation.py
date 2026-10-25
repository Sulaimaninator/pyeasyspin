"""Lab-to-sample rotation matrices from Exp.SampleFrame and Exp.SampleRotation."""
import numpy as np

from .._registry import F, nout
from .._mat import a2, vec, isempty, numel


@nout(2)
def p_samplerotation(Exp=None):
    """(R_L2S, rotatedSample): list of 3x3 lab-to-sample rotation matrices (one per sample
    orientation and rotation angle) and whether any of them is a rotation."""
    if not isempty(Exp.SampleFrame):
        SF = a2(np.asarray(Exp.SampleFrame, dtype=float))
        nSamples = SF.shape[0]
        F.logmsg(1, '  %d sample orientation(s) given in Exp.SampleFrame', nSamples)
        R_L2S = [np.asarray(F.erot(SF[s, :]), dtype=float) for s in range(nSamples)]
    else:
        R_L2S = [np.eye(3)]
        nSamples = 1
    SR = Exp.SampleRotation
    if not isempty(SR):
        if not isinstance(SR, (list, tuple)) or len(SR) != 2:
            raise ValueError('Exp.SampleRotation must be of the form {nL,rho}, with the lab-frame rotation axis nL '
                             'and the rotation angle rho.')
        rotaxis, rho = SR
        if isinstance(rotaxis, str):
            rotaxis = F.letter2vec(rotaxis)
        elif numel(rotaxis) != 3:
            raise ValueError('Rotation axis (first element in Exp.SampleRotation) must be a 3-element vector or a '
                             'letter code.')
        rho = vec(np.asarray(rho, dtype=float))
        Rrot = [None] * rho.size
        for r in range(rho.size - 1, -1, -1):
            Rrot[r] = np.asarray(F.rotaxi2mat(rotaxis, rho[r]), dtype=float)
        R_L2S = [R_L2S[s] @ Rrot[r] for s in range(nSamples) for r in range(rho.size)]
    angles = np.zeros((len(R_L2S), 3))
    for idx in range(len(R_L2S) - 1, -1, -1):
        angles[idx, :] = vec(np.asarray(F.eulang(R_L2S[idx]), dtype=float))
    return R_L2S, bool(np.any(angles != 0))

"""Molecular-to-lab Euler angles for all sample orientations and crystal sites."""
import numpy as np

from .._registry import F, nout
from .._struct import Struct
from .._mat import a2, vec, isempty


@nout(4)
def p_crystalorientations(Exp=None, Opt=None):
    """(angles_M2L (n x 3), nOrientations, nSites, averageOverChi)."""
    MF = np.asarray(Exp.MolFrame, dtype=float)
    if (MF.shape[1] if MF.ndim == 2 else (3 if MF.size == 3 else MF.size)) != 3:
        raise ValueError('Exp.MolFrame requires three numbers per row (three Euler angles).')
    Exp, Opt = Struct(Exp), Struct(Opt)          # MATLAB modifies local copies only
    pepperCall = 'rotatedSample' in Opt
    if not pepperCall:
        Opt.R_L2S, Opt.rotatedSample = F.p_samplerotation(Exp)[:2]
        Opt.crystalSample = True
    if bool(np.all(np.asarray(Opt.crystalSample) != 0)):
        if isempty(Exp.CrystalSymmetry):
            Exp.CrystalSymmetry = 'P1'
            F.logmsg(1, '  no crystal symmetry given: assuming space group P1 (#1)')
        Rsite_C = list(F.sitetransforms(Exp.CrystalSymmetry))
        nSites = len(Rsite_C)
        F.logmsg(1, '  crystal symmetry: %d magnetically distinct sites in unit cell', nSites)
        if 'Sites' in Opt and not isempty(Opt.Sites):
            sites = vec(np.asarray(Opt.Sites, dtype=float))
            if np.any(sites < 1) or np.any(sites > nSites):
                raise ValueError(f'For the given crystal symmetry, Opt.Sites must contain numbers between 1 and {nSites}.')
            Rsite_C = [Rsite_C[int(s) - 1] for s in sites]
            nSites = len(Rsite_C)
        if not isempty(Exp.MolFrame):
            F.logmsg(1, '  molecular frame given in Exp.MolFrame')
            R_S2M = np.asarray(F.erot(Exp.MolFrame), dtype=float)
        else:
            R_S2M = np.eye(3)
        xyzM0_S = R_S2M.T @ np.eye(3)
        xyzM_S = [np.asarray(Rsite_C[i], dtype=float) @ xyzM0_S for i in range(nSites)]
        R_L2S = Opt.R_L2S if isinstance(Opt.R_L2S, list) else [Opt.R_L2S]
        nSampleOrientations = len(R_L2S)
        angles_M2L = np.zeros((nSampleOrientations * nSites, 3))
        idx = 0
        for iSampleOri in range(nSampleOrientations):
            R_S2L_ = np.asarray(R_L2S[iSampleOri], dtype=float).T
            for iSite in range(nSites):
                angles_M2L[idx, :] = vec(np.asarray(F.eulang(R_S2L_ @ xyzM_S[iSite], True), dtype=float))
                idx += 1
        nOrientations = nSampleOrientations * nSites
        averageOverChi = False
    else:
        angles_M2L = -np.fliplr(a2(MF))
        nOrientations = angles_M2L.shape[0]
        averageOverChi = True
        nSites = 1
        if not isempty(Exp.CrystalSymmetry):
            F.logmsg(1, '  powder simulation; disregarding Exp.CrystalSymmetry')
        if not isempty(Exp.MolFrame):
            F.logmsg(1, '  powder simulation; disregarding Exp.MolFrame')
    return angles_M2L, float(nOrientations), float(nSites), averageOverChi

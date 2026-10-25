"""Orientation grid (powders) or sample orientations (crystals) for pepper and salt."""
import numpy as np

from .._registry import F, nout
from .._struct import Struct
from .._mat import a2, vec, isempty, numel


def _t(x):
    a = np.asarray(x)
    return a.size > 0 and bool(np.all(a != 0))


@nout(3)
def p_gridsetup(Sys=None, Exp=None, Opt=None):
    """(Exp, Opt, nOrientations): for disordered/partially ordered samples the sphgrid orientations
    (Exp.phi, Exp.theta, Exp.MolFrame, Exp.OriWeights, Exp.tri, Opt.GridParams, Opt.nOctants);
    for crystals the number of sample orientations."""
    Exp, Opt = Struct(Exp), Struct(Opt)
    F.logmsg(1, '-grid/orientations-------------------------------------')
    useGrid = _t(Opt.disorderedSample) or _t(Opt.partiallyOrderedSample)
    if useGrid:
        F.logmsg(1, '  disordered sample (randomly oriented centers)')
        if isempty(Opt.GridSymmetry):
            if _t(Opt.disorderedSample):
                Opt.GridSymmetry, Opt.GridFrame = F.hamsymm(Sys)[:2]
                nonEquiPops = 'initState' in Sys and not isempty(Sys.initState)
                if nonEquiPops and Opt.GridSymmetry == 'Dinfh':
                    F.logmsg(1, '  Hamiltonian symmetry is axial (Dinfh), non-equilibrium state given in Sys.initState\\n'
                                '   -> reducing symmetry to rhombic (D2h).')
                    Opt.GridSymmetry = 'D2h'
            elif isempty(Opt.GridSymmetry):
                Opt.GridSymmetry = 'Ci'
            msg = '  grid: automatically determined grid symmetry'
        else:
            msg = '  grid: user-specified grid symmetry'
        F.logmsg(1, msg)
        if _t(Opt.partiallyOrderedSample) and Opt.GridSymmetry != 'Ci':
            raise ValueError('For partially ordered samples, Ci grid symmetry is required.')
        if isempty(Opt.GridFrame):
            Opt.GridFrame = np.eye(3)
            msg = '  grid: default grid frame orientation'
        else:
            msg = '  grid: user-specified grid frame orientation'
        F.logmsg(1, msg)
        GridFrame = np.asarray(Opt.GridFrame, dtype=float)
        tiltedFrame = np.sum(np.diag(GridFrame)) != 3
        F.logmsg(1, '  grid: symmetry %s, %s frame', Opt.GridSymmetry, 'tilted' if tiltedFrame else 'non-tilted')
        if tiltedFrame:
            ang = vec(np.asarray(F.eulang(GridFrame, 1), dtype=float)) * 180 / np.pi
            F.logmsg(1, '  grid: frame orientation in molecular frame (Euler angles, deg):\\n    [%s]',
                     ''.join('%0.3f ' % a for a in ang))
        GridSize = vec(np.asarray(Opt.GridSize, dtype=float))
        grid, tri = F.sphgrid(Opt.GridSymmetry, GridSize[0], nargout=2)[:2]
        Opt.nOctants = grid.nOctants
        Exp.OriWeights = grid.weights
        Exp.tri = tri
        phi, theta = F.vec2ang(GridFrame @ np.asarray(grid.vecs, dtype=float))[:2]
        Exp.phi = np.asarray(phi, dtype=float).reshape(1, -1)
        Exp.theta = np.asarray(theta, dtype=float).reshape(1, -1)
        chi = np.zeros(Exp.theta.shape)
        Exp.MolFrame = np.vstack([-chi, -Exp.theta, -Exp.phi]).T
        nOrientations = Exp.phi.size
        nOct = int(np.asarray(Opt.nOctants).ravel()[0])
        Opt.GridParams = np.array([[nOrientations, GridSize[0], float(np.asarray(grid.closedPhi).ravel()[0]),
                                    float(nOct), float(np.asarray(grid.maxPhi).ravel()[0])]])
        maxPhi = float(np.asarray(grid.maxPhi).ravel()[0])
        names = {-1: 'north pole (single point)', 0: 'a quarter of a meridian', 4: 'upper hemisphere',
                 2: 'half of upper hemisphere', 8: 'full sphere'}
        if nOct == 1:
            s = '%d octant of the upper hemisphere, maxPhi = %g*pi' % (nOct, maxPhi / np.pi)
        elif nOct in names:
            s = names[nOct]
        else:
            raise ValueError(f'Unsupported number {nOct} of grid octants.')
        F.logmsg(1, '  integration region: ' + s)
        if nOrientations == 1:
            F.logmsg(1, '  1 orientation (GridSize = %d)', GridSize[0])
        else:
            F.logmsg(1, '  %d orientations (GridSize = %d)', nOrientations, GridSize[0])
    else:
        SF = a2(np.asarray(Exp.SampleFrame, dtype=float))
        nC1, nC2 = SF.shape
        if nC2 != 3:
            raise ValueError(f'Exp.SampleFrame must be a Nx3 array, yours is {nC1}x{nC2}.')
        nOrientations = nC1
        if 'SampleRotation' in Exp and not isempty(Exp.SampleRotation):
            nOrientations *= numel(Exp.SampleRotation[1])
        Opt.nOctants = -2.0
        Exp.OriWeights = np.ones((1, nOrientations)) * 4 * np.pi
        if nOrientations == 1:
            F.logmsg(1, '  crystal sample, single orientation')
        else:
            F.logmsg(1, '  crystal sample, %d orientations', nOrientations)
    return Exp, Opt, float(nOrientations)

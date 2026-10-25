"""Convert MD trajectory to orientational potential"""
import numpy as np
from .._registry import F, nout
from .._matlab import linspace, smooth3_gaussian as _smooth3_gaussian


@nout(2)
def mdtraj2oripot(FrameTraj=None, nBins=None):
    """mdtraj2oripot(FrameTraj[, nBins=90]) -> (potential, pdf): Euler-angle histogram of the frame
    trajectory (3x3xnSteps), smoothed with smooth3 'gaussian' [3 3 3], floored at 1e-14;
    potential = -log(pdf)."""
    if nBins is None:
        nBins = 90
    nBins = np.atleast_1d(np.asarray(nBins, dtype=float)).ravel(order='F')
    if nBins.size == 1:
        nBins = np.array([nBins[0], nBins[0] / 2, nBins[0]])
    if nBins.size != 3:
        raise ValueError('Second input (nBins) must be a single number of an array of three numbers.')
    qTemp = np.squeeze(F.rotmat2quat(FrameTraj))
    phi, theta, psi = F.quat2euler(qTemp, 'active', nargout=3)
    phi = np.mod(phi, 2 * np.pi)
    psi = np.mod(psi, 2 * np.pi)
    EulerTraj = np.column_stack([np.ravel(phi, order='F'), np.ravel(theta, order='F'), np.ravel(psi, order='F')])
    edges = [linspace(0, 2 * np.pi, nBins[0] + 1), linspace(0, np.pi, nBins[1] + 1), linspace(0, 2 * np.pi, nBins[2] + 1)]
    pdf = F.histcountsn(EulerTraj, edges)
    pdf = pdf[0] if isinstance(pdf, tuple) else pdf
    pdf = _smooth3_gaussian(pdf, (3, 3, 3))
    thr = 1e-14
    pdf[pdf < thr] = thr
    potential = -np.log(pdf)
    return potential, pdf

"""Propagate rotational trajectories of orientational quaternions (Euler-Maruyama or
Leimkuhler-Matthews steps with random angular increments)"""
import numpy as np
from .._registry import F
from .._mat import isempty

# EasySpin's Leimkuhler-Matthews steps average consecutive draws of randn(3,nTraj,nSteps), which leaves
# nSteps-1 steps for nSteps-1 updates plus the step index nSteps -> MATLAB stops with an index error at
# the last step. The port draws nSteps+1 normals. True restores EasySpin (DEVIATIONS).
EASYSPIN_LM_PAGES = False


def gen_rand_steps(Sim, rng=None):
    """EasySpin's genRandSteps: Gaussian angular steps sqrt(2*Diff*dt)*randn (3 x nTraj x nSteps;
    one page fewer for Leimkuhler-Matthews). Draws come from rng (numpy Generator/RandomState,
    default the global numpy stream); MATLAB's randn sequence is not reproduced."""
    rng = np.random if rng is None else rng
    nTraj, nSteps = int(Sim['nTraj']), int(Sim['nSteps'])
    LM = Sim['Integrator'] == 'Leimkuhler-Matthews' and not EASYSPIN_LM_PAGES
    randns = rng.standard_normal((3, nTraj, nSteps + 1 if LM else nSteps))
    Diff = np.asarray(Sim['Diff'], dtype=float).reshape(-1, 1, 1)
    sc = np.sqrt(2 * Diff * Sim['dt'])
    if Sim['Integrator'] == 'Euler-Maruyama':
        randns *= sc
        return randns
    if Sim['Integrator'] == 'Leimkuhler-Matthews':
        return (randns[:, :, :-1] + randns[:, :, 1:]) / 2 * sc
    raise ValueError("Unrecognized function or variable 'randAngStep'.")


def stochtraj_proprottraj(q=None, Sim=None, iter=None, rng=None, randAngSteps=None):
    """stochtraj_proprottraj(q, Sim, iter) -> q (4 x nTraj x nSteps). Random: the angular steps are
    drawn by gen_rand_steps from rng; randAngSteps (3 x nTraj x nSteps) can be passed instead
    to supply the draws. Numeric potentials use Sim['interpGrad'] = three callables taking an N x 3 array
    of (alpha, beta, gamma)."""
    nSteps = int(Sim['nSteps'])
    nTraj = int(Sim['nTraj'])
    dt = Sim['dt']
    Diff = np.asarray(Sim['Diff'], dtype=float).reshape(-1, 1)
    lam = Sim['lambda'] if 'lambda' in Sim else None
    LMK = Sim['LMK'] if 'LMK' in Sim else None
    interpGrad = Sim['interpGrad'] if 'interpGrad' in Sim else []
    isEigenPot = not isempty(lam)
    isNumericPot = not isempty(interpGrad)
    startStep = 1 if iter > 1 else 2
    if randAngSteps is None:
        randAngSteps = gen_rand_steps(Sim, rng)
    q = np.array(q, dtype=float)
    if q.ndim == 2:
        q = q.reshape(4, q.shape[1], 1) if nSteps == 1 else q.reshape(4, nTraj, -1)
    for iStep in range(startStep, nSteps + 1):
        if iStep == 1:
            qLast = q[:, :, -1].copy()
            q = np.zeros((4, nTraj, nSteps))
        else:
            qLast = q[:, :, iStep - 2]
        if iStep > randAngSteps.shape[2]:
            raise IndexError(f'Index in position 3 exceeds array bounds. Index must not exceed {randAngSteps.shape[2]}.')
        currRandAngStep = randAngSteps[:, :, iStep - 1]
        if isEigenPot:
            torque = F.stochtraj_calcanistorque(LMK, lam, qLast)
            AngStep = torque * (Diff * dt) + currRandAngStep
        elif isNumericPot:
            alpha, beta, gamma = F.quat2euler(qLast, 'active', nargout=3)
            pts = np.column_stack([np.ravel(alpha), np.ravel(beta), np.ravel(gamma)])
            pxint = np.ravel(interpGrad[0](pts))
            pyint = np.ravel(interpGrad[1](pts))
            pzint = np.ravel(interpGrad[2](pts))
            torque = np.vstack([-pxint, -pyint, -pzint])
            AngStep = torque * (Diff * dt) + currRandAngStep
        else:
            AngStep = currRandAngStep
        theta = np.sqrt(np.sum(AngStep ** 2, axis=0))
        with np.errstate(invalid='ignore', divide='ignore'):
            ux = AngStep[0] / theta
            uy = AngStep[1] / theta
            uz = AngStep[2] / theta
        st = np.sin(theta / 2)
        ct = np.cos(theta / 2)
        q1, q2, q3, q4 = qLast[0], qLast[1], qLast[2], qLast[3]
        q[0, :, iStep - 1] = q1 * ct - q2 * ux * st - q3 * uy * st - q4 * uz * st
        q[1, :, iStep - 1] = q2 * ct + q1 * ux * st - q4 * uy * st + q3 * uz * st
        q[2, :, iStep - 1] = q3 * ct + q4 * ux * st + q1 * uy * st - q2 * uz * st
        q[3, :, iStep - 1] = q4 * ct - q3 * ux * st + q2 * uy * st + q1 * uz * st
    return q

"""Generate stochastic trajectories of Markovian jumps using kinetic Monte Carlo"""
import math
import warnings
import numpy as np
import scipy.linalg
from .._registry import F, nout
from .._struct import Struct
from .._matlab import linspace


def _isvector(a):
    s = a.shape if a.ndim >= 2 else (1, a.size)
    return len(s) == 2 and (s[0] == 1 or s[1] == 1)


def _randi(rng, imax, size):
    if hasattr(rng, 'integers'):
        return rng.integers(1, int(imax) + 1, size)
    return rng.randint(1, int(imax) + 1, size)


@nout(-1)
def stochtraj_jump(Sys=None, Par=None, Opt=None, nargout=None, rng=None, u=None):
    """stochtraj_jump(Sys, Par=None, Opt=None, nargout=3) -> nargout 2: (t, stateTraj) if Opt.statesOnly
    else (t, RTraj); 3: (t, RTraj, qTraj); 4: (t, RTraj, qTraj, stateTraj). Random: start states
    (randi) and the uniform jump draws come from rng (numpy Generator/RandomState, default the global
    numpy stream). u (nSteps x nTraj) supplies the uniform draws. nargout=0 computes and returns
    None (no plot)."""
    if Sys is None:
        return None
    rng = np.random if rng is None else rng
    Sys = Struct(Sys)
    Par = Struct() if Par is None else Struct(Par)
    Opt = Struct() if Opt is None else Struct(Opt)
    if nargout is None:
        nargout = 3
    if nargout not in (0, 2, 3, 4):
        raise ValueError('Incorrect number of output arguments.')
    if 'Verbosity' not in Opt:
        Opt.Verbosity = 0
    if 'statesOnly' not in Opt:
        Opt.statesOnly = False or nargout == 0
    from ..config import config
    config.LogLevel = Opt.Verbosity
    if 'Potential' in Sys:
        warnings.warn('Sys.Potential is given. This field is not used by stochtraj_jump.')
    if 'dt' not in Par:
        raise ValueError('The time step Par.dt must be specified.')
    tcorr = None
    if 'TransRates' in Sys:
        TRM = np.atleast_2d(np.asarray(Sys.TransRates, dtype=float))
        if TRM.ndim != 2 or TRM.shape[0] != TRM.shape[1]:
            raise ValueError('Sys.TransRates must be a square matrix.')
        if np.any(np.diag(TRM) > 0) or np.any((TRM - np.diag(np.diag(TRM))) < 0):
            raise ValueError('Sys.TransRates must contain strictly negative diagonal and positive off-diagonal entries.')
        if np.any(np.abs(np.sum(TRM, axis=1) / np.max(np.abs(TRM))) > 1e-13):
            raise ValueError('In Sys.TransRates, the sum over each row must be zero.')
        TPM = scipy.linalg.expm(Par.dt * TRM)
        D = np.linalg.eigvals(TRM)
        idx = np.abs(D) / np.max(np.abs(D)) > 1e-11
        tcorr = -1 / D[idx]
    elif 'TransProb' in Sys:
        TPM = np.atleast_2d(np.asarray(Sys.TransProb, dtype=float))
        if TPM.ndim != 2 or TPM.shape[0] != TPM.shape[1]:
            raise ValueError('Sys.TransProb must be a square matrix.')
        if np.any(TPM < 0):
            raise ValueError('All elements in Sys.TransProp must positive (or zero).')
        rs = np.sum(TPM, axis=1)
        if np.any(np.abs(rs[rs != 0] - 1) > 1e-12):
            raise ValueError('The rows of Sys.TransProb must sum to 1.')
    else:
        raise ValueError('A transition rate matrix (Sys.TransRates) or a transition probability matrix (Sys.TransProb) '
                         'is required.')
    nStates = TPM.shape[0]
    cumulTPM = np.cumsum(TPM, axis=1)
    if not Opt.statesOnly:
        if 'Orientations' in Sys:
            Orientations = np.atleast_2d(np.asarray(Sys.Orientations, dtype=float))
            if Orientations.shape == (3, nStates):
                pass
            elif Orientations.shape == (nStates, 3):
                Orientations = Orientations.T
            else:
                raise ValueError('The size of Sys.Orientations must be (3,nStates) or (nStates,3).')
        else:
            raise ValueError(f'Orientations for the {nStates} states are required. Give a {nStates}x3 array in '
                             f'Sys.Orientations.\nIf only states are desired, use Opt.statesOnly = 1.')
    if 'nSteps' in Par and 'dt' in Par:
        dt = Par.dt
        nSteps = Par.nSteps
    elif 'tMax' in Par and 'dt' in Par:
        dt = Par.dt
        if dt <= 0:
            raise ValueError('Par.dt must be positive.')
        nSteps = math.ceil(Par.tMax / dt)
    else:
        dt = Par.dt
        if 'TransRates' in Sys:
            nSteps = math.ceil(200 * float(np.max(np.real(tcorr))) / dt)
        else:
            nSteps = 500
        F.logmsg(0, '-- Number of time steps not given. Using %d steps.', nSteps)
    Par.nSteps = int(nSteps)
    Par.dt = dt
    if 'Integrator' not in Par:
        Par.Integrator = 'Euler-Maruyama'
    if Par.Integrator not in ('Euler-Maruyama', 'Leimkuhler-Matthews'):
        raise ValueError('Input for integrator method not recognized.')
    if 'nTraj' not in Par:
        Par.nTraj = 1
    nTraj = int(Par.nTraj)
    if 'StatesStart' in Par:
        StatesStart = np.asarray(Par.StatesStart, dtype=float)
        if not _isvector(StatesStart) or StatesStart.size != nTraj:
            raise ValueError('Par.StatesStart should be of size (1,1) or (1,Par.nTraj).')
        if np.any(StatesStart < 1) or np.any(StatesStart > nStates):
            raise ValueError('Each entry in Par.StatesStart needs to be equal to an integer between 1 and nStates.')
        StatesStart = StatesStart.ravel(order='F')
    else:
        StatesStart = _randi(rng, nStates, nTraj).astype(float)
    if 'checkConvergence' in Opt:
        if Opt.checkConvergence and nTraj == 1:
            raise ValueError('Checking convergence of a single trajectory using the R statistic is not supported.\n')
    nSteps = Par.nSteps
    statesOnly = Opt.statesOnly
    if not statesOnly:
        qStates = np.asarray(F.euler2quat(Orientations)).reshape(4, -1, order='F')
        qTraj = np.zeros((4, nSteps, nTraj))
        for iTraj in range(nTraj):
            qTraj[:, 0, iTraj] = qStates[:, int(StatesStart[iTraj]) - 1]
    F.logmsg(2, '-- Calculating stochastic trajectories -----------------------')
    stateTraj = np.zeros((nSteps, nTraj))
    stateTraj[0, :] = StatesStart
    # MATLAB draws u = rand(1,nSteps) per trajectory (same stream order as rng.random((nTraj, nSteps)))
    U = rng.random((nTraj, nSteps)).T if u is None else np.asarray(u, dtype=float).reshape(nSteps, nTraj, order='F')
    state = stateTraj[0, :].astype(np.int64)
    for iStep in range(1, nSteps):
        # state = find(cumulTPM(state,:) > u(iStep-1), 1), for all trajectories at once
        k = np.sum(cumulTPM[state - 1] <= U[iStep - 1][:, None], axis=1)
        if np.any(k >= nStates):
            raise IndexError('Unable to perform assignment because the left and right sides have a different number of elements.')
        state = k + 1
        stateTraj[iStep, :] = state
    if not statesOnly:
        for iTraj in range(nTraj):
            qTraj[:, 1:, iTraj] = qStates[:, stateTraj[1:, iTraj].astype(int) - 1]
    t = linspace(0, nSteps * Par.dt, nSteps).reshape(-1, 1)
    F.logmsg(2, '-- Propagation finished --------------------------------------')
    F.logmsg(2, '--------------------------------------------------------------')
    config.LogLevel = None
    if nargout == 0:
        return None
    if nargout == 2:
        if Opt.statesOnly:
            return t, stateTraj
        return t, F.quat2rotmat(_mdrop(qTraj))
    RTraj = F.quat2rotmat(_mdrop(qTraj))
    if nargout == 3:
        return t, RTraj, _mdrop(qTraj)
    return t, RTraj, _mdrop(qTraj), stateTraj


def _mdrop(a):
    while a.ndim > 2 and a.shape[-1] == 1:
        a = a[..., 0]
    return a

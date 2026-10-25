"""Generate stochastic rotational trajectories"""
import math
import numpy as np
from .._registry import F, nout
from .._struct import Struct
from .._mat import isempty
from .._matlab import linspace, griddedinterpolant_linear as _interpolant

EPS = np.finfo(float).eps

# True restores EasySpin's start-orientation sampling for Sys.Potential = [L M K lambda] rows,
# which omits the sin(beta) volume element (DEVIATIONS).
EASYSPIN_NO_SINBETA = False


def _mdrop(a):
    """MATLAB drops trailing singleton dimensions beyond the second."""
    while a.ndim > 2 and a.shape[-1] == 1:
        a = a[..., 0]
    return a


def _isvector(a):
    s = a.shape if a.ndim >= 2 else (1, a.size)
    return len(s) == 2 and (s[0] == 1 or s[1] == 1)


def gradient_euler(Data, aGrid, bGrid, gGrid):
    """EasySpin's gradient of a potential on an Euler-angle grid (body-frame torque components)."""
    aGrid, bGrid, gGrid = (np.ravel(np.asarray(g, dtype=float)) for g in (aGrid, bGrid, gGrid))
    Data = np.asarray(Data, dtype=float)
    if Data.ndim != 3:
        raise ValueError('Expected a 3-dimensional array for input data.')
    if Data.shape != (aGrid.size, bGrid.size, gGrid.size):
        raise ValueError('Dimensions of input data array must match the sizes of the grids.')
    _, BGrid, GGrid = np.meshgrid(aGrid, bGrid, gGrid, indexing='ij')
    dAlpha, dBeta, dGamma = aGrid[1] - aGrid[0], bGrid[1] - bGrid[0], gGrid[1] - gGrid[0]
    da = np.zeros(Data.shape)
    da[1:-1] = Data[2:] - Data[:-2]
    da[0] = 4 * Data[1] - 3 * Data[0] - Data[2]
    da[-1] = 3 * Data[-1] + Data[-3] - 4 * Data[-2]
    da = da / 2 / dAlpha
    db = np.zeros(Data.shape)
    db[:, 1:-1] = Data[:, 2:] - Data[:, :-2]
    db[:, 0] = 4 * Data[:, 1] - 3 * Data[:, 0] - Data[:, 2]
    db[:, -1] = 3 * Data[:, -1] + Data[:, -3] - 4 * Data[:, -2]
    db = db / 2 / dBeta
    dg = np.zeros(Data.shape)
    dg[:, :, 1:-1] = Data[:, :, 2:] - Data[:, :, :-2]
    dg[:, :, 0] = 4 * Data[:, :, 1] - 3 * Data[:, :, 0] - Data[:, :, 2]
    dg[:, :, -1] = 3 * Data[:, :, -1] + Data[:, :, -3] - 4 * Data[:, :, -2]
    dg = dg / 2 / dGamma
    csc = 1 / np.sin(BGrid)
    cot = 1 / np.tan(BGrid)
    dx = -csc * np.cos(GGrid) * da + np.sin(GGrid) * db + cot * np.cos(GGrid) * dg
    dy = csc * np.sin(GGrid) * da + np.cos(GGrid) * db - cot * np.sin(GGrid) * dg
    dz = dg
    return dx, dy, dz


@nout(-1)
def stochtraj_diffusion(Sys=None, Par=None, Opt=None, nargout=None, rng=None):
    """stochtraj_diffusion(Sys, Par=None, Opt=None, nargout=3) -> (t, RTraj, qTraj) (nargout 2: (t, RTraj)).
    t is nSteps x 1, RTraj 3x3xnStepsxnTraj, qTraj 4xnStepsxnTraj (trailing singletons dropped as in
    MATLAB). Random: steps come from rng (numpy Generator/RandomState, default the global numpy
    stream); nargout=0 computes the trajectories and returns None (no plot)."""
    if Sys is None:
        return None
    Par = Struct() if Par is None else Struct(Par)
    Opt = Struct() if Opt is None else Struct(Opt)
    Sys = Struct(Sys)
    if nargout is None:
        nargout = 3
    if nargout not in (0, 2, 3):
        raise ValueError('Incorrect number of output arguments.')
    if 'Verbosity' not in Opt:
        Opt.Verbosity = 0
    from ..config import config
    config.LogLevel = Opt.Verbosity
    Dynamics, Sim = F.validate_dynord('stochtraj_diffusion', Sys)
    Sim = Struct(Sim)
    Sim.Diff = Dynamics.Diff
    tcorr = 1 / 6 / np.asarray(Dynamics.Diff, dtype=float)
    if 'Potential' not in Sys:
        Sys.Potential = np.zeros((0, 0))
    Pot = Sys.Potential
    isUserPotFun = callable(Pot) or (isinstance(Pot, np.ndarray) and Pot.ndim == 3)
    if isUserPotFun:
        PseudoPotFun = Pot
        if not callable(PseudoPotFun):
            PseudoPotFun = np.asarray(PseudoPotFun, dtype=float)
            if np.any(np.isnan(PseudoPotFun)):
                raise ValueError('NaN detected in Sys.Potential.')
            if np.any(np.isinf(PseudoPotFun)):
                raise ValueError('Inf detected in Sys.Potential.')
            nGrid = np.array(PseudoPotFun.shape) + np.array([0, 2, 0])
        else:
            import inspect
            if len(inspect.signature(PseudoPotFun).parameters) != 3:
                raise ValueError('The function provided in Sys.Potential must accept 3 inputs.')
            nGrid = np.array([91, 46, 91])
        alphaGrid = linspace(0, 2 * np.pi, nGrid[0])
        betaGrid = linspace(0, np.pi, nGrid[1])[1:-1]
        gammaGrid = linspace(0, 2 * np.pi, nGrid[2])
        if callable(PseudoPotFun):
            AG, BG, GG = np.meshgrid(alphaGrid, betaGrid, gammaGrid, indexing='ij')
            PseudoPotFun = np.asarray(PseudoPotFun(AG, BG, GG), dtype=float)
        dx, dy, dz = gradient_euler(PseudoPotFun, alphaGrid, betaGrid, gammaGrid)
        grids = (alphaGrid, betaGrid, gammaGrid)
        Sim.interpGrad = [_interpolant(grids, dx), _interpolant(grids, dy), _interpolant(grids, dz)]
    else:
        Pot = np.asarray(Pot)
        if Pot.size:
            Pot = np.atleast_2d(Pot)
            if Pot.shape[1] != 4:
                raise ValueError('Sys.Potential need four numbers (L, M, K, lambda) per row.')
            Lvals, Mvals, Kvals, lam = np.real(Pot[:, 0]), np.real(Pot[:, 1]), np.real(Pot[:, 2]), Pot[:, 3]
            if np.any(Lvals < 1):
                raise ValueError('In Sys.Potential, all values of L must be greater than or equal to one.')
            if np.any(np.abs(Mvals) > Lvals) or np.any(Mvals < 0):
                raise ValueError('In Sys.Potential, all values of M must be between 0 and L.')
            if np.any(np.abs(Kvals) > Lvals):
                raise ValueError('In Sys.Potential, all values of K must be between -L and L.')
            Mzero = Mvals == 0
            if np.any(Kvals[Mzero] < 0):
                raise ValueError('In Sys.Potential, for M = 0 the values of K must be between 0 and L.')
            L00 = (Mvals == 0) & (Kvals == 0)
            if np.iscomplexobj(lam) and np.any(np.imag(lam[L00]) != 0):
                raise ValueError('In Sys.Potential, for M = K = 0 lambda must be real-valued.')
            Sim.LMK = np.real(Pot[:, 0:3]).astype(float)
            l4 = Pot[:, 3:4]
            Sim['lambda'] = l4 if (np.iscomplexobj(l4) and np.any(np.imag(l4) != 0)) else np.real(l4).astype(float)
        else:
            Sim.LMK = np.zeros((0, 0))
            Sys['lambda'] = np.zeros((0, 0))    # EasySpin sets Sys.lambda here (Sim.lambda stays [])
    if 't' in Par:
        t = np.ravel(np.asarray(Par.t, dtype=float), order='F')
        nSteps = t.size
        dt = t[1] - t[0]
        if abs(dt - np.max(t) / nSteps) > EPS:
            raise ValueError('t is not linearly spaced.')
    elif 'nSteps' in Par and 'dt' in Par:
        dt = Par.dt
        nSteps = Par.nSteps
    elif 'nSteps' in Par and 'tMax' in Par:
        nSteps = Par.nSteps
        dt = Par.tMax / nSteps
    elif 'tMax' in Par and 'dt' in Par:
        dt = Par.dt
        if dt <= 0:
            raise ValueError('Par.dt must be positive.')
        nSteps = math.ceil(Par.tMax / dt)
    else:
        dt = float(np.min(tcorr)) / 10
        F.logmsg(0, '-- No time parameters given. Using time step of %0.5g s.', dt)
        if 'nSteps' in Par:
            nSteps = Par.nSteps
        else:
            nSteps = math.ceil(200 * float(np.max(tcorr)) / dt)
            F.logmsg(0, '-- Number of time steps not given. Using %d steps.', nSteps)
    nSteps = int(nSteps)
    Sim.nSteps = nSteps
    Sim.dt = dt
    if 'Integrator' not in Par:
        Par.Integrator = 'Euler-Maruyama'
    if Par.Integrator not in ('Euler-Maruyama', 'Leimkuhler-Matthews'):
        raise ValueError('Input for integrator method not recognized.')
    Sim.Integrator = Par.Integrator
    if 'nTraj' not in Par:
        Par.nTraj = 1
    Sim.nTraj = int(Par.nTraj)
    OriStart = Par.OriStart if 'OriStart' in Par else np.zeros((0, 0))
    potEmpty = (not callable(Sys.Potential)) and np.size(Sys.Potential) == 0
    if isempty(OriStart):
        if not potEmpty:
            if isUserPotFun:
                pdf = np.exp(-PseudoPotFun) * np.sin(betaGrid).reshape(1, -1, 1)
                a_, b_, g_ = F.cardamom_rejectionsample3d(pdf, alphaGrid.reshape(1, -1), betaGrid.reshape(1, -1),
                                                          gammaGrid.reshape(1, -1), Sim.nTraj, rng=rng)
                OriStart = np.vstack([np.ravel(a_), np.ravel(b_), np.ravel(g_)])
            else:
                N = 100
                aG = linspace(0, 2 * np.pi, N + 1)
                bG = linspace(0, np.pi, N / 2 + 1)
                gG = linspace(0, 2 * np.pi, N + 1)
                AG, BG, GG = np.meshgrid(aG, bG, gG, indexing='ij')
                U = 0
                P = np.atleast_2d(np.asarray(Sys.Potential))
                for iTerm in range(P.shape[0]):
                    L, M, K = (float(np.real(P[iTerm, c])) for c in range(3))
                    lam = P[iTerm, 3]
                    if M == 0 and K == 0:
                        Yfun = np.real(F.wignerd(np.array([[L, M, K]]), AG, BG, GG))
                    else:
                        Yfun = 2 * np.real(F.wignerd(np.array([[L, M, K]]), AG, BG, GG))
                    U = U + -lam * Yfun
                # EasySpin samples exp(-U) here, without the Euler-angle volume element sin(beta) that the
                # numeric-potential branch above includes; fixed (DEVIATIONS).
                pdfU = np.exp(-U) if EASYSPIN_NO_SINBETA else np.exp(-U) * np.sin(BG)
                a_, b_, g_ = F.cardamom_rejectionsample3d(pdfU, aG.reshape(1, -1), bG.reshape(1, -1),
                                                          gG.reshape(1, -1), Par.nTraj, rng=rng)
                OriStart = np.vstack([np.ravel(a_), np.ravel(b_), np.ravel(g_)])
        else:
            gridPts = linspace(-1, 1, Sim.nTraj)
            gridTheta = np.arccos(gridPts)
            gridPhi = np.sqrt(np.pi * Sim.nTraj) * np.arcsin(gridPts)
            gridPsi = np.sqrt(np.pi * Sim.nTraj) * np.arcsin(gridPts)
            OriStart = np.vstack([gridPhi, gridTheta, gridPsi])
    OriStart = np.asarray(OriStart, dtype=float)
    if OriStart.ndim < 2:
        OriStart = OriStart.reshape(-1, 1)
    if _isvector(OriStart):
        OriStart = OriStart.reshape(-1, 1, order='F')
    if OriStart.shape[1] == 1 and Sim.nTraj > 1:
        OriStart = np.tile(OriStart, (1, Sim.nTraj))
    if OriStart.shape[1] != Sim.nTraj:
        raise ValueError('Number of starting orientations must be equal to Par.nTraj.')
    if OriStart.shape[0] == 3:
        q0 = F.euler2quat(OriStart, 'active')
    elif OriStart.shape[0] == 4:
        q0 = OriStart
    else:
        raise ValueError('The size of OriStart must be (3,1) or (3,nTraj) for Euler angles, or (4,1) or (4,nTraj) for quaternions.')
    qTraj = np.zeros((4, Sim.nTraj, Sim.nSteps))
    qTraj[:, :, 0] = np.asarray(q0).reshape(4, Sim.nTraj, order='F')
    if 'convTolerance' not in Opt:
        Opt.convTolerance = 1e-6
    if 'checkConvergence' in Opt:
        checkConvergence = Opt.checkConvergence
        if checkConvergence and Sim.nTraj == 1:
            raise ValueError('Checking convergence of a single trajectory using the R statistic is not supported.')
    else:
        checkConvergence = False
    converged = False
    it = 1
    F.logmsg(2, '-- Calculating stochastic trajectories -----------------------')
    while not converged:
        if it == 1:
            qTraj = F.stochtraj_proprottraj(qTraj, Sim, it, rng=rng)
        else:
            F.logmsg(3, '-- Convergence not obtained -------------------------------------')
            F.logmsg(3, '-- Propagation extended to %dth iteration -----------------------', it)
            Sim.nSteps = int(max(np.max(np.ceil(np.ravel(tcorr) / Sim.dt)), math.ceil(1.2 * Sim.nSteps)))
            qTraj = F.stochtraj_proprottraj(qTraj, Sim, it, rng=rng)
        if checkConvergence:
            gr = F.stochtraj_grstat(qTraj)
            converged = bool(np.all(np.ravel(gr) < 1 + Opt.convTolerance))
        else:
            converged = True
        it += 1
        if it > 15 and not converged:
            F.logmsg(1, 'Warning: restarting trajectory set due to lack of convergence.\n'
                        'Consider increasing length or number of trajectories.\n')
            it = 1
            Sim.nSteps = nSteps
            q0 = F.euler2quat(OriStart, 'active')
            qTraj = np.zeros((4, Sim.nTraj, Sim.nSteps))
            qTraj[:, :, 0] = np.asarray(q0).reshape(4, Sim.nTraj, order='F')
    qTraj = np.asarray(qTraj)
    if qTraj.ndim == 2:
        qTraj = qTraj[:, :, None]
    totSteps = qTraj.shape[2]
    qTraj = _mdrop(np.transpose(qTraj, (0, 2, 1)))
    t = linspace(0, totSteps * Sim.dt, totSteps).reshape(-1, 1)
    RTraj = F.quat2rotmat(qTraj)
    F.logmsg(2, '-- Propagation finished --------------------------------------')
    F.logmsg(2, '--------------------------------------------------------------')
    config.LogLevel = None
    if nargout == 0:
        return None
    if nargout == 2:
        return t, RTraj
    return t, RTraj, qTraj

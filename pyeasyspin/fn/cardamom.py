"""Trajectory-based simulation of CW-EPR spectra."""
import math
import numpy as np
from .._registry import F, nout
from .._struct import Struct
from .._mat import isempty
from .._matlab import linspace, colon, interp1, round_half_away, cumtrapz, cumtrapz_rows as _cumtrapz_rows
from ..config import config

# True restores EasySpin's qTraj indexing in the 'diffusion' model with Sys.Potential (DEVIATIONS).
EASYSPIN_QTRAJ_INDEXING = False


def _mdrop(a):
    while a.ndim > 2 and a.shape[-1] == 1:
        a = a[..., 0]
    return a


def _perm34(x):
    x = np.asarray(x)
    x = x.reshape(x.shape + (1,) * (4 - x.ndim)) if x.ndim < 4 else x
    return _mdrop(np.transpose(x, (0, 1, 3, 2)))


def _size(x, k):
    s = np.shape(x)
    return s[k - 1] if k <= len(s) else 1


def _isempty_md(MD):
    return MD is None or (isinstance(MD, np.ndarray) and MD.size == 0) or (isinstance(MD, (list, dict)) and len(MD) == 0)


@nout(-1)
def cardamom(Sys=None, Exp=None, Par=None, Opt=None, MD=None, nargout=None, plot=False, rng=None):
    """cardamom(Sys, Exp, Par, Opt=None, MD=None, nargout=2) -> nargout 1: spc; 2: (B, spc); 3: (B, spc,
    TDSignal); 4: (B, spc, TDSignal, t). Random: trajectories (and the MD-HMM start states) come from
    rng (numpy Generator/RandomState, default the global numpy stream), so spectra agree with MATLAB
    statistically, not draw by draw. plot=True draws the spectrum (MATLAB's nargout=0)."""
    nargin = sum(a is not None for a in (Sys, Exp, Par)) + (Opt is not None) + (MD is not None)
    if Sys is None and Exp is None and Par is None:
        return None
    if Par is None:
        raise ValueError('At least 3 inputs (Sys,Exp,Par) are required.')
    rng = np.random if rng is None else rng
    Opt = Struct() if Opt is None else Struct(Opt)
    if MD is None:
        MD = np.zeros((0, 0))
    if nargout is None:
        nargout = 2
    if nargout not in (0, 1, 2, 3, 4):
        raise ValueError('Incorrect number of output arguments.')
    Par = Struct(Par)
    Exp = Struct(Exp)
    if 'Verbosity' not in Opt:
        Opt.Verbosity = 0
    config.LogLevel = Opt.Verbosity
    Sys, err = F.validatespinsys(Sys)
    if err:
        raise ValueError(err)
    Sys = Struct(Sys)
    if Sys.nElectrons > 1:
        raise ValueError('cardamom does not support more than one electron.')
    if Sys.nNuclei > 1:
        raise ValueError('cardamom does not support more than one nucleus.')
    isBroadening = bool(np.any(np.asarray(Sys.lw) > 0)) if 'lw' in Sys else False
    F.logmsg(1, 'Experimental settings:')
    Exp, FieldSweep, CenterField, CenterFreq, _ = F.validate_exp('cardamom', Sys, Exp)
    Exp = Struct(Exp)
    if FieldSweep:
        omega0 = 2 * np.pi * Exp.mwFreq * 1e9
    else:
        omega0 = 2 * np.pi * CenterFreq * 1e9
    if 'Model' not in Par:
        raise ValueError('Please specify a simulation model in Par.Model.')
    if not isinstance(Par.Model, str):
        raise ValueError('Par.Model must be a string.')
    if Par.Model in ('diffusion', 'jump'):
        if not _isempty_md(MD):
            raise ValueError('For stochastic diffusion/jump model, do not provide an MD trajectory.')
    elif Par.Model in ('MD-direct', 'MD-HBD', 'MD-HMM'):
        if _isempty_md(MD):
            raise ValueError(f"For the model '{Par.Model}', MD simulation information must be provided in the input argument 'MD'.")
    else:
        raise ValueError(f"Setting '{Par.Model}' for Par.Model not recognized.")
    LocalDynamicsModel = Par.Model
    useMDdirect = LocalDynamicsModel == 'MD-direct'
    useMD = not _isempty_md(MD)
    HMM = None
    if useMD:
        MD = Struct(MD)
        F.logmsg(1, '  using MD trajectory data')
        if 'dt' not in MD:
            raise ValueError('The MD trajectory time step MD.dt must be given.')
        if 'DiffGlobal' not in MD:
            MD.DiffGlobal = np.zeros((0, 0))
        if 'LabelName' not in MD:
            MD.LabelName = 'R1'
        if 'FrameTraj' not in MD:
            raise ValueError('The spin label frame trajectory MD.FrameTraj must be given.')
        if _size(MD.FrameTraj, 1) != 3 or _size(MD.FrameTraj, 2) != 3:
            raise ValueError('Frame trajectory in MD must be of size (3,3,nSteps,nTraj).')
        dimsSwapped = _size(MD.FrameTraj, 3) < _size(MD.FrameTraj, 4)
        if dimsSwapped:
            MD.FrameTraj = _perm34(MD.FrameTraj)
            MD.FrameTrajwrtProt = _perm34(MD.FrameTrajwrtProt)
            MD.dihedrals = _perm34(MD.dihedrals)
        if 'FrameTrajwrtProt' not in MD:
            raise ValueError('The spin label frame trajectory MD.FrameTrajwrtProt must be given.')
        if 'removeGlobal' not in MD:
            MD.removeGlobal = True
        MD.RTraj = MD.FrameTrajwrtProt if MD.removeGlobal else MD.FrameTraj
        MD.nTraj = _size(MD.RTraj, 4)
        MD.nSteps = _size(MD.RTraj, 3)
        if MD.nTraj != 1:
            raise ValueError('Can only handle MD data with a single trajectory.')
        RT = np.asarray(MD.RTraj)
        RT4 = RT.reshape(RT.shape + (1,) * (4 - RT.ndim))
        RTrajInv = np.ascontiguousarray(np.transpose(RT4, (1, 0, 2, 3)).reshape(RT.shape))
        eyes = np.ascontiguousarray(np.broadcast_to(np.eye(3).reshape(3, 3, 1, 1), RT4.shape).reshape(RT.shape))
        if not F.allclose(F.multimatmult(RT, RTrajInv), eyes, 1e-10):
            raise ValueError('Rotation matrices in frame trajectory are not orthogonal.')
        del RTrajInv, eyes
        F.logmsg(1, '    label: %s', MD.LabelName)
        F.logmsg(1, '    number of trajectories: %d', MD.nTraj)
        F.logmsg(1, '    number of time steps: %d', MD.nSteps)
        F.logmsg(1, '    size of time step: %g ps', MD.dt / 1e-12)
        F.logmsg(1, '    remove global diffusion: %d', MD.removeGlobal)
        if LocalDynamicsModel == 'MD-HMM':
            F.logmsg(1, 'Building HMM model')
            if 'HMM' in MD:
                F.logmsg(1, '  using provided HMM parameters')
                HMM = Struct(MD.HMM)
                if 'TransProb' not in HMM:
                    raise ValueError('The transition probability matrix must be provided in HMM.transmat.')
                if 'nLag' not in HMM:
                    raise ValueError('The time lag (an integer multiple of the MD time step) must be provided in HMM.nLag.')
                if 'eqDistr' not in HMM:
                    raise ValueError('The equilibrium state distribution must be provided in HMM.eqdistr.')
                if 'viterbiTraj' not in HMM:
                    raise ValueError('The Viterbi trajectory must be provided in HMM.viterbiTraj.')
            else:
                F.logmsg(1, '  constructing HMM from MD')
                nLag = round_half_away(MD.tLag / MD.dt)
                HMM = Struct(F.mdhmm(MD.dihedrals, MD.dt, MD.nStates, nLag, Opt))
            vit = np.asarray(HMM.viterbiTraj, dtype=float)
            MD.viterbiTraj = (vit.reshape(-1, 1) if vit.ndim == 1 else vit).T
            MD.nStates = HMM.nStates
            Par.dtSpatial = MD.tLag
        FT2 = np.asarray(MD.FrameTraj) ** 2
        FrameAcorr = np.squeeze(F.autocorrfft(np.squeeze(FT2), 3, 1, 1, 1))
        N = int(round_half_away(MD.nSteps / 2))
        time = linspace(0, N * MD.dt, N)
        tauR = np.max(_cumtrapz_rows(time, FrameAcorr[:, :N]), axis=1)
        tauR = np.mean(tauR)
        DiffLocal = 1 / 6 / tauR
        MD.tauR = tauR
    isDiffSim = (useMD and LocalDynamicsModel == 'MD-HBD') or LocalDynamicsModel == 'diffusion'
    dynamInfoGiven = any(k in Sys for k in ('tcorr', 'Diff', 'logtcorr', 'logDiff'))
    if useMD and LocalDynamicsModel == 'MD-HBD' and not dynamInfoGiven:
        F.logmsg(1, 'Sys.Diff not specified, estimating from MD trajectory data')
        stopFitT = math.floor(MD.nSteps / 2) * MD.dt
        res = F.cardamom_estimatedifftensor(np.squeeze(np.asarray(MD.FrameTraj)), MD.dt, stopFitT)
        Sys.Diff = res[0] if isinstance(res, tuple) else res
        F.logmsg(1, 'Estimated Sys.Diff eigenvalues:  (%g, %g, %g) rad^2/µs', *np.ravel(Sys.Diff / 1e6))
    Dynamics = Struct(F.validate_dynord('cardamom', Sys, FieldSweep, isDiffSim))
    if isDiffSim:
        DiffLocal = Dynamics.Diff
        Sys.Diff = DiffLocal
    if useMD:
        Dynamics.DiffGlobal = MD.DiffGlobal
    includeGlobalDynamics = not isempty(Dynamics.DiffGlobal)
    if 'Potential' in Sys:
        useLocalPotential = True
        LocalPotential = Sys.Potential
    else:
        useLocalPotential = False
    fastMethodApplicable = Sys.nElectrons == 1 and Sys.S == 0.5 and Sys.nNuclei <= 1
    if 'Method' not in Opt:
        Opt.Method = 'fast' if fastMethodApplicable else 'ISTOs'
    if Opt.Method == 'fast' and not fastMethodApplicable:
        raise ValueError("Opt.Method = 'fast' is not applicable for this spin system.")
    if 'FFTWindow' not in Opt:
        Opt.FFTWindow = True
    FFTWindow = Opt.FFTWindow
    if 'specCon' not in Opt:
        Opt.specCon = False
    checkConvergence = Opt.specCon
    if 'LagTime' not in Opt:
        Opt.LagTime = 2e-9
    if 'dtSpin' not in Par:
        raise ValueError('Par.dtSpin (spin propagation time step) must be given.')
    if 'nSteps' not in Par:
        raise ValueError('Par.nSteps must be given.')
    ns = np.asarray(Par.nSteps)
    if not np.issubdtype(ns.dtype, np.number) or ns.size != 1 or np.iscomplexobj(ns) or float(ns) % 1 or float(ns) < 1:
        raise ValueError('Par.nSteps must be a positive integer.')
    if 'dtSpatial' in Par:
        if useMDdirect:
            raise ValueError('For MD-direct simulations, Par.dtSpatial is not allowed. The time step is taken from the MD input.')
        if Par.dtSpin < Par.dtSpatial:
            raise ValueError('The spatial dynamics time step Par.dtSpatial must be less than or equal to the spin dynamics time step Par.dtSpin.')
    else:
        if not useMDdirect:
            raise ValueError(f'The spatial dynamics time step Par.dtSpatial must be specified when using an {LocalDynamicsModel} model.')
        Par.dtSpatial = MD.dt
    r = Par.dtSpin / Par.dtSpatial
    if abs(r - round_half_away(r)) > 1e-4 or r < 1:
        raise ValueError('The spin propagation time step (Par.dtSpin) must be a multiple of the trajectory time step (Par.dtSpatial or MD.dt).')
    r = round_half_away(r)
    Par.dtSpatial = Par.dtSpin / r
    Par.BlockLength = r
    nStepsSpin = int(Par.nSteps)
    nStepsSpatial = nStepsSpin * int(round_half_away(Par.dtSpin / Par.dtSpatial))
    dtSpin = Par.dtSpin
    dtSpatial = Par.dtSpatial
    if not useMDdirect and 'nTraj' not in Par:
        Par.nTraj = 100
    if useMDdirect:
        Par.lag = math.ceil(Opt.LagTime / Par.dtSpin)
        nBlocks = math.floor(MD.nSteps / Par.BlockLength)
        if nBlocks < Par.nSteps:
            raise ValueError('MD trajectory is too short (%g ns) for the required FID length (%g ns.'
                             % (MD.nSteps * MD.dt, Par.nSteps * Par.dtSpin))
        Par.nTraj = math.floor((nBlocks - Par.nSteps) / Par.lag) + 1
    else:
        Par.lag = 0
    if 'nOrients' not in Par:
        Par.nOrients = 100
    F.logmsg(1, 'Parameter settings:')
    F.logmsg(1, "  Local dynamics model:   '%s'", LocalDynamicsModel)
    if includeGlobalDynamics:
        F.logmsg(1, '  Global correlation time:  %g rad^2/µs', *np.ravel(np.asarray(Dynamics.DiffGlobal) / 1e6))
    else:
        F.logmsg(1, '  Global correlation time:  none')
    F.logmsg(1, '  Number of orientations: %d', Par.nOrients)
    F.logmsg(1, '  Number of trajectories: %d', Par.nTraj)
    F.logmsg(1, '  Spin propagation:       %d steps of %g ns (%g ns total)', nStepsSpin, dtSpin / 1e-9, nStepsSpin * dtSpin / 1e-9)
    F.logmsg(1, '  Spatial propagation:    %d steps of %g ns (%g ns total)', nStepsSpatial, dtSpatial / 1e-9, nStepsSpatial * dtSpatial / 1e-9)
    F.logmsg(1, '  Lag time:               %g MD steps', Par.lag)
    if LocalDynamicsModel == 'diffusion':
        DiffLocal = Dynamics.Diff
    elif LocalDynamicsModel in ('MD-direct', 'MD-HBD', 'MD-HMM'):
        RTrajLocal = MD.RTraj
        qTrajLocal = F.rotmat2quat(RTrajLocal)
        if LocalDynamicsModel == 'MD-HBD':
            useLocalPotential = True
            LocalPotential = F.mdtraj2oripot(MD.FrameTraj)
            LocalPotential = LocalPotential[0] if isinstance(LocalPotential, tuple) else LocalPotential
        elif LocalDynamicsModel == 'MD-HMM':
            RT = np.asarray(RTrajLocal)
            RTrajLocal = _mdrop(RT.reshape(RT.shape + (1,) * (4 - RT.ndim))[:, :, ::int(HMM.nLag), :])
            qTrajLocal = F.rotmat2quat(RTrajLocal)
        for fld in ('FrameTraj', 'RTraj', 'FrameTrajwrtProt', 'RProtDiff'):
            MD.pop(fld, None)
    nOrientations = int(Par.nOrients)
    Orientations = np.zeros((0, 0))
    if 'Orients' in Par:
        Orientations = np.asarray(Par.Orients, dtype=float)
        if Orientations.ndim < 2 or min(Orientations.shape) == 1:
            Orientations = np.tile(Orientations.reshape(-1, 1, order='F'), (1, nOrientations))
    if Orientations.size:
        gridPhi = Orientations[0, :]
        gridTheta = Orientations[1, :]
        weight = np.ones(gridPhi.shape)
        weight = weight / np.sum(weight)
    else:
        if checkConvergence:
            nOrientations = math.ceil(nOrientations / 2)
            skip = 0
            gridPts = 2 * F.cardamom_sobol_generate(1, nOrientations, skip) - 1
            gridPhi = np.sqrt(np.pi * nOrientations) * np.arcsin(gridPts)
            gridTheta = np.arccos(gridPts)
            weight = np.ones(gridPhi.shape)
        else:
            gridPts = linspace(-1, 1, nOrientations)
            gridPhi = np.sqrt(np.pi * nOrientations) * np.arcsin(gridPts)
            gridTheta = np.arccos(gridPts)
            weight = np.ones(gridPhi.shape)
            weight = weight / np.sum(weight)
    gridPhi, gridTheta, weight = np.ravel(gridPhi), np.ravel(gridTheta), np.ravel(weight)
    F.logmsg(1, "Quantum propagation method: '%s'", Opt.Method)
    F.logmsg(1, 'Running simulation')
    cache = {}                                    # clear cardamom_propagatedm (its persistent caches)
    it = 1
    spcArray = None
    nOrientsTot = 0
    t = linspace(0, nStepsSpin * Par.dtSpin, nStepsSpin).reshape(-1, 1)
    converged = False
    spcLast = 0
    Sys_ = Struct()
    rmsdLast = None
    while not converged:
        iTDSignal = [None] * nOrientations
        itCell = [None] * nOrientations
        for iOri in range(nOrientations):
            if LocalDynamicsModel == 'diffusion':
                Sys.Diff = Dynamics.Diff
                Par.dt = dtSpatial
                Par.nSteps = nStepsSpatial
                if useLocalPotential:
                    Sys.Potential = LocalPotential
                    Par.nSteps = 2 * nStepsSpatial
                _, RTrajLocal, qTrajLocal = F.stochtraj_diffusion(Sys, Par, Opt, nargout=3, rng=rng)
                if useLocalPotential:
                    RTrajLocal = np.asarray(RTrajLocal)[:, :, nStepsSpatial:]
                    # EasySpin: qTrajLocal(:,:,nStepsSpatial+1:end,:) selects trajectories, not time steps
                    # (qTraj is 4 x nSteps x nTraj); fixed to the time-step dimension (DEVIATIONS).
                    qTrajLocal = (np.asarray(qTrajLocal)[:, :, nStepsSpatial:] if EASYSPIN_QTRAJ_INDEXING
                                  else np.asarray(qTrajLocal)[:, nStepsSpatial:])
            elif LocalDynamicsModel == 'jump':
                Par.dt = dtSpatial
                Par.nSteps = nStepsSpatial
                _, RTrajLocal, qTrajLocal = F.stochtraj_jump(Sys, Par, Opt, nargout=3, rng=rng)
            elif LocalDynamicsModel == 'MD-direct':
                pass
            elif LocalDynamicsModel == 'MD-HBD':
                Sys.Diff = DiffLocal
                Par.nSteps = nStepsSpatial
                if useLocalPotential:
                    Sys.Potential = LocalPotential
                    Par.nSteps = 2 * nStepsSpatial
                Par.dt = dtSpatial
                _, RTrajLocal, qTrajLocal = F.stochtraj_diffusion(Sys, Par, Opt, nargout=3, rng=rng)
                if useLocalPotential:
                    RTrajLocal = np.asarray(RTrajLocal)[:, :, nStepsSpatial:]
                    qTrajLocal = np.asarray(qTrajLocal)[:, nStepsSpatial:]   # EasySpin indexes qTraj(:,k:end,:)
            elif LocalDynamicsModel == 'MD-HMM':
                Sys.TransProb = HMM.TransProb
                Par.dt = dtSpatial
                Par.nSteps = 2 * nStepsSpatial
                CumulDist = np.cumsum(np.ravel(HMM.eqDistr)) / np.sum(HMM.eqDistr)
                ss = np.ravel(np.asarray(Par.StatesStart, dtype=float)) if 'StatesStart' in Par else np.zeros(0)
                nT = int(Par.nTraj)
                if ss.size < nT:
                    ss = np.concatenate([ss, np.zeros(nT - ss.size)])
                for k in range(nT):
                    ss[k] = np.flatnonzero(CumulDist > rng.random())[0] + 1
                Par.StatesStart = ss.reshape(1, -1)
                Opt.statesOnly = True
                _, stateTraj = F.stochtraj_jump(Sys, Par, Opt, nargout=2, rng=rng)
                stateTraj = np.asarray(stateTraj)
                stateTraj = stateTraj.reshape(stateTraj.shape[0], -1)[nStepsSpatial:, :]
                Par.stateTraj = stateTraj
            if Opt.Method == 'ISTOs':
                Par.qTraj = qTrajLocal
            else:
                Par.RTraj = RTrajLocal
            qLab = np.asarray(F.euler2quat(0, gridTheta[iOri], gridPhi[iOri], 'active')).reshape(4, 1, 1)
            qLab = _mdrop(np.tile(qLab, (1, nStepsSpin, int(Par.nTraj))))
            if includeGlobalDynamics:
                Sys_.Diff = Dynamics.DiffGlobal
                Par.dt = dtSpin
                Par.nSteps = nStepsSpin
                _, _, qTrajGlobal = F.stochtraj_diffusion(Sys_, Par, Opt, nargout=3, rng=rng)
                qLab = F.quatmult(qLab, qTrajGlobal)
            if Opt.Method == 'ISTOs':
                Par.qLab = qLab
            else:
                Par.RLab = F.quat2rotmat(qLab)
            Par.nSteps = nStepsSpin
            Par.dtSpin = dtSpin
            Par.dtSpatial = dtSpatial
            Sprho = np.asarray(F.cardamom_propagatedm(Sys, Par, Opt, MD, omega0, CenterField, cache=cache))
            S3 = Sprho.reshape(Sprho.shape[0], Sprho.shape[1], -1)
            sig = 0
            for k in range(S3.shape[0]):
                sig = sig + S3[k, k, :].reshape(-1, 1)
            iTDSignal[iOri] = weight[iOri] * sig
            itCell[iOri] = t
        TDSignal = iTDSignal
        tCell = itCell
        if FFTWindow:
            TDSignal = [s * (0.5 * (1 + np.cos(np.pi * x / np.max(x)))) for s, x in zip(TDSignal, tCell)]
        Bres = 0.1
        tReq = 1 / (F.unitconvert(Bres / 10, 'mT->MHz') * 1e6)
        tMax = max(float(np.max(x)) for x in tCell)
        M = math.ceil(tReq / Par.dtSpin) if tMax < tReq else math.ceil(tMax / Par.dtSpin)
        TD = np.zeros((M, len(TDSignal)), dtype=complex)
        for j, s in enumerate(TDSignal):
            TD[:s.size, j] = np.ravel(s)
        TDSignal = TD
        tLong = linspace(0, M * Par.dtSpin, M).reshape(-1, 1)
        if isBroadening:
            lw = np.ravel(np.asarray(Sys.lw, dtype=float))
            if lw[0] > 0:
                fwhm = F.unitconvert(lw[0], 'mT->MHz') * 1e6
                alpha = np.pi ** 2 * fwhm ** 2 / (4 * np.log(2))
                TDSignal = np.exp(-alpha * tLong ** 2) * TDSignal
            if lw.size == 2 and lw[1] > 0:
                TL = Dynamics.T2
                TDSignal = np.exp(-tLong / TL) * TDSignal
        newspc = np.imag(np.fft.fftshift(np.fft.fft(TDSignal * tLong, axis=0)))   # MATLAB fftshift(X): all dims
        spcArray = newspc if spcArray is None else np.concatenate([spcArray, newspc], axis=1)
        spcNew = np.mean(spcArray, axis=1, keepdims=True)
        if checkConvergence:
            span = np.max(spcNew) - np.min(spcNew)
            rmsdNew = np.sqrt(np.mean((spcNew - spcLast) ** 2)) / span
            if rmsdNew < 2e-2:
                converged = True
            elif it > 2:
                rmsdRelativeChange = abs((rmsdNew - rmsdLast) / rmsdLast)
                converged = rmsdRelativeChange < 0.5
        else:
            converged = True
        if not converged:
            if Opt.Verbosity:
                print('Convergence not achieved. Propagation is being extended.')
            rmsdLast = rmsdNew
            spcLast = spcNew
            nOrientsTot = nOrientsTot + nOrientations
            skip = it * nOrientsTot
            nOrientations = nOrientsTot
            if Orientations.size:
                # EasySpin: repmat(Orientations(:,1),[2^(iter-1),1]) (a column of phi/theta of the first orientation)
                gridPhi = np.tile(Orientations[:, 0], 2 ** (it - 1))
                gridTheta = np.tile(Orientations[:, 1], 2 ** (it - 1))
            else:
                gridPts = np.ravel(2 * F.cardamom_sobol_generate(1, nOrientations, skip) - 1)
                gridPhi = np.sqrt(np.pi * nOrientations) * np.arcsin(gridPts)
                gridTheta = np.arccos(gridPts)
            it += 1
        else:
            spcAvg = spcNew
    freq = 1 / (Par.dtSpin * M) * colon(-M / 2, 1, M / 2 - 1)
    if FieldSweep:
        fftAxis = F.unitconvert(freq / 1e6 + Exp.mwFreq * 1e3, 'MHz->mT', np.mean(Sys.g))
    else:
        fftAxis = freq / 1e9 + F.unitconvert(Exp.Field, 'mT->MHz', np.mean(Sys.g)) / 1e3
    if FieldSweep:
        R = np.ravel(Exp.Range)
        xAxis = linspace(R[0], R[1], Exp.nPoints)
    else:
        R = np.ravel(Exp.mwRange)
        xAxis = linspace(R[0], R[1], Exp.nPoints)
    spcAvg = np.ravel(spcAvg)
    if FieldSweep:
        outspc = interp1(fftAxis, spcAvg, xAxis)
    else:
        spcAvg = spcAvg[::-1]
        fa = np.ravel(fftAxis)
        outspc = interp1(fa, spcAvg, xAxis, 'spline')
        outspc[(xAxis < np.min(fa)) | (xAxis > np.max(fa))] = 0          # interp1(..., 'spline', 0)
        outspc = cumtrapz(xAxis[::-1], outspc)
    TDSignal = np.mean(TDSignal, axis=1, keepdims=True)
    TDSignal = TDSignal[:int(Par.nSteps)]
    xAxis = xAxis.reshape(1, -1)
    outspc = np.asarray(outspc).reshape(1, -1)
    config.LogLevel = None
    if plot or nargout == 0:
        if plot:
            import matplotlib.pyplot as plt
            plt.plot(np.ravel(xAxis), np.ravel(outspc))
            plt.xlabel('magnetic field (mT)' if FieldSweep else 'frequency (GHz)')
            plt.ylabel('intensity (arb.u.)')
        if nargout == 0:
            return None
    if nargout == 1:
        return outspc
    if nargout == 2:
        return xAxis, outspc
    if nargout == 3:
        return xAxis, outspc, TDSignal
    return xAxis, outspc, TDSignal, t

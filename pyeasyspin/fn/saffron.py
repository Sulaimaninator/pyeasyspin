"""Simulate pulse EPR signals and spectra"""
import datetime
import itertools
import math
import warnings

import numpy as np
import scipy.linalg

from .._registry import F, nout
from .._struct import Struct
from .._mat import isempty, numel, vec
from .._matlab import colon, linspace
from ..config import config


def _c(x):
    return x[0] if isinstance(x, tuple) else x


def _now():
    return datetime.datetime.now().strftime('%d-%b-%Y %H:%M:%S')


def _dense(x):
    return x.toarray() if hasattr(x, 'toarray') else np.asarray(x)


def _eig(H):
    """MATLAB eig for a full matrix: Hermitian solver (ascending) when H is exactly Hermitian."""
    H = _dense(H)
    if np.array_equal(H, H.conj().T):
        return scipy.linalg.eigh(H)
    E, V = np.linalg.eig(H)
    return E, V


def _num(x):
    return float(np.ravel(np.asarray(x, dtype=float))[0])


def _row(x):
    return np.asarray(x, dtype=float).reshape(1, -1, order='F') if np.ndim(x) else np.array([[float(x)]])


def _cpow(base, p):
    """MATLAB scalar power base^p (exact for integer-valued p)."""
    return complex(base) ** float(p)


@nout(-1)
def saffron(Sys=None, Exp=None, Opt=None, nargout=None, plot=False):
    """Pulse EPR (ESEEM, HYSCORE, Mims ENDOR, user sequences) of powders and crystals.
    nargout: 1 -> data; 2 (default) -> (x, data); 3 -> (x, data, info). plot=True runs MATLAB's
    nargout-0 branch (saffron_plot / s_plotting) and still returns (x, data).
    Fast mode: frequency-domain peak binning (sf_peaks); thyme mode: time-domain propagation
    (s_thyme). Times in us, frequencies in MHz, field in mT."""
    plotting = plot or nargout == 0
    if nargout is None:
        nargout = 2
    if Sys is None:
        raise ValueError('Please supply a spin system as first input argument.')
    if Exp is None:
        raise ValueError('Please supply experimental parameters as second input argument.')
    if nargout > 3:
        raise ValueError('Too many output arguments.')
    if Opt is None or isempty(Opt):
        Opt = Struct()
    if not isinstance(Sys, (dict, list)):
        raise ValueError('The first input (Sys) must be a structure or a cell array of structures.')
    if not isinstance(Exp, dict):
        raise ValueError('The second input argument (Exp) must be a structure.')
    if not isinstance(Opt, dict):
        raise ValueError('The third input argument (Opt) must be a structure.')
    Exp, Opt = Struct(Exp), Struct(Opt)
    if 'Verbosity' not in Opt:
        Opt.Verbosity = 0
    config.LogLevel = Opt.Verbosity
    if 'separate' not in Opt:
        Opt.separate = ''
    separateOutput, err = F.parseoption(Opt, 'separate', ['', 'components', 'transitions', 'orientations', 'sites'],
                                        nargout=2)
    if not isempty(err):
        raise ValueError(err)
    separateComponentSpectra = separateOutput == 2
    separateTransitionSpectra = separateOutput == 3
    separateOrientationSpectra = separateOutput == 4
    separateSiteSpectra = separateOutput == 5
    if separateTransitionSpectra or separateSiteSpectra or separateOrientationSpectra:
        separateComponentSpectra = True
    if separateTransitionSpectra:
        raise ValueError("saffron does not support Opt.separate='transitions'.")
    if separateSiteSpectra:
        raise ValueError("saffron does not support Opt.separate='sites'.")
    if separateOrientationSpectra:
        raise ValueError("saffron does not support Opt.separate='orientations'.")
    if 'Sequence' not in Exp:
        raise ValueError('Unrecognized field name "Sequence".')
    isENDOR = isinstance(Exp.Sequence, str) and Exp.Sequence == 'MimsENDOR'
    if 'IsoCutoff' not in Opt:
        Opt.IsoCutoff = 1e-4
    singleIsotopologue = isinstance(Sys, dict) and 'singleiso' in Sys and bool(Sys.singleiso)
    if not singleIsotopologue:
        thirdOutput = nargout >= 3 or plotting
        autoRange = False
        x, data, info = F.compisoloop(F.saffron, Sys, Exp, Opt, autoRange, thirdOutput,
                                      separateComponentSpectra)[:3]
        if plotting:
            if 'td' in info:
                _saffron_plot(x, info, isENDOR, Sys, Exp, Opt)
            else:
                Opt.SinglePointDetection = info.SinglePointDetection
                F.s_plotting(x, data, Exp, Opt)
            if nargout == 0:
                return x, data
        if nargout == 1:
            return data
        if nargout == 2:
            return x, data
        return x, data, info

    F.logmsg(1, '=begin=saffron====' + _now() + '=================')
    F.logmsg(2, '  log level %d', config.LogLevel)
    F.logmsg(1, '-general-----------------------------------------------')
    Sys, err = F.validatespinsys(Sys)
    if not isempty(err):
        raise ValueError(err)
    Sys = Struct(Sys)
    if bool(Sys.MO_present):
        raise ValueError('saffron does not support Sys.Ham* parameters.')
    if np.any(vec(np.asarray(Sys.L, dtype=float)) != 0) if not isempty(Sys.L) else False:
        raise ValueError('saffron does not support Sys.L.')
    if (np.any(np.asarray(Sys.gStrain) != 0) or np.any(np.asarray(Sys.AStrain) != 0)
            or np.any(np.asarray(Sys.DStrain) != 0)):
        raise ValueError('saffron does not support Sys.gStrain, Sys.AStrain, or Sys.DStrain.')
    for field_ in ('initState', 'eqState'):
        if field_ in Sys and not isempty(Sys[field_]):
            raise ValueError('Sys.%s is specific to spidyan and cannot be used with saffron. Please remove.' % field_)
    DefaultExp = Struct(Temperature=np.zeros((0, 0)), Ordering=np.zeros((0, 0)), SampleRotation=np.zeros((0, 0)),
                        SampleFrame=np.zeros((0, 0)), CrystalSymmetry='', MolFrame=np.zeros((0, 0)))
    Exp = Struct(F.adddefaults(Exp, DefaultExp))
    if 'Orientations' in Exp:
        raise ValueError('Exp.Orientations is no longer supported, use Exp.SampleFrame instead.')
    if 'CrystalOrientation' in Exp:
        raise ValueError('Exp.CrystalOrientation is no longer supported, use Exp.SampleFrame instead.')
    if 'lightBeam' in Exp and not isempty(Exp.lightBeam):
        raise ValueError('Photoselection (via Exp.lightBeam) is not supported.')
    if 'Field' not in Exp:
        raise ValueError('Exp.Field is missing. Give a magnetic field in mT.')
    if 'Sequence' not in Exp:
        raise ValueError('Exp.Sequence is missing. Provide a pre-defined or user-defined pulse sequence.')
    if not isempty(Exp.Temperature):
        raise ValueError('Exp.Temperature is not supported for pulse EPR simulations.')
    Exp, Opt = F.p_sampletype(Exp, Opt)[:2]
    Exp, Opt = Struct(Exp), Struct(Opt)
    if bool(Opt.partiallyOrderedSample):
        raise ValueError('Partially ordered samples are not implemented in saffron.')
    nDataPoints = int(np.prod(np.asarray(Exp.nPoints, dtype=float))) if 'nPoints' in Exp else 1
    for f in ('GridSymmetry', 'GridFrame', 'Transitions', 'Sites'):
        if f not in Opt:
            Opt[f] = np.zeros((0, 0))
    if 'nKnots' in Opt:
        raise ValueError('Options.nKnots is obsolete. Use Options.GridSize instead, e.g. Options.GridSize = 91.')
    for field_ in ('StateTrajectories', 'ExcOperator'):
        if field_ in Opt:
            del Opt[field_]
            warnings.warn('Opt.' + field_ + ' is specific to spidyan and can not be used with saffron. It will be ignored.')
    if 'Relaxation' in Opt and numel(Opt.Relaxation) > 1:
        raise ValueError('Opt.Relaxation has to be true or false.')
    elif 'Relaxation' not in Opt and ('T1' in Sys or 'T2' in Sys):
        Opt.Relaxation = True
    elif 'Relaxation' not in Opt:
        Opt.Relaxation = False
    if 'GridSize' not in Opt:
        Opt.GridSize = 30 + 1
    if numel(Opt.GridSize) > 1:
        raise ValueError('Only one number allowed in Opt.GridSize. saffron does not support interpolation.')
    if _num(Opt.GridSize) < 7:
        raise ValueError('Opt.GridSize must be at least 7. You gave %d.' % _num(Opt.GridSize))
    if 'SimulationMode' not in Opt:
        Opt.SimulationMode = 'fast'
    Events, Vary, Opt, oldSeq = F.s_sequencer(Exp, Opt, nargout=4)[:4]
    Opt = Struct(Opt)
    if not isempty(oldSeq):
        Exp.Flip = oldSeq.Flip
        Exp.Phase = oldSeq.Phase
        Exp.tp = oldSeq.tp
        Exp.t = oldSeq.t
        for f in ('Inc', 'dt'):
            if f not in oldSeq:
                raise ValueError(f'Unrecognized field name "{f}".')
        Exp.Inc = oldSeq.Inc
        Exp.dt = oldSeq.dt
    _, err = F.parseoption(Opt, 'SimulationMode', ['fast', 'thyme'], nargout=2)
    if not isempty(err):
        raise ValueError(err)
    fastSimulationMode = Opt.SimulationMode == 'fast'
    if not fastSimulationMode and plotting and 'nPoints' in Exp:
        if Opt.SinglePointDetection and numel(Exp.nPoints) > 2:
            raise ValueError('Not enough output arguments for more than two indirect dimensions with single point detections. The data can not be generally plotted.')
        elif not Opt.SinglePointDetection and numel(Exp.nPoints) > 1:
            raise ValueError('Not enough output arguments for more than one indirect dimensions with transient detections. The data can not be generally plotted.')
    if fastSimulationMode:
        if Sys.nElectrons > 1:
            raise ValueError('Fast mode does not work for more than one electron spin.')
        if 'n' in Sys and np.any(vec(np.asarray(Sys.n, dtype=float)) != 1):
            raise ValueError('Fast mode does not work for equivalent nuclei.')
        if 'nn' in Sys and np.any(vec(np.asarray(Sys.nn, dtype=float)) != 0):
            raise ValueError('Fast mode does not work in the presence of nuclear-nuclear couplings.')
    if 'OriWeights' not in Exp:
        Exp, Opt, nOrientations = F.p_gridsetup(Sys, Exp, Opt)[:3]
        Exp, Opt = Struct(Exp), Struct(Opt)
    Orientations, nOrientations, nSites = F.p_crystalorientations(Exp, Opt)[:3]
    Orientations = np.atleast_2d(np.asarray(Orientations, dtype=float))
    nOrientations = int(nOrientations)
    nSites = int(nSites)
    if numel(Exp.OriWeights) != nOrientations:
        Exp.OriWeights = np.tile(_row(Exp.OriWeights), (1, nSites))
    OriWeights = vec(np.asarray(Exp.OriWeights, dtype=float))

    if fastSimulationMode:
        out = _fast(Sys, Exp, Opt, Orientations, nOrientations, OriWeights, nargout)
    else:
        out = _thyme(Sys, Exp, Opt, Events, Vary, Orientations, nOrientations, OriWeights, nDataPoints,
                     nargout, plotting)
    F.logmsg(1, '=end=saffron======%s=================\n', _now())
    return out


# ----------------------------------------------------------------------------- fast mode
def _fast(Sys, Exp, Opt, Orientations, nOrientations, OriWeights, nargout):
    spidyanExpFields = ('DetSequence', 'DetOperator')
    if any(f in Exp for f in spidyanExpFields):
        raise ValueError('Exp.DetSequence and Exp.DetOperator are specific to spidyan and can not be used with saffron. Please remove them.')
    if 'T1T2' in Sys:
        raise ValueError('T1 and T2 should be provided in Sys.T1 and Sys.T2 and not in Sys.T1T2.')
    T1T2 = np.array([0.0, 0.0])
    if 'T1' in Sys:
        if np.any(np.array(np.shape(np.atleast_2d(Sys.T1))) > 1):
            raise ValueError('The fast algorithm does not support transition selective relaxation times, please provide a single value for Sys.T1.')
        T1T2[0] = _num(Sys.T1)
    if 'T2' in Sys:
        if np.any(np.array(np.shape(np.atleast_2d(Sys.T2))) > 1):
            raise ValueError('The fast algorithm does not support transition selective relaxation times, please provide a single value for Sys.T2.')
        T1T2[1] = _num(Sys.T2)
    T1T2[T1T2 == 0] = np.inf
    if np.any(T1T2 <= 0):
        raise ValueError('T1 and T2 in Sys.T1 and Sys.T2 must be positive, in microseconds.')
    Sys.T1T2 = T1T2.reshape(1, 2)
    predefinedExperiment = 'Sequence' in Exp and not isempty(Exp.Sequence) and isinstance(Exp.Sequence, str)
    ExperimentNames = ['2pESEEM', '3pESEEM', '4pESEEM', 'HYSCORE', 'MimsENDOR']
    if predefinedExperiment:
        if 'Filter' in Exp:
            raise ValueError('Exp.Filter can only be used with custom sequences.')
        ids = [k + 1 for k, nm in enumerate(ExperimentNames) if nm == Exp.Sequence]
        if not ids:
            raise ValueError("Exp.Sequence '%s' not recognized." % Exp.Sequence)
        Exp.ExperimentID = ids[0]
        F.logmsg(1, 'Sequence: %s', ExperimentNames[Exp.ExperimentID - 1])
        if 'tp' in Exp and np.any(np.asarray(Exp.tp) != 0):
            raise ValueError('You cannot use predefined sequences (Exp.Sequence) with real pulses (Exp.tp).')
        isENDOR = False
        tauRequired = True
        ID = Exp.ExperimentID
        if ID == 1:
            nIntervals, nDimensions, IncSchemeID = 2, 1, 2
            nPathways = 1
            tauRequired = False
        elif ID == 2:
            nIntervals, nDimensions, IncSchemeID = 3, 1, 1
            nPathways = 2
        elif ID == 3:
            nIntervals, nDimensions, IncSchemeID = 4, 1, 2
            nPathways = 2
        elif ID == 4:
            nIntervals, nDimensions, IncSchemeID = 4, 2, 11
            nPathways = 2
        else:
            isENDOR = True
            nIntervals, nDimensions, IncSchemeID = 3, 1, 0
            nPathways = 2
        if tauRequired:
            if 'tau' not in Exp:
                raise ValueError('Exp.tau is missing.')
            if numel(Exp.tau) != 1:
                raise ValueError('Exp.tau must contain a single value (in µs).')
            if _num(Exp.tau) == 0:
                raise ValueError('Exp.tau must be larger than 0.')
        else:
            if 'tau' not in Exp:
                Exp.tau = 0
            if numel(Exp.tau) != 1:
                raise ValueError('Exp.tau must contain a single value (in µs).')
    else:
        F.logmsg(1, 'User-specified pulse experiment.')
        Exp.ExperimentID = -1
        isENDOR = False
        if np.any(~np.isinf(T1T2)):
            raise ValueError('Relaxation times T1 and T2 for custom sequences not supported.')
        if 'Flip' not in Exp:
            raise ValueError('Fast sequence engine: the pulse flip angles (Exp.Flip) are missing.')
        nIntervals = numel(Exp.Flip)
        if 'Inc' not in Exp:
            raise ValueError('Fast sequence engine: the incrementation scheme (Exp.Inc) is missing.')
        if numel(Exp.Inc) != nIntervals:
            raise ValueError('Exp.Inc must contain the same number of elements as Exp.Flip.')
        Inc = vec(np.asarray(Exp.Inc, dtype=float))
        IncScheme = list(Inc[Inc != 0])
        schemes = {(1,): 1, (1, 1): 2, (1, -1): 3, (1, 2): 11, (1, 2, 1): 12, (1, 2, 2): 13, (1, 1, 2): 14,
                   (1, 2, 2, 1): 15, (1, 2, -2, 1): 16, (1, 1, 2, 2): 17}
        key = tuple(int(v) if v == int(v) else v for v in IncScheme)
        if key not in schemes:
            raise ValueError('Unsupported incrementation scheme in Exp.Inc.')
        IncSchemeID = schemes[key]
        nDimensions = int(np.max(np.abs(Inc)))
        if 't' not in Exp:
            Exp.t = np.zeros((1, nIntervals))
        if numel(Exp.t) != nIntervals:
            raise ValueError('Exp.t must contain the same number of elements as Exp.Flip')
        if np.all(vec(np.asarray(Exp.t, dtype=float)) == 0) and len(IncScheme) < nIntervals:
            F.logmsg(0, 'Some delays are zero, but are not incremented!')
    if not isENDOR:
        if 'dt' not in Exp:
            raise ValueError('Exp.dt is missing.')
        if numel(Exp.dt) == 1:
            Exp.dt = _num(Exp.dt) * np.ones((1, nDimensions))
        elif numel(Exp.dt) != nDimensions:
            raise ValueError('Exp.dt needs either 1 or %d elements, one per dimension. You gave %d.'
                             % (nDimensions, numel(Exp.dt)))
    if 'nPoints' not in Exp:
        if isENDOR:
            Exp.nPoints = 1001
        elif nDimensions == 1:
            Exp.nPoints = 512
        else:
            Exp.nPoints = np.array([[1, 1]]) * 256
    if numel(Exp.nPoints) == 1:
        Exp.nPoints = _num(Exp.nPoints) * np.ones((1, nDimensions))
    elif numel(Exp.nPoints) != nDimensions:
        raise ValueError('Exp.nPoints needs either 1 or %d elements, one per dimension. You gave %d.'
                         % (nDimensions, numel(Exp.nPoints)))
    nPts = [int(v) for v in vec(np.asarray(Exp.nPoints, dtype=float))]
    if 'tp' not in Exp:
        Exp.tp = np.zeros((1, nIntervals))
    elif numel(Exp.tp) != nIntervals:
        raise ValueError('Exp.tp contains a wrong number of elements.')
    tp = vec(np.asarray(Exp.tp, dtype=float))
    idealPulse = tp == 0
    realPulse = ~idealPulse
    dtv = vec(np.asarray(Exp.dt, dtype=float)) if 'dt' in Exp else None

    if not predefinedExperiment:
        tE = vec(np.asarray(Exp.t, dtype=float))
        for iInterval in range(nIntervals):
            iDim = int(abs(Inc[iInterval]))
            if iDim == 0:
                t_range = tE[iInterval] * np.array([1.0, 1.0])
            else:
                t_range = tE[iInterval] + np.sign(Inc[iInterval]) * (nPts[iDim - 1] - 1) * dtv[iDim - 1]
            if np.any(np.asarray(t_range) < 0):
                raise ValueError('Negative delay after pulse %d.' % (iInterval + 1))
        F.logmsg(1, '  determining pathways contributing to the echo')
        if 'Pathways' in Exp:
            code0 = {'a': 1, 'b': 2, '+': 3, '-': 4}
            pw = Exp.Pathways if isinstance(Exp.Pathways, list) else [Exp.Pathways]
            pathwayList = np.array([[code0[c] for c in s] for s in pw], dtype=float)
            if pathwayList.shape[1] != nIntervals:
                raise ValueError('Exp.Pathways contains pathways longer or shorter than the pulse sequence.')
        else:
            pathwayList = np.atleast_2d(np.asarray(F.sf_pathways(Exp.t, Exp.Inc), dtype=float))
        nPathways = pathwayList.shape[0]
        if nPathways == 0:
            raise ValueError('Sorry, no focused echo with this sequence and timings.')
        if 'Filter' not in Exp:
            Exp.Filter = np.zeros((0, 0))
        if not isempty(Exp.Filter):
            F.logmsg(1, '  applying user-supplied coherence filters')
            if not isinstance(Exp.Filter, str):
                raise ValueError("Exp.Filter must be a string containing '0', '1', 'a', 'b', '+', '-' and/or '.'.")
            if len(Exp.Filter) != nIntervals:
                raise ValueError('Exp.Filter must have %d elements instead of the given %d elements.'
                                 % (nIntervals, len(Exp.Filter)))
            for iInt in range(nIntervals):
                col = pathwayList[:, iInt]
                c = Exp.Filter[iInt]
                if c == '0':
                    keep = (col == 1) | (col == 2)
                elif c == '1':
                    keep = (col == 3) | (col == 4)
                elif c == 'a':
                    keep = col == 1
                elif c == 'b':
                    keep = col == 2
                elif c == '+':
                    keep = col == 3
                elif c == '-':
                    keep = col == 4
                elif c in '.*':
                    keep = col != 0
                else:
                    raise ValueError("Exp.Filter(%d) = '%s' is invalid. Use '0', '1', 'a', 'b', '+', '-' or '.'."
                                     % (iInt + 1, c))
                pathwayList = pathwayList[keep, :]
            nPathways = pathwayList.shape[0]
            if nPathways == 0:
                raise ValueError('Exp.Filter is too restrictive: no echo at detection point left after applying the filter.')
        else:
            F.logmsg(1, '  no user-supplied coherence filters')
        idxFreeL, idxFreeR, idxPulseL, idxPulseR = _pathwayparser(pathwayList)
        if 'Phase' not in Exp:
            Exp.Phase = np.ones((1, nIntervals))
        Phase = vec(np.asarray(Exp.Phase, dtype=float))
        Flip = vec(np.asarray(Exp.Flip, dtype=float))
        pathwayprefactor = np.ones(nPathways, dtype=complex)
        for iPulse in range(Flip.size):
            if idealPulse[iPulse]:
                theta = Flip[iPulse] * math.pi / 2
                c = math.cos(theta / 2)
                s = math.sin(theta / 2)
                for iPathway in range(nPathways):
                    kL = idxPulseL[iPathway, iPulse]
                    pL = c if kL in (1, 2) else (-1j * s * _cpow(-1j, Phase[iPulse]) if kL == 3
                                                 else -1j * s * _cpow(1j, Phase[iPulse]))
                    kR = idxPulseR[iPathway, iPulse]
                    pR = c if kR in (1, 2) else (1j * s * _cpow(1j, Phase[iPulse]) if kR == 3
                                                 else 1j * s * _cpow(-1j, Phase[iPulse]))
                    pathwayprefactor[iPathway] = pL * pathwayprefactor[iPathway] * pR
        rmv = np.abs(pathwayprefactor) < 1e-6
        pathwayList = pathwayList[~rmv, :]
        idxFreeL = idxFreeL[~rmv, :]
        idxFreeR = idxFreeR[~rmv, :]
        idxPulseL = idxPulseL[~rmv, :]
        idxPulseR = idxPulseR[~rmv, :]
        pathwayprefactor = pathwayprefactor[~rmv]
        nPathways = pathwayList.shape[0]
        idxIncL = idxFreeL[:, Inc != 0]
        idxIncR = idxFreeR[:, Inc != 0]
        if config.LogLevel and config.LogLevel > 0:
            F.logmsg(1, '  Pathways and prefactors:')
            Str = 'ab+-'
            for iPathway in range(nPathways):
                F.logmsg(1, '    %d. (%s)   %+4.3f%+4.3fi', iPathway + 1,
                         ''.join(Str[int(k) - 1] for k in pathwayList[iPathway, :]),
                         pathwayprefactor[iPathway].real, pathwayprefactor[iPathway].imag)

    if isENDOR:
        lwEndor = vec(np.asarray(Sys.lwEndor, dtype=float))
        if np.all(lwEndor == 0):
            raise ValueError('Positive ENDOR line width in Sys.lwEndor needed.')
        Sys.lwEndor = float(lwEndor[0])
        if 'Range' not in Exp:
            raise ValueError('Frequency range (Exp.Range) must be given for an ENDOR experiment.')
        if 'tprf' not in Exp:
            Exp.tprf = 20
    if 'T1' not in Exp:
        Exp.T1 = 0
    if 'T2' not in Exp:
        Exp.T2 = 0
    if predefinedExperiment:
        if Exp.ExperimentID in (2, 5, 3):
            if 'T' not in Exp:
                Exp.T = 0
        elif Exp.ExperimentID == 4:
            if 't1' not in Exp:
                Exp.t1 = 0
            if 't2' not in Exp:
                Exp.t2 = 0
            if numel(Exp.t1) > 1:
                raise ValueError('Exp.t1 must a single positive number, in units of µs.')
            if numel(Exp.t2) > 1:
                raise ValueError('Exp.t2 must a single positive number, in units of µs.')
            if _num(Exp.t1) != _num(Exp.t2):
                print('Exp.t1 and Exp.t2 are not identical, so the resulting spectrum might be asymmetric.')
    OrientationSelection = 'mwFreq' in Exp
    if OrientationSelection:
        F.logmsg(1, 'Microwave frequency given: orientation selection is on.')
        if 'ExciteWidth' not in Exp:
            raise ValueError("Orientation selection: Exp.ExciteWidth (in MHz) missing. It should be about the inverse of the first pulse length (100MHz for 10ns). If you don't want orientation selection, set it to a very large number (1e6) or remove the microwave frequency.")
    else:
        F.logmsg(1, 'no orientation selection (infinite bandwidth).')
    if 'ExciteWidth' in Exp and 'mwFreq' not in Exp:
        raise ValueError('Exp.ExciteWidth is given, but Exp.mwFreq is missing. Please give Exp.mwFreq.')
    if 'HStrain' in Exp:
        raise ValueError('You gave Exp.HStrain, but it should be Sys.HStrain (in the system, not the experiment structure).')
    nNuclei = int(Sys.nNuclei)
    if 'Nuclei' in Opt:
        if isempty(Opt.Nuclei):
            raise ValueError('Opt.Nuclei must contain the indices of nuclei to include in the simulation.')
        nn_ = vec(np.asarray(Opt.Nuclei, dtype=float))
        if np.any(nn_ > nNuclei) or np.any(nn_ < 0):
            raise ValueError('Index in Opt.Nuclei out of range.')
        shfNuclei = [int(v) for v in nn_]
    else:
        shfNuclei = list(range(1, nNuclei + 1))
        if shfNuclei:
            if np.all(idealPulse) and 'ExciteWidth' in Exp:
                EW = _num(Exp.ExciteWidth)
                A_ = np.atleast_2d(np.asarray(Sys.A, dtype=float))
                if bool(Sys.fullA):
                    idxStrong = [np.max(np.abs(A_[(iNuc - 1) * 3:(iNuc - 1) * 3 + 3, :])) > EW
                                 for iNuc in range(1, nNuclei + 1)]
                else:
                    idxStrong = list(np.max(np.abs(A_), axis=1) > EW)
                shfNuclei = [k for k, s in zip(shfNuclei, idxStrong) if not s]
    twoElectronManifolds = (Sys.nElectrons == 1 and _num(Sys.S) == 0.5 and len(shfNuclei) == nNuclei)
    if 'ProductRule' not in Opt:
        Opt.ProductRule = 0
    if nNuclei == 1:
        Opt.ProductRule = 0
    ProductRule = bool(Opt.ProductRule)
    if isENDOR:
        if 'EndorMethod' not in Opt:
            Opt.EndorMethod = 0 if (len(shfNuclei) == 1 or ProductRule) else 1
        if Opt.EndorMethod == 0:
            F.logmsg(1, 'using population swaps of adjacent nuclear sublevels')
        elif Opt.EndorMethod == 1:
            F.logmsg(1, 'using bandwidth-filtered Iy pi pulse, sum over transitions')
        elif Opt.EndorMethod == 2:
            F.logmsg(1, 'using bandwidth-filtered Iy pi pulse, full RF sweep')
        else:
            raise ValueError('Unknown setting for Opt.EndorMethod. Must be 0, 1, or 2.')
        if Opt.EndorMethod == 0 and len(shfNuclei) > 1 and not ProductRule:
            raise ValueError('Opt.EndorMethod=0 gives incorrect results for multiple nuclei.')
    if 'Expand' not in Opt:
        Opt.Expand = 4 if nDimensions == 1 else 2
    maxExpand = 8
    if numel(Opt.Expand) != 1 or _num(Opt.Expand) < 0 or _num(Opt.Expand) > maxExpand or _num(Opt.Expand) % 1:
        raise ValueError('Opt.Expand must be an integer between 0 and %d.' % maxExpand)
    if np.any(realPulse) and ProductRule:
        raise ValueError('saffron: Cannot apply product rule and real pulses at the same time.')
    if 'OriThreshold' not in Opt:
        Opt.OriThreshold = 0.005
    if 'Window' not in Opt:
        Opt.Window = 'ham+'
    if 'ZeroFillFactor' not in Opt:
        Opt.ZeroFillFactor = 2
    processData = True
    if 'logplot' not in Opt:
        Opt.logplot = 0
    if 'PartialIFFT' not in Opt:
        Opt.PartialIFFT = 1
    if 'TimeDomain' not in Opt:
        Opt.TimeDomain = 0
    TimeDomain = bool(Opt.TimeDomain)

    F.logmsg(1, '-Hamiltonians------------------------------------------')
    F.logmsg(1, 'setting up electronic Hamiltonian...')
    if not twoElectronManifolds:
        coreSys = Struct(F.nucspinrmv(Sys, np.array(shfNuclei, dtype=float)))
        coreSys.processed = False
        coreSys.lw = 0
        if 'A_' in coreSys:
            del coreSys['A']
        H0, mux, muy, muz = (_dense(m) for m in F.ham(coreSys, nargout=4)[:4])
        Sx = _dense(F.sop(coreSys, np.array([[1.0, 1.0]])))
        Sy = _dense(F.sop(coreSys, np.array([[1.0, 2.0]])))
        Sz = _dense(F.sop(coreSys, np.array([[1.0, 3.0]])))
    F.logmsg(1, 'computing nuclear spin sub-Hamiltonians...')
    if shfNuclei:
        nshf = len(shfNuclei)
        if ProductRule:
            F.logmsg(1, '  separate subspace for each of the %d superhyperfine nuclei', nshf)
            NucHams = [Struct() for _ in range(nshf)]
        else:
            F.logmsg(1, '  complete nuclear state space of all %d superhyperfine nuclei', nshf)
            NucHams = [Struct(Hnzx=0, Hnzy=0, Hnzz=0, Hhfx=0, Hhfy=0, Hhfz=0, Hnq=0)]
        if isENDOR:
            for f in ('Ix', 'Iy', 'Iz'):
                NucHams[0][f] = 0
        Ivec = vec(np.asarray(Sys.I, dtype=float))
        I = Ivec[np.array(shfNuclei) - 1]
        gn = vec(np.asarray(Sys.gn, dtype=float))
        A_all = np.atleast_2d(np.asarray(Sys.A, dtype=float))
        nm = _c(F.nmagn())
        pl = _c(F.planck())
        for iiNuc in range(nshf, 0, -1):
            if ProductRule:
                Ix = _dense(F.sop(float(I[iiNuc - 1]), np.array([[1.0, 1.0]])))
                Iy = _dense(F.sop(float(I[iiNuc - 1]), np.array([[1.0, 2.0]])))
                Iz = _dense(F.sop(float(I[iiNuc - 1]), np.array([[1.0, 3.0]])))
                NH = NucHams[iiNuc - 1]
            else:
                Ix = _dense(F.sop(I.reshape(1, -1), np.array([[float(iiNuc), 1.0]])))
                Iy = _dense(F.sop(I.reshape(1, -1), np.array([[float(iiNuc), 2.0]])))
                Iz = _dense(F.sop(I.reshape(1, -1), np.array([[float(iiNuc), 3.0]])))
                NH = NucHams[0]
            iNuc = shfNuclei[iiNuc - 1]
            if isENDOR:
                if ProductRule:
                    NH.Ix, NH.Iy, NH.Iz = Ix, Iy, Iz
                else:
                    NH.Ix = NH.Ix + Ix
                    NH.Iy = NH.Iy + Iy
                    NH.Iz = NH.Iz + Iz
            pre = -gn[iNuc - 1] * nm / 1e3 / pl / 1e6
            if ProductRule:
                NH.Hnzx, NH.Hnzy, NH.Hnzz = pre * Ix, pre * Iy, pre * Iz
            else:
                NH.Hnzx = NH.Hnzx + pre * Ix
                NH.Hnzy = NH.Hnzy + pre * Iy
                NH.Hnzz = NH.Hnzz + pre * Iz
            if bool(Sys.fullA):
                A = A_all[(iNuc - 1) * 3:(iNuc - 1) * 3 + 3, :]
            else:
                if 'AFrame' in Sys:
                    R_A2M = _dense(F.erot(np.atleast_2d(np.asarray(Sys.AFrame, dtype=float))[iNuc - 1, :])).T
                else:
                    R_A2M = np.eye(3)
                A = R_A2M @ np.diag(A_all[iNuc - 1, :]) @ R_A2M.T
                A = (A + A.T) / 2
            hx = A[0, 0] * Ix + A[0, 1] * Iy + A[0, 2] * Iz
            hy = A[1, 0] * Ix + A[1, 1] * Iy + A[1, 2] * Iz
            hz = A[2, 0] * Ix + A[2, 1] * Iy + A[2, 2] * Iz
            if ProductRule:
                NH.Hhfx, NH.Hhfy, NH.Hhfz = hx, hy, hz
            else:
                NH.Hhfx = NH.Hhfx + A[0, 0] * Ix + A[0, 1] * Iy + A[0, 2] * Iz
                NH.Hhfy = NH.Hhfy + A[1, 0] * Ix + A[1, 1] * Iy + A[1, 2] * Iz
                NH.Hhfz = NH.Hhfz + A[2, 0] * Ix + A[2, 1] * Iy + A[2, 2] * Iz
            Hnq_ = 0
            if Ivec[iNuc - 1] >= 1:
                Q_all = np.atleast_2d(np.asarray(Sys.Q, dtype=float))
                if bool(Sys.fullQ):
                    Q = Q_all[(iNuc - 1) * 3:(iNuc - 1) * 3 + 3, :]
                else:
                    if 'QFrame' in Sys:
                        R_Q2M = _dense(F.erot(np.atleast_2d(np.asarray(Sys.QFrame, dtype=float))[iNuc - 1, :])).T
                    else:
                        R_Q2M = np.eye(3)
                    Q = R_Q2M @ np.diag(Q_all[iNuc - 1, :]) @ R_Q2M.T
                    Q = (Q + Q.T) / 2
                Hnq_ = (Ix @ (Q[0, 0] * Ix + Q[0, 1] * Iy + Q[0, 2] * Iz)
                        + Iy @ (Q[1, 0] * Ix + Q[1, 1] * Iy + Q[1, 2] * Iz)
                        + Iz @ (Q[2, 0] * Ix + Q[2, 1] * Iy + Q[2, 2] * Iz))
            if ProductRule:
                NH.Hnq = Hnq_
            else:
                NH.Hnq = NH.Hnq + Hnq_
        nSubSpaces = len(NucHams)
        F.logmsg(1, '  %d nuclei, %d subspaces', nshf, nSubSpaces)
    else:
        F.logmsg(1, '  no subspace factorization')
        nSubSpaces = 0
        NucHams = []

    Field = _num(Exp.Field)
    orientationPreSelection = OrientationSelection and twoElectronManifolds
    if orientationPreSelection:
        F.logmsg(1, 'pre-computing orientation selecton from g tensor alone...')
        zLab = np.asarray(F.ang2vec(Orientations[:, 0], Orientations[:, 1]))
        g_ = np.asarray(Sys.g, dtype=float)
        geff = (g_ @ zLab) if bool(Sys.fullg) else (np.diag(vec(g_)) @ zLab)
        geff = np.sqrt(np.sum(geff ** 2, axis=0))
        nu = geff * _c(F.bmagn()) * Field / 1e3 / _c(F.planck()) / 1e6
        lw = np.diag(vec(np.asarray(Sys.HStrain, dtype=float))) @ zLab
        lw2 = np.sum(lw ** 2, axis=0) + _num(Exp.ExciteWidth) ** 2
        gOriSelWeight = np.exp(-(nu - 1000 * _num(Exp.mwFreq)) ** 2 / lw2)

    F.logmsg(1, 'Preparation...')
    nDim = nDimensions
    if isENDOR:
        Rg_ = vec(np.asarray(Exp.Range, dtype=float))
        rf = linspace(Rg_[0], Rg_[1], nPts[0])
        endorspc = np.zeros(nPts[0], dtype=complex)
    else:
        if not TimeDomain:
            ExpansionFactor = 2 ** int(_num(Opt.Expand))
            nPointsF = [ExpansionFactor * p for p in nPts]
            siz = (1, nPointsF[0]) if nDim == 1 else tuple(nPointsF)
            if ProductRule:
                pathwaybuffRe = [[np.zeros(siz, order='F') for _ in range(nSubSpaces)] for _ in range(nPathways)]
                pathwaybuffIm = [[np.zeros(siz, order='F') for _ in range(nSubSpaces)] for _ in range(nPathways)]
            else:
                buffRe = np.zeros(siz, order='F')
                buffIm = np.zeros(siz, order='F')
        totaltd = np.zeros(tuple(nPts)) if nDim == 2 else np.zeros((1, nPts[0]))
    F.logmsg(1, 'Looping over %d orientations...', nOrientations)
    if np.any(realPulse):
        if 'nOffsets' not in Opt:
            Opt.nOffsets = 291
        if 'lwOffset' not in Opt:
            Opt.lwOffset = 100
        offsets = linspace(-1, 1, int(_num(Opt.nOffsets))) * _num(Opt.lwOffset) * 2
        offsetWeight = np.exp(-(offsets / _num(Opt.lwOffset)) ** 2)
        offsetWeight = offsetWeight / np.sum(offsetWeight)
    else:
        offsets = np.array([0.0])
        offsetWeight = np.array([1.0])
        Opt.nOffsets = 1
    nOffsets = int(_num(Opt.nOffsets))
    if twoElectronManifolds:
        g = np.asarray(Sys.g, dtype=float)
        if not bool(Sys.fullg):
            Rg = _dense(F.erot(np.asarray(Sys.gFrame, dtype=float))).T
            g = Rg @ np.diag(vec(g)) @ Rg.T
    if not predefinedExperiment:
        tE = vec(np.asarray(Exp.t, dtype=float))
        Flip = vec(np.asarray(Exp.Flip, dtype=float))
    nSkippedOrientations = 0
    pathwaytd = None
    for iOri in range(nOrientations):
        _, yLab_M, zLab_M = F.erot(Orientations[iOri, :], 'rows', nargout=3)[:3]
        yLab_M, zLab_M = vec(yLab_M), vec(zLab_M)
        if twoElectronManifolds:
            OriSelWeight = np.array([gOriSelWeight[iOri] if OrientationSelection else 1.0])
            if OriSelWeight[0] < _num(Opt.OriThreshold):
                nSkippedOrientations += 1
                continue
            qa = g.T @ zLab_M
            qa = qa / np.linalg.norm(qa)
            Manifold = [Struct(S=-0.5 * qa), Struct(S=+0.5 * qa)]
            Transitions = np.array([[1, 2]])
            nTransitions = 1
            ManifoldsInvolved = [1, 2]
        else:
            muzL = zLab_M[0] * mux + zLab_M[1] * muy + zLab_M[2] * muz
            H = H0 - Field * muzL
            eE, eV = _eig(H)
            eE = np.real(eE)
            SyLab = yLab_M[0] * Sx + yLab_M[1] * Sy + yLab_M[2] * Sz
            SyLab = np.abs(eV.conj().T @ SyLab @ eV)
            maxSyy = np.max(SyLab)
            if OrientationSelection:
                dE = (eE.reshape(-1, 1) - eE.reshape(1, -1)) - _num(Exp.mwFreq) * 1e3
                excitationAmplitude = np.exp(-(dE / _num(Exp.ExciteWidth)) ** 2)
                SyLab = SyLab * excitationAmplitude
            SyLab[SyLab < _num(Opt.OriThreshold) * maxSyy] = 0
            if not isempty(Opt.Transitions):
                rmvTransition = np.ones(H.shape)
                TR = np.atleast_2d(np.asarray(Opt.Transitions, dtype=int))
                for tr in TR:
                    rmvTransition[tr[0] - 1, tr[1] - 1] = 0
                    rmvTransition[tr[1] - 1, tr[0] - 1] = 0
                SyLab[rmvTransition != 0] = 0
            T = np.tril(SyLab, -1)
            u_idx, v_idx = np.nonzero(T.T)     # find(): column-major order; v = row, u = column
            OriSelWeight = T[v_idx, u_idx]
            Transitions = np.stack([u_idx + 1, v_idx + 1], axis=1)
            nTransitions = Transitions.shape[0]
            if nTransitions == 0:
                nSkippedOrientations += 1
                continue
            mi = np.zeros(Sx.shape[0], dtype=bool)
            mi[u_idx] = True
            mi[v_idx] = True
            ManifoldsInvolved = list(np.nonzero(mi)[0] + 1)
            Manifold = [Struct() for _ in range(max(ManifoldsInvolved))]
            for iM in ManifoldsInvolved:
                vcol = eV[:, iM - 1]
                Manifold[iM - 1].S = np.real(np.array([vcol.conj() @ Sx @ vcol, vcol.conj() @ Sy @ vcol,
                                                       vcol.conj() @ Sz @ vcol]))
        F.logmsg(2, 'orientation %d of %d: %d transitions', iOri + 1, nOrientations, nTransitions)
        for iSpace in range(nSubSpaces):
            NH = NucHams[iSpace]
            Hnuc = Field * (zLab_M[0] * NH.Hnzx + zLab_M[1] * NH.Hnzy + zLab_M[2] * NH.Hnzz)
            Hnuc = Hnuc + NH.Hnq
            for iM in ManifoldsInvolved:
                S = Manifold[iM - 1].S
                H = Hnuc + S[0] * NH.Hhfx + S[1] * NH.Hhfy + S[2] * NH.Hhfz
                EE, VV = scipy.linalg.eigh((H + H.conj().T) / 2)
                Manifold[iM - 1].setdefault('V', [None] * nSubSpaces)[iSpace] = VV
                Manifold[iM - 1].setdefault('E', [None] * nSubSpaces)[iSpace] = np.real(EE)
        for iT in range(nTransitions):
            b = int(Transitions[iT, 0])
            a = int(Transitions[iT, 1])
            for iSpace in range(nSubSpaces):
                Ea = Manifold[a - 1].E[iSpace]
                Eb = Manifold[b - 1].E[iSpace]
                Ma = Manifold[a - 1].V[iSpace]
                Mb = Manifold[b - 1].V[iSpace]
                M = Ma.conj().T @ Mb
                Mt = M.conj().T
                nNucStates = Ea.size
                eyeN = np.eye(nNucStates)
                if np.any(realPulse):
                    idxa = np.arange(nNucStates)
                    idxb = idxa + nNucStates
                if isENDOR:
                    NH = NucHams[iSpace]
                    IyLab = yLab_M[0] * NH.Ix + yLab_M[1] * NH.Iy + yLab_M[2] * NH.Iz
                PulseSubPropagator = {}
                for iOffset in range(nOffsets):
                    for iInt in range(nIntervals):
                        if realPulse[iInt]:
                            nu1 = (math.pi / 2 * Flip[iInt]) / tp[iInt] / 2 / math.pi
                            Hpulse = np.block([[np.diag(Ea + offsets[iOffset] / 2), +M * nu1 / 2j],
                                               [-Mt * nu1 / 2j, np.diag(Eb - offsets[iOffset] / 2)]])
                            FPP = scipy.linalg.expm(-2j * math.pi * tp[iInt] * Hpulse)
                            PulseSubPropagator[(iInt, 1)] = FPP[np.ix_(idxa, idxa)]
                            PulseSubPropagator[(iInt, 2)] = FPP[np.ix_(idxb, idxb)]
                            PulseSubPropagator[(iInt, 3)] = FPP[np.ix_(idxa, idxb)]
                            PulseSubPropagator[(iInt, 4)] = FPP[np.ix_(idxb, idxa)]
                    if isENDOR:
                        prefactor = OriWeights[iOri] * OriSelWeight[iT]
                    else:
                        prefactor = OriWeights[iOri] * OriSelWeight[iT] if not ProductRule else 1
                        if not TimeDomain and ProductRule:
                            for iP in range(nPathways):
                                pathwaybuffRe[iP][iSpace] = np.zeros(siz, order='F')
                                pathwaybuffIm[iP][iSpace] = np.zeros(siz, order='F')
                    prefactor = prefactor * offsetWeight[iOffset]
                    if not predefinedExperiment:
                        increments = Inc
                        BlockL, BlockR = [], []
                        for iPathway in range(nPathways):
                            iBlock = 0
                            Left = prefactor * pathwayprefactor[iPathway]
                            Right = 1
                            for iInt in range(nIntervals):
                                if idealPulse[iInt]:
                                    kL = idxPulseL[iPathway, iInt]
                                    if kL == 3:
                                        Left = M @ Left if np.ndim(Left) else M * Left
                                    elif kL == 4:
                                        Left = Mt @ Left if np.ndim(Left) else Mt * Left
                                    kR = idxPulseR[iPathway, iInt]
                                    if kR == 3:
                                        Right = Right @ Mt if np.ndim(Right) else Right * Mt
                                    elif kR == 4:
                                        Right = Right @ M if np.ndim(Right) else Right * M
                                else:
                                    PL = PulseSubPropagator[(iInt, int(idxPulseL[iPathway, iInt]))]
                                    PR = PulseSubPropagator[(iInt, int(idxPulseR[iPathway, iInt]))]
                                    Left = PL @ Left if np.ndim(Left) else PL * Left
                                    Right = Right @ PR.conj().T if np.ndim(Right) else Right * PR.conj().T
                                t = tE[iInt]
                                if t > 0:
                                    E = Ea if idxFreeL[iPathway, iInt] == 1 else Eb
                                    dg = np.diag(np.exp(-2j * math.pi * E * t))
                                    Left = dg @ Left if np.ndim(Left) else dg * Left
                                    E = Ea if idxFreeR[iPathway, iInt] == 1 else Eb
                                    dg = np.diag(np.exp(+2j * math.pi * E * t))
                                    Right = Right @ dg if np.ndim(Right) else Right * dg
                                if increments[iInt] != 0:
                                    if iBlock == 0:
                                        G = (Left @ Right) if (np.ndim(Left) and np.ndim(Right)) else Left * Right
                                        if np.ndim(G) == 0:
                                            G = G * eyeN
                                    else:
                                        BL = eyeN if np.ndim(Left) == 0 else Left
                                        BR = eyeN if np.ndim(Right) == 0 else Right
                                        _setcell(BlockL, iBlock, BL)
                                        _setcell(BlockR, iBlock, BR)
                                    iBlock += 1
                                    Left = 1
                                    Right = 1
                            if increments[-1] != 0:
                                D = M
                            else:
                                RM = (Right @ M) if np.ndim(Right) else Right * M
                                D = (RM @ Left) if np.ndim(Left) else RM * Left
                            if ProductRule:
                                raise ValueError('Product rule for user-defined experiments not implemented.')
                            if TimeDomain:
                                totaltd = totaltd + _sf_evolve(IncSchemeID, nPts, dtv, idxIncL[iPathway, :],
                                                               idxIncR[iPathway, :], Ea, Eb, G, D, *BlockL, *BlockR)
                            else:
                                F.sf_peaks(IncSchemeID, buffRe, buffIm, Exp.dt, idxIncL[iPathway, :],
                                           idxIncR[iPathway, :], Ea, Eb, G, D, *BlockL, *BlockR)
                    else:
                        ID = Exp.ExperimentID
                        tau = _num(Exp.tau)
                        if ID == 5:
                            if not np.all(idealPulse):
                                raise ValueError('Pre-defined Mims ENDOR with real pulses not supported.')
                            Q_ = np.outer(np.exp(-2j * math.pi * Ea * tau), np.exp(-2j * math.pi * Eb * tau).conj())
                            G_ = prefactor * Q_ * M
                            D_ = np.conj(Q_) * M
                            G1 = G_ @ Mt
                            D1 = D_ @ Mt
                            G2 = Mt @ G_
                            D2 = Mt @ D_
                            if _num(Exp.T) != 0:
                                q_ = np.exp(-2j * math.pi * Ea * _num(Exp.T))
                                G1 = np.outer(q_, q_.conj()) * G1
                                q_ = np.exp(-2j * math.pi * Eb * _num(Exp.T))
                                G2 = np.outer(q_, q_.conj()) * G2
                            G1 = np.diag(np.diag(G1))
                            G2 = np.diag(np.diag(G2))
                            traceG1D1 = np.trace(G1 @ D1)
                            traceG2D2 = np.trace(G2 @ D2)
                            nu1 = np.abs(Ea.reshape(-1, 1) - Ea.reshape(1, -1))
                            nu2 = np.abs(Eb.reshape(-1, 1) - Eb.reshape(1, -1))
                            Iy1 = Ma.conj().T @ IyLab @ Ma
                            Iy2 = Mb.conj().T @ IyLab @ Mb
                            fwhm = 1 / _num(Exp.tprf)
                            Gamma = fwhm / (2 * math.sqrt(math.log(2)))
                            theta = math.pi
                            Gam = Sys.lwEndor / math.sqrt(2 * math.log(2))
                            pre = math.sqrt(2 / math.pi) / Gam
                            method = int(_num(Opt.EndorMethod))
                            if method == 2:
                                BWthreshold = 0.01
                                for irf in range(rf.size):
                                    BW1 = np.exp(-((nu1 - rf[irf]) / Gamma) ** 2)
                                    BW2 = np.exp(-((nu2 - rf[irf]) / Gamma) ** 2)
                                    if np.any(BW1 > BWthreshold):
                                        Prfa = scipy.linalg.expm(-1j * theta * (Iy1 * BW1))
                                        G1_ = Prfa @ G1 @ Prfa.conj().T
                                        G1_ = np.diag(np.diag(G1_))
                                        endorspc[irf] = endorspc[irf] + traceG1D1 - np.trace(G1_ @ D1)
                                    if np.any(BW2 > BWthreshold):
                                        Prfb = scipy.linalg.expm(-1j * theta * (Iy2 * BW2))
                                        G2_ = Prfb @ G2 @ Prfb.conj().T
                                        G2_ = np.diag(np.diag(G2_))
                                        endorspc[irf] = endorspc[irf] + traceG2D2 - np.trace(G2_ @ D2)
                            elif method == 0:
                                for j in range(1, nNucStates):
                                    i = j + 1
                                    G1_ = G1.copy()
                                    q = G1_[i - 1, i - 1]
                                    G1_[i - 1, i - 1] = G1_[j - 1, j - 1]
                                    G1_[j - 1, j - 1] = q
                                    G2_ = G2.copy()
                                    q = G2_[i - 1, i - 1]
                                    G2_[i - 1, i - 1] = G2_[j - 1, j - 1]
                                    G2_[j - 1, j - 1] = q
                                    ampl = [traceG1D1 - np.trace(G1_ @ D1), traceG2D2 - np.trace(G2_ @ D2)]
                                    freq = [Ea[i - 1] - Ea[j - 1], Eb[i - 1] - Eb[j - 1]]
                                    endorspc = endorspc + ampl[0] * pre * np.exp(-2 * ((rf - freq[0]) / Gam) ** 2)
                                    endorspc = endorspc + ampl[1] * pre * np.exp(-2 * ((rf - freq[1]) / Gam) ** 2)
                            elif method == 1:
                                ij = np.array(list(itertools.combinations(range(1, nNucStates + 1), 2))).reshape(-1, 2)
                                for (i, j) in ij:
                                    freq1 = nu1[i - 1, j - 1]
                                    freq2 = nu2[i - 1, j - 1]
                                    BW1 = np.exp(-((nu1 - freq1) / Gamma) ** 2)
                                    BW2 = np.exp(-((nu2 - freq2) / Gamma) ** 2)
                                    Prfa = scipy.linalg.expm(-1j * theta * (Iy1 * BW1))
                                    Prfb = scipy.linalg.expm(-1j * theta * (Iy2 * BW2))
                                    G1_ = np.diag(np.diag(Prfa @ G1 @ Prfa.conj().T))
                                    G2_ = np.diag(np.diag(Prfb @ G2 @ Prfb.conj().T))
                                    sig1 = traceG1D1 - np.trace(G1_ @ D1)
                                    sig2 = traceG2D2 - np.trace(G2_ @ D2)
                                    endorspc = endorspc + sig1 * pre * np.exp(-((rf - freq1) / Gam) ** 2)
                                    endorspc = endorspc + sig2 * pre * np.exp(-((rf - freq2) / Gam) ** 2)
                        elif ID in (4, 3):
                            name = 'HYSCORE' if ID == 4 else '4p-ESEEM'
                            if not np.all(idealPulse):
                                raise ValueError(f'Pre-defined {name} with real pulses not supported.')
                            Q_ = np.outer(np.exp(-2j * math.pi * Ea * tau), np.exp(-2j * math.pi * Eb * tau).conj())
                            G_ = prefactor * Q_ * M
                            D_ = np.conj(Q_) * M
                            if ID == 4:
                                G1 = G_ @ Mt
                                G2 = Mt @ G_
                                D1 = Mt @ D_
                                D2 = D_ @ Mt
                            else:
                                G1 = G_ @ Mt
                                D1 = Mt @ D_
                                G2 = Mt @ G_
                                D2 = D_ @ Mt
                            Tl1, Tr1 = Mt, M
                            Tl2, Tr2 = M, Mt
                            if ID == 4:
                                t1_, t2_ = _num(Exp.t1), _num(Exp.t2)
                                if t1_ != 0:
                                    q_ = np.exp(-2j * math.pi * t1_ * Ea)
                                    G1 = np.outer(q_, q_.conj()) * G1
                                    q_ = np.exp(-2j * math.pi * t1_ * Eb)
                                    G2 = np.outer(q_, q_.conj()) * G2
                                if t2_ != 0:
                                    q_ = np.exp(+2j * math.pi * t2_ * Eb)
                                    D1 = np.outer(q_, q_.conj()) * D1
                                    q_ = np.exp(+2j * math.pi * t2_ * Ea)
                                    D2 = np.outer(q_, q_.conj()) * D2
                            else:
                                T_ = _num(Exp.T)
                                if T_ != 0:
                                    q_ = np.exp(-2j * math.pi * T_ * Ea)
                                    G1 = np.outer(q_, q_.conj()) * G1
                                    q_ = np.exp(-2j * math.pi * T_ * Eb)
                                    G2 = np.outer(q_, q_.conj()) * G2
                                    q_ = np.exp(+2j * math.pi * T_ * Eb)
                                    D1 = np.outer(q_, q_.conj()) * D1
                                    q_ = np.exp(+2j * math.pi * T_ * Ea)
                                    D2 = np.outer(q_, q_.conj()) * D2
                            a1 = (IncSchemeID, nPts, dtv, [1, 2], [1, 2], Ea, Eb, G1, D1, Tl1, Tr1)
                            a2 = (IncSchemeID, nPts, dtv, [2, 1], [2, 1], Ea, Eb, G2, D2, Tl2, Tr2)
                            pathwaytd = _predef_accumulate(TimeDomain, ProductRule, iSpace, a1, a2, pathwaytd,
                                                           nPathways, nSubSpaces)
                            if TimeDomain and not ProductRule:
                                totaltd = totaltd + _sf_evolve(*a1)
                                totaltd = totaltd + _sf_evolve(*a2)
                            elif not TimeDomain:
                                bR1, bI1 = ((pathwaybuffRe[0][iSpace], pathwaybuffIm[0][iSpace]) if ProductRule
                                            else (buffRe, buffIm))
                                bR2, bI2 = ((pathwaybuffRe[1][iSpace], pathwaybuffIm[1][iSpace]) if ProductRule
                                            else (buffRe, buffIm))
                                F.sf_peaks(IncSchemeID, bR1, bI1, Exp.dt, np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]]),
                                           Ea, Eb, G1, D1, Tl1, Tr1)
                                F.sf_peaks(IncSchemeID, bR2, bI2, Exp.dt, np.array([[2.0, 1.0]]), np.array([[2.0, 1.0]]),
                                           Ea, Eb, G2, D2, Tl2, Tr2)
                        elif ID == 2:
                            if not np.all(idealPulse):
                                raise ValueError('Pre-defined 3pESEEM not supported for real pulses.')
                            Q_ = np.outer(np.exp(-2j * math.pi * Ea * tau), np.exp(-2j * math.pi * Eb * tau).conj())
                            G_ = prefactor * Q_ * M
                            D_ = np.conj(Q_) * M
                            G1 = G_ @ Mt
                            D1 = D_ @ Mt
                            G2 = Mt @ G_
                            D2 = Mt @ D_
                            T_ = _num(Exp.T)
                            if T_ != 0:
                                q_ = np.exp(-2j * math.pi * T_ * Ea)
                                G1 = np.outer(q_, q_.conj()) * G1
                                q_ = np.exp(-2j * math.pi * T_ * Eb)
                                G2 = np.outer(q_, q_.conj()) * G2
                            a1 = (IncSchemeID, nPts, dtv, 1, 1, Ea, Eb, G1, D1)
                            a2 = (IncSchemeID, nPts, dtv, 2, 2, Ea, Eb, G2, D2)
                            pathwaytd = _predef_accumulate(TimeDomain, ProductRule, iSpace, a1, a2, pathwaytd,
                                                           nPathways, nSubSpaces)
                            if TimeDomain and not ProductRule:
                                totaltd = totaltd + _sf_evolve(*a1)
                                totaltd = totaltd + _sf_evolve(*a2)
                            elif not TimeDomain:
                                bR1, bI1 = ((pathwaybuffRe[0][iSpace], pathwaybuffIm[0][iSpace]) if ProductRule
                                            else (buffRe, buffIm))
                                bR2, bI2 = ((pathwaybuffRe[1][iSpace], pathwaybuffIm[1][iSpace]) if ProductRule
                                            else (buffRe, buffIm))
                                F.sf_peaks(IncSchemeID, bR1, bI1, Exp.dt, 1.0, 1.0, Ea, Eb, G1, D1)
                                F.sf_peaks(IncSchemeID, bR2, bI2, Exp.dt, 2.0, 2.0, Ea, Eb, G2, D2)
                        elif ID == 1:
                            if not np.all(idealPulse):
                                raise ValueError('Pre-defined 2pESEEM not supported for real pulses.')
                            G = prefactor * M
                            D = M
                            T1left = Mt
                            T1right = Mt
                            if tau > 0:
                                Q_ = np.outer(np.exp(-2j * math.pi * tau * Ea), np.exp(-2j * math.pi * tau * Eb).conj())
                                G = Q_ * G
                                D = np.conj(Q_) * D
                            a1 = (IncSchemeID, nPts, dtv, [1, 2], [2, 1], Ea, Eb, G, D, T1left, T1right)
                            if TimeDomain:
                                if ProductRule:
                                    pathwaytd = _ptd_set(pathwaytd, nPathways, nSubSpaces, 0, iSpace, _sf_evolve(*a1))
                                else:
                                    totaltd = totaltd + _sf_evolve(*a1)
                            else:
                                bR1, bI1 = ((pathwaybuffRe[0][iSpace], pathwaybuffIm[0][iSpace]) if ProductRule
                                            else (buffRe, buffIm))
                                F.sf_peaks(IncSchemeID, bR1, bI1, Exp.dt, np.array([[1.0, 2.0]]), np.array([[2.0, 1.0]]),
                                           Ea, Eb, G, D, T1left, T1right)
            if ProductRule and not isENDOR:
                for iPathway in range(nPathways):
                    td_ = 1
                    for iSpace in range(nSubSpaces):
                        if TimeDomain:
                            thistd = pathwaytd[iPathway][iSpace]
                        else:
                            pathwaybuff = pathwaybuffRe[iPathway][iSpace] + 1j * pathwaybuffIm[iPathway][iSpace]
                            if nDim == 1:
                                thistd = np.fft.ifft(pathwaybuff, axis=1) * nPointsF[0]
                                thistd = thistd[:, :nPts[0]]
                            else:
                                thistd = _ifft2dpartial(pathwaybuff, nPts, Opt.PartialIFFT) * int(np.prod(nPointsF))
                        td_ = td_ * thistd
                    totaltd = totaltd + OriWeights[iOri] * OriSelWeight[iT] * td_
    F.logmsg(1, 'end of orientation/transition loop')
    F.logmsg(1, '%d of %d orientations skipped', nSkippedOrientations, nOrientations)

    Ivec = vec(np.asarray(Sys.I, dtype=float)) if not isempty(Sys.I) else np.zeros(0)
    EqDensityTrace = float(np.prod(2 * Ivec + 1))
    if isENDOR:
        endorspc = np.real(endorspc)
        endorspc = endorspc / EqDensityTrace
        endorspc = endorspc / nPathways
        if Opt.EndorMethod == 2:
            endorspc = vec(F.convspec(endorspc.reshape(1, -1), rf[1] - rf[0], Sys.lwEndor))
    else:
        if ProductRule:
            td = totaltd
        elif TimeDomain:
            td = totaltd
        else:
            F.logmsg(1, 'Postprocessing...')
            buff = buffRe + 1j * buffIm
            if nDim == 1:
                td = np.fft.ifft(buff, axis=1) * buff.size
                td = td[:, :nPts[0]]
            else:
                td = _ifft2dpartial(buff, nPts, Opt.PartialIFFT) * buff.size
        td = td / EqDensityTrace
        td = td / nPathways
        if not predefinedExperiment:
            td = td / np.max(np.abs(pathwayprefactor))
    if not isENDOR:
        ID = Exp.ExperimentID
        if ID == -1:
            if nDim == 1:
                t1 = np.arange(nPts[0]) * dtv[0]
            else:
                t1 = np.arange(nPts[0]) * dtv[0]
                t2 = np.arange(nPts[1]) * dtv[1]
        elif ID == 1:
            t1 = np.arange(nPts[0]) * dtv[0] + _num(Exp.tau)
        elif ID == 2:
            t1 = np.arange(nPts[0]) * dtv[0] + _num(Exp.tau) + _num(Exp.T)
        elif ID == 3:
            t1 = np.arange(nPts[0]) * dtv[0] + _num(Exp.T)
        elif ID == 4:
            t1 = np.arange(nPts[0]) * dtv[0] + _num(Exp.t1)
            t2 = np.arange(nPts[1]) * dtv[1] + _num(Exp.t2)
        decayAdded = False
        T1T2 = vec(Sys.T1T2)
        if np.any(~np.isinf(T1T2)):
            F.logmsg(1, 'Adding relaxation decays...')
            T1, T2 = T1T2[0], T1T2[1]
            tdecay = None
            if ID == 1:
                if not np.isinf(T2):
                    tdecay = np.exp(-2 * t1 / T2)
                    td = td * tdecay
            elif ID == 2:
                if not np.isinf(T1):
                    tdecay = np.exp(-2 * _num(Exp.tau) / T2) * np.exp(-t1 / T1)
                    td = td * tdecay
            elif ID == 4:
                if not np.isinf(T1):
                    tdecay = np.outer(np.exp(-t1 / T1), np.exp(-t2 / T1))
                    tdecay = np.exp(-2 * _num(Exp.tau) / T2) * tdecay
                    td = td * tdecay
            decayAdded = tdecay is not None
    F.logmsg(1, '-final-------------------------------------------------')
    F.logmsg(1, 'Data processing...')
    info = Struct()
    if not isENDOR:
        if nDim == 1:
            td = td.reshape(1, -1)
            if processData:
                if decayAdded:
                    tdfit = F.exponfit(t1, td, 1)[2]
                    tdx = td - np.asarray(tdfit).reshape(1, -1)
                else:
                    tdx = td
                tdx = tdx - np.mean(tdx)
                win = np.asarray(F.apowin(Opt.Window, tdx.size)).reshape(1, -1)
                tdx = tdx * win
                fd = np.fft.fft(tdx, int(_num(Opt.ZeroFillFactor)) * tdx.size, axis=1)
                fd = np.fft.fftshift(fd, axes=1)
                f1 = F.fdaxis(dtv[0], fd.size)
        else:
            if processData:
                if decayAdded:
                    tdx = F.basecorr(td, np.zeros((0, 0)), np.array([[2, 2]]))[0]
                else:
                    tdx = F.basecorr(td, np.zeros((0, 0)), np.array([[0, 0]]))[0]
                w1 = np.asarray(F.apowin(Opt.Window, nPts[0])).reshape(-1, 1)
                w2 = np.asarray(F.apowin(Opt.Window, nPts[1])).reshape(-1, 1)
                zf = int(_num(Opt.ZeroFillFactor))
                fd = np.fft.fftshift(np.fft.fftn(np.asarray(tdx) * (w1 @ w2.T), s=(zf * nPts[0], zf * nPts[1]), axes=(0, 1)))
                f1 = F.fdaxis(dtv[0], fd.shape[0])
                f2 = F.fdaxis(dtv[1], fd.shape[1])
        if np.max(np.abs(fd)) < 1e-300:
            fd = fd * 0
        if processData:
            if nDim == 2:
                info.f1 = f1
                info.f2 = f2
            else:
                info.f = f1
            info.fd = fd
    else:
        f1 = rf
        fd = endorspc
        if np.max(np.abs(fd)) < 1e-300:
            fd = fd * 0
    if isENDOR:
        x_out = f1.reshape(1, -1)
        y_out = fd.reshape(1, -1)
    else:
        x_out = [t1.reshape(1, -1), t2.reshape(1, -1)] if nDim == 2 else t1.reshape(1, -1)
        y_out = td
    if isENDOR:
        info.fd = fd.reshape(1, -1)
        info.td = np.zeros((0, 0))
    else:
        info.td = td
        info.fd = fd
    if nargout == 1:
        return y_out
    if nargout == 3:
        return x_out, y_out, info
    return x_out, y_out


def _setcell(L, k, v):
    while len(L) < k:
        L.append(np.zeros((0, 0)))
    L[k - 1] = v


def _ptd_set(ptd, nP, nS, iP, iS, val):
    if ptd is None:
        ptd = [[None] * nS for _ in range(max(nP, 2))]
    ptd[iP][iS] = val
    return ptd


def _predef_accumulate(TimeDomain, ProductRule, iSpace, a1, a2, ptd, nP, nS):
    if TimeDomain and ProductRule:
        ptd = _ptd_set(ptd, nP, nS, 0, iSpace, _sf_evolve(*a1))
        ptd = _ptd_set(ptd, nP, nS, 1, iSpace, _sf_evolve(*a2))
    return ptd


def _ifft2dpartial(spc, N, usePartialIFFT):
    if bool(usePartialIFFT):
        td = np.fft.ifft(spc, axis=0)
        td = np.fft.ifft(td[:N[0], :], axis=1)
        return td[:, :N[1]]
    td = np.fft.ifft2(spc)
    return td[:N[0], :N[1]]


def _sf_evolve(IncSchemeID, nPoints, dt, incL, incR, Ea, Eb, G, D, *T):
    """Time-domain evolution kernel (Opt.TimeDomain = 1)."""
    incL = vec(np.asarray(incL, dtype=float)).astype(int)
    incR = vec(np.asarray(incR, dtype=float)).astype(int)
    if len(T) != 2 * (incL.size - 1):
        raise ValueError('Time-domain evolution kernel: Wrong number of transfer matrices.')
    nPoints = [int(p) for p in np.atleast_1d(nPoints)]
    dt = vec(np.asarray(dt, dtype=float))
    G = np.asarray(G)
    NN = G.size
    Density = G
    Detector = np.asarray(D).T.reshape(1, NN, order='F')
    E = [vec(Ea), vec(Eb)]
    Signal = np.zeros((1, nPoints[0]), dtype=complex) if len(nPoints) == 1 else np.zeros(tuple(nPoints), dtype=complex)
    if IncSchemeID == 1:
        FinalDensity = Density.reshape(-1, 1, order='F')
        UUt = np.outer(np.exp(-2j * math.pi * E[incL[0] - 1] * dt[0]), np.exp(-2j * math.pi * E[incR[0] - 1] * dt[0]).conj())
        UUt = UUt.reshape(-1, 1, order='F')
        for k1 in range(nPoints[0]):
            Signal.ravel(order='F')
            Signal[0, k1] = (Detector @ FinalDensity)[0, 0]
            FinalDensity = UUt * FinalDensity
    elif IncSchemeID == 2:
        T1l, T1r = np.asarray(T[0]), np.asarray(T[1])
        UUleft = np.outer(np.exp(-2j * math.pi * E[incL[1] - 1] * dt[0]), np.exp(-2j * math.pi * E[incL[0] - 1] * dt[0]))
        UUright = np.outer(np.exp(+2j * math.pi * E[incR[0] - 1] * dt[0]), np.exp(+2j * math.pi * E[incR[1] - 1] * dt[0]))
        for k in range(nPoints[0]):
            FinalDensity = T1l @ Density @ T1r
            Signal[0, k] = (Detector @ FinalDensity.reshape(-1, 1, order='F'))[0, 0]
            T1l = UUleft * T1l
            T1r = UUright * T1r
    elif IncSchemeID == 3:
        raise ValueError('Time-domain evolution for incrementation scheme [1 -1] not supported.')
    elif IncSchemeID == 11:
        T1l, T1r = np.asarray(T[0]), np.asarray(T[1])
        UUt1 = np.outer(np.exp(-2j * math.pi * E[incL[0] - 1] * dt[0]), np.exp(-2j * math.pi * E[incR[0] - 1] * dt[0]).conj())
        UUt2 = np.outer(np.exp(-2j * math.pi * E[incL[1] - 1] * dt[1]), np.exp(-2j * math.pi * E[incR[1] - 1] * dt[1]).conj())
        UUt2 = UUt2.reshape(-1, 1, order='F')
        for k1 in range(nPoints[0]):
            FinalDensity = (T1l @ Density @ T1r).reshape(NN, 1, order='F')
            for k2 in range(nPoints[1]):
                Signal[k1, k2] = (Detector @ FinalDensity)[0, 0]
                FinalDensity = UUt2 * FinalDensity
            Density = UUt1 * Density
    else:
        raise ValueError('Time-domain evolution for this experiment not supported.')
    return Signal


def _pathwayparser(ctp):
    """(FreeL, FreeR, PulsL, PulsR): manifold indices (1 alpha, 2 beta) of the left/right free
    evolution propagators and the pulse sub-propagator codes (1..4) for each pathway and interval."""
    PropL = np.array([[1, 3, 1, 3], [4, 2, 4, 2], [1, 3, 1, 3], [4, 2, 4, 2]])
    PropR = np.array([[1, 3, 3, 1], [4, 2, 2, 4], [4, 2, 2, 4], [1, 3, 3, 1]])
    if isinstance(ctp, str):
        ctp = np.array([['ab+-'.index(c) + 1 for c in ctp]])
    ctp = np.atleast_2d(np.asarray(ctp, dtype=int))
    before = np.concatenate([2 * np.ones((ctp.shape[0], 1), dtype=int), ctp[:, :-1]], axis=1)
    idx = ctp + (before - 1) * 4                       # linear (column-major) indices into PropL/R
    PulsL = PropL.ravel(order='F')[idx - 1]
    PulsR = PropR.ravel(order='F')[idx - 1]
    idx = 5 * ctp - 4
    FreeL = PropL.ravel(order='F')[idx - 1]
    FreeR = PropR.ravel(order='F')[idx - 1]
    return FreeL, FreeR, PulsL, PulsR


# ----------------------------------------------------------------------------- thyme mode
def _rotatesystem(Sys_M, angles_M2L):
    Sys_L = Struct(Sys_M)
    R_M2L = _dense(F.erot(angles_M2L))
    R_L2M = R_M2L.T
    tensors = ['g', 'A', 'Q', 'D', 'ee', 'nn']
    ang_L2T = None
    for tname in [t for t in tensors if t in Sys_M]:
        tFrame_ = tname + 'Frame'
        if tFrame_ not in Sys_M:
            raise ValueError(f'Unrecognized field name "{tFrame_}".')
        ang_M2T = np.atleast_2d(np.asarray(Sys_M[tFrame_], dtype=float))
        if ang_M2T.size == 0:
            ang_M2T = ang_M2T.reshape(0, 0)
        for iRow in range(ang_M2T.shape[0]):
            for iCol in range(ang_M2T.shape[1] // 3):
                colidx = slice(3 * iCol, 3 * iCol + 3)
                R_M2T = _dense(F.erot(ang_M2T[iRow, colidx]))
                R_L2T = R_M2T @ R_L2M
                val = vec(F.eulang(R_L2T))
                # ang_L2T(iRow,colidx) = ... (the variable persists across tensors, as in EasySpin)
                need = (iRow + 1, 3 * iCol + 3)
                if ang_L2T is None:
                    ang_L2T = np.zeros(need)
                elif ang_L2T.shape[0] < need[0] or ang_L2T.shape[1] < need[1]:
                    B = np.zeros((max(ang_L2T.shape[0], need[0]), max(ang_L2T.shape[1], need[1])))
                    B[:ang_L2T.shape[0], :ang_L2T.shape[1]] = ang_L2T
                    ang_L2T = B
                ang_L2T[iRow, colidx] = val
        if ang_L2T is None:
            raise NameError("Unrecognized function or variable 'ang_L2T'.")
        Sys_L[tFrame_] = ang_L2T.copy()
    return Sys_L


def _thyme(Sys, Exp, Opt, Events, Vary, Orientations, nOrientations, OriWeights, nDataPoints, nargout, plotting):
    F.logmsg(1, '-processing Sys structure------------------------------')
    Sys, Sigma, DetOps, Events, Relaxation = F.s_propagationsetup(Sys, Events, Opt)[:5]
    Sys = Struct(Sys)
    nElectrons = numel(Sys.S)
    FrameShift = 0.0 if isempty(Opt.FrameShift) else float(Opt.FrameShift)
    if FrameShift != 0:
        F.logmsg(1, '  adapting Sys.g to the simulation frame')
    Field = vec(np.asarray(Exp.Field, dtype=float))
    gshift = (FrameShift * 1e9) * _c(F.planck()) / _c(F.bmagn()) / (Field[-1] * 1e-3)
    g = np.asarray(Sys.g, dtype=float)
    fullg = g.ndim == 2 and g.shape == (3 * nElectrons, 3)
    if fullg:
        Sys.g = g - np.tile(gshift * np.eye(3), (nElectrons, 1))
    else:
        Sys.g = g - gshift
    rawSignals = [None] * nOrientations
    timeAxis = [None] * nOrientations
    F.logmsg(1, '-starting orientation loop-----------------------------')
    if bool(Opt.Relaxation):
        F.logmsg(1, '  relaxation is active during simulation')
    else:
        F.logmsg(1, '  no relaxation during propagation')
    F.logmsg(1, '  propagating %d orientations', nOrientations)
    FieldS = _num(Exp.Field)
    # parfor: without a pool MATLAB runs the iterations from last to first; the same order here
    # keeps the callee sequence identical to the recorded one (results do not depend on it)
    for iOrientation in range(nOrientations - 1, -1, -1):
        Sys_L = _rotatesystem(Sys, Orientations[iOrientation, :])
        Ham = _dense(F.ham(Sys_L, FieldS * np.array([[0.0, 0.0, 1.0]])))
        Relaxation_ = Relaxation
        if not isempty(Relaxation_):
            F.logmsg(1, '  adapting relaxation superoperator to system frame')
            _, U = _eig(Ham)
            R = np.kron(U.T, U.conj().T)
            Relaxation_ = Struct(Relaxation_)
            Relaxation_.Gamma = R.conj().T @ _dense(Relaxation_.Gamma) @ R
        res = F.s_thyme(Sigma, Ham, DetOps, Events, Relaxation_, Vary)
        timeAxis[iOrientation], rawSignals[iOrientation] = res[0], res[1]
    timeAxis = timeAxis[0]
    Signal = 0
    for iOrientation in range(nOrientations):
        Signal = Signal + np.asarray(rawSignals[iOrientation]) * OriWeights[iOrientation]
    if 'DetPhase' in Exp:
        F.logmsg(1, '  applying detection phase: %d*pi', _num(Exp.DetPhase) / math.pi)
        phase = np.exp(-1j * _num(Exp.DetPhase))
    else:
        F.logmsg(1, '  applying default detection phase: 0')
        phase = np.exp(-1j * 0)
    Signal = Signal * phase
    if not Opt.SinglePointDetection and 'DetIntegrate' in Exp and bool(Exp.DetIntegrate):
        F.logmsg(1, '-integrating echos-------------------------------------')
        sizeSig = list(np.shape(Signal))
        Signal = np.reshape(Signal, (nDataPoints, sizeSig[-1]), order='F')
        Signal = np.sum(Signal, axis=1)
        newshape = sizeSig[:-2] + [1]
        if len(newshape) < 2:
            newshape = newshape + [1] * (2 - len(newshape))
        Signal = np.reshape(Signal, newshape, order='F')
        Opt.SinglePointDetection = True
        F.logmsg(1, '  integrated %d transients', Signal.size)
    if not Opt.SinglePointDetection:
        F.logmsg(1, '-processing transients---------------------------------')
        if 'DetFrequency' not in Exp and 'mwFreq' in Exp:
            F.logmsg(1, '  using Exp.mwFreq as detection frequency')
            Exp.DetFrequency = Exp.mwFreq
        FreqTranslation = -(np.asarray(Exp.DetFrequency, dtype=float) - Events[0]['FrameShift'])
        Signal = np.asarray(F.signalprocessing(timeAxis, Signal, FreqTranslation))
        while Signal.ndim > 2 and Signal.shape[-1] == 1:      # MATLAB drops trailing singletons
            Signal = Signal[..., 0]
        if Signal.ndim == 2 and Signal.shape[1] == 1:
            Signal = Signal.T
        dt = Events[-1]['TimeStep']
        DW = vec(np.asarray(Exp.DetWindow, dtype=float))
        timeAxis = colon(DW[0], dt, DW[1]).reshape(1, -1)
    F.logmsg(1, '-setting up axes for output----------------------------')
    if 'nPoints' not in Exp:
        F.logmsg(1, '  no indirect dimensions')
        nIndirectDimensions = 0
        x = np.zeros((0, 0))
    else:
        nP = [int(v) for v in vec(np.asarray(Exp.nPoints, dtype=float))]
        nIndirectDimensions = len(nP)
        F.logmsg(1, '  setting up %d axes for indirect dimensions', nIndirectDimensions)
        x = [None] * nIndirectDimensions
        predefinedExperiment = 'Sequence' in Exp and not isempty(Exp.Sequence) and isinstance(Exp.Sequence, str)
        if predefinedExperiment:
            ExperimentNames = ['2pESEEM', '3pESEEM', '4pESEEM', 'HYSCORE', 'MimsENDOR']
            ids = [k + 1 for k, nm in enumerate(ExperimentNames) if nm == Exp.Sequence]
            Exp.ExperimentID = ids[0] if ids else np.zeros((0, 0))
            dtE = _num(Exp.dt) if 'dt' in Exp else None
            ID = Exp.ExperimentID
            if ID == 1:
                taustart = _num(Exp.tau) if 'tau' in Exp else 0
                x[0] = taustart + dtE * np.arange(nP[0]).reshape(1, -1)
            elif ID == 2:
                Tstart = _num(Exp.T) if 'T' in Exp else 0
                taustart = _num(Exp.tau) if 'tau' in Exp else 0
                x[0] = taustart + Tstart + dtE * np.arange(nP[0]).reshape(1, -1)
            elif ID == 3:
                Tstart = _num(Exp.T) if 'T' in Exp else 0
                x[0] = Tstart + dtE * np.arange(nP[0]).reshape(1, -1)
            elif ID == 4:
                for iDim in range(2):
                    nm = 't' + str(iDim + 1)
                    tstart = _num(Exp[nm]) if nm in Exp else 0
                    x[iDim] = tstart + dtE * np.arange(nP[iDim]).reshape(1, -1)
        else:
            from .s_sequencer import _dimrows
            for iDim in range(nIndirectDimensions):
                rows = _dimrows(Exp['Dim' + str(iDim + 1)])
                axes_ = np.zeros((len(rows), nP[iDim]))
                for iParameter, (_, val) in enumerate(rows):
                    if np.size(val) == 1:
                        axes_[iParameter, :] = _num(val) * np.arange(nP[iDim])
                    else:
                        axes_[iParameter, :] = np.arange(1, nP[iDim] + 1)
                x[iDim] = axes_
    if not Opt.SinglePointDetection:
        F.logmsg(1, '  adding axis of direct dimension')
        if not isinstance(x, list):
            x = []
        while len(x) < nIndirectDimensions + 1:
            x.append(np.zeros((0, 0)))
        x[nIndirectDimensions] = timeAxis
    if isinstance(x, list) and len(x) == 1:
        x = x[0]
    info = Struct(SinglePointDetection=Opt.SinglePointDetection)
    if plotting:
        F.s_plotting(timeAxis, Signal, Exp, Opt)
        if nargout == 0:
            return x, Signal
    if nargout == 1:
        return Signal
    if nargout == 3:
        return x, Signal, info
    return x, Signal


# ----------------------------------------------------------------------------- plot
def _saffron_plot(x, info, isENDOR, Sys, Exp, Opt):
    """Figure of MATLAB's nargout-0 branch (matplotlib); returns the figure."""
    import matplotlib.pyplot as plt
    F.logmsg(1, 'Plotting...')
    twoDim = isinstance(x, list)
    x1 = vec(x[0]) if twoDim else vec(x)
    fig = plt.figure()
    if isENDOR:
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x1, vec(info.fd))
        ax.set_xlabel('frequency (MHz)')
        ax.set_ylabel('intensity (arb.u.)')
        tau = _num(Exp.tau) if 'tau' in Exp else 0
        if 'tau' in Exp and tau < 1:
            ax.set_title('Mims ENDOR, %g mT, tau = %g ns' % (_num(Exp.Field), 1000 * tau), fontweight='bold')
        else:
            ax.set_title('Mims ENDOR, %g mT, tau = %g µs' % (_num(Exp.Field), tau), fontweight='bold')
        return fig
    plotFreqDomain = ('f' in info or 'f1' in info) and 'fd' in info
    if not twoDim:
        ax = fig.add_subplot(2, 1, 1) if plotFreqDomain else fig.add_subplot(1, 1, 1)
        predefined = 'ExperimentID' in Exp and _num(Exp.ExperimentID) > 0
        td = vec(info.td)
        quad = not predefined and np.iscomplexobj(td) and np.any(td.imag != 0)
        if quad:
            ax.plot(x1, td.real, label='I')
            ax.plot(x1, td.imag, label='Q')
            ax.legend(frameon=False)
        else:
            ax.plot(x1, td.real)
        if predefined:
            xlb = ['τ (µs)', 'τ+T (µs)', 'T (µs)', '...', 'frequency (MHz)']
            ID = int(_num(Exp.ExperimentID))
            ax.set_xlabel(xlb[ID - 1])
            ax.set_ylabel('echo amplitude')
            ax.set_title(['2pESEEM', '3pESEEM', '4pESEEM', 'HYSCORE', 'MimsENDOR'][ID - 1] + ', TD signal')
        else:
            ax.set_xlabel('t (µs)')
            ax.set_ylabel('echo amplitude (arb.u.)')
            ax.set_title('User-defined experiment, TD signal')
        if plotFreqDomain:
            ax2 = fig.add_subplot(2, 1, 2)
            f = vec(info.f)
            fd = vec(info.fd)
            idx = np.arange(int(np.nonzero(f == 0)[0][0]), f.size)
            ax2.plot(f[idx], np.abs(fd[idx]), label='abs')
            ax2.plot(f[idx], fd[idx].real, label='in-phase')
            if quad:
                ax2.plot(f[idx], fd[idx].imag, label='quadrature')
            ax2.legend(frameon=False)
            ax2.set_xlim(0, np.max(f))
            ax2.set_xlabel('ν (MHz)')
            ax2.set_ylabel('intensity (arb.u.)')
            ax2.set_title('Spectrum')
            if 'Nucs' in Sys:
                nuI = vec(F.larmorfrq(Sys.Nucs, Exp.Field))
                for v in nuI:
                    ax2.axvline(abs(v), color=(0.8, 0.8, 0.8), zorder=0)
    else:
        x2 = vec(x[1])
        ax = fig.add_subplot(1, 3, 1) if plotFreqDomain else fig.add_subplot(1, 1, 1)
        ax.pcolormesh(x1, x2, np.real(np.asarray(info.td)).T, shading='auto')
        ax.set_title('Time domain (real part)')
        ax.set_xlabel('t1 (µs)')
        ax.set_ylabel('t2 (µs)')
        if plotFreqDomain:
            ax2 = fig.add_subplot(1, 3, (2, 3))
            fx1, fx2 = vec(info.f1), vec(info.f2)
            fd = np.abs(np.asarray(info.fd))
            if 'logplot' in Opt and bool(Opt.logplot):
                fd = np.log(fd)
                maxfd = np.max(fd)
                fd[fd < maxfd - 6] = maxfd - 6
            ax2.contour(fx1, fx2, fd.T, 20)
            ax2.set_aspect('equal')
            ax2.set_title('Frequency domain')
            ax2.set_xlabel('ν1 (MHz)')
            ax2.set_ylabel('ν2 (MHz)')
            fm = max(np.max(np.abs(fx1)), np.max(np.abs(fx2)))
            ax2.plot([0, 0], [-fm, fm], 'k')
            ax2.plot([-fm, fm], [-fm, fm], 'k:')
            ax2.plot([fm, -fm], [-fm, fm], 'k:')
            ax2.set_ylim(0, fm)
    return fig

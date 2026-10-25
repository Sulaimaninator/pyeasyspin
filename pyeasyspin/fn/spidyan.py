"""Simulate spin dynamics during pulse EPR experiments"""
import datetime
import numpy as np
import scipy.linalg
from .._registry import F
from .._struct import Struct
from .._mat import isempty, numel
from ..config import config


def _c(x):
    return x[0] if isinstance(x, tuple) else x


def _clock():
    n = datetime.datetime.now()
    return [n.year, n.month, n.day, n.hour, n.minute, n.second + n.microsecond / 1e6]


def _now():
    return datetime.datetime.now().strftime('%d-%b-%Y %H:%M:%S')


def _setg(G, i, n, val):
    """MATLAB G(i,1:n) = val with growth (i 0-based)."""
    G = np.zeros((0, 0)) if G is None else np.atleast_2d(np.asarray(G, dtype=float))
    r, c = G.shape
    if i >= r or n > c:
        H = np.zeros((max(r, i + 1), max(c, n)))
        H[:r, :c] = G
        G = H
    else:
        G = G.copy()
    G[i, :n] = val
    return G


def _applyPhaseShift(RawSignal, Phase):
    Phase = np.asarray(Phase).ravel()
    nPhases = Phase.size
    if isinstance(RawSignal, list):
        # bsxfun(@times, RawSignal{i}, Phase') -- Phase' is the conjugate transpose
        return [np.asarray(R) * np.conj(Phase).reshape(-1, 1) for R in RawSignal]
    R = np.asarray(RawSignal)
    SignalSize = list(R.shape) if R.ndim >= 2 else [1] * (2 - R.ndim) + list(R.shape)
    nPoints = int(np.prod(SignalSize[:-2])) if len(SignalSize) > 2 else 1
    R = R.reshape([nPoints, SignalSize[-2], SignalSize[-1]], order='F')
    P = np.zeros(R.shape, dtype=complex)
    for iPhase in range(nPhases):
        P[:, iPhase, :] = R[:, iPhase, :] * Phase[iPhase]
    return P.reshape(SignalSize, order='F')


def _unique_rows_count(A):
    A = np.asarray(A)
    return np.unique(A, axis=0).shape[0] if A.size else 0


def spidyan(Sys=None, Exp=None, Opt=None, nargout=None, plot=False):
    """Time-domain simulation of spin dynamics during pulse EPR experiments (Hilbert or
    Liouville space, with relaxation). nargout: 1 -> Signal; 2 (default) -> (TimeAxis, Signal);
    3 -> (TimeAxis, Signal, info) with info.FinalState, .StateTrajectories, .Events.
    plot=True runs EasySpin's no-output branch (checks and s_plotting) and still returns the
    outputs. Times in us, frequencies in GHz, field in mT."""
    nargin = 2 if Opt is None else 3
    if Sys is None or Exp is None:
        raise ValueError('Wrong number of input arguments!')
    plotting = plot or nargout == 0
    if nargout is None:
        nargout = 2
    if nargout > 3:
        raise ValueError('Too many output arguments.')
    if nargin < 3 or isempty(Opt):
        Opt = Struct()
    if not isinstance(Sys, dict):
        raise TypeError('First input argument (Sys) must be a structure!')
    if not isinstance(Exp, dict):
        raise TypeError('Second input argument (Exp) must be a structure!')
    if not isinstance(Opt, dict):
        raise TypeError('Third input argument (Opt) must be a structure!')
    Sys, Exp, Opt = Struct(Sys), Struct(Exp), Struct(Opt)
    StartTime = _clock()
    if 'Verbosity' not in Opt:
        Opt.Verbosity = 0
    config.LogLevel = Opt.Verbosity
    F.logmsg(1, '=begin=spidyan====' + _now() + '=================')
    F.logmsg(2, '  log level %d', config.LogLevel)

    if 'Field' not in Exp:
        F.logmsg(1, '  no Exp.Field given')
        F.logmsg(1, '  checking if Sys allows for running without a field...')
        if 'g' in Sys:
            raise ValueError('Exp.Field is required for Sys.g.')
        if 'ZeemanFreq' not in Sys:
            raise ValueError('Unrecognized field name "ZeemanFreq".')
        zf = np.asarray(Sys.ZeemanFreq, dtype=float).ravel()
        if zf.size and np.all(zf != 0):
            Exp.Field = (zf[0] * 1e6) * F.planck() / _c(F.bmagn()) / (_c(F.gfree()) * 1e-6)
            F.logmsg(1, '  Exp.Field is needed for calculating simulation frame frequencies')
            F.logmsg(1, '  setting Exp.Field to %.1f mT', Exp.Field)
        else:
            raise ValueError('Exp.Field is required for given spin system.')
    elif numel(Exp.Field) != 1:
        raise ValueError('Exp.Field must be a single number (in mT).')

    Opt.SimulationMode = 'thyme'
    if ('T1' in Sys or 'T2' in Sys) and 'Relaxation' not in Opt:
        Opt.Relaxation = True

    Events, Vary, Opt = F.s_sequencer(Exp, Opt)
    Opt = Struct(Opt)

    if plotting and 'nPoints' in Exp:
        if Opt.SinglePointDetection and numel(Exp.nPoints) > 2:
            raise ValueError('spidyan can not plot more than two indirect dimensions in combination with single point detection.')
        elif not Opt.SinglePointDetection and numel(Exp.nPoints) > 1:
            raise ValueError('spidyan can not plot more than one indirect dimension in combination with transient detection.')

    F.logmsg(1, '-validating spin system--------------------------------')
    if 'S' not in Sys:
        if 'ZeemanFreq' in Sys and numel(Sys.ZeemanFreq) != 1:
            raise ValueError('If you provide more than one value in Sys.ZeemanFreq, Sys.S must be specified')
    if 'ZeemanFreq' in Sys and Opt.FrameShift != 0:
        F.logmsg(1, 'adapting Sys.ZeemanFreq to the simulation frame')
        Sys.ZeemanFreq = np.asarray(Sys.ZeemanFreq, dtype=float) - Opt.FrameShift
    if 'g' in Sys:
        nElectrons = numel(Sys.S) if 'S' in Sys else 1
        if Opt.FrameShift != 0:
            F.logmsg(1, '  adapting Sys.g to the simulation frame')
        Field = np.asarray(Exp.Field, dtype=float).ravel()
        gshift = (Opt.FrameShift * 1e9) * F.planck() / _c(F.bmagn()) / (Field[-1] * 1e-3)
        g = np.asarray(Sys.g, dtype=float)
        fullg = g.ndim == 2 and g.shape == (3 * nElectrons, 3)
        if fullg:
            Sys.g = g - np.tile(gshift * np.eye(3), (nElectrons, 1))
        else:
            Sys.g = g - gshift if g.ndim else float(g) - gshift
    if 'ZeemanFreq' in Sys:
        F.logmsg(1, '  translating Sys.ZeemanFreq into g values')
        ZF = np.asarray(Sys.ZeemanFreq, dtype=float) * 1000
        Sys.ZeemanFreq = ZF
        dgTensor = np.atleast_2d(np.asarray(Sys.g)).shape[1] if 'g' in Sys else 1
        zf = ZF.ravel(order='F')
        for ieSpin in range(zf.size):
            if zf[ieSpin] != 0:
                gv = (zf[ieSpin] * 1e6) * F.planck() / _c(F.bmagn()) / (Exp.Field * 1e-3)
                G = _setg(Sys.get('g'), ieSpin, dgTensor, gv)
                Sys.g = float(G[0, 0]) if G.size == 1 else G
    if 'ZeemanFreq' in Sys:
        F.logmsg(1, '  removing Sys.ZeemanFreq from Sys structure')
        del Sys['ZeemanFreq']

    if 'DetOperator' in Exp:
        if not isinstance(Exp.DetOperator, list):
            Exp.DetOperator = [Exp.DetOperator]
        Opt.DetOperator = Exp.DetOperator
    if 'DetPhase' in Exp:
        if 'DetOperator' in Opt and numel(Exp.DetPhase) != len(Opt.DetOperator):
            raise ValueError('Exp.DetPhase has to have the same length as Exp.DetOperator.')
        elif 'DetOperator' not in Opt and numel(Exp.DetPhase) != 1:
            raise ValueError('Exp.DetPhase has to have length one if no Exp.DetOperator is provided.')

    F.logmsg(1, '  parsing the Sys structure...')
    Sys, Sigma, DetOps, Events, Relaxation = F.s_propagationsetup(Sys, Events, Opt)

    F.logmsg(1, '  computing lab frame Hamiltonian')
    Ham = F.ham(Sys, Exp.Field * np.array([0.0, 0.0, 1.0]))
    if not isempty(Relaxation):
        F.logmsg(1, '  adapting relaxation superoperator to system frame')
        Hd = Ham.toarray() if hasattr(Ham, 'toarray') else np.asarray(Ham)
        if np.array_equal(Hd, Hd.conj().T):
            _, U = scipy.linalg.eigh(Hd)
        else:
            _, U = np.linalg.eig(Hd)
        R = np.kron(U.T, U.conj().T)
        Relaxation = Struct(Relaxation)
        Relaxation.Gamma = R.conj().T @ np.asarray(Relaxation.Gamma) @ R

    F.logmsg(1, '-starting propagation----------------------------------')
    F.logmsg(1, '  depending on the Exp setup this may take a while...')
    TimeAxis, RawSignal, FinalState, StateTrajectories, Events = \
        F.s_thyme(Sigma, Ham, DetOps, Events, Relaxation, Vary)
    F.logmsg(1, '  propagation finished!')

    F.logmsg(1, '-validating and processing outout----------------------')
    nDetOps = len(DetOps) if isinstance(DetOps, list) else numel(DetOps)
    if not isempty(RawSignal) and 'DetPhase' in Exp:
        DetPhase = np.exp(-1j * np.asarray(Exp.DetPhase, dtype=float))
        F.logmsg(1, '  applying detection phase: %d*pi', np.asarray(Exp.DetPhase) / np.pi)
        RawSignal = _applyPhaseShift(RawSignal, DetPhase)

    if Opt.SinglePointDetection:
        F.logmsg(1, '  single point detection...')
        if 'nPoints' in Exp and nDetOps != 1:
            R = np.asarray(RawSignal)
            R = R if R.ndim >= 2 else R.reshape(1, -1)
            Signal = np.expand_dims(R, axis=R.ndim - 1)    # permute(.,[1:D-1 D+1 D])
        else:
            Signal = RawSignal
    else:
        if not isempty(RawSignal):
            if 'DetFreq' in Exp:
                F.logmsg(1, '  veryfing Exp.DetFreq...')
                if 'DetOperator' in Exp:
                    if numel(Exp.DetFreq) != numel(Exp.DetOperator):
                        raise ValueError('Exp.DetOperator and Exp.DetFreq must contain the same number of elements.')
                elif numel(Exp.DetFreq) != 1:
                    raise ValueError('If Exp.DetOperator is not provided the default detection operator (S+ for all '
                                     'electrons) is assumed. In this case Exp.DetFreq must contain only one frequency.')
            else:
                if 'DetOperator' not in Exp and 'mwFreq' in Exp:
                    F.logmsg(1, '  using Exp.mwFreq as detection frequency')
                    Exp.DetFreq = Exp.mwFreq
            F.logmsg(1, '  processing transients...')
            FreqTranslation = np.zeros((1, nDetOps))
            if 'DetFreq' in Exp and not isempty(Exp.DetFreq):
                Ev1 = Events[0] if isinstance(Events, list) else Events
                df = np.array(Exp.DetFreq, dtype=float).ravel()
                df[df > 0] = df[df > 0] - Ev1['FrameShift']
                df[df < 0] = df[df < 0] + Ev1['FrameShift']
                Exp.DetFreq = df
                if df.size > nDetOps:
                    FreqTranslation = np.concatenate([FreqTranslation, np.zeros((1, df.size - nDetOps))], axis=1)
                FreqTranslation[0, :df.size] = -df
            Signal = F.signalprocessing(TimeAxis, RawSignal, FreqTranslation)
            if not isinstance(TimeAxis, list):
                T = np.asarray(TimeAxis)
                T = T if T.ndim >= 2 else T.reshape(1, -1)
                SizeT = T.shape
                linearTimeAxis = T.reshape([int(np.prod(SizeT[:-1])), SizeT[-1]], order='F')
                if _unique_rows_count(linearTimeAxis) == 1:
                    TimeAxis = linearTimeAxis[0:1, :]
        else:
            F.logmsg(1, '  nothing was detected...')
            Signal = np.zeros((0, 0))

    FS = np.asarray(FinalState) if not isinstance(FinalState, list) else FinalState
    if isinstance(FS, np.ndarray) and FS.ndim == 3 and FS.shape[0] == 1:
        FinalState = FS[0]
    if not isempty(StateTrajectories):
        if isinstance(StateTrajectories, list) and len(StateTrajectories) == 1:
            StateTrajectories = StateTrajectories[0]

    if plotting:
        F.logmsg(1, '-no output requested------------------------------------')
        F.logmsg(1, '  switching to graphical output')
        if isempty(Signal):
            print('Detection was switched off, nothing to display.')
        else:
            F.s_plotting(TimeAxis, Signal, Exp, Opt)
    if nargout in (0, 2):
        result = (TimeAxis, Signal)
    elif nargout == 1:
        result = Signal
    elif nargout == 3:
        info = Struct(FinalState=FinalState, StateTrajectories=StateTrajectories, Events=Events)
        result = (TimeAxis, Signal, info)
    else:
        raise ValueError('Incorrect number of output arguments. 1,2, or 3 expected.')
    hmsString = F.elapsedtime(StartTime, _clock())
    hmsString = hmsString[0] if isinstance(hmsString, tuple) else hmsString
    F.logmsg(1, 'spidyan took ' + str(hmsString))
    F.logmsg(1, '=end=spidyan======%s=================\n', _now())
    return result

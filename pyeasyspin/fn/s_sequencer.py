"""Pulse sequence parser for saffron and spidyan: builds the event list and the Vary structure."""
import math
import re
import warnings

import numpy as np

from .._registry import F, nout
from .._struct import Struct
from .._mat import isempty, numel, vec
from .._matlab import Cell, round_sig as _round_sig   # noqa: F401 (Cell re-exported)


def _setrow(A, i, v):
    """MATLAB A(i,:) = v (i 1-based) with growth; A None means a new variable."""
    v = np.asarray(v).reshape(-1)
    if A is None or (isinstance(A, np.ndarray) and A.size == 0):
        A = np.zeros((i, v.size), dtype=v.dtype)
    elif A.shape[1] != v.size:
        raise ValueError('Unable to perform assignment because the size of the left side is '
                         f'1-by-{A.shape[1]} and the size of the right side is 1-by-{v.size}.')
    if A.shape[0] < i:
        A = np.concatenate([A, np.zeros((i - A.shape[0], A.shape[1]), dtype=A.dtype)], axis=0)
    if np.iscomplexobj(v) and not np.iscomplexobj(A):
        A = A.astype(complex)
    else:
        A = A.copy()
    A[i - 1, :] = v
    return A


def _addtomessage(old, toadd):
    return toadd if isempty(old) else old + ', ' + toadd


def _dimrows(D):
    """Rows (string, value) of an Exp.DimN cell (n x 2): a column-major flat list or nested rows."""
    if isinstance(D, (list, tuple)) and D and all(isinstance(r, (list, tuple)) for r in D):
        return [(r[0], r[1]) for r in D]
    D = list(D) if isinstance(D, tuple) else D
    if isinstance(D, np.ndarray) and D.dtype == object:
        D = list(D.ravel(order='F')) if D.ndim == 2 and D.shape[1] == 2 else list(D.ravel())
    n = len(D) // 2
    return [(D[i], D[i + n]) for i in range(n)]


def _mlength(x):
    if isinstance(x, (list, tuple, str)):
        return len(x)
    s = np.shape(x)
    return int(max(s)) if len(s) and np.size(x) else (1 if np.ndim(x) == 0 else 0)


def _nrows(x):
    s = np.shape(x)
    return 1 if len(s) == 0 else (s[0] if len(s) > 1 else 1)


def _num2str(x):
    x = float(x)
    if x == int(x):
        return str(int(x))
    return ('%.4f' % x).rstrip('0').rstrip('.') if abs(x) >= 1e-4 else '%.5g' % x


def _str2double(s):
    try:
        return float(s)
    except ValueError:
        return float('nan')


def _is_pulse(x):
    return isinstance(x, dict)


def _row(x):
    a = np.asarray(x)
    return a.reshape(1, -1) if a.ndim < 2 else a


def _get_user_cell(c, idx, Points, moddims):
    """Element of a user-provided cell (Pulse.IQ / Pulse.t) at 1-based subscripts idx. Cells from
    MATLAB records arrive as flat column-major lists; their shape is then taken as Points(d) along
    the modified dimensions and 1 elsewhere."""
    if isinstance(c, Cell):
        return c.getitem(idx)
    if isinstance(c, np.ndarray) and c.dtype == object:
        return c[tuple(int(i) - 1 for i in idx[:c.ndim])]
    if not isinstance(c, list):
        c = [c]
    if len(idx) == 1:
        return c[int(idx[0]) - 1]
    shape = [int(Points[d]) if d in moddims else 1 for d in range(len(Points))]
    lin, stride = 0, 1
    for k, i in enumerate(idx):
        lin += (int(i) - 1) * stride
        stride *= shape[k] if k < len(shape) else 1
    return c[lin]


@nout(-1)
def s_sequencer(Exp=None, Opt=None, nargout=None):
    """(Events, Vary, Opt[, oldSyntax]). Fast mode (Opt.SimulationMode 'fast' and a sequence it can
    handle): ([], [], Opt, Exp_oldSyntax with t, Flip, tp, Phase[, Inc, dt, nPoints]), or [] for a
    predefined sequence. Thyme mode: the list of event Structs (pulse waveforms via pulse/rfmixer/
    resonator, delays, detection, relaxation, frame shift), the Vary Struct for indirect dimensions
    ([] without Exp.nPoints) and the updated Opt. nargout defaults to 3 (as spidyan calls it)."""
    if nargout is None:
        nargout = 3
    Exp = Struct(Exp)
    Opt = Struct(Opt)
    F.logmsg(1, '-parsing pulse sequence-----------------------------')
    predefinedExperiment = 'Sequence' in Exp and isinstance(Exp.Sequence, str)
    message = []
    if 'SimulationMode' not in Opt:
        Opt.SimulationMode = ''
    origSimulationMode = Opt.SimulationMode
    saffronSpecificFieldsExp = ['ExciteWidth', 'Filter']
    saffronSpecificFieldsOpt = ['TimeDomain', 'Expand', 'ProductRule', 'EndorMethod', 'nOffsets',
                                'lwOffset', 'logplot', 'Window', 'ZeroFillFactor']
    if 'DetWindow' in Exp:
        Opt.SimulationMode = 'thyme'
        message = _addtomessage(message, 'Exp.DetWindow does not work with the fast method')
    if 'ResonatorFrequency' in Exp or 'ResonatorQL' in Exp or 'FrequencyResponse' in Exp:
        Opt.SimulationMode = 'thyme'
        message = _addtomessage(message, 'Exp defintion contains resonator fields')
    if predefinedExperiment and 'DetWindow' in Exp:
        Opt.SimulationMode = 'thyme'
        message = _addtomessage(message, 'predefined experiments can not be run in combination with Exp.DetWindow')
    Exp_oldSyntax = None
    if Opt.SimulationMode == 'fast':
        if predefinedExperiment:
            Exp_oldSyntax = np.zeros((0, 0))
        else:
            Exp_oldSyntax = Struct()
            Sequence = Exp.Sequence
            Pulses = np.array([_is_pulse(s) for s in Sequence])
            PulsePositions = np.nonzero(Pulses)[0] + 1
            DelayPositions = np.nonzero(~Pulses)[0] + 1
            nPulses = len(PulsePositions)
            nDelays = len(DelayPositions)
            if np.any(np.diff(PulsePositions) == 0):
                Opt.SimulationMode = 'thyme'
                message = _addtomessage(message, 'two or more pulses are not separated by an interpulse delay')
            Flip = np.zeros((1, nPulses))
            tp = np.zeros((1, nPulses))
            Phase = np.ones((1, nPulses))
            for iPulse in range(nPulses):
                pulse_ = Sequence[PulsePositions[iPulse] - 1]
                if 'Type' in pulse_ and pulse_['Type'] != 'rectangular' and pulse_['Type'] != 'rectangular/none':
                    message = _addtomessage(message, 'The fast algorithm only supports monochromatic rectangular pulses')
                    Opt.SimulationMode = 'thyme'
                if 'Flip' in pulse_:
                    Flip[0, iPulse] = pulse_['Flip'] / (math.pi / 2)
                else:
                    Opt.SimulationMode = 'thyme'
                    message = _addtomessage(message, 'flip angles must be provided')
                if 'Phase' in pulse_:
                    Phase[0, iPulse] = np.mod(Phase[0, iPulse] + pulse_['Phase'] / (math.pi / 2), 4)
                if 'tp' in pulse_ and pulse_['tp'] != 0:
                    tp[0, iPulse] = pulse_['tp']
            t = np.zeros((1, nDelays))
            for iDelay, p in enumerate(DelayPositions):
                t[0, iDelay] = Sequence[p - 1]
            Exp_oldSyntax.t = t
            Exp_oldSyntax.Flip = Flip
            Exp_oldSyntax.tp = tp
            Exp_oldSyntax.Phase = Phase
            if 'nPoints' in Exp:
                nDimensions = numel(Exp.nPoints)
                Inc = np.zeros((1, nDelays))
                if nDimensions > 2:
                    message = _addtomessage(message, 'more than 2 indirect dimensions were provided')
                    Opt.SimulationMode = 'thyme'
                dt = np.zeros((1, nDimensions))
                for iDimension in range(1, nDimensions + 1):
                    Field2Get = 'Dim' + str(iDimension)
                    for (FullString, val) in _dimrows(Exp[Field2Get]):
                        for piece in FullString.split(','):
                            Strings = piece.split('.')
                            EventType = Strings[0][0]
                            EventSpecificIndex = _str2double(Strings[0][1:])
                            if EventType == 'p':
                                message = _addtomessage(message, 'a pulse parameter is changed along indirect dimension no. ' + str(iDimension))
                                Opt.SimulationMode = 'thyme'
                            else:
                                k = int(EventSpecificIndex)
                                if k > Inc.shape[1]:
                                    Inc = np.concatenate([Inc, np.zeros((1, k - Inc.shape[1]))], axis=1)
                                Inc[0, k - 1] = iDimension
                            v = np.asarray(val, dtype=float)
                            if dt[0, iDimension - 1] == 0:
                                if v.size != 1:
                                    raise ValueError('Unable to perform assignment because the left and right sides have a different number of elements.')
                                dt[0, iDimension - 1] = float(v.ravel()[0])
                            elif np.all(dt[0, iDimension - 1] != v) and v.size:
                                Opt.SimulationMode = 'thyme'
                                message = _addtomessage(message, 'the increment of dimension no. ' + str(iDimension) + ' is not linear')
                Exp_oldSyntax.Inc = Inc
                Exp_oldSyntax.dt = dt
                Exp_oldSyntax.nPoints = Exp.nPoints
    if origSimulationMode != Opt.SimulationMode:
        message = "The 'thyme' simulation mode has to be used." + '\\n' + 'The reason for this was: ' + \
            (message if not isempty(message) else '') + '\\n'
        F.logmsg(1, message)
    if Opt.SimulationMode == 'thyme':
        if any(f in Exp for f in saffronSpecificFieldsExp):
            msg = []
            for f in saffronSpecificFieldsExp:
                if f in Exp:
                    msg = _addtomessage(msg, f)
            print('The following fields in the Exp structure are specific to the fast algorithm and will be ignored: ' + msg)
        if any(f in Opt for f in saffronSpecificFieldsOpt):
            msg = []
            for f in saffronSpecificFieldsOpt:
                if f in Opt:
                    msg = _addtomessage(msg, f)
            print('The following fields in the Opt structure are specific to the fast algorithm and will be ignored: ' + msg)
    if Opt.SimulationMode == 'fast':
        out = (np.zeros((0, 0)), np.zeros((0, 0)), Opt, Exp_oldSyntax)
        return out[:max(nargout, 1)] if nargout != 1 else out[0]

    # ------------------------------------------------------------------ thyme
    Vary = np.zeros((0, 0))
    if 'mwFreq' not in Exp:
        raise ValueError('Exp.mwFreq is required for the thyme method.')
    Opt.SinglePointDetection = False
    if predefinedExperiment:
        Exp = Struct(F.s_predefinedexperiments(Exp))
    Seq = [Struct(s) if _is_pulse(s) else s for s in Exp.Sequence]
    Exp.Sequence = Seq
    isPulse = np.array([_is_pulse(s) for s in Seq])
    PulsePositions = np.nonzero(isPulse)[0] + 1
    idealPulses = False
    mwFreqrequired = False
    msgtp = 'Please provide pulse lenghts (Par.tp) for the following pulses:'
    msgFrequency = 'Please provide Par.Frequency for the following pulses:'
    for Pos in PulsePositions:
        PulseNumber = int(np.nonzero(PulsePositions == Pos)[0][0]) + 1
        P = Seq[Pos - 1]
        if 'tp' not in P and 'IQ' not in P:
            idealPulses = True
            msgtp = msgtp + ' ' + str(PulseNumber) + ','
        if 'Frequency' not in P and 'mwFreq' not in Exp and 'IQ' not in P:
            mwFreqrequired = True
            msgFrequency = msgFrequency + ' ' + str(PulseNumber) + ','
        if 'IQ' in P and 'IntTimeStep' not in Opt:
            raise ValueError('If you use userprovided IQs for pulses, Opt.IntTimeStep must be given.')
    if msgtp[-1] == ',':
        msgtp = msgtp[:-1]
    if msgFrequency[-1] == ',':
        msgFrequency = msgFrequency[:-1]
    if idealPulses and mwFreqrequired:
        errmsg = 'Real pulses are required for the thyme-method.\n' + msgtp + '\n' + msgFrequency
    elif idealPulses:
        errmsg = 'Real pulses are required for the thyme-method.\n' + msgtp
    elif mwFreqrequired:
        errmsg = 'Real pulses are required for the thyme-method.\n' + msgFrequency
    else:
        errmsg = ''
    if errmsg:
        raise ValueError(errmsg)

    if 'DetWindow' in Exp:
        F.logmsg(1, '  setting up detection window:')
        if 'DetSequence' in Exp:
            warnings.warn('You provided Exp.DetWindow and Exp.DetSequence. Exp.DetSequence will be ignored.')
        DW = vec(np.asarray(Exp.DetWindow, dtype=float))
        if np.any(DW < 0) and _is_pulse(Seq[-1]):
            raise ValueError('You provided Exp.DetWindow with a negative value, but the last element in Exp.Sequence is a pulse. Detection into a pulse is not possible. Please adapt your detection or append an appropriate free evolution event.')
        elif np.min(DW) < 0 and (abs(np.min(DW)) > Seq[-1]):
            raise ValueError('Your detection window is extending beyond the last free evolution into a pulse. Please shorten detection window or adapt the length of the free evolution.')
        if _is_pulse(Seq[-1]):
            if DW[0] > 0:
                Seq.append(float(DW[0]))
        else:
            Seq[-1] = Seq[-1] + DW[0]
        if DW.size == 1 or DW[0] == DW[1]:
            Opt.SinglePointDetection = True
            Seq.append(0.0)
            F.logmsg(1, '  single point detection')
        else:
            Seq.append(float(np.diff(DW)[0]))
            F.logmsg(1, '  transient detection')
        Exp.DetSequence = np.zeros((1, len(Seq)))
        Exp.DetSequence[0, -1] = True
    elif 'DetSequence' in Exp:
        F.logmsg(1, '  setting up detection:')
        if isinstance(Exp.DetSequence, str):
            if Exp.DetSequence == 'last':
                Exp.DetSequence = np.zeros((1, len(Seq)))
                Exp.DetSequence[0, -1] = True
                F.logmsg(1, '  detection is active during the last element in Exp.Sequence')
            elif Exp.DetSequence == 'all':
                Exp.DetSequence = np.ones((1, len(Seq)))
                F.logmsg(1, '  all elements in Exp.Sequence are detected')
            else:
                raise ValueError('The string you provided in Exp.DetSequence was not recognized')
        else:
            if numel(Exp.DetSequence) != 1 and numel(Exp.DetSequence) != len(Seq):
                raise ValueError('The lengths of Exp.Sequence and Exp.DetSequence do not match. Length of Exp.DetSequence has to be 1 or the same as Exp.Sequence.')
            F.logmsg(1, '  detection set according to Exp.DetSequence')
        DS = vec(np.asarray(Exp.DetSequence, dtype=float))
        if np.sum(DS) == 1:
            sel = [Seq[i] for i in np.nonzero(DS == 1)[0] if i < len(Seq)]
            if len(sel) == 1 and not _is_pulse(sel[0]) and sel[0] == 0:
                Opt.SinglePointDetection = True
                F.logmsg(1, '  single point detection')
    else:
        F.logmsg(1, '  no detection specified, detection is active during the entire sequence')
        Exp.DetSequence = np.ones((1, len(Seq)))

    includeResonator = False
    Resonator = Struct()
    if 'ResonatorFrequency' in Exp or 'ResonatorQL' in Exp or 'FrequencyResponse' in Exp:
        F.logmsg(1, '  resonator present')
        if 'ResonatorFrequency' in Exp and 'ResonatorQL' in Exp:
            Resonator.Arg1 = Exp.ResonatorFrequency
            Resonator.Arg2 = Exp.ResonatorQL
            includeResonator = True
        elif 'ResonatorFrequency' in Exp:
            raise ValueError('Exp.ResonatorFrequency provided, but Exp.ResonatorQL is missing')
        elif 'ResonatorQL' in Exp:
            raise ValueError('Exp.ResonatorQL provided, but Exp.ResonatorFrequency is missing')
        if 'FrequencyResponse' in Exp:
            if includeResonator:
                F.logmsg(1, '  also found FrequencyResponse of the resonator, ignoring ResonatorFrequency and QL')
            FR = np.atleast_2d(np.asarray(Exp.FrequencyResponse))
            Resonator.Arg1 = FR[0:1, :]
            Resonator.Arg2 = FR[1:2, :]
            includeResonator = True
        if 'ResonatorMode' in Exp:
            if Exp.ResonatorMode in ('simulate', 'compensate'):
                Resonator.Arg3 = Exp.ResonatorMode
            else:
                raise ValueError("Resonator.Mode must be 'simulate' or 'compensate'.")
        else:
            Resonator.Arg3 = 'simulate'

    freqShift = 0
    if 'mwFreq' in Exp:
        freqShift = freqShift + Exp.mwFreq
    F.logmsg(1, '  setting up simulation frame:')
    if 'SimFreq' in Opt:
        if Opt.SimFreq == 0:
            FrameShift = False
        else:
            freqShift = freqShift - Opt.SimFreq
            FrameShift = Opt.SimFreq
    else:
        minPulseFreq = Exp.mwFreq if ('mwFreq' in Exp and Exp.mwFreq != 0) else None
        for ev in Seq:
            if _is_pulse(ev) and 'Frequency' in ev:
                fr = vec(np.asarray(ev.Frequency, dtype=float)) / 1000
                if minPulseFreq is None:
                    minPulseFreq = float(np.min(fr))
                else:
                    minPulseFreq = float(np.min(np.concatenate([[minPulseFreq], fr + freqShift])))
        if minPulseFreq is None:
            FrameShift = np.zeros((0, 0))
        else:
            FrameShift = math.floor(minPulseFreq - 2)
            if FrameShift > 0:
                freqShift = freqShift - FrameShift
    fsnz = (not isempty(FrameShift)) and FrameShift != 0
    if fsnz:
        F.logmsg(1, '  simulation frame frequency is %d GHz', FrameShift)
    else:
        F.logmsg(1, '  simulating in the lab frame')
    Opt.FrameShift = FrameShift
    Opt.FreqShift = freqShift
    F.logmsg(1, '  determining minimal required time step')
    maxPulseFreq = freqShift
    for ev in Seq:
        if _is_pulse(ev) and 'Frequency' in ev:
            fr = vec(np.asarray(ev.Frequency, dtype=float)) / 1000
            maxPulseFreq = float(np.max(np.concatenate([fr + freqShift, [maxPulseFreq]])))
    NyquistFreq = 2 * maxPulseFreq
    NyquistTime = 1 / NyquistFreq / 1000
    IntStepFactor = 50
    DetStepFactor = IntStepFactor / 2
    RefIntTimeStep = _round_sig(NyquistTime / IntStepFactor, 2)
    RefDetTimeStep = DetStepFactor * RefIntTimeStep
    if 'IntTimeStep' in Opt and 'DetTimeStep' not in Exp:
        if Opt.IntTimeStep > NyquistTime:
            raise ValueError('Your integration time step (Opt.IntTimeStep) does not fulfill the Nyquist criterion for the pulses you provided. For best results, change it to '
                             + ('%10.1e' % RefIntTimeStep) + ' µs or less.')
        elif Opt.IntTimeStep > NyquistTime / (1 / 2 * IntStepFactor):
            warnings.warn('Although your integration time step (Opt.IntTimeStep) fulfills the Nyquist criterion for the pulses you provided, it might not be small enough for accurate results. You might want to change it to '
                          + ('%10.1e' % RefIntTimeStep) + ' µs or less.')
        Exp.DetTimeStep = RefDetTimeStep
    elif 'IntTimeStep' not in Opt and 'DetTimeStep' in Exp:
        factor = math.ceil(Exp.DetTimeStep / RefIntTimeStep)
        Opt.IntTimeStep = Exp.DetTimeStep / factor
    elif 'IntTimeStep' in Opt and 'DetTimeStep' in Exp:
        if math.fmod(Opt.IntTimeStep, Exp.DetTimeStep) if Exp.DetTimeStep != 0 else Opt.IntTimeStep:
            raise ValueError('The provided integration time step (Opt.IntTimeStep) and detection time step (Exp.DetTimeStep) do not match. Exp.DetTimeStep has to be a multiple of Opt.IntTimeStep.')
        if Opt.IntTimeStep > NyquistTime:
            raise ValueError('Your integration time step (Opt.IntTimeStep) does not fulfill the Nyquist criterion for the pulses you provided. For best results adapt it to '
                             + ('%10.1e' % RefIntTimeStep) + ' µs or less.')
        elif Opt.IntTimeStep > NyquistTime / (2 / 3 * IntStepFactor):
            warnings.warn('Although your integration time step (Opt.IntTimeStep) fulfills the Nyquist criterion for the pulses you provided, it might not be small enough for accurate results. You might want to adapt it to '
                          + ('%10.1e' % RefIntTimeStep) + ' µs or less.')
    else:
        Opt.IntTimeStep = RefIntTimeStep
        Exp.DetTimeStep = RefDetTimeStep
    F.logmsg(1, '  the integration time step is %0.2e microseconds', Opt.IntTimeStep)
    F.logmsg(1, '  the detection time step is %0.2e microseconds', Exp.DetTimeStep)

    nEvents = len(Seq)
    Events = [Struct() for _ in range(nEvents)]
    Intervals = np.zeros(nEvents)
    if 'Relaxation' in Opt:
        if numel(Opt.Relaxation) != 1 and numel(Opt.Relaxation) != nEvents:
            raise ValueError('The lengths of Exp.Sequence and Opt.Relaxation do not match. Length of Opt.Relaxation has to be 1 or the same as Exp.Sequence.')
    if 'StateTrajectories' in Opt:
        if numel(Opt.StateTrajectories) != 1 and numel(Opt.StateTrajectories) != nEvents:
            raise ValueError('The lengths of Exp.Sequence and Opt.StateTrajectories do not match. Length of Opt.StateTrajectories has to be 1 or the same as Exp.Sequence.')
    F.logmsg(1, '  parsing Exp.Sequence:')
    isPulse = np.array([_is_pulse(s) for s in Seq])
    pulseIndices = np.nonzero(isPulse)[0] + 1
    delayIndices = np.nonzero(~isPulse)[0] + 1
    nPulses = len(pulseIndices)
    nDelays = len(delayIndices)
    if 'DetWindow' in Exp:
        F.logmsg(1, '  found %d pulse(s), %d free evolution period(s) and a detection window', nPulses, nDelays - 1)
    else:
        F.logmsg(1, '  found %d pulse(s) and %d free evolution period(s)', nPulses, nDelays)
    if 'mwPolarization' in Exp and not isinstance(Exp.mwPolarization, str):
        raise ValueError("Exp.mwPolarization has to be 'linear' or 'circular'.")
    Pulses = [None] * nPulses
    iPulse = 1
    Pulse = None
    t = None
    ShiftedUserIQ = None
    for iEvent in range(1, nEvents + 1):
        E = Events[iEvent - 1]
        if isPulse[iEvent - 1]:
            E.type = 'pulse'
            PC = Exp.get('PhaseCycle', None)
            if PC is not None and not isinstance(PC, list):
                PC = [PC]
            if PC is not None and iPulse <= len(PC) and not isempty(PC[iPulse - 1]):
                ThisPhaseCycle = np.atleast_2d(np.asarray(PC[iPulse - 1], dtype=float))
                nPhaseSteps = ThisPhaseCycle.shape[0]
            else:
                ThisPhaseCycle = 0.0
                nPhaseSteps = 1
            S = Seq[iEvent - 1]
            if 'IQ' not in S:
                if 'Type' not in S:
                    if 'Frequency' not in S:
                        S.Frequency = 0
                        S.Type = 'rectangular'
                        if 'mwFreq' not in Exp:
                            raise ValueError('If you do not provide a frequency for the pulse, you need to give Exp.mwFreq.')
                    else:
                        fr = vec(np.asarray(S.Frequency, dtype=float))
                        if fr.size > 1 and fr[1] != fr[0]:
                            raise ValueError('Pulse at position ' + str(iEvent) + ' in Exp.Sequence: no Pulse.Type specified, assuming a monochromatice rectangular pulse, but the field Pulse.Frequency looks like a frequency-swept pulse.')
                elif S.Type == 'rectangular' and 'Frequency' not in S:
                    S.Frequency = 0
                Pulse = Struct(S)
                if 'tp' not in Pulse:
                    raise ValueError('Pulse at position %d in Exp.Sequence: no Pulse.tp specified. Please provide a pulse length.' % iEvent)
                if 'Frequency' not in Pulse:
                    Pulse.Frequency = np.array([[0.0, 0.0]])
                Pulse.PhaseCycle = ThisPhaseCycle
                if 'Flip' in S:
                    Pulse.Flip = S.Flip
                elif 'Qcrit' not in Pulse and 'Amplitude' not in Pulse:
                    raise ValueError('Flip angle for pulse no. %d is missing.' % iPulse)
                if 'Phase' not in Pulse:
                    Pulse.Phase = 0
                Pulse.TimeStep = Opt.IntTimeStep
                phase0 = Pulse.Phase
                IQrows = None
                for iPCstep in range(1, nPhaseSteps + 1):
                    Pulse.Phase = phase0 + np.atleast_2d(Pulse.PhaseCycle)[iPCstep - 1, 0]
                    t, IQ = F.pulse(Pulse, nargout=2)[:2]
                    if includeResonator:
                        tOrig = vec(t)[-1]
                        t, IQ = F.resonator(t, IQ, Exp.mwFreq, Resonator.Arg1, Resonator.Arg2, Resonator.Arg3)[:2]
                        E.Ringing = vec(t)[-1] - tOrig
                    if freqShift != 0:
                        Opt.dt = Opt.IntTimeStep
                        t, IQ = F.rfmixer(t, IQ, freqShift, 'IQshift', Opt)[:2]
                    IQrows = _setrow(IQrows, iPCstep, IQ)
                E.IQ = IQrows
            else:
                if isinstance(S.IQ, list):
                    userIQ = S.IQ[0]
                else:
                    userIQ = S.IQ
                userIQ = np.atleast_2d(np.asarray(userIQ))
                if userIQ.ndim > 2:
                    raise ValueError('The data structure of the userdefined IQ of pulse on position ' + str(iEvent) + ' in Exp.Sequence is not recognized.')
                d1, d2 = userIQ.shape
                if d1 != nPhaseSteps and d2 != nPhaseSteps:
                    raise ValueError('The dimensionality of the IQ signal provided of the pulse on position ' + str(iEvent) + ' in Exp.Sequence  is not in agreement with the phasecycle for this pulse. For user-defined waveforms the array must contain all IQs.')
                elif d2 == nPhaseSteps:
                    userIQ = userIQ.conj().T
                Pulse = Struct() if Pulse is None else Struct(Pulse)
                if 't' not in S:
                    raise ValueError('A userdefined IQ was used for the pulse on position ' + str(iEvent) + ' in Exp.Sequence but the time axis is missing. Please provide it through Pulse.t.')
                if isinstance(S.t, list):
                    t0 = vec(S.t[0])
                    Pulse.TimeStep = t0[1] - t0[0]
                    tIQ = S.t[0]
                else:
                    t0 = vec(S.t)
                    Pulse.TimeStep = t0[1] - t0[0]
                    tIQ = S.t
                Pulse.userIQ = Struct(t=S.t)
                for iPhaseStep in range(1, nPhaseSteps + 1):
                    Opt.dt = Opt.IntTimeStep
                    if includeResonator:
                        if t is None:
                            raise NameError("Unrecognized function or variable 't'.")
                        tOrig = vec(t)[-1]
                        tIQ, currentIQ = F.resonator(t, userIQ[iPhaseStep - 1:iPhaseStep, :], Exp.mwFreq,
                                                     Resonator.Arg1, Resonator.Arg2, Resonator.Arg3)[:2]
                        E.Ringing = vec(t)[-1] - tOrig
                    else:
                        currentIQ = userIQ[iPhaseStep - 1:iPhaseStep, :]
                    t, sIQ = F.rfmixer(tIQ, currentIQ, -Opt.FrameShift, 'IQshift', Opt)[:2]
                    ShiftedUserIQ = _setrow(ShiftedUserIQ, iPhaseStep, sIQ)
                E.IQ = ShiftedUserIQ
                Pulse.userIQ.IQ = S.IQ
                Pulse.PhaseCycle = ThisPhaseCycle
            E.t = t
            E.PhaseCycle = ThisPhaseCycle
            if 'mwPolarization' in Exp and not isempty(Exp.mwPolarization) and Exp.mwPolarization == 'circular':
                E.ComplexExcitation = True
            else:
                E.ComplexExcitation = False
            Pulse.EventIndex = iEvent
            Pulses[iPulse - 1] = Struct(Pulse)
            iPulse += 1
            Intervals[iEvent - 1] = vec(t)[-1]
        else:
            E.type = 'free evolution'
            E.t = Seq[iEvent - 1]
            Intervals[iEvent - 1] = Seq[iEvent - 1]
        if 'Relaxation' not in Opt:
            E.Relaxation = False
        else:
            E.Relaxation = Opt.Relaxation if numel(Opt.Relaxation) == 1 else vec(Opt.Relaxation)[iEvent - 1]
            if isinstance(E.Relaxation, np.ndarray):
                E.Relaxation = E.Relaxation.item() if E.Relaxation.size == 1 else E.Relaxation
        if 'DetSequence' not in Exp or isempty(Exp.DetSequence):
            E.Detection = False
        else:
            DS = vec(np.asarray(Exp.DetSequence))
            E.Detection = DS[0].item() if DS.size == 1 else DS[iEvent - 1].item()
        if 'StateTrajectories' not in Opt:
            E.StateTrajectories = False
        else:
            ST = vec(np.asarray(Opt.StateTrajectories))
            E.StateTrajectories = ST[0].item() if ST.size == 1 else ST[iEvent - 1].item()
        E.Propagation = np.zeros((0, 0))
        E.TimeStep = Opt.IntTimeStep
        E.FrameShift = FrameShift

    if includeResonator:
        F.logmsg(1, '  checking for pulse overlap due to ringing from resonator:')
        for iEvent in pulseIndices:
            FollowingEvent = iEvent + 1
            if FollowingEvent <= nEvents and Events[FollowingEvent - 1].type == 'pulse':
                raise ValueError('When using a resonator, pulses need to be separated by inter pulse delays to accomodate for ringing from the resonator.')
            elif FollowingEvent <= nEvents:
                ShortenedDelay = Events[FollowingEvent - 1].t - Events[iEvent - 1].Ringing
                if ShortenedDelay < 0:
                    raise ValueError('Event ' + str(FollowingEvent) + ' (a delay) is too short to accomodate ringing of the preceding pulse.')
        F.logmsg(1, '  all good!')

    if 'nPoints' in Exp:
        F.logmsg(1, '-validating indirect dimensions------------------------')
        nPoints = vec(np.asarray(Exp.nPoints, dtype=float))
        nDimensions = nPoints.size
        Vary = Struct()
        Vary.Points = Exp.nPoints
        Vary.IncrementationTable = [np.zeros((0, 0)) for _ in range(nDimensions)]
        PulseModifications = [None] * nPulses
        F.logmsg(1, '  found %d indirect dimension(s)', nDimensions)
        F.logmsg(1, '  with a total %d data points', int(np.prod(nPoints)))
        for iDimension in range(1, nDimensions + 1):
            F.logmsg(1, '  parsing dimension no. %d', iDimension)
            nP = int(nPoints[iDimension - 1])
            IncrementationTable = np.zeros((nEvents, nP))
            if nDimensions == 1 and 'Dim' in Exp:
                raise ValueError('You provided Exp.Dim, please always provide Dim with a number, e.g Exp.Dim1.')
            Field2Get = 'Dim' + str(iDimension)
            if Field2Get not in Exp:
                raise ValueError('You requested a ' + str(nDimensions) + '-dimensional experiment, but Exp.Dim'
                                 + str(iDimension) + ' is missing.')
            for (FullString, val) in _dimrows(Exp[Field2Get]):
                for piece in re.split(',', FullString):
                    FieldIndex = None
                    Strings = re.split(r'\.', piece)
                    EventType = Strings[0][0]
                    EventSpecificIndex = _str2double(Strings[0][1:])
                    if len(Strings) == 2:
                        pars = re.split(r'\(|\)', Strings[1])
                        Strings[1] = pars[0]
                        if len(pars) > 1:
                            FieldIndex = _str2double(pars[1])
                    L = _mlength(val)
                    if len(Strings) != 2 or Strings[1] != 'IQ':
                        if (len(Strings) != 2 or Strings[1] != 'Frequency' or L == 1
                                or not np.any(_nrows(val) == np.array([1, nP - 1]))):
                            if L != 1 and L != nP:
                                raise ValueError('The number of points provided for Dimension ' + str(iDimension)
                                                 + ' does not match the length of the vector in the Exp.Dim structure.')
                    if EventType == 'p':
                        EventSpecificIndex = int(EventSpecificIndex)
                        EventNumber = Pulses[EventSpecificIndex - 1].EventIndex
                        PulseNumber = EventSpecificIndex
                        if len(Strings) == 1:
                            raise ValueError('You requested a pulse to be changed in Exp.' + Field2Get + ' but did not specify the field.')
                        Field = Strings[1]
                        if Field == 't':
                            Field = 'tp'
                        if Field == 'Position':
                            se = [EventNumber - 1, EventNumber + 1]
                            if any(s > nEvents for s in se):
                                raise ValueError('Moving pulses can not be the first or last event in your Exp structure.')
                            if se[0] < 1:
                                raise IndexError('Index in position 1 is invalid. Array indices must be positive integers or logical values.')
                            dt = np.asarray(val, dtype=float)
                            if dt.size == 1:
                                ramp = np.arange(nP) * float(dt.ravel()[0])
                                IncrementationTable[se[0] - 1, :] = IncrementationTable[se[0] - 1, :] + ramp
                                IncrementationTable[se[1] - 1, :] = IncrementationTable[se[1] - 1, :] - ramp
                            else:
                                IncrementationTable[se[0] - 1, :] = IncrementationTable[se[0] - 1, :] + vec(dt)
                                IncrementationTable[se[1] - 1, :] = IncrementationTable[se[1] - 1, :] - vec(dt)
                        elif Field == 'IQ':
                            if PulseModifications[PulseNumber - 1] is None:
                                PulseModifications[PulseNumber - 1] = [[iDimension, Field, None, None]]
                            else:
                                PulseModifications[PulseNumber - 1].append([iDimension, Field, None, None])
                        else:
                            row = [iDimension, Field, val, FieldIndex]
                            if PulseModifications[PulseNumber - 1] is None:
                                PulseModifications[PulseNumber - 1] = [row]
                            else:
                                PulseModifications[PulseNumber - 1].append(row)
                    elif EventType == 'd':
                        EventNumber = int(delayIndices[int(EventSpecificIndex) - 1])
                        dt = np.asarray(val, dtype=float)
                        if dt.size == 1:
                            IncrementationTable[EventNumber - 1, :] += np.arange(nP) * float(dt.ravel()[0])
                        else:
                            IncrementationTable[EventNumber - 1, :] += vec(dt)
            if np.any(IncrementationTable):
                Vary.IncrementationTable[iDimension - 1] = IncrementationTable
        nDataPoints = int(np.prod(nPoints))
        DimensionIndices = np.ones(nDimensions, dtype=int)
        InitialPulses = Pulses
        Vary.Pulses = [np.zeros((0, 0)) for _ in range(nPulses)]
        F.logmsg(1, '  creating the Vary structure that contains all required wave forms and delay changes')
        ArrayIndex = None
        for iDataPoint in range(nDataPoints):
            Pulses = [Struct(p) for p in InitialPulses]
            EventLengths = Intervals.copy()
            for iPulse in range(1, nPulses + 1):
                mods = PulseModifications[iPulse - 1]
                if mods is None:
                    continue
                P = Pulses[iPulse - 1]
                P.ArrayIndex = np.ones(len(DimensionIndices), dtype=int)
                IQindex = sum(1 for m in mods if m[1] == 'IQ')
                VP = Vary.Pulses[iPulse - 1]
                if not isinstance(VP, dict):
                    VP = Struct()
                    Vary.Pulses[iPulse - 1] = VP
                if not IQindex:
                    for (Dimension, Field, Increment, FieldIndex) in mods:
                        k = int(DimensionIndices[Dimension - 1])
                        Inc = np.asarray(Increment, dtype=float)
                        cur = np.array(P[Field], dtype=float) if not np.isscalar(P[Field]) else float(P[Field])
                        if Inc.size == 1 or (Field == 'Frequency' and _nrows(Increment) == 1):
                            if FieldIndex is None:
                                P[Field] = cur + (Inc.item() if Inc.size == 1 else Inc) * (k - 1)
                            else:
                                fi = int(FieldIndex) - 1
                                cur = np.atleast_1d(cur).astype(float)
                                shp = np.shape(P[Field])
                                flat = cur.ravel(order='F').copy()
                                if Inc.size == 1:
                                    flat[fi] = flat[fi] + Inc.item() * (k - 1)
                                else:
                                    flat[fi] = flat[fi] + vec(Inc)[k - 1]
                                P[Field] = flat.reshape(shp, order='F') if len(shp) else flat[0]
                        else:
                            if Field == 'Frequency':
                                P[Field] = cur + np.atleast_2d(Inc)[k - 1, :]
                            elif FieldIndex is None:
                                P[Field] = cur + vec(Inc)[k - 1]
                            else:
                                fi = int(FieldIndex) - 1
                                shp = np.shape(P[Field])
                                flat = np.atleast_1d(cur).astype(float).ravel(order='F').copy()
                                flat[fi] = flat[fi] + vec(Inc)[k - 1]
                                P[Field] = flat.reshape(shp, order='F') if len(shp) else flat[0]
                        P.ArrayIndex[Dimension - 1] = k
                    ArrayIndex = list(P.ArrayIndex)
                    phase0 = P.Phase
                    PCm = np.atleast_2d(np.asarray(P.PhaseCycle, dtype=float))
                    for iPCstep in range(1, PCm.shape[0] + 1):
                        P.Phase = phase0 + PCm[iPCstep - 1, 0]
                        t, IQ = F.pulse(P, nargout=2)[:2]
                        if includeResonator:
                            tOrig = vec(t)[-1]
                            t, IQ = F.resonator(t, IQ, Exp.mwFreq, Resonator.Arg1, Resonator.Arg2, Resonator.Arg3)[:2]
                            VP.Ringing = _setnd(VP.get('Ringing'), ArrayIndex, vec(t)[-1] - tOrig)
                        if freqShift != 0:
                            Opt.dt = Opt.IntTimeStep
                            IQ = F.rfmixer(t, IQ, freqShift, 'IQshift', Opt)[1]
                        if 'IQs' not in VP:
                            VP.IQs = Cell()
                        try:
                            cur = VP.IQs.getitem(ArrayIndex)
                        except IndexError:
                            cur = None
                        if not _inside(VP.IQs, ArrayIndex):
                            cur = None
                        VP.IQs.setitem(ArrayIndex, _setrow(cur, iPCstep, IQ))
                    if 'ts' not in VP:
                        VP.ts = Cell()
                    VP.ts.setitem(ArrayIndex, t)
                else:
                    if IQindex < len(mods):
                        raise ValueError('It is not possible to combine changing pulse parameters with user defined IQ, please check your Dim input for Pulse '
                                         + str(iPulse) + '.')
                    nD = len(DimensionIndices)
                    moddims = {m[0] - 1 for m in mods}
                    if len(mods) == 1:
                        Dimension = mods[0][0]
                        IndexToLoad = int(DimensionIndices[Dimension - 1])
                        # IndexToSave = ones(n) is n x n in MATLAB; num2cell gives n^2 subscripts,
                        # the extra ones are trailing singletons
                        IndexToSave = np.ones(nD * nD, dtype=int)
                        IndexToSave[Dimension - 1] = DimensionIndices[Dimension - 1]
                        ArrayIndex = list(IndexToSave)
                        userIQ = _get_user_cell(P.userIQ.IQ, [IndexToLoad], nPoints, moddims)
                        tIQ = _get_user_cell(P.userIQ.t, [IndexToLoad], nPoints, moddims)
                    else:
                        IndexToLoad = np.ones((nD, nD), dtype=int)
                        flat = IndexToLoad.reshape(-1, order='F')
                        for m in mods:
                            flat[m[0] - 1] = DimensionIndices[m[0] - 1]
                        IndexToLoad = flat.reshape((nD, nD), order='F')
                        ArrayIndex = list(IndexToLoad.ravel(order='F'))
                        userIQ = _get_user_cell(P.userIQ.IQ, ArrayIndex, nPoints, moddims)
                        tIQ = _get_user_cell(P.userIQ.t, ArrayIndex, nPoints, moddims)
                    PCm = np.atleast_2d(np.asarray(P.PhaseCycle, dtype=float))
                    nPhaseSteps = PCm.shape[0]
                    userIQ = np.atleast_2d(np.asarray(userIQ))
                    d1, d2 = userIQ.shape
                    if d1 != nPhaseSteps and d2 != nPhaseSteps:
                        raise ValueError('The dimensionality of the IQ signal provided for event ' + str(nEvents)
                                         + ' is not in agreement with the phasecycle for this pulse. For user-defined waveforms the array must contain all IQs.')
                    elif d2 == nPhaseSteps:
                        userIQ = userIQ.conj().T
                    SU = None
                    for iPhaseStep in range(1, nPhaseSteps + 1):
                        Opt.dt = Opt.IntTimeStep
                        if includeResonator:
                            tOrig = vec(tIQ)[-1]
                            tIQ, currentIQ = F.resonator(tIQ, userIQ[iPhaseStep - 1:iPhaseStep, :], Exp.mwFreq,
                                                         Resonator.Arg1, Resonator.Arg2, Resonator.Arg3)[:2]
                            VP.Ringing = _setnd(VP.get('Ringing'), ArrayIndex, vec(tIQ)[-1] - tOrig)
                        else:
                            currentIQ = userIQ[iPhaseStep - 1:iPhaseStep, :]
                        t, sIQ = F.rfmixer(tIQ, currentIQ, -Opt.FrameShift, 'IQshift', Opt)[:2]
                        SU = _setrow(SU, iPhaseStep, sIQ)
                    if 'IQs' not in VP:
                        VP.IQs = Cell()
                    VP.IQs.setitem(ArrayIndex, SU)
                    if 'ts' not in VP:
                        VP.ts = Cell()
                    VP.ts.setitem(ArrayIndex, t)
                ev = int(P.EventIndex) - 1
                if includeResonator:
                    EventLengths[ev] = vec(t)[-1] - _getnd(VP.Ringing, ArrayIndex)
                else:
                    EventLengths[ev] = vec(t)[-1]
            for iDimension in range(nDimensions):
                T = Vary.IncrementationTable[iDimension]
                if not isempty(T):
                    colk = T[:, DimensionIndices[iDimension] - 1]
                    for ev in np.nonzero(colk)[0]:
                        EventLengths[ev] = EventLengths[ev] + colk[ev]
            NewSequence, NewEventLengths = F.s_reorder_events(EventLengths.reshape(1, -1), isPulse.reshape(1, -1))[:2]
            NewSequence = vec(NewSequence).astype(int)
            NewEventLengths = vec(NewEventLengths)
            if includeResonator:
                for iPulse in range(1, nPulses + 1):
                    ThisEvent = int(np.nonzero(NewSequence == pulseIndices[iPulse - 1])[0][0]) + 1
                    if ThisEvent == len(NewSequence):
                        break
                    FollowingEvent = NewSequence[ThisEvent]
                    if Events[FollowingEvent - 1].type == 'pulse':
                        raise ValueError('When using a resonator, pulses need to be separated by inter pulse delays to accomodate for ringing from the resonator.')
                    VP = Vary.Pulses[iPulse - 1]
                    if isinstance(VP, dict) and VP:
                        Ringing = _getnd(VP.Ringing, ArrayIndex)
                    else:
                        Ringing = Events[pulseIndices[iPulse - 1] - 1].Ringing
                    ShortenedDelay = NewEventLengths[ThisEvent] - Ringing
                    if ShortenedDelay < 0:
                        raise ValueError('The delay ' + str(FollowingEvent) + ' is too short to accomodate for ringing of the preceeding pulse.')
            for iEvent in range(1, nEvents + 1):
                other = Events[NewSequence[iEvent - 1] - 1]
                if other.Detection != Events[iEvent - 1].Detection:
                    raise ValueError('Due to a moving pulse, the events ' + str(iEvent) + ' and ' + str(NewSequence[iEvent - 1])
                                     + ' are being interchanged, but they do not have the same setting with respect to detection.')
                if other.Relaxation != Events[iEvent - 1].Relaxation:
                    raise ValueError('Due to a moving pulse, the events ' + str(iEvent) + ' and ' + str(NewSequence[iEvent - 1])
                                     + ' are being interchanged, but they do not have the same setting with respect to relaxation.')
            for d in range(nDimensions - 1, -1, -1):
                if DimensionIndices[d] < nPoints[d]:
                    DimensionIndices[d] += 1
                    break
                else:
                    DimensionIndices[d] = 1
    out = (Events, Vary, Opt, np.zeros((0, 0)))
    F.logmsg(1, '  pulse sequence parsed successfully!')
    return out[0] if nargout == 1 else out[:max(nargout, 1)]


def _inside(c, idx):
    """True if the 1-based subscripts idx address an existing element of Cell c."""
    shp = list(c.mshape) + [1] * len(idx)
    return all(int(i) <= shp[k] for k, i in enumerate(idx)) and len(c) > 0


def _setnd(A, idx, value):
    """MATLAB A(idx{:}) = value for a numeric N-D array with growth (A None = new)."""
    idx = [int(i) for i in idx]
    if A is None or np.size(A) == 0:
        A = np.zeros([1] * max(len(idx), 2))
        if len(idx) == 1:
            A = np.zeros((1, 1))
    A = np.array(A, dtype=float)
    if len(idx) == 1:
        idx = [1, idx[0]] if A.shape[0] == 1 else [idx[0], 1]
    shp = list(A.shape) + [1] * (len(idx) - A.ndim)
    new = [max(s, i) for s, i in zip(shp, idx + [1] * (len(shp) - len(idx)))]
    if new != list(A.shape):
        B = np.zeros(new)
        B[tuple(slice(0, s) for s in A.shape)] = A.reshape(A.shape)
        A = B
    A[tuple(i - 1 for i in idx) + (0,) * (A.ndim - len(idx))] = value
    while A.ndim > 2 and A.shape[-1] == 1:
        A = A[..., 0]
    return A


def _getnd(A, idx):
    A = np.asarray(A)
    idx = [int(i) for i in idx]
    if A.ndim == 0:
        return float(A)
    if len(idx) == 1:
        return A.ravel(order='F')[idx[0] - 1]
    sub = [i - 1 for i in idx[:A.ndim]] + [0] * max(0, A.ndim - len(idx))
    return A[tuple(sub)]

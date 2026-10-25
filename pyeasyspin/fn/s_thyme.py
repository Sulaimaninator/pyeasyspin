"""Time domain evolution of density matrix (the 'thyme' propagation engine of spidyan/saffron)."""
import itertools
import math

import numpy as np
import scipy.linalg
from numba import njit

from .._registry import F, nout
from .._struct import Struct
from .._mat import isempty, vec
from .._matlab import colon, Cell


def _mround(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _dense(x):
    return x.toarray() if hasattr(x, 'toarray') else np.asarray(x)


def _mshape(a):
    """Array in MATLAB shape: trailing singleton dimensions beyond the second are dropped."""
    a = np.asarray(a)
    s = list(a.shape)
    while len(s) > 2 and s[-1] == 1:
        s.pop()
    return a.reshape(s) if list(a.shape) != s else a


def _msize(a):
    if isinstance(a, Cell):
        return list(a.mshape)
    s = list(np.shape(a))
    if len(s) < 2:
        s = [1] * (2 - len(s)) + s if len(s) else [1, 1]
    while len(s) > 2 and s[-1] == 1:
        s.pop()
    return s


def _flat_shape(n, Points):
    """Shape of a recorded (flattened) N-D cell of n elements built along some of the dimensions
    Points (s_sequencer: size Points(d) along varied dimensions, 1 elsewhere). The first matching
    set of dimensions is taken; a 3x1 and a 1x3 cell (Points [3 3]) cannot be told apart."""
    P = [int(p) for p in vec(np.asarray(Points, dtype=float))]
    if len(P) <= 1:
        return [1, n]
    for r in range(len(P) + 1):
        for S in itertools.combinations(range(len(P)), r):
            if int(np.prod([P[d] for d in S])) == n:
                return [P[d] if d in S else 1 for d in range(len(P))]
    return [1, n]


def _load(Array, ArrayIndex, Points):
    """LoadfromArray: element of a cell/numeric array at ArrayIndex, falling back to index 1 along
    dimensions the array does not extend into."""
    ArrayIndex = [int(i) for i in np.atleast_1d(ArrayIndex)]
    if isinstance(Array, list) and not isinstance(Array, Cell):
        nPer = _flat_shape(len(Array), Points) if len(ArrayIndex) > 1 else [1, len(Array)]
    else:
        nPer = _msize(Array)
    IndexToLoad = [1] * len(ArrayIndex)
    if len(ArrayIndex) == 1:
        if ArrayIndex[0] <= nPer[1]:
            IndexToLoad = ArrayIndex
    else:
        for i in range(len(nPer)):
            if ArrayIndex[i] <= nPer[i]:
                IndexToLoad[i] = ArrayIndex[i]
    if isinstance(Array, list):
        if len(IndexToLoad) == 1:
            return Array[IndexToLoad[0] - 1]
        lin, stride = 0, 1
        for k, i in enumerate(IndexToLoad):
            lin += (i - 1) * stride
            stride *= nPer[k] if k < len(nPer) else 1
        return Array[lin]
    A = np.asarray(Array)
    if len(IndexToLoad) == 1:
        return A.ravel(order='F')[IndexToLoad[0] - 1]
    sub = [i - 1 for i in IndexToLoad[:A.ndim]]
    if A.ndim > len(sub):
        sub += [0] * (A.ndim - len(sub))
    return A[tuple(sub)]


def _Propagator(Ham, dt):
    return scipy.linalg.expm(-1j * Ham * dt)


def _Liouvillian(Ham, Gamma, equilibriumState, dt):
    n = Ham.shape[0]
    HamSuOp = np.kron(np.eye(n), Ham) - np.kron(Ham.T, np.eye(n))
    L = -1j * HamSuOp - Gamma
    SigmaSS = Gamma @ equilibriumState
    SigmaSS = np.linalg.solve(-L, SigmaSS)
    L = scipy.linalg.expm(L * dt)
    return L, SigmaSS


def _buildPropagators(Ham0, xOp, dt, vertRes, scale):
    UTable = []
    rx = np.real(xOp)
    for iRes in range(0, vertRes + 1):
        Ham1 = scale * (iRes - vertRes / 2) * rx
        UTable.append(_Propagator(Ham0 + Ham1, dt))
    return UTable


def _buildLiouvillians(Ham0, Gamma, equilibriumState, xOp, dt, vertRes, scale):
    LTable, SigmassTable = [], []
    rx = np.real(xOp)
    for ivertRes in range(0, vertRes + 1):
        Ham1 = scale * (ivertRes - vertRes / 2) * rx
        L, SigmaSS = _Liouvillian(Ham0 + Ham1, Gamma, equilibriumState, dt)
        LTable.append(L)
        SigmassTable.append(SigmaSS)
    return LTable, SigmassTable


@njit(cache=True)
def _prop_kernel(Us, fixed, nSteps, S0, detect, D, sig, col0, store, traj):
    """Hilbert-space steps Sigma = U*Sigma*U' (U = Us[0] if fixed, else Us[k]); after step k the
    detected values D*Sigma(:) go to sig[:, col0+k] and, if store, Sigma to traj[k]."""
    n = S0.shape[0]
    S = S0.copy()
    T = np.empty((n, n), dtype=np.complex128)
    nDet = D.shape[0]
    for k in range(nSteps):
        U = Us[0] if fixed else Us[k]
        for i in range(n):
            for j in range(n):
                acc = 0j
                for l in range(n):
                    acc += U[i, l] * S[l, j]
                T[i, j] = acc
        for i in range(n):
            for j in range(n):
                acc = 0j
                for l in range(n):
                    acc += T[i, l] * np.conj(U[j, l])
                S[i, j] = acc
        if detect:
            for d in range(nDet):
                acc = 0j
                for j in range(n):
                    for i in range(n):
                        acc += D[d, i + j * n] * S[i, j]
                sig[d, col0 + k] = acc
        if store:
            traj[k] = S
    return S


@njit(cache=True)
def _liou_kernel(Ls, SSs, fixed, nSteps, SV0, n, detect, D, sig, col0, store, traj):
    """Liouville-space steps SigmaVector = SigmaSS + L*(SigmaVector - SigmaSS) (Sigma(:) order)."""
    N2 = SV0.size
    SV = SV0.copy()
    tmp = np.empty(N2, dtype=np.complex128)
    nDet = D.shape[0]
    for k in range(nSteps):
        L = Ls[0] if fixed else Ls[k]
        SS = SSs[0] if fixed else SSs[k]
        for i in range(N2):
            tmp[i] = SV[i] - SS[i]
        for i in range(N2):
            acc = 0j
            for j in range(N2):
                acc += L[i, j] * tmp[j]
            SV[i] = SS[i] + acc
        if detect:
            for d in range(nDet):
                acc = 0j
                for i in range(N2):
                    acc += D[d, i] * SV[i]
                sig[d, col0 + k] = acc
        if store:
            for i in range(N2):
                traj[k, i % n, i // n] = SV[i]
    return SV


def _stack_vectors(St, iPC, nSteps, nPC, N2):
    if nSteps == 0:
        return np.zeros((0, N2), dtype=np.complex128)
    return np.ascontiguousarray(np.stack([np.asarray(_cellget(St, iPC, k, nPC)).reshape(-1)
                                          for k in range(1, nSteps + 1)]), dtype=np.complex128)


def _run_liou(Ls, SSs, fixed, nSteps, Sigma, detect, DetArray, currentSignal, col0, traj, DensityMatrices, start):
    if nSteps <= 0:
        return Sigma
    n = Sigma.shape[0]
    SV0 = np.ascontiguousarray(_vecF(Sigma), dtype=np.complex128)
    D = np.ascontiguousarray(DetArray, dtype=np.complex128) if detect else np.zeros((1, n * n), dtype=np.complex128)
    sig = currentSignal if detect else np.zeros((1, col0 + nSteps), dtype=np.complex128)
    tb = np.empty((nSteps, n, n), dtype=np.complex128) if traj else np.empty((1, 1, 1), dtype=np.complex128)
    SV = _liou_kernel(Ls, SSs, fixed, nSteps, SV0, n, detect, D, sig, col0, traj, tb)
    if traj:
        for k in range(nSteps):
            _setlist(DensityMatrices, start + k, tb[k].copy())
    return SV.reshape(n, n, order='F')


def _stack_steps(Ut, iPC, nSteps, nPC, n):
    if nSteps == 0:
        return np.zeros((0, n, n), dtype=np.complex128)
    return np.ascontiguousarray(np.stack([_dense(_cellget(Ut, iPC, k, nPC)) for k in range(1, nSteps + 1)]),
                                dtype=np.complex128)


def _run_kernel(Us, fixed, nSteps, Sigma, detect, DetArray, currentSignal, col0, traj, DensityMatrices, start):
    """Propagate Sigma through nSteps (non-relaxation) steps, filling currentSignal columns col0...
    and the trajectory list from position start (1-based) as the MATLAB loops do."""
    if nSteps <= 0:
        return Sigma
    n = Sigma.shape[0]
    S0 = np.ascontiguousarray(Sigma, dtype=np.complex128)
    D = np.ascontiguousarray(DetArray, dtype=np.complex128) if detect else np.zeros((1, n * n), dtype=np.complex128)
    sig = currentSignal if detect else np.zeros((1, col0 + nSteps), dtype=np.complex128)
    tb = np.empty((nSteps, n, n), dtype=np.complex128) if traj else np.empty((1, 1, 1), dtype=np.complex128)
    S = _prop_kernel(Us, fixed, nSteps, S0, detect, D, sig, col0, traj, tb)
    if traj:
        for k in range(nSteps):
            _setlist(DensityMatrices, start + k, tb[k].copy())
    return S


def _ncols(c, nrows):
    if isinstance(c, Cell) and len(c.mshape) == 2:
        return c.mshape[1]
    return len(c) // max(nrows, 1)


def _cellget(c, i, j, nrows):
    """c{i,j} (1-based) of an nrows x m cell stored column-major."""
    return c[(i - 1) + (j - 1) * nrows]


def _vecF(A):
    return np.asarray(A).reshape(-1, order='F')


def _isfield(s, name):
    return isinstance(s, dict) and name in s


@nout(5)
def s_thyme(Sigma=None, Ham0=None, Det=None, Events=None, Relaxation=None, Vary=None):
    """(TimeArray, SignalArray, FinalStates, StateTrajectories, Events): propagates the density
    matrix Sigma under the static Hamiltonian Ham0 (MHz) through the event list of s_sequencer /
    s_propagationsetup (pulses as piecewise-constant propagators from a 1024-level amplitude table,
    free evolutions, optional Liouville-space relaxation, phase cycles, indirect dimensions in Vary)
    and records <Det> (normalized) during detected events. Arrays are in MATLAB shape; cell outputs
    are Cell (column-major lists)."""
    nargin = 4 + (Relaxation is not None) + (Vary is not None)
    if Sigma is None or Ham0 is None or Det is None or Events is None:
        raise ValueError('Wrong number of input arguments!')
    if nargin < 4 or nargin > 6:
        raise ValueError('Wrong number of input arguments!')
    if Relaxation is None:
        Relaxation = np.zeros((0, 0))
    if Vary is None:
        Vary = np.zeros((0, 0))
    if not isinstance(Events, list):
        Events = [Events]
    Events = [Struct(e) for e in Events]
    for e in Events:
        if isinstance(e.get('Propagation'), dict):
            e.Propagation = Struct(e.Propagation)
    nEvents = len(Events)
    Det = [_dense(d) for d in (Det if isinstance(Det, list) else [Det])]
    nDet = len(Det)
    Ham0 = _dense(Ham0) * 2 * math.pi
    Sigma = _dense(Sigma)
    if not isempty(Vary):
        Points = vec(np.asarray(Vary['Points'], dtype=float)).astype(int)
        nDimensions = Points.size
        DimensionIndices = np.ones(nDimensions, dtype=int)
        nPoints = int(np.prod(Points))
        IndirectDimensions = [int(p) for p in Points]
    else:
        Points = np.array([1])
        nDimensions = 0
        nPoints = 1
        IndirectDimensions = [1]
        DimensionIndices = np.ones(1, dtype=int)
    n = Sigma.shape[1]
    FinalStates = np.zeros(tuple(IndirectDimensions) + (n, n))
    StateTrajectories = np.zeros((0, 0))
    initialSigma = Sigma
    Resonator = False
    isPulse = np.zeros(nEvents)
    PulsePositions = []
    nPulses = 0
    for iEvent in range(nEvents):
        if Events[iEvent].type == 'pulse':
            isPulse[iEvent] = 1
            nPulses += 1
            PulsePositions.append(iEvent + 1)
            if 'Ringing' in Events[iEvent]:
                Resonator = True
    StoreInCellArray = False
    InitialEventLengths = np.zeros(nEvents)
    for iEvent in range(nEvents):
        InitialEventLengths[iEvent] = vec(Events[iEvent].t)[-1]

    VP = Vary['Pulses'] if nDimensions > 0 else []
    if not isinstance(VP, list):
        VP = [VP]

    def vary_pulse(iPulse):
        p = VP[iPulse - 1] if iPulse - 1 < len(VP) else None
        return p if isinstance(p, dict) and len(p) else None

    def increment_lengths(EventLengths):
        for iDim in range(nDimensions):
            T = Vary['IncrementationTable'][iDim]
            if not isempty(T) and np.any(np.asarray(T)):
                T = np.atleast_2d(np.asarray(T, dtype=float))
                colk = T[:, DimensionIndices[iDim] - 1]
                for ev in np.nonzero(colk)[0]:
                    EventLengths[ev] = EventLengths[ev] + colk[ev]

    if nDimensions > 0:
        for iPoint in range(1, nPoints + 1):
            EventLengths = InitialEventLengths.copy()
            DetectionTime = 0
            increment_lengths(EventLengths)
            for iPulse in range(1, nPulses + 1):
                vp = vary_pulse(iPulse)
                if vp is not None:
                    iEvent = PulsePositions[iPulse - 1]
                    t = _load(vp['ts'], DimensionIndices, Points)
                    if Resonator:
                        Ringing = _load(vp['Ringing'], DimensionIndices, Points)
                        Ringing = _mround(Ringing * (10 ** 10)) / (10 ** 10)
                        EventLengths[iEvent - 1] = vec(t)[-1] - Ringing
                    else:
                        EventLengths[iEvent - 1] = vec(t)[-1]
            Sequence, NewEventLengths = F.s_reorder_events(EventLengths.reshape(1, -1), isPulse.reshape(1, -1))[:2]
            Sequence = vec(Sequence).astype(int)
            NewEventLengths = vec(NewEventLengths).astype(float).copy()
            if Resonator:
                for iPulse in range(1, nPulses + 1):
                    ThisEventIndex = PulsePositions[iPulse - 1]
                    ThisEvent = int(np.nonzero(Sequence == ThisEventIndex)[0][0]) + 1
                    vp = vary_pulse(iPulse)
                    if vp is not None:
                        Ringing = _load(vp['Ringing'], DimensionIndices, Points)
                    else:
                        Ringing = Events[Sequence[ThisEventIndex - 1] - 1].Ringing
                    Ringing = _mround(Ringing * (10 ** 10)) / (10 ** 10)
                    NewEventLengths[ThisEvent - 1] += Ringing
                    if ThisEvent < nEvents:
                        FollowingEvent = Sequence[ThisEvent]
                        if Events[FollowingEvent - 1].type == 'pulse':
                            raise ValueError('When using a resonator, pulses need to be separated by inter pulse delays to accomodate for ringing from the resonator.')
                        NewEventLengths[FollowingEvent - 1] -= Ringing
                        if NewEventLengths[FollowingEvent - 1] < 0:
                            raise ValueError('The delay ' + str(FollowingEvent) + ' is too short to accomodate for ringing of the preceding pulse.')
            for iSequence in range(nEvents):
                if Events[Sequence[iSequence] - 1].Detection:
                    DetectionTime = DetectionTime + NewEventLengths[iSequence]
            if iPoint == 1:
                ReferenceDetection = DetectionTime
            else:
                if abs(ReferenceDetection - DetectionTime) > 1e-12:
                    StoreInCellArray = True
                    break
            for d in range(nDimensions - 1, -1, -1):
                if DimensionIndices[d] < Points[d]:
                    DimensionIndices[d] += 1
                    break
                else:
                    DimensionIndices[d] = 1
        DimensionIndices = np.ones(nDimensions, dtype=int)
    elif Resonator:
        for iEvent in range(nEvents - 1):
            if Events[iEvent].type == 'pulse':
                if Events[iEvent + 1].type == 'pulse':
                    raise ValueError('When using a resonator, pulses need to be separated by inter pulse delays to accomodate for ringing from the resonator.')
                Ringing = _mround(Events[iEvent].Ringing * (10 ** 10)) / (10 ** 10)
                Events[iEvent + 1].t = Events[iEvent + 1].t - Ringing

    SignalArray = TimeArray = None
    firstDetection = True
    for iPoints in range(1, nPoints + 1):
        EventLengths = InitialEventLengths.copy()
        Sigma = initialSigma
        if not isempty(Vary):
            increment_lengths(EventLengths)
            for iPulse in range(1, nPulses + 1):
                vp = vary_pulse(iPulse)
                if vp is not None:
                    iEvent = PulsePositions[iPulse - 1]
                    IQ = _load(vp['IQs'], DimensionIndices, Points)
                    t = _load(vp['ts'], DimensionIndices, Points)
                    ev = Events[iEvent - 1]
                    if not (np.shape(np.atleast_2d(ev.IQ)) == np.shape(np.atleast_2d(IQ))
                            and np.array_equal(np.atleast_2d(ev.IQ), np.atleast_2d(IQ))):
                        ev.IQ = IQ
                        ev.t = t
                        ev.Propagation = np.zeros((0, 0))
                    EventLengths[iEvent - 1] = vec(ev.t)[-1]
            Sequence, NewEventLengths = F.s_reorder_events(EventLengths.reshape(1, -1), isPulse.reshape(1, -1))[:2]
            Sequence = vec(Sequence).astype(int)
            NewEventLengths = vec(NewEventLengths)
            for iEvent in range(1, nEvents + 1):
                NewPosition = int(np.nonzero(Sequence == iEvent)[0][0])
                ev = Events[iEvent - 1]
                if ev.type == 'free evolution' and NewEventLengths[NewPosition] != ev.t:
                    ev.t = NewEventLengths[NewPosition]
                    ev.Propagation = np.zeros((0, 0))
        else:
            Sequence = np.arange(1, nEvents + 1)

        # ---- propagators of pulses
        for iEvent in range(nEvents):
            cE = Events[iEvent]
            prop = cE.Propagation
            if isempty(prop) or not (_isfield(prop, 'Utotal') or _isfield(prop, 'Ltotal')):
                if cE.type == 'pulse':
                    PCm = np.atleast_2d(np.asarray(cE.PhaseCycle, dtype=float))
                    nPhaseCycle = PCm.shape[0]
                    IQm = np.atleast_2d(np.asarray(cE.IQ))
                    rMax = np.max(np.abs(np.real(IQm)))
                    iMax = np.max(np.abs(np.imag(IQm)))
                    MaxWave = max(rMax, iMax)
                    if 'vertRes' in cE:
                        vertRes = int(cE.vertRes) + 1 if cE.vertRes % 2 else int(cE.vertRes)
                    else:
                        vertRes = 1024
                    scale = 2 * 2 * 2 * math.pi * MaxWave / vertRes
                    realArbitrary = np.real(IQm) / MaxWave
                    realBinary = _mround(vertRes * (realArbitrary + 1) / 2).astype(int)
                    ComplexExcitation = bool(cE.ComplexExcitation)
                    if ComplexExcitation:
                        imagArbitrary = np.imag(IQm) / MaxWave
                        imagBinary = _mround(vertRes * (imagArbitrary + 1) / 2).astype(int)
                    relax = bool(cE.Relaxation)
                    nWave = max(realBinary.shape)
                    if (not relax and not _isfield(prop, 'Utotal')) or (relax and not _isfield(prop, 'Ltotal')):
                        if not isinstance(cE.Propagation, dict):
                            cE.Propagation = Struct()
                        xOp = _dense(cE.xOp)
                        if relax:
                            nS = Sigma.shape[0]
                            Gamma = _dense(Relaxation['Gamma'])
                            equilibriumState = _vecF(_dense(Relaxation['equilibriumState'])).reshape(-1, 1)
                        if not ComplexExcitation and not relax:
                            UTable = _buildPropagators(Ham0, xOp, cE.TimeStep, vertRes, scale)
                            cE.Propagation.UTable = UTable
                        elif not ComplexExcitation and relax:
                            LTable, SigmassTable = _buildLiouvillians(Ham0, Gamma, equilibriumState, xOp,
                                                                      cE.TimeStep, vertRes, scale)
                            cE.Propagation.LTable = LTable
                            cE.Propagation.SigmassTable = SigmassTable
                        detect = bool(cE.Detection)
                        if detect and not relax:
                            Utotal = [None] * (nPhaseCycle * nWave)
                            ushape = (nPhaseCycle, nWave)
                        elif not relax:
                            Utotal = [None] * nPhaseCycle
                            ushape = (nPhaseCycle, 1)
                        else:
                            Ltotal = [None] * (nPhaseCycle * nWave)
                            SigmaSStotal = [None] * (nPhaseCycle * nWave)
                        for iPC in range(1, nPhaseCycle + 1):
                            for iWP in range(1, nWave + 1):
                                if not ComplexExcitation:
                                    rb = realBinary[iPC - 1, iWP - 1]
                                    if not relax:
                                        U = UTable[rb]
                                    else:
                                        L = LTable[rb]
                                        SigmaSS = SigmassTable[rb]
                                else:
                                    Ham1 = (scale / 2 * (realBinary[iPC - 1, iWP - 1] - vertRes / 2) * np.real(xOp)
                                            + 1j * scale / 2 * (imagBinary[iPC - 1, iWP - 1] - vertRes / 2) * np.imag(xOp))
                                    Ham = Ham0 + Ham1
                                    if not relax:
                                        U = _Propagator(Ham, cE.TimeStep)
                                    else:
                                        L, SigmaSS = _Liouvillian(Ham, Gamma, equilibriumState, cE.TimeStep)
                                if not relax:
                                    if detect:
                                        Utotal[(iPC - 1) + (iWP - 1) * nPhaseCycle] = U
                                    elif iWP == 1:
                                        Utotal[iPC - 1] = U
                                    else:
                                        Utotal[iPC - 1] = U @ Utotal[iPC - 1]
                                else:
                                    Ltotal[(iPC - 1) + (iWP - 1) * nPhaseCycle] = L
                                    SigmaSStotal[(iPC - 1) + (iWP - 1) * nPhaseCycle] = SigmaSS
                        if not relax:
                            cE.Propagation.Utotal = Cell(Utotal, ushape)
                        else:
                            cE.Propagation.Ltotal = Cell(Ltotal, (nPhaseCycle, nWave))
                            cE.Propagation.SigmaSStotal = Cell(SigmaSStotal, (nPhaseCycle, nWave))

        # ---- propagation and detection
        ttotal = 0
        firstDetection = True
        firstDensityMatrix = True
        CurrentStateTrajectories = Cell()
        counter = 0
        AcquisitionIndex = [int(i) for i in DimensionIndices]
        sig_cols, t_cols = [], []
        Signal = t = None
        for iEvent in Sequence:
            cE = Struct(Events[iEvent - 1])
            detect = bool(cE.Detection)
            relax = bool(cE.Relaxation)
            nS = Sigma.shape[0]
            if detect:
                if cE.type == 'pulse':
                    tvector = vec(cE.t).astype(float)
                    currentSignal = np.zeros((nDet, np.atleast_2d(cE.IQ).shape[1] + 1), dtype=complex)
                else:
                    tvector = colon(0, cE.TimeStep, cE.t)
                    currentSignal = np.zeros((nDet, tvector.size), dtype=complex)
                DetArray = np.zeros((nDet, nS * nS), dtype=complex)
                for iDet in range(nDet):
                    Det[iDet] = Det[iDet].T.reshape(1, nS * nS, order='F') if Det[iDet].ndim == 2 and Det[iDet].shape[0] == nS \
                        else np.asarray(Det[iDet]).reshape(1, -1)
                    normDet_ = (Det[iDet] @ Det[iDet].conj().T)[0, 0]
                    if np.iscomplexobj(normDet_) and normDet_.imag == 0:
                        normDet_ = normDet_.real          # MATLAB drops an all-zero imaginary part
                    DetArray[iDet, :] = Det[iDet][0] / normDet_
                currentSignal[:, 0] = DetArray @ _vecF(Sigma)
            else:
                if cE.type == 'pulse':
                    tvector = vec(cE.t).astype(float)
                else:
                    tvector = np.array([0.0, float(cE.t)])
                    cE.TimeStep = cE.t
                currentSignal = None
            traj = bool(cE.StateTrajectories)
            if traj:
                DensityMatrices = [Sigma]
            else:
                DensityMatrices = None
            if cE.type == 'pulse':
                Nt = vec(cE.t).size
                tv = np.zeros(max(tvector.size, Nt + 1))
                tv[:tvector.size] = tvector
                tv[0] = 0
                tv[1:Nt + 1] = vec(cE.t) + cE.TimeStep
                tvector = tv
                PCm = np.atleast_2d(np.asarray(cE.PhaseCycle, dtype=float))
                nPhaseCycle = PCm.shape[0]
                if nPhaseCycle > 1:
                    StateBeforePC = Sigma
                    loopState = np.zeros(Sigma.shape)
                    PCnorm = np.sum(np.abs(PCm[:, 1]))
                    PCSignal = None
                for iPC in range(1, nPhaseCycle + 1):
                    if not relax:
                        Ut = cE.Propagation['Utotal']
                        nSteps = _ncols(Ut, nPhaseCycle)
                        Us = _stack_steps(Ut, iPC, nSteps, nPhaseCycle, nS)
                        Sigma = _run_kernel(Us, False, nSteps, Sigma, detect, DetArray if detect else None,
                                            currentSignal, 1, traj, DensityMatrices, 2)
                    else:                                   # Liouville space (relaxation)
                        Lt = cE.Propagation['Ltotal']
                        St = cE.Propagation['SigmaSStotal']
                        nSteps = _ncols(Lt, nPhaseCycle)
                        Ls = _stack_steps(Lt, iPC, nSteps, nPhaseCycle, nS * nS)
                        SSs = _stack_vectors(St, iPC, nSteps, nPhaseCycle, nS * nS)
                        Sigma = _run_liou(Ls, SSs, False, nSteps, Sigma, detect, DetArray if detect else None,
                                          currentSignal, 1, traj, DensityMatrices, 2)
                    if nPhaseCycle > 1:
                        PCweight = PCm[iPC - 1, 1] / PCnorm
                        loopState = loopState + PCweight * Sigma
                        if detect:
                            PCSignal = PCweight * currentSignal if iPC == 1 else PCSignal + PCweight * currentSignal
                        if iPC != nPhaseCycle:
                            Sigma = StateBeforePC
                if nPhaseCycle > 1:
                    Sigma = loopState
                    if detect:
                        currentSignal = PCSignal
            else:
                prop = Events[iEvent - 1].Propagation
                if not relax:
                    if _isfield(prop, 'Utotal'):
                        U = _dense(prop['Utotal'])
                    else:
                        U = _Propagator(Ham0, cE.TimeStep)
                        if not isinstance(Events[iEvent - 1].Propagation, dict):
                            Events[iEvent - 1].Propagation = Struct()
                        Events[iEvent - 1].Propagation.Utotal = U
                else:
                    if _isfield(prop, 'Ltotal'):
                        L = _dense(prop['Ltotal'])
                        SigmaSS = np.asarray(prop['SigmaSStotal']).reshape(-1, 1)
                    else:
                        Gamma = _dense(Relaxation['Gamma'])
                        equilibriumState = _vecF(_dense(Relaxation['equilibriumState'])).reshape(-1, 1)
                        L, SigmaSS = _Liouvillian(Ham0, Gamma, equilibriumState, cE.TimeStep)
                        if not isinstance(Events[iEvent - 1].Propagation, dict):
                            Events[iEvent - 1].Propagation = Struct()
                        Events[iEvent - 1].Propagation.Ltotal = L
                        Events[iEvent - 1].Propagation.SigmaSStotal = SigmaSS
                if not relax:
                    Us = np.ascontiguousarray(np.asarray(U, dtype=np.complex128).reshape(1, nS, nS))
                    Sigma = _run_kernel(Us, True, tvector.size - 1, Sigma, detect, DetArray if detect else None,
                                        currentSignal, 1, traj, DensityMatrices, 2)
                else:
                    Ls = np.ascontiguousarray(np.asarray(L, dtype=np.complex128).reshape(1, nS * nS, nS * nS))
                    SSs = np.ascontiguousarray(np.asarray(SigmaSS, dtype=np.complex128).reshape(1, nS * nS))
                    Sigma = _run_liou(Ls, SSs, True, tvector.size - 1, Sigma, detect, DetArray if detect else None,
                                      currentSignal, 1, traj, DensityMatrices, 2)
            if currentSignal is not None and currentSignal.size:
                if firstDetection:
                    sig_cols = [currentSignal[:, 0:1]]
                    t_cols = [np.array([float(ttotal)])]
                    firstDetection = False
                sig_cols.append(currentSignal[:, 1:])
                t_cols.append(tvector[1:] + ttotal)
            if traj:
                if firstDensityMatrix:
                    for iCell, Dm in enumerate(DensityMatrices):
                        CurrentStateTrajectories.setitem([iCell + 1], Dm)
                    firstDensityMatrix = False
                    counter = len(DensityMatrices)
                elif detect or iEvent == Sequence[-1]:
                    iStart = counter
                    for iDensity in range(2, len(DensityMatrices) + 1):
                        CurrentStateTrajectories.setitem([iStart + iDensity - 1], DensityMatrices[iDensity - 1])
                    counter = counter + len(DensityMatrices) - 1
            ttotal = ttotal + tvector[-1]
        if not firstDetection:
            Signal = np.concatenate(sig_cols, axis=1)
            t = np.concatenate(t_cols).reshape(1, -1)
            if iPoints == 1:
                if not StoreInCellArray:
                    SignalSize = Signal.shape
                    SignalArray = np.zeros(tuple(IndirectDimensions) + SignalSize, dtype=complex)
                    TimeArray = np.zeros(tuple(IndirectDimensions) + (SignalSize[1],))
                else:
                    shp = (1, IndirectDimensions[0]) if len(IndirectDimensions) == 1 else tuple(IndirectDimensions)
                    SignalArray = Cell([np.zeros((0, 0))] * int(np.prod(shp)), Cell._trim(shp))
                    TimeArray = Cell([np.zeros((0, 0))] * int(np.prod(shp)), Cell._trim(shp))
            if not StoreInCellArray:
                ms = _msize(TimeArray)
                if t.size > 1 and ms[-1] != t.size:
                    StoreInCellArray = True
                    shp = (1, IndirectDimensions[0]) if len(IndirectDimensions) == 1 else tuple(IndirectDimensions)
                    NewSignalArray = Cell([np.zeros((0, 0))] * int(np.prod(shp)), Cell._trim(shp))
                    NewTimeArray = Cell([np.zeros((0, 0))] * int(np.prod(shp)), Cell._trim(shp))
                    for jSignal in range(1, iPoints):
                        if len(IndirectDimensions) == 1:
                            indexID = [jSignal]
                        else:
                            indexID = [int(i) + 1 for i in np.unravel_index(jSignal - 1, IndirectDimensions, order='F')]
                        sub = tuple(i - 1 for i in indexID)
                        S_ = SignalArray[sub]
                        NewSignalArray.setitem(indexID, S_.reshape(nDet, -1))
                        NewTimeArray.setitem(indexID, TimeArray[sub].reshape(1, -1))
                    NewSignalArray.setitem(AcquisitionIndex, Signal)
                    NewTimeArray.setitem(AcquisitionIndex, t)
                    SignalArray = NewSignalArray
                    TimeArray = NewTimeArray
                else:
                    sub = tuple(i - 1 for i in AcquisitionIndex)
                    TimeArray[sub] = t.ravel()
                    SignalArray[sub] = Signal
            else:
                SignalArray.setitem(AcquisitionIndex, Signal)
                TimeArray.setitem(AcquisitionIndex, t)
        sub = tuple(i - 1 for i in AcquisitionIndex)
        if np.iscomplexobj(Sigma) and not np.iscomplexobj(FinalStates):
            FinalStates = FinalStates.astype(complex)
        FinalStates[sub] = Sigma
        if len(CurrentStateTrajectories):
            if not isinstance(StateTrajectories, Cell):
                StateTrajectories = Cell()
            StateTrajectories.setitem(AcquisitionIndex, CurrentStateTrajectories)
        if not isempty(Vary):
            for d in range(nDimensions - 1, -1, -1):
                if DimensionIndices[d] < Points[d]:
                    DimensionIndices[d] += 1
                    break
                else:
                    DimensionIndices[d] = 1
    if firstDetection:
        SignalArray = np.zeros((0, 0))
        TimeArray = np.zeros((0, 0))
    if isinstance(SignalArray, np.ndarray):
        SignalArray = _mshape(SignalArray)
        TimeArray = _mshape(TimeArray)
    FinalStates = _mshape(FinalStates)
    return TimeArray, SignalArray, FinalStates, StateTrajectories, Events


def _setlist(L, k, v):
    """MATLAB L{k} = v for a growing row cell held as a Python list (k 1-based)."""
    while len(L) < k:
        L.append(np.zeros((0, 0)))
    L[k - 1] = v

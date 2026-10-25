"""Compute propagation operator"""
import math
import warnings
import numpy as np
import scipy.linalg
from .._mat import vec


def _dense(A):
    return A.toarray() if hasattr(A, 'toarray') else np.asarray(A)


def propint(*varargin, nargout=None):
    """U = propint(H0, H1, tlim, Freq, Phase=0, nIntervals=256) or propint(Ustore, tlim, Freq,
    Phase=0): propagator under H0 + H1*cos(2*pi*Freq*t + Phase). nargout=2 also returns the
    stored one-period propagators (list). Times in us, frequencies in MHz, as in EasySpin."""
    if nargout is None:
        nargout = 1
    if nargout in (0, 1):
        SaveU = False
    elif nargout == 2:
        SaveU = True
    else:
        raise ValueError('Wrong number of output arguments')
    nargin = len(varargin)
    UseUstore = isinstance(varargin[0], (list, tuple))
    if UseUstore:
        Ustore, tlim, Frequency = varargin[0:3]
        Ustore = [_dense(u) for u in Ustore]
        nIntervals = len(Ustore) - 1
        Phase = varargin[3] if nargin > 3 else 0
    else:
        H0, H1, tlim, Frequency = varargin[0:4]
        H0 = _dense(H0)
        H1 = _dense(H1)
        Phase = varargin[4] if nargin > 4 else 0
        nIntervals = varargin[5] if nargin > 5 else 256
    nIntervals = int(nIntervals)
    tlim = vec(np.asarray(tlim))
    if tlim.size == 1:
        tlim = np.array([0, tlim[0]])
    Fr = np.asarray(Frequency)
    if Fr.size != 1 or np.iscomplexobj(Fr) or float(Fr) <= 0:
        raise ValueError('Frequency must be real and positive!')
    Frequency = float(Fr)
    if tlim.size != 2 or np.iscomplexobj(tlim):
        raise ValueError('tlim is not a real 1x2 array!')
    tlim = tlim.astype(float)
    if not UseUstore and np.all(H1 == 0):
        return scipy.linalg.expm(-2j * np.pi * (tlim[1] - tlim[0]) * H0)

    tPeriod = 1 / Frequency
    dt = tPeriod / nIntervals
    t = (np.arange(nIntervals) + 0.5) * dt
    nSteps = (tlim[1] - tlim[0]) / dt
    if abs(math.trunc(nSteps) / nSteps - 1) > 1e-2:
        warnings.warn('Pulse too short. Increase number of integration points!')
    j = np.trunc(nIntervals * np.fmod(tlim / tPeriod, 1) + 1).astype(int)   # 1-based
    nPeriods = tlim / tPeriod

    if not UseUstore:
        UPeriod = np.eye(H0.shape[0])
        cH0 = -2j * np.pi * dt * H0
        cH1 = -2j * np.pi * dt * H1
        ct = np.cos(2 * np.pi * Frequency * t + Phase)
        if SaveU:
            Ustore = [None] * (nIntervals + 1)
            Ustore[0] = UPeriod
        U1 = U2 = None
        for k in range(1, nIntervals + 1):
            if k == j[0]:
                U1 = UPeriod
            if k == j[1]:
                U2 = UPeriod
            UPeriod = scipy.linalg.expm(cH0 + cH1 * ct[k - 1]) @ UPeriod
            if SaveU:
                Ustore[k] = UPeriod
    else:
        UPeriod = Ustore[-1]
        # EasySpin takes Ustore{j+1} here, one interval later than the direct path (which
        # stores UPeriod at k == j, i.e. after j-1 intervals = Ustore{j}); corrected (DEVIATIONS)
        U1 = Ustore[j[0] - 1]
        U2 = Ustore[j[1] - 1]
    if U1 is None or U2 is None:
        raise NameError("Unrecognized function or variable 'U1'.")
    p = int(np.trunc(nPeriods[1]) - np.trunc(nPeriods[0]))
    U = U2 @ np.linalg.matrix_power(UPeriod, p) @ U1.conj().T
    if nargout == 2:
        return U, Ustore
    return U

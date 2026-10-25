"""Steady-state solutions of Bloch equations"""
import datetime
import math
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from .._registry import F
from .._struct import Struct
from ..config import config


def _ifft_sym(x, n):
    """MATLAB real(ifft(x,'symmetric')) for a length-n column: only the first floor(n/2)+1
    entries are used (conjugate symmetry assumed), as FFTW's c2r transform does."""
    return np.fft.irfft(x[:n // 2 + 1], n)


def blochsteady(g=None, T1=None, T2=None, DeltaB0=None, B1=None, modAmp=None, modFreq=None,
                Options=None, nargout=None, plot=False):
    """Periodic steady state of the Bloch equations under sinusoidal field modulation.
    g, T1/T2 (us), DeltaB0/B1/modAmp (mT), modFreq (kHz); Options.nPoints, .kmax, .Method
    ('fft' or 'td'), .Verbosity. nargout 2 -> (t, My); 4 (default) -> (t, Mx, My, Mz);
    7 -> (t, Mx, My, Mz, Xk, Yk, Zk); t in us. plot=True draws the MATLAB nargout=0 figure
    (and, as there, repeats the first point at the end of one period)."""
    args = (g, T1, T2, DeltaB0, B1, modAmp, modFreq, Options)
    nargin = 0
    for k, a in enumerate(args):
        if a is not None:
            nargin = k + 1
    msgs = {1: 'Second input (T1) is missing.', 2: 'Third input (T2) is missing.',
            3: 'Fourth input (deltaB0) is missing.', 4: 'Fifth input (B1) is missing.',
            5: 'Sixth input (ModAmp) is missing.', 6: 'Seventh input (ModFreq) is missing.'}
    if nargin in msgs:
        raise ValueError(msgs[nargin])
    if nargin == 7:
        Options = Struct()
    if not isinstance(Options, dict):
        raise TypeError('Eighth input (Options) must be a structure.')
    Options = Struct(Options)
    if plot:
        nargout = 0
    elif nargout is None:
        nargout = 4
    if 'Verbosity' not in Options:
        Options.Verbosity = 0
    config.LogLevel = Options.Verbosity
    try:
        return _blochsteady(g, T1, T2, DeltaB0, B1, modAmp, modFreq, Options, nargout)
    finally:
        config.LogLevel = None


def _blochsteady(g, T1, T2, DeltaB0, B1, modAmp, modFreq, Options, nargout):
    F.logmsg(1, '=begin=blochsteady======' + _now() + '=================')
    if 'nPoints' not in Options:
        Options.nPoints = np.zeros((0, 0))
    if 'Method' not in Options:
        Options.Method = 'fft'
    onlyAbsorption = nargout == 2

    T1 = np.asarray(T1, dtype=float) * 1e-6
    T2 = np.asarray(T2, dtype=float) * 1e-6
    DeltaB0 = np.asarray(DeltaB0, dtype=float) * 1e-3
    B1 = np.asarray(B1, dtype=float) * 1e-3
    modAmp = np.asarray(modAmp, dtype=float) * 1e-3
    modFreq = np.asarray(modFreq, dtype=float) * 1e3

    M0 = 1
    bm = F.bmagn()
    bm = bm[0] if isinstance(bm, tuple) else bm
    gamma = bm / F.hbar() * np.asarray(g, dtype=float)
    if np.size(g) != 1:
        raise ValueError('g (1st input) must be a single number.')
    if T1.size != 1 or T1 <= 0:
        raise ValueError('T1 (2nd input) must be a single positive number.')
    if T2.size != 1 or T2 <= 0:
        raise ValueError('T2 (3rd input) must be a single positive number.')
    if T2 > T1:
        raise ValueError('T2 cannot be larger than T1.')
    if DeltaB0.size != 1:
        raise ValueError('The field offset (4th input_ must be a single number.')
    if B1.size != 1 or B1 <= 0:
        raise ValueError('The microwave field amplitude B1 (5th input) must be a single positive number.')
    if modAmp.size != 1 or modAmp <= 0:
        raise ValueError('The modulation amplitude (6th input) must be a single positive number.')
    if modFreq.size != 1 or modFreq <= 0:
        raise ValueError('The modulation frequency (7th input) must be a single positive number.')
    T1, T2, DeltaB0, B1 = float(T1), float(T2), float(DeltaB0), float(B1)
    modAmp, modFreq, gamma = float(modAmp), float(modFreq), float(gamma)
    omegam = 2 * np.pi * modFreq

    F.logmsg(1, 'Determination of maximum Fourier order (kmax)')
    if 'kmax' not in Options or np.size(Options.kmax) == 0:
        maxfieldoffset = max(modAmp / 2 - DeltaB0, DeltaB0 + modAmp / 2)
        maxfieldoffset = min(maxfieldoffset, modAmp)
        maxfreqoffset = (gamma / 2 / np.pi) * maxfieldoffset
        kmax = maxfreqoffset / modFreq
        kmax = math.ceil(kmax * 1.4)
        F.logmsg(1, '  based on max. field offset: %d', kmax)
        threshold = 1e-6
        thresholdorder = -np.log(threshold) / 2 * gamma * modAmp * T2
        thresholdorder = math.ceil(thresholdorder)
        F.logmsg(1, '  based on T2: %d', thresholdorder)
        minkmax = 20
        kmax = max(kmax, thresholdorder)
        kmax = max(kmax, minkmax)
        F.logmsg(1, '  larger of the two, but at least %d: %d', minkmax, kmax)
    else:
        kmax = int(Options.kmax)
        F.logmsg(1, '  user-supplied: %d', kmax)

    F.logmsg(1, 'Frequency-domain steady-state solution')
    F.logmsg(1, '  Set up diagonals')
    k = np.arange(-(kmax + 1), kmax + 2, dtype=float)
    a = 1j * k * omegam
    b = gamma * DeltaB0
    c = gamma * B1
    d = gamma * modAmp / 4
    tau1 = 1 / (a + 1 / T1)
    tau2 = 1 / (a + 1 / T2)
    q = np.arange(1, 2 * kmax + 2)          # 0-based positions of MATLAB q = (1:2*kmax+1)+1
    c2m = d ** 2 * tau2[q - 1]
    c2p = d ** 2 * tau2[q + 1]
    c1m = b * d * (tau2[q] + tau2[q - 1])
    c1p = b * d * (tau2[q] + tau2[q + 1])
    c0 = a[q] + 1 / T2 + b ** 2 * tau2[q] + c ** 2 * tau1[q] + d ** 2 * (tau2[q - 1] + tau2[q + 1])
    cL = c * M0 * tau1[kmax + 1] / T1

    F.logmsg(1, '  Assemble sparse pentadiagonal matrix')
    nRows = 2 * kmax + 1
    C0 = np.zeros(nRows, dtype=complex)
    C0[kmax] = cL
    # spdiags([c2m c1m c0 c1p c2p],[0 1 2 3 4],nRows,nRows+4) with nRows < nRows+4 takes the
    # diagonal entries row-wise: row i holds B(i,k) at column i+d(k); then columns 3:end-2.
    rows, cols, vals = [], [], []
    for off, col in zip((-2, -1, 0, 1, 2), (c2m, c1m, c0, c1p, c2p)):
        i = np.arange(nRows)
        j = i + off
        ok = (j >= 0) & (j < nRows)
        rows.append(i[ok]); cols.append(j[ok]); vals.append(col[ok])
    P = scipy.sparse.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                shape=(nRows, nRows))
    F.logmsg(1, '  Solve sparse linear system for y Fourier coefficients')
    F.logmsg(1, '    matrix size: %d x %d', nRows, nRows)
    Y = scipy.sparse.linalg.spsolve(P, C0)

    F.logmsg(1, '  Calculate x and z Fourier coefficients')
    q = np.arange(1, 2 * kmax)              # 0-based positions of MATLAB q = 2:2*kmax
    Xk = Zk = None
    if not onlyAbsorption:
        Xk = tau2[q + 1] * (b * Y[q] + d * (Y[q - 1] + Y[q + 1]))
        deltak0 = np.zeros(2 * kmax - 1)
        deltak0[kmax - 1] = 1
        Zk = tau1[q + 1] * (-c * Y[q] + M0 * deltak0 / T1)
    Yk = Y[q]

    F.logmsg(1, 'Calculation of time-domain signal.')
    tPeriod = 1 / modFreq
    nPoints = Options.nPoints
    if np.size(nPoints) == 0:
        nPoints = 2 * kmax - 1
    nPoints = int(nPoints)
    t = np.linspace(0, tPeriod, nPoints + 1)[:-1]

    Mx = Mz = None
    if Options.Method == 'td':
        F.logmsg(1, '  Method: explicit evolution with complex exponentials')
        My = 0
        if not onlyAbsorption:
            Mx = 0
            Mz = 0
        for kk in range(-(kmax - 1), kmax):
            idx = kk + kmax - 1
            phase = np.exp(1j * kk * omegam * t)
            if not onlyAbsorption:
                Mx = Mx + Xk[idx] * phase
            My = My + Yk[idx] * phase
            if not onlyAbsorption:
                Mz = Mz + Zk[idx] * phase
        My = np.real(My)
        if not onlyAbsorption:
            Mx = np.real(Mx)
            Mz = np.real(Mz)
    elif Options.Method == 'fft':
        F.logmsg(1, '  Method: Inverse Fourier transform')
        n = Yk.size
        F.logmsg(1, '    nPoints: %d, Mky: %d', nPoints, n)
        if nPoints <= n:
            FT = lambda m: n * _ifft_sym(np.fft.ifftshift(m), n)
            My = FT(Yk)
            if not onlyAbsorption:
                Mx = FT(Xk)
                Mz = FT(Zk)
            if nPoints < n:
                tt = np.linspace(0, tPeriod, n + 1)[:-1]
                My = np.interp(t, tt, My)
                if not onlyAbsorption:
                    Mx = np.interp(t, tt, Mx)
                    Mz = np.interp(t, tt, Mz)
        else:
            def fill(m):
                return np.concatenate([m[:kmax], np.zeros(nPoints - 2 * kmax + 1), m[kmax:]])
            FT = lambda m: nPoints * _ifft_sym(fill(np.fft.ifftshift(m)), nPoints)
            My = FT(Yk)
            if not onlyAbsorption:
                Mx = FT(Xk)
                Mz = FT(Zk)
    else:
        raise ValueError('Unknown method.')

    addLastPoint = nargout == 0
    if addLastPoint:
        t = np.append(t, 1 / modFreq)
        My = np.append(My, My[0])
        if not onlyAbsorption:
            Mx = np.append(Mx, Mx[0])
            Mz = np.append(Mz, Mz[0])
    t = t / 1e-6
    col = lambda v: np.asarray(v).reshape(-1, 1)
    F.logmsg(1, '=end=blochsteady========%s=================\n', _now())
    if nargout == 2:
        return col(t), col(My)
    if nargout == 7:
        return col(t), col(Mx), col(My), col(Mz), col(Xk), col(Yk), col(Zk)
    if nargout == 0:
        _plotresults(t, My, T1, T2, DeltaB0, B1, modAmp, modFreq, omegam)
    return col(t), col(Mx), col(My), col(Mz)


def _now():
    return datetime.datetime.now().strftime('%d-%b-%Y %H:%M:%S')


def _plotresults(t, My, T1, T2, DeltaB0, B1, modAmp, modFreq, omegam):
    import matplotlib.pyplot as plt
    tx1 = np.arccos(-DeltaB0 * 2 / modAmp + 0j).real / omegam * 1e6
    tx2 = 1 / modFreq * 1e6 - tx1
    refc = (0.7, 0.7, 0.7)
    fig = plt.gcf()
    fig.clf()
    ax1 = fig.add_subplot(3, 1, (1, 2))
    ax1.plot(t, My)
    ax1.autoscale(tight=True)
    ax1.set_ylabel('absorption signal')
    ax1.set_title('T_1 = %g µs, T_2 = %g µs, modAmp = %g mT, modFreq =  %g kHz, B_1 = %g mT, ΔB = %g mT'
                  % (T1 * 1e6, T2 * 1e6, modAmp * 1e3, modFreq / 1e3, B1 * 1e3, DeltaB0 * 1e3), fontsize=8)
    for x in (tx1, tx2):
        ax1.axvline(x, color=refc)
    ax2 = fig.add_subplot(3, 1, 3)
    Bmod = modAmp / 2 * 1e3 * np.cos(omegam * t * 1e-6)
    ax2.plot(t, Bmod)
    ax2.autoscale(tight=True)
    ax2.set_xlabel('time (µs)')
    ax2.set_ylabel('modulation field (mT)')
    ax2.axhline(-DeltaB0 * 1e3, color=refc)
    for x in (tx1, tx2):
        ax2.axvline(x, color=refc)
    return fig

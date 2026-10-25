"""HYSCORE powder spectrum outline"""
import math
import numpy as np
import scipy.linalg
from .._registry import F
from .._struct import Struct, isfield
from .._mat import vec


def _v(x):
    return x[0] if isinstance(x, tuple) else x


def nucfrq2d(sys=None, B0=None, tau=None, plot=False):
    """nucfrq2d(Sys, B0, tau=0) -> Struct with the HYSCORE powder outline data that EasySpin plots:
    FreqsAlpha, FreqsBeta (nOri x nfreq, MHz), LarmorFreqs, maxFrq, the blind-spot map (freq,
    Modulation; only for tau > 0), grid and tri. MATLAB only plots; plot=True draws the same figure."""
    if sys is None:
        return None
    if B0 is None:
        raise ValueError('Wrong number of input arguments!')
    if isfield(sys, 'S'):
        if np.any(np.asarray(sys['S']) > 1 / 2):
            raise ValueError('Systems with S>1/2 are not supported.')
    Options = Struct(GridSize=31, nPoints=100, expand=1.1, baColor=np.array([1, 0, 0]) * 0.6,
                     abColor=np.array([0, 0, 1]) * 0.6, QuadraticAxes=0)
    if tau is None:
        tau = 0.0
    tau = vec(np.asarray(tau, dtype=float))
    computeBlindSpots = bool(np.all(tau > 0)) and not Options.QuadraticAxes

    H0, mux, muy, muz = F.ham(sys)
    H0, mux, muy, muz = (np.asarray(x.toarray() if hasattr(x, 'toarray') else x) for x in (H0, mux, muy, muz))
    N = H0.shape[0]
    nn = N // 2
    triup = np.triu(np.ones((nn, nn)), 1)
    BetaMask = scipy.linalg.block_diag(triup, np.zeros((nn, nn))).astype(bool)
    AlphaMask = scipy.linalg.block_diag(np.zeros((nn, nn)), triup).astype(bool)
    nfreq = int(np.count_nonzero(BetaMask))
    thisSymm = F.hamsymm(sys)
    if isinstance(thisSymm, tuple):
        thisSymm = thisSymm[0]
    grid, tri = F.sphgrid(thisSymm, Options.GridSize, nargout=2)
    x = np.asarray(grid.vecs)
    nOct = grid.nOctants
    nOri = x.shape[1]
    FreqsAlpha = np.zeros((nOri, nfreq))
    FreqsBeta = np.zeros((nOri, nfreq))
    B = B0 * x
    # column-major selection, as MATLAB EE(Mask)
    aF = AlphaMask.ravel(order='F')
    bF = BetaMask.ravel(order='F')
    for k in range(nOri):
        H = H0 - (B[0, k] * mux + B[1, k] * muy + B[2, k] * muz)
        E = np.linalg.eigvalsh(H)
        EE = np.tile(E.reshape(-1, 1), (1, N))
        EE = EE.T - EE
        FreqsAlpha[k, :] = EE.ravel(order='F')[aF]
        FreqsBeta[k, :] = EE.ravel(order='F')[bF]
    if Options.QuadraticAxes:
        FreqsAlpha = FreqsAlpha ** 2
        FreqsBeta = FreqsBeta ** 2
    maxFrq = max(np.max(FreqsAlpha), np.max(FreqsBeta))
    out = Struct(FreqsAlpha=FreqsAlpha, FreqsBeta=FreqsBeta, maxFrq=maxFrq, grid=grid, tri=tri)
    if computeBlindSpots:
        nPoints = 2 * Options.nPoints + 1
        freq = Options.expand * maxFrq * np.linspace(-1, 1, nPoints)
        Modulation = np.zeros((nPoints, nPoints))
        for k in range(tau.size):
            Modulation1D = np.abs(np.sin(math.pi * tau[k] * freq))
            Modulation = Modulation + np.outer(Modulation1D, Modulation1D)
        Modulation = Modulation / np.max(Modulation)
        out.freq = freq
        out.Modulation = Modulation
    LarmorFreqs = _v(F.nmagn()) * B0 * vec(F.nucgval(sys['Nucs'])) / _v(F.planck()) / 1e9
    out.LarmorFreqs = LarmorFreqs
    if plot:
        _plot(out, Options, nOct, nfreq, computeBlindSpots)
    return out


def _plot(o, Options, nOct, nfreq, computeBlindSpots):
    import matplotlib.pyplot as plt
    e = Options.expand * o.maxFrq
    fig, ax = plt.subplots()
    if computeBlindSpots:
        xy = o.freq ** 2 if Options.QuadraticAxes else o.freq
        ax.pcolormesh(xy, xy, o.Modulation, cmap='gray', vmin=-3.5, vmax=1, shading='gouraud')
    ax.plot([0, e], [0, e], 'k')
    ax.plot([0, -e], [0, e], 'k')
    ax.plot([0, 0], [0, e], 'k')
    for L in np.ravel(o.LarmorFreqs):
        ax.plot(np.array([-2, 0, 2]) * L, np.array([0, 2, 0]) * L, 'k--')
    FA, FB = o.FreqsAlpha, o.FreqsBeta
    for i1 in range(nfreq):
        for i2 in range(nfreq):
            for s in (1, -1):
                ax.plot(s * FB[:, i1], FA[:, i2], '.' if nOct >= 1 else '-', color=Options.baColor, lw=2, ms=2)
                ax.plot(s * FA[:, i1], FB[:, i2], '.' if nOct >= 1 else '-', color=Options.abColor, lw=2, ms=2)
    ax.set_aspect('equal')
    ax.set_xlim(-e, e)
    ax.set_ylim(0, e)
    ax.set_xlabel('nu_1 (MHz)')
    ax.set_ylabel('nu_2 (MHz)')
    plt.show()

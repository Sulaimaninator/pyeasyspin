"""Excitation profile calculation for arbitrary pulses"""
import math
import numpy as np
from .._registry import F, nout
from .._struct import Struct, isfield
from .._mat import vec
from .._matlab import nextpow2 as _nextpow2, linspace as _linspace


@nout(2)
def exciteprofile(*varargin, nargout=None, plot=False):
    """exciteprofile(t, IQ[, offsets]) -> (offsets, Mag): offsets (MHz, 1xn) and the 3 x n
    magnetization (x, y, z) after the pulse, starting from -z (spin-1/2, no relaxation).
    plot=True draws pulse and profiles (MATLAB nargout=0)."""
    if nargout is None:
        nargout = 0 if plot else 2
    if len(varargin) == 0:
        return None
    if len(varargin) == 2:
        t, IQ = varargin
        Opt = Struct()
    elif len(varargin) == 3:
        t, IQ = varargin[:2]
        Opt = Struct(Offsets=varargin[2])
    else:
        raise ValueError('exciteprofile() needs 2 or 3 input arguments.')
    t = vec(np.asarray(t, dtype=float))
    IQv = vec(np.asarray(IQ))
    if t.size != IQv.size:
        raise ValueError('The number of points of the time axis (%d) and the number of points in the pulse '
                         'shape function IQ (%d) do not match.' % (t.size, IQv.size))
    plotResults = nargout == 0

    if not isfield(Opt, 'nOffsets'):
        Opt.nOffsets = 201
    if not isfield(Opt, 'Offsets'):
        if _nextpow2(t.size) < 10:
            zf = 2 ** 10
        else:
            zf = 4 * 2 ** _nextpow2(t.size)
        IQft = np.abs(np.fft.fftshift(np.fft.fft(IQv, zf)))
        f = np.ravel(F.fdaxis(t[1] - t[0], zf))
        indbw = np.flatnonzero(IQft > 0.5 * np.max(IQft))
        BW = abs(f[indbw[-1]] - f[indbw[0]])
        CenterFreq = np.mean([f[indbw[-1]], f[indbw[0]]])
        offsets = _linspace(-BW, BW, Opt.nOffsets) + CenterFreq
        offsets_out = offsets.reshape(1, -1)
    else:
        offsets_out = Opt.Offsets
        offsets = vec(np.asarray(Opt.Offsets, dtype=float))

    nPoints = t.size
    nOffsets = offsets.size

    Sx = np.asarray(F.sop(1 / 2, 'x'))
    Sy = np.asarray(F.sop(1 / 2, 'y'))
    Sz = np.asarray(F.sop(1 / 2, 'z'))
    Density0 = -Sz

    Isignal = np.real(IQv)
    Qsignal = np.imag(IQv)
    dt = t[1] - t[0]
    eye2 = np.eye(2)
    # all offsets at once: arrays of 2x2 matrices, shape (nOffsets, 2, 2)
    Ham0 = offsets[:, None, None] * Sz[None, :, :]

    def step(M):
        q = np.sqrt(M[:, 0, 0] ** 2 - np.abs(M[:, 0, 1]) ** 2 + 0j)
        small = np.abs(q) < 1e-10
        qs = np.where(small, 1.0, q)
        U = np.cosh(q)[:, None, None] * eye2 + (np.sinh(q) / qs)[:, None, None] * M
        if np.any(small):
            U[small] = eye2 + M[small]
        return U

    if _all_equal(IQv):
        Ham = Isignal[0] * Sx + Qsignal[0] * Sy + Ham0
        tp = (t[1] - t[0]) * (nPoints - 1)
        UPulse = step(-2j * math.pi * tp * Ham)
    else:
        UPulse = np.broadcast_to(eye2, (nOffsets, 2, 2)).astype(complex)
        for it in range(nPoints - 1):
            Ham = Isignal[it] * Sx + Qsignal[it] * Sy + Ham0
            dU = step(-2j * math.pi * dt * Ham)
            UPulse = dU @ UPulse
    Density = UPulse @ Density0 @ np.conj(np.swapaxes(UPulse, 1, 2))
    Mag = np.zeros((3, nOffsets))
    for k, S in enumerate((Sx, Sy, Sz)):
        Mag[k, :] = -2 * np.real(np.sum(S[None, :, :] * np.swapaxes(Density, 1, 2), axis=(1, 2)))

    if plotResults:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
        ax[0].plot(t, Isignal, 'b', label='I')
        ax[0].plot(t, Qsignal, 'r', label='Q')
        ax[0].set_xlabel('t (µs)')
        ax[0].set_ylabel('amplitude (MHz)')
        ax[0].legend()
        for i, (c, lab) in enumerate(zip(((0, 0.5, 0), 'r', 'b'), 'xyz')):
            ax[1].plot(offsets, Mag[i, :], color=c, label=lab, visible=(i == 2))
        ax[1].set_xlabel('frequency (MHz)')
        ax[1].set_ylabel('M_i/M_0')
        ax[1].set_ylim(-1, 1)
        plt.show()

    if nargout == 0:
        return None
    if nargout == 2:
        return offsets_out, Mag
    raise ValueError('Two output arguments are required.')


def _all_equal(z):
    """MATLAB min(IQ)==max(IQ) for a vector (true only if all elements are equal)."""
    z = vec(z)
    return bool(np.all(z == z[0])) if z.size else True

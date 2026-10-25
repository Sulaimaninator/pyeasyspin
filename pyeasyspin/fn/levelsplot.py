"""Plot energy level (Zeeman) diagram"""
import inspect
import numpy as np
from .._registry import F
from .._struct import Struct


def _c(x):
    return x[0] if isinstance(x, tuple) else x


def _unit_convert(E_MHz, toUnit):
    if toUnit == 'GHz':
        return E_MHz / 1e3
    if toUnit == 'cm^-1':
        return F.unitconvert(E_MHz, 'MHz->cm^-1')
    if toUnit == 'eV':
        return F.unitconvert(E_MHz, 'MHz->eV')
    raise ValueError("Unsupported unit '%s' in Opt.Units." % toUnit)


def _call(fn, *args, nargout):
    """Call a ported EasySpin function with nargout if it takes one (multi-output callees)."""
    try:
        takes = 'nargout' in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        takes = False
    return fn(*args, nargout=nargout) if takes else fn(*args)


def levelsplot(Sys=None, *varargin, plot=True, ax=None):
    """levelsplot(Sys, Ori='z', B=[0 1400], mwFreq=[], Opt): energy-level (Zeeman) diagram
    along a field orientation, with the resonant transitions at mwFreq (GHz) drawn as vertical
    lines coloured by relative intensity. Opt: Units ('GHz', 'cm^-1', 'eV'), nPoints,
    PlotThreshold, SlopeColor, StickSpectrum.

    Returns a Struct with the plotted data: B (field axis, in fieldUnit), E (levels, nB x
    nLevels, in energyUnit), fieldUnit, energyUnit, resonFields (mT), intensity (normalized,
    sorted by |intensity|), Transitions (1-based level pairs), transE (energies of the two
    levels of each drawn transition, energyUnit) and transColor; and axes (matplotlib axes,
    plot=True only). plot=False skips matplotlib entirely."""
    nargin = 1 + len(varargin)
    Ori = 'z'
    B = [0, 1400]
    mwFreq = np.zeros((0, 0))
    Opt = Struct()
    if nargin == 5:
        Ori, B, mwFreq, Opt = varargin
    elif nargin == 4:
        Ori, B, mwFreq = varargin
    elif nargin == 3:
        Ori, B = varargin
    elif nargin == 2:
        Ori = varargin[0]
        print('Third input argument (magnetic field range [Bmin Bmax]) missing; assuming [%d %d] mT.' % (B[0], B[1]))
    elif nargin == 1:
        print("Second input argument (orientation) missing; assuming '%s'." % Ori)
        print('Third input argument (magnetic field range [Bmin Bmax]) missing; assuming [%d %d] mT.' % (B[0], B[1]))
    else:
        raise ValueError('Too many input arguments! At most 5 (Sys,Ori,B,mwFreq,Opt) are possible.')
    if not isinstance(Opt, dict):
        raise TypeError('Fifth input argument (options) must be a structure.')
    Opt = Struct(Opt)
    Opt.setdefault('Units', 'GHz')
    Opt.setdefault('nPoints', 201)
    Opt.setdefault('PlotThreshold', 1e-6)
    Opt.AllowedColor = np.array([1.0, 0, 0])
    Opt.ForbiddenColor = np.array([1.0, 1, 1]) * 0.8
    Opt.setdefault('SlopeColor', False)
    Opt.setdefault('StickSpectrum', False)

    if isinstance(Ori, dict):
        raise TypeError("Second input argument (Ori) wrong: can't be a structure.")
    if isinstance(Ori, str):
        n = F.letter2vec(Ori)
        phi, theta = F.vec2ang(n)
        chi = 0.0
    else:
        o = np.asarray(Ori, dtype=float).ravel()
        if o.size == 2:
            phi, theta, chi = o[0], o[1], 0.0
        elif o.size == 3:
            phi, theta, chi = o
        else:
            raise ValueError("Orientation (2nd input argument) must be a string ('x','y','z','xy','xz','yz','xyz'), "
                             "a two-element array [phi theta], or a three-element array [phi theta chi].")
    phi, theta, chi = float(phi), float(theta), float(chi)
    if isinstance(B, dict):
        raise TypeError("Third input argument (B) wrong: can't be a structure.")
    B = np.asarray(B, dtype=float).ravel()
    if B.size <= 1:
        raise ValueError('Input argument that specifies B range needs at least two elements, [Bmin Bmax].')
    Bvec = np.linspace(B[0], B[1], int(Opt.nPoints)) if B.size == 2 else B
    if np.max(Bvec) >= 2000:
        Bscale, fieldUnit = 1e-3, 'T'
    else:
        Bscale, fieldUnit = 1.0, 'mT'
    mw = np.asarray(mwFreq)
    if mw.size:
        if mw.dtype.kind not in 'iuf' or mw.size != 1:
            raise ValueError('Fourth input argument (mwFreq) must be a single number.')
        mwFreq = float(mw)
    computeResonances = mw.size > 0
    if not computeResonances and Opt.StickSpectrum:
        import warnings
        warnings.warn('Cannot plot stick spectrum, since microwave frequency is missing.')

    E_MHz = np.asarray(F.levels(Sys, np.array([phi, theta, chi]), Bvec))
    E_MHz = E_MHz.reshape(Bvec.size, -1) if E_MHz.ndim != 2 else E_MHz
    E = _unit_convert(E_MHz, Opt.Units)
    nLevels = E.shape[1]
    if Opt.Units == 'GHz':
        energyUnit, Escale = 'GHz', 1.0
    elif Opt.Units == 'cm^-1':
        if np.max(np.abs(E)) < 1e-2:
            energyUnit, Escale = '10^{-4} cm^{-1}', 1e4
        else:
            energyUnit, Escale = 'cm^{-1}', 1.0
    elif Opt.Units == 'eV':
        if np.max(np.abs(E)) < 1:
            energyUnit, Escale = 'meV', 1e3
        else:
            energyUnit, Escale = 'eV', 1.0
    else:
        raise ValueError("Unsupported unit '%s' in Opt.Units." % Opt.Units)

    out = Struct(B=Bvec * Bscale, E=E * Escale, fieldUnit=fieldUnit, energyUnit=energyUnit,
                 resonFields=np.zeros(0), intensity=np.zeros(0), Transitions=np.zeros((0, 2)),
                 transE=np.zeros((0, 2)), transColor=np.zeros((0, 3)))
    if computeResonances:
        resfieldsOpt = Struct(Threshold=0, Freq2Field=0)
        Exp = Struct(mwFreq=mwFreq, Range=B[[0, -1]].reshape(1, 2))
        Exp.SampleFrame = np.array([[-chi, -theta, -phi]])
        res = _call(F.resfields, Sys, Exp, resfieldsOpt, nargout=4)
        resonFields = np.asarray(res[0], dtype=float).ravel()
        intensity = np.asarray(res[1], dtype=float).ravel()
        Transitions = np.asarray(res[3]).reshape(-1, 2) if np.size(res[3]) else np.zeros((0, 2))
        if resonFields.size:
            tpMax = np.max(np.abs(intensity))
            if tpMax > 0:
                intensity = intensity / tpMax
            absintensity = np.abs(intensity)
            idx = np.argsort(absintensity, kind='stable')
            absintensity = absintensity[idx]
            resonFields = resonFields[idx]
            Transitions = Transitions[idx, :]
            intensity = intensity[idx]
            zL = F.ang2vec(phi, theta)
            H0, muzL = F.ham(Sys, zL, nargout=2)
            H0 = H0.toarray() if hasattr(H0, 'toarray') else np.asarray(H0)
            muzL = muzL.toarray() if hasattr(muzL, 'toarray') else np.asarray(muzL)
            tE, tC, keep = [], [], []
            for iF in range(resonFields.size - 1, -1, -1):
                if absintensity[iF] < Opt.PlotThreshold:
                    continue
                H = H0 - muzL * resonFields[iF]
                Ei = np.linalg.eigvalsh((H + H.conj().T) / 2)      # sort(eig(H)), H Hermitian
                Ei = _unit_convert(Ei, Opt.Units)
                tE.append(Escale * Ei[Transitions[iF, :].astype(int) - 1])
                tC.append(absintensity[iF] * Opt.AllowedColor + (1 - absintensity[iF]) * Opt.ForbiddenColor)
                keep.append(iF)
            out.resonFields = resonFields
            out.intensity = intensity
            out.Transitions = Transitions
            out.transIdx = np.array(keep, dtype=int)
            out.transE = np.array(tE).reshape(-1, 2)
            out.transColor = np.array(tC).reshape(-1, 3)
    if plot:
        out.axes = _draw(out, Opt, computeResonances, mwFreq, Ori, phi, theta, chi, Escale, ax)
    return out


def _draw(out, Opt, computeResonances, mwFreq, Ori, phi, theta, chi, Escale, ax):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    linecolor = (0, 0.4470, 0.7410)
    fig = plt.gcf() if ax is None else ax.figure
    stick = bool(Opt.StickSpectrum) and computeResonances
    if ax is None:
        fig.clf()
        if stick:
            gs = fig.add_gridspec(4, 1)
            ax = fig.add_subplot(gs[0:3, 0])
        else:
            ax = fig.add_subplot(1, 1, 1)
    else:
        ax.cla()
    Bp, Ep = out.B, out.E
    if Opt.SlopeColor:
        for iLevel in range(Ep.shape[1] - 1, -1, -1):
            col = np.abs(F.deriv(Ep[:, iLevel] / Escale)).ravel()
            pts = np.column_stack([Bp, Ep[:, iLevel]]).reshape(-1, 1, 2)
            seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(seg, cmap='viridis_r')
            lc.set_array((col[:-1] + col[1:]) / 2)
            ax.add_collection(lc)
        ax.autoscale()
    else:
        ax.plot(Bp, Ep, color=linecolor, linewidth=0.8)
    ax.margins(x=0, y=0)
    ax.set_ylabel('energy (%s)' % out.energyUnit)
    if computeResonances:
        if out.resonFields.size:
            sc = 1e-3 if out.fieldUnit == 'T' else 1
            for k, iF in enumerate(out.transIdx):
                ax.plot(out.resonFields[iF] * np.array([1, 1]) * sc, out.transE[k], color=out.transColor[k],
                        linewidth=1)
            if stick:
                xl = ax.get_xlim()
                ax2 = fig.add_subplot(fig.axes[0].get_gridspec()[3, 0], sharex=ax)
                ax2.axhline(0, color='k', linewidth=0.5)
                sc = 1e-3 if out.fieldUnit == 'T' else 1
                for iF in range(out.resonFields.size):
                    ax2.plot([out.resonFields[iF] * sc] * 2, [0, out.intensity[iF]], color=linecolor, linewidth=2)
                ax2.set_ylim([-1.1, 1.1] if np.any(out.intensity < 0) else [0, 1.1])
                ax2.set_xlim(xl)
        else:
            ax.text(0.0, 0.0, ' no resonances in field range', color='r', va='bottom', transform=ax.transAxes)
    ax.set_xlabel('magnetic field (%s)' % out.fieldUnit)
    mwstr = ''
    if computeResonances and np.isfinite(mwFreq):
        if Opt.Units == 'GHz':
            mwstr = '  %g GHz\n' % mwFreq
        elif Opt.Units == 'cm^-1':
            mwstr = '  %0.3g cm^{-1} (%g GHz)\n' % (mwFreq * 1e9 / F.clight() / 100, mwFreq)
        elif Opt.Units == 'eV':
            ev = mwFreq * 1e9 * F.planck() / F.evolt()
            mwstr = ('  %0.3g meV (%g GHz)\n' % (ev * 1e3, mwFreq) if Escale == 1e3
                     else '  %0.3g eV (%g GHz)\n' % (ev, mwFreq))
    if isinstance(Ori, str):
        oristr0, chiGiven = "'" + Ori + "', ", False
    else:
        oristr0, chiGiven = '', np.size(Ori) == 3
    if chiGiven:
        oristr = '  %s (φ,θ,χ) = (%0.1f, %0.1f, %0.1f)°\n%s' % (oristr0, phi * 180 / np.pi, theta * 180 / np.pi,
                                                              chi * 180 / np.pi, mwstr)
    else:
        oristr = '  %s (φ,θ) = (%0.1f, %0.1f)°\n%s' % (oristr0, phi * 180 / np.pi, theta * 180 / np.pi, mwstr)
    ax.text(0.0, 1.0, oristr, va='top', transform=ax.transAxes)
    return ax

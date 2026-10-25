"""Compute resonance fields for cw EPR (perturbation theory, 1 electron spin)."""
import sys
import numpy as np
from .._registry import F, nout
from .._struct import Struct
from .._mat import a2, vec, row, isempty, numel, reshape
from ..config import config


def _c(x):
    """First output of a constant function (bmagn/nmagn return (value, sigma))."""
    return x[0] if isinstance(x, tuple) else x


def _first(x):
    return x[0] if isinstance(x, tuple) else x


def _err(err):
    if err:
        raise ValueError(err)


def _any(x):
    return x is not None and bool(np.any(np.asarray(x) != 0))


def _mif(x):
    """MATLAB `if x` for an array: true if non-empty and all elements nonzero."""
    a = np.asarray(x)
    return a.size > 0 and bool(np.all(a != 0))


def _num(x):
    return float(vec(np.asarray(x, dtype=float))[0])


def _called_from(name):
    """MATLAB `d = dbstack; numel(d)>1 && strcmp(d(2).name, name)`: is the caller `name`?"""
    f = sys._getframe(2)
    return f is not None and f.f_code.co_name == name


def _norm(v):
    return np.sqrt(np.sum(np.abs(v) ** 2))


@nout(5)
def resfields_perturb(Sys=None, Exp=None, Opt=None):
    """(B, Int, Wid, Transitions, spec): resonance fields (mT) by 1st/2nd order perturbation theory.
    B, Int, Wid are (2S*nNucSublevels) x nOrientations; with Opt.ImmediateBinning, only spec."""
    if Sys is None:
        return None
    if Exp is None:
        raise ValueError('Use two or three inputs: refields_perturb(Sys,Exp) or refields_perturb(Sys,Exp,Opt)!')
    Opt = Struct() if isempty(Opt) else Struct(Opt)
    if 'Verbosity' not in Opt:
        Opt.Verbosity = 0.0
    config.LogLevel = Opt.Verbosity

    Sys, err = F.validatespinsys(Sys)
    _err(err)
    Sys = Struct(Sys)
    Sv = vec(np.asarray(Sys.S, dtype=float))
    highSpin = bool(np.any(Sv > 0.5))
    if int(Sys.nElectrons) != 1:
        err = f'Perturbation theory available only for systems with 1 electron. Yours has {int(Sys.nElectrons)}.'
    if _any(Sys.L):
        err = 'Perturbation theory not available for elctron spin combined with orbital angular momentum!'
    if highSpin and _any(Sys.DStrain) and np.any(np.mod(Sv, 1)):
        err = ('D strain not supported for half-integer spins with perturbation theory. Use matrix '
               'diagonalization or remove Sys.DStrain.')
    if _any(Sys.DStrain) and _any(Sys.DFrame):
        err = 'D stain cannot be used with tilted D tensors.'
    if 'nn' in Sys and _any(Sys.nn):
        err = 'Perturbation theory not available for nuclear-nuclear couplings (Sys.nn).'
    _err(err)

    if 'gAStrainCorr' not in Sys:
        Sys.gAStrainCorr = 1.0
    else:
        Sys.gAStrainCorr = float(np.sign(vec(Sys.gAStrainCorr)[0]))

    if Sys.fullg:
        g = a2(Sys.g).astype(float)
    else:
        R_g2M = np.asarray(F.erot(Sys.gFrame)).T
        g = R_g2M @ np.diag(vec(Sys.g)) @ R_g2M.T
    if highSpin:
        if not Sys.fullD:
            R_D2M = np.asarray(F.erot(Sys.DFrame)).T
            D = R_D2M @ np.diag(vec(Sys.D)) @ R_D2M.T
        else:
            D = a2(Sys.D).astype(float)   # EasySpin leaves D undefined here and crashes (DEVIATIONS.md)
        D = D - np.eye(3) * np.trace(D) / 3

    S = float(Sv[0])
    n2S = int(round(2 * S))
    nTransitions = 2 * S
    I = vec(np.asarray(Sys.I, dtype=float)) if numel(Sys.I) else np.zeros(0)
    nNuclei = int(Sys.nNuclei)
    nNucStates = 2 * I + 1 if nNuclei > 0 else np.array([1.0])
    SysA = a2(Sys.A).astype(float) if nNuclei > 0 else None
    if nNuclei > 0 and not Sys.fullA:
        if np.any(vec(SysA) == 0):
            raise ValueError('All hyperfine coupling constants must be non-zero.')
    A = [None] * nNuclei
    mI = [None] * nNuclei
    idxn = [None] * nNuclei
    for iNuc in range(nNuclei - 1, -1, -1):
        if Sys.fullA:
            A[iNuc] = SysA[3 * iNuc:3 * iNuc + 3, :]
        else:
            R_A2M = np.asarray(F.erot(a2(Sys.AFrame)[iNuc, :])).T
            A[iNuc] = R_A2M @ np.diag(SysA[iNuc, :]) @ R_A2M.T
        mI[iNuc] = -I[iNuc] + np.arange(int(nNucStates[iNuc]))
        idxn[iNuc] = np.arange(1, int(nNucStates[iNuc]) + 1, dtype=float)

    DefaultExp = Struct(mwFreq=np.nan, Range=np.nan, CenterSweep=np.nan, Temperature=np.nan, mwMode='',
                        SampleFrame=np.zeros((1, 3)), CrystalSymmetry=1.0, MolFrame=np.zeros((1, 3)),
                        SampleRotation=np.zeros((0, 0)))
    Exp = Struct(F.adddefaults(Exp, DefaultExp))
    if 'CrystalOrientation' in Exp:
        raise ValueError('Exp.CrystalOrientation is no longer supported, use Exp.SampleFrame/Exp.MolFrame instead.')
    if 'Mode' in Exp:
        raise ValueError('Exp.Mode is no longer supported. Use Exp.mwMode instead.')
    if _mif(np.isnan(np.asarray(Exp.mwFreq, dtype=float))):
        raise ValueError('Experiment.mwFreq is missing!')
    if _mif(~np.isnan(np.asarray(Exp.CenterSweep, dtype=float))):
        cs = vec(np.asarray(Exp.CenterSweep, dtype=float))
        Exp.Range = np.maximum(cs[0] + np.array([[-1.0, 1.0]]) * cs[1] / 2, 0)
    if 'SearchRange' in Exp:
        Exp.Range = Exp.SearchRange
    if not np.iscomplexobj(Exp.Range) and _mif(np.isnan(np.asarray(Exp.Range, dtype=float))):
        raise ValueError('Experiment.Range/Exp.CenterSweep is missing!')
    Rg = vec(np.asarray(Exp.Range))
    if (np.iscomplexobj(Rg) or np.any(np.diff(Rg) <= 0) or np.any(~np.isfinite(Rg)) or np.any(Rg < 0)):
        raise ValueError('Exp.Range is not valid!')
    Rg = Rg.astype(float)

    xi1, xik, nB1, nk, nB0_L, mwmode = F.p_excitationgeometry(Exp.mwMode)[:6]
    nB1 = vec(np.asarray(nB1, dtype=float))
    nk = vec(np.asarray(nk, dtype=float))
    nB0_L = vec(np.asarray(nB0_L, dtype=float))
    if 'Temperature' in Exp:
        if numel(Exp.Temperature) > 1:
            err = 'Exp.Temperature must be a single number.'
        if _mif(np.isinf(np.asarray(Exp.Temperature, dtype=float))):
            err = 'If given, Exp.Temperature must have a finite value.'
    else:
        Exp.Temperature = np.nan
    _err(err)

    if 'lightBeam' not in Exp:
        Exp.lightBeam = ''
    if 'lightScatter' not in Exp:
        Exp.lightScatter = 0.0
    usePhotoSelection = not isempty(Exp.lightBeam) and Exp.lightScatter < 1
    if usePhotoSelection:
        if 'tdm' not in Sys or isempty(Sys.tdm):
            raise ValueError('To include photoselection weights, Sys.tdm must be given.')
        if isinstance(Exp.lightBeam, str):
            k = np.array([[0.0], [1.0], [0.0]])
            alphas = {'perpendicular': -np.pi / 2, 'parallel': np.pi, 'unpolarized': np.nan}
            if Exp.lightBeam not in alphas:
                raise ValueError("Unknown string in Exp.lightBeam. Use '', 'perpendicular', 'parallel' or 'unpolarized'.")
            Exp.lightBeam = [k, alphas[Exp.lightBeam]]
        elif not isinstance(Exp.lightBeam, (list, tuple)) or len(Exp.lightBeam) != 2:
            raise ValueError('Exp.lightBeam should be a 2-element cell {k alpha}.')

    Orientations, nOrientations, nSites, averageOverChi = F.p_crystalorientations(Exp, Opt)[:4]
    Orientations = a2(np.asarray(Orientations, dtype=float))
    nOrientations, nSites = int(nOrientations), int(nSites)
    if 'Sites' not in Opt:
        Opt.Sites = np.zeros((0, 0))
    if 'PerturbOrder' not in Opt:
        Opt.PerturbOrder = 2.0
    if numel(Opt.PerturbOrder) != 1 or np.iscomplexobj(Opt.PerturbOrder):
        raise ValueError('Opt.PerturbOrder must be either 1 or 2.')
    po = _num(Opt.PerturbOrder)
    if po == 1:
        secondOrder = False
    elif po == 2:
        secondOrder = True
    else:
        raise ValueError('Opt.PerturbOrder must be either 1 or 2.')
    F.logmsg(1, '2nd order perturbation theory' if secondOrder else '1st order perturbation theory')
    if 'ImmediateBinning' not in Opt:
        Opt.ImmediateBinning = 0.0

    E0 = Exp.mwFreq * 1e3
    detA = np.zeros(nNuclei)
    invA = [None] * nNuclei
    trAA = np.zeros(nNuclei)
    for iNuc in range(nNuclei - 1, -1, -1):
        A_ = A[iNuc]
        detA[iNuc] = np.linalg.det(A_)
        invA[iNuc] = np.linalg.inv(A_)
        trAA[iNuc] = np.trace(A_.T @ A_)
    if highSpin:
        trDD = np.trace(D @ D)
    II1 = I * (I + 1)

    immediateBinning = _mif(Opt.ImmediateBinning)
    nNucStatesI = [int(v) for v in nNucStates]
    maxStates = max(nNucStatesI)
    if not immediateBinning:
        if nNuclei > 0:
            idxn = np.rint(a2(F.allcombinations(*idxn))).astype(int) - 1     # 0-based
        else:
            idxn = np.zeros((1, 1), dtype=int)
        nNucTrans = idxn.shape[0]
        rows = nNucTrans
    else:
        nNucTrans = None
        rows = maxStates
        Baxis = np.linspace(Rg[0], Rg[1], int(_num(Exp.nPoints)))
        dB = Baxis[1] - Baxis[0]
        spec = np.zeros((1, int(_num(Exp.nPoints))))
        AccumWeights = vec(Exp.AccumWeights)

    if _mif(~np.isnan(np.asarray(Exp.Temperature, dtype=float))):
        Populations = np.exp(-F.planck() * (2 * S - np.arange(n2S + 1)).reshape(-1, 1) * Exp.mwFreq * 1e9
                             / F.boltzm() / _num(Exp.Temperature))
        Populations = Populations / np.sum(Populations)
        Polarization = np.diff(Populations, axis=0)[::-1, 0]
    else:
        Polarization = np.ones(n2S)

    gg = g @ g.T
    trgg = np.trace(gg)
    mSv = S - np.arange(n2S)                                       # S:-1:-S+1
    c = _c(F.bmagn()) / 2 * np.sqrt(S * (S + 1) - mSv * (mSv - 1))
    c = c / F.planck() / 1e9
    c2 = c ** 2

    vecs = np.zeros((3, nOrientations))
    geff = np.zeros(nOrientations)
    TransitionRate = np.zeros((n2S, nOrientations))
    Intensity = np.zeros((n2S, nOrientations))
    if not immediateBinning:
        Bfinal = [np.zeros((nOrientations, nNucTrans)) for _ in range(n2S)]
    E1A = np.zeros((rows, nNuclei))
    E2A = np.zeros((rows, nNuclei))
    E2DA = np.zeros((rows, nNuclei)) if highSpin else 0.0
    kvec = np.zeros((3, nNuclei))
    for iOri in range(nOrientations - 1, -1, -1):
        R_L2M = np.asarray(F.erot(Orientations[iOri, :])).T
        n0 = R_L2M @ nB0_L
        vecs[:, iOri] = n0
        geff[iOri] = _norm(g.T @ n0)
        u = g.T @ n0 / geff[iOri]
        preOri = 1e6 * F.planck() / (geff[iOri] * _c(F.bmagn()))
        if usePhotoSelection:
            k = Exp.lightBeam[0]
            alpha = Exp.lightBeam[1]
            ori = Orientations[iOri, 0:2] if averageOverChi else Orientations[iOri, 0:3]
            photoWeight = F.photoselect(Sys.tdm, ori.reshape(1, -1), k, alpha)
            photoWeight = (1 - Exp.lightScatter) * photoWeight + Exp.lightScatter
        else:
            photoWeight = 1
        TR = None
        if averageOverChi:
            if mwmode.linearpolarizedMode:
                TR = c2 / 2 * (1 - xi1 ** 2) * (trgg - _norm(g @ u) ** 2)
            elif mwmode.unpolarizedMode:
                TR = c2 / 4 * (1 + xik ** 2) * (trgg - _norm(g @ u) ** 2)
            elif mwmode.circpolarizedMode:
                TR = (c2 / 2 * (1 + xik ** 2) * (trgg - _norm(g @ u) ** 2)
                      + mwmode.circSense * 2 * c2 * xik ** 2 * np.linalg.det(g) / _norm(g.T @ n0))
        else:
            if mwmode.linearpolarizedMode:
                nB1_ = R_L2M @ nB1
                TR = c2 * _norm(np.cross(g.T @ nB1_, u)) ** 2
            elif mwmode.unpolarizedMode:
                nk_ = R_L2M @ nk
                TR = c2 / 2 * (trgg - _norm(g @ u) ** 2 - _norm(np.cross(g.T @ nk_, u)) ** 2)
            elif mwmode.circpolarizedMode:
                nk_ = R_L2M @ nk
                TR = (c2 * (trgg - _norm(g @ u) ** 2 - _norm(np.cross(g.T @ nk_, u)) ** 2)
                      + mwmode.circSense * 2 * c2 * np.linalg.det(g) * xik / _norm(g.T @ n0))
        if TR is None:
            raise ValueError('Index exceeds the number of array elements.')   # TransitionRate(:,iOri) unset
        TransitionRate[:, iOri] = TR
        dBdE = (F.planck() / _c(F.bmagn()) * 1e9) / geff[iOri]
        Intensity[:, iOri] = Polarization * TransitionRate[:, iOri] * dBdE * photoWeight
        if highSpin:
            Du = D @ u
            uDu = u @ Du
            uDDu = Du @ Du
            D1sq = uDDu - uDu ** 2
            D2sq = 2 * trDD + uDu ** 2 - 4 * uDDu
        for imS in range(n2S):
            mS = mSv[imS]
            E1D = -uDu / 2 * (3 - 6 * mS) if highSpin else 0.0
            if nNuclei > 0:
                for iNuc in range(nNuclei):
                    K = A[iNuc] @ u
                    nK = _norm(K)
                    kvec[:, iNuc] = K / nK
                    E1A_ = mI[iNuc] * nK
                    if immediateBinning:
                        E1A[:nNucStatesI[iNuc], iNuc] = E1A_
                    else:
                        E1A[:, iNuc] = E1A_[idxn[:, iNuc]]
                E1Ax = E1A
            else:
                E1Ax = 0.0
            E2D = 0.0
            if secondOrder:
                if highSpin:
                    x = (D1sq * (4 * S * (S + 1) - 3 * (8 * mS ** 2 - 8 * mS + 3))
                         - D2sq / 4 * (2 * S * (S + 1) - 3 * (2 * mS ** 2 - 2 * mS + 1)))
                    E2D = -x / (2 * E0)
                if nNuclei > 0:
                    for n in range(nNuclei):
                        k_ = kvec[:, n]
                        Ak = A[n].T @ k_
                        kAu = Ak @ u
                        kAAk = _norm(Ak) ** 2
                        A1sq = kAAk - kAu ** 2
                        A2 = detA[n] * (u @ invA[n] @ k_)
                        A3 = trAA[n] - _norm(A[n] @ u) ** 2 - kAAk + kAu ** 2
                        x = A1sq * mI[n] ** 2 - A2 * (1 - 2 * mS) * mI[n] + A3 / 2 * (II1[n] - mI[n] ** 2)
                        E2A_ = +x / (2 * E0)
                        if immediateBinning:
                            E2A[:nNucStatesI[n], n] = E2A_
                        else:
                            E2A[:, n] = E2A_[idxn[:, n]]
                        if highSpin:
                            DA = Du @ Ak - uDu * kAu
                            y = DA * (3 - 6 * mS) * mI[n]
                            E2DA_ = -y / E0
                            if immediateBinning:
                                E2DA[:nNucStatesI[n], n] = E2DA_
                            else:
                                E2DA[:, n] = E2DA_[idxn[:, n]]
                    E2Ax, E2DAx = E2A, E2DA
                else:
                    E2Ax, E2DAx = 0.0, 0.0
            else:
                E2Ax, E2DAx = 0.0, 0.0
            if immediateBinning:
                B0 = (E0 - E1D - E2D) * preOri * 1e3
                Bshifts = (-(E1Ax + E2Ax + E2DAx)) * preOri * 1e3
                spec = spec + Intensity[imS, iOri] * AccumWeights[iOri] * \
                    F.multinucstick(B0, nNucStates, Bshifts, Baxis[0], dB, Exp.nPoints)
            else:
                if secondOrder:
                    Bfinal[imS][iOri, :] = (E0 - E1D - E2D - np.sum(np.atleast_2d(E1Ax + E2Ax + E2DAx), axis=1)) * preOri
                else:
                    Bfinal[imS][iOri, :] = (E0 - E1D - np.sum(np.atleast_2d(E1Ax), axis=1)) * preOri

    if immediateBinning:
        B = np.zeros((0, 0))
        Int = np.zeros((0, 0))
        Wid = np.zeros((0, 0))
        Transitions = np.zeros((0, 0))
        spec = spec / dB / np.prod(nNucStates)
        spec = spec * (2 * np.pi)
    else:
        B = np.hstack(Bfinal).T
        B = B * 1e3
        nNucSublevels = int(np.prod(nNucStates))
        Int = np.repeat(Intensity, nNucSublevels, axis=0) / nNucSublevels
        Int = np.flipud(Int)
        HS = row(np.asarray(Sys.HStrain, dtype=float))
        if np.any(HS):
            lw2 = np.sum(HS ** 2 @ vecs ** 2, axis=0)
            lw = np.sqrt(lw2) * 1e6 * F.planck() / geff / _c(F.bmagn()) * 1e3
            Wid = np.tile(lw, (nNucTrans * n2S, 1))
        else:
            Wid = 0.0
        gS = a2(Sys.gStrain)
        AS = vec(np.asarray(Sys.AStrain, dtype=float))
        if np.any(gS) or np.any(AS):
            if np.any(a2(Sys.gFrame)):
                raise ValueError('g strain cannot be used with tilted g tensors.')
            # EasySpin raises this also without A strain (gStrain + tilted A); only A strain needs it (DEVIATIONS.md)
            if nNuclei > 0 and np.any(a2(Sys.AFrame)[0, :]) and np.any(AS):
                raise ValueError('A strain cannot be used with a tilted A tensor.')
            if np.any(gS):
                gStrainMatrix = np.diag(gS[0, :] / a2(Sys.g)[0, :]) * Exp.mwFreq * 1e3
                if np.any(a2(Sys.gFrame)):
                    R_g2M = np.asarray(F.erot(a2(Sys.gFrame)[0, :])).T
                    gStrainMatrix = R_g2M @ gStrainMatrix @ R_g2M.T
            else:
                gStrainMatrix = np.zeros((3, 3))
            if np.any(AS) and nNuclei > 0:
                AStrainMatrix = np.diag(AS)
                if 'AFrame' in Sys:
                    Rp = np.asarray(F.erot(a2(Sys.AFrame)[0, :])).T
                    AStrainMatrix = Rp @ AStrainMatrix @ Rp.T
                corr = Sys.gAStrainCorr
                mI1 = -I[0] + np.arange(int(2 * I[0]) + 1)
                lw2 = np.zeros((mI1.size, nOrientations))
                for idx in range(mI1.size):
                    StrainMatrix = gStrainMatrix + corr * (mI1[idx]) * AStrainMatrix
                    for iOri in range(nOrientations):
                        lw2[idx, iOri] = vecs[:, iOri] @ StrainMatrix ** 2 @ vecs[:, iOri]
                Wid_gA = np.sqrt(lw2) * F.planck() * 1e6 / np.tile(geff, (mI1.size, 1)) / _c(F.bmagn()) * 1e3
                idx = np.repeat(np.arange(mI1.size), int(round(2 * S * nNucTrans / mI1.size)))
                Wid = np.sqrt(Wid_gA[idx, :] ** 2 + np.asarray(Wid) ** 2)
            else:
                StrainMatrix = gStrainMatrix
                lw2 = np.zeros((1, nOrientations))
                for iOri in range(nOrientations):
                    lw2[0, iOri] = vecs[:, iOri] @ StrainMatrix ** 2 @ vecs[:, iOri]
                Wid_gA = np.sqrt(lw2) * F.planck() * 1e6 / geff / _c(F.bmagn()) * 1e3
                Wid = np.sqrt(np.tile(Wid_gA ** 2, (n2S * nNucTrans, 1)) + np.asarray(Wid) ** 2)
        elif _any(Sys.DStrain):
            if _any(Sys.DFrame):
                raise ValueError('Cannot use D/E strain with tilted D tensor.')
            x, y, z = vecs[0, :], vecs[1, :], vecs[2, :]
            mSc = (S - np.arange(n2S + 1)).reshape(-1, 1)            # (S:-1:-S).'
            mSS = mSc ** 2 - S * (S + 1) / 3
            dHdD_ = mSS * ((3 * z ** 2 - 1) / 2).reshape(1, -1)
            dHdE_ = mSS * (3 / 2 * (x ** 2 - y ** 2)).reshape(1, -1)
            DS = vec(a2(Sys.DStrain))
            DeltaD, DeltaE = DS[0], DS[1]
            rDE = _num(Sys.DStrainCorr)
            if rDE != 0:
                R12 = rDE * DeltaD * DeltaE
                CovMatrix = np.array([[DeltaD ** 2, R12], [R12, DeltaE ** 2]])
                L, V = np.linalg.eigh(CovMatrix)
                L = np.sqrt(L)
                dHdD0_ = dHdD_
                dHdD_ = L[0] * (V[0, 0] * dHdD0_ + V[0, 1] * dHdE_)
                # EasySpin uses the already-updated dHdD_ here; getdstrainops uses the original (DEVIATIONS.md)
                dHdE_ = L[1] * (V[1, 0] * dHdD0_ + V[1, 1] * dHdE_)
            else:
                dHdD_ = dHdD_ * DeltaD
                dHdE_ = dHdE_ * DeltaE
            lwD = np.diff(dHdD_, axis=0)
            lwE = np.diff(dHdE_, axis=0)
            MHz2mT = (F.planck() / _c(F.bmagn()) * 1e9) / geff
            lwD = lwD * MHz2mT
            lwE = lwE * MHz2mT
            # rows of B are mS-major blocks of nNucTrans sub-lines: repelem, not EasySpin's repmat (DEVIATIONS.md)
            Wid2_DE = np.repeat(lwD ** 2 + lwE ** 2, nNucTrans, axis=0)
            Wid = np.sqrt(Wid2_DE + np.asarray(Wid) ** 2)
        elif np.size(Wid) > 1 or _mif(np.asarray(Wid) > 0):
            # EasySpin's `elseif Wid>0` drops all H-strain widths if any one is 0 (DEVIATIONS.md)
            pass
        else:
            Wid = np.zeros((0, 0))
        lowerLevels = np.arange(1, nNucSublevels + 1, dtype=float).reshape(-1, 1)
        T = []
        for _ in range(n2S):
            upperLevels = lowerLevels + nNucSublevels
            T.append(np.hstack([lowerLevels, upperLevels]))
            lowerLevels = upperLevels
        Transitions = np.vstack(T)
        spec = 0.0

    pepperCall = _called_from('pepper')
    if nSites > 1 and not pepperCall:
        siz = (int(round(nTransitions * nSites)), int(round(np.size(B) / nTransitions / nSites)))
        B = reshape(B, siz)
        if not isempty(Int):
            Int = reshape(Int, siz)
        if not isempty(Wid):
            Wid = reshape(Wid, siz)
    return B, Int, Wid, Transitions, spec

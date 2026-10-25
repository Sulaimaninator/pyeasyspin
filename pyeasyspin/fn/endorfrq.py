"""Compute ENDOR frequencies and intensities (matrix diagonalization)."""
import sys
import numpy as np
import scipy.sparse as sps
from .._registry import F, nout
from .._struct import Struct
from .._mat import a2, vec, isempty, numel, reshape
from ..config import config


def _c(x):
    return x[0] if isinstance(x, tuple) else x


def _err(err):
    if err:
        raise ValueError(err)


def _num(x):
    return float(vec(np.asarray(x, dtype=float))[0])


def _d(x):
    return np.asarray(x.toarray() if sps.issparse(x) else x)


def _called_from(name):
    """MATLAB `d = dbstack; numel(d)>1 && strcmp(d(2).name, name)`."""
    f = sys._getframe(2)
    return f is not None and f.f_code.co_name == name


def _sandwich(V, X):
    """abs(V'*X*V).^2"""
    return np.abs(V.conj().T @ X @ V) ** 2


@nout(4)
def endorfrq(Sys=None, Exp=None, Opt=None, nargout=4):
    """(Pdat, Idat, Transitions, Info): ENDOR transition frequencies (MHz, nTransitions x
    nOrientations), intensities (computed only for nargout > 1), level pairs (1-based), Info."""
    if Sys is None:
        return None
    if Exp is None:
        raise ValueError('Wrong number of input arguments!')
    if nargout < 0:
        raise ValueError('Not enough output arguments.')
    if nargout > 4:
        raise ValueError('Too many output arguments.')
    if isempty(Opt):
        Opt = Struct()
    if not isinstance(Sys, dict):
        raise ValueError('First input argument (Sys) must be a structure!')
    if not isinstance(Exp, dict):
        raise ValueError('Second input argument (Exp) must be a structure!')
    if not isinstance(Opt, dict):
        raise ValueError('Third input argument (Opt) must be a structure!')
    Opt = Struct(Opt)
    if 'Verbosity' not in Opt:
        Opt.Verbosity = 0.0
    config.LogLevel = Opt.Verbosity
    EasySpinLogLevel = Opt.Verbosity
    Info = Struct()
    Sys, err = F.validatespinsys(Sys)
    _err(err)
    if int(Sys.nNuclei) == 0:
        raise ValueError("The system doesn't contain any nuclei.")
    DefaultSys = Struct(lw=0.0, HStrain=np.array([[0.0, 0.0, 0.0]]))
    Sys = Struct(F.adddefaults(Sys, DefaultSys))
    DefaultExp = Struct(mwFreq=np.nan, Temperature=np.nan, ExciteWidth=np.inf, Field=np.nan,
                        SampleFrame=np.zeros((1, 3)), CrystalSymmetry=1.0, MolFrame=np.zeros((1, 3)),
                        SampleRotation=np.zeros((0, 0)), Range=np.zeros((0, 0)))
    if 'Field' not in Exp:
        raise ValueError('Exp.Field is missing.')
    Exp = Struct(F.adddefaults(Exp, DefaultExp))
    mwFreq = Exp.mwFreq * 1e3
    computeNonEquiPops = 'initState' in Sys and not isempty(Sys.initState)
    if computeNonEquiPops:
        computeBoltzmann = False
    else:
        if np.any(np.isinf(np.asarray(Exp.Temperature, dtype=float))):
            raise ValueError('If given, Params.Temperature must be a finite value.')
        computeBoltzmann = not np.all(np.isnan(np.asarray(Exp.Temperature, dtype=float)))
    if 'ExciteWidth' in Exp and 'mwFreq' not in Exp:
        raise ValueError('Par.mwFreq is missing.')
    if 'Sites' not in Opt:
        Opt.Sites = np.zeros((0, 0))
    Orientations, nOrientations, nSites, AverageOverChi = F.p_crystalorientations(Exp, Opt)[:4]
    Orientations = a2(np.asarray(Orientations, dtype=float))
    nOrientations, nSites = int(nOrientations), int(nSites)
    for name in ('nTransitions',):
        if name in Opt:
            raise ValueError(f'Options.{name} is obsolete. Please remove from code!')
    DefaultOpt = Struct(Transitions=np.zeros((0, 0)), Threshold=1e-3, Nuclei=np.zeros((0, 0)), Intensity='on',
                        Enhancement='off', OriWeights=np.zeros((0, 0)), OriThreshold=1e-4, TPSGridSize=4.0,
                        TPSGridSymm='D2h', SelectionThreshold=0.01)
    Opt = Struct(F.adddefaults(Opt, DefaultOpt))
    nNuclei = int(Sys.nNuclei)
    if isempty(Opt.Nuclei):
        Opt.Nuclei = np.arange(1, nNuclei + 1, dtype=float).reshape(1, -1)
    Nuclei = vec(np.asarray(Opt.Nuclei, dtype=float))
    if np.any(Nuclei < 1) or np.any(Nuclei > nNuclei):
        raise ValueError('Opt.Nuclei is out of range.')
    IntensitySwitch = _c(F.parseoption(Opt, 'Intensity', ['off', 'on'])) - 1
    EnhancementSwitch = _c(F.parseoption(Opt, 'Enhancement', ['off', 'on'])) - 1
    UseOriWeights = not isempty(Opt.OriWeights) and Opt.OriThreshold > 0
    OriWeights = vec(np.asarray(Opt.OriWeights, dtype=float)) if UseOriWeights else None
    if UseOriWeights and OriWeights.size != nOrientations:
        raise ValueError(f'Opt.OriWeights has wrong number of elements ({OriWeights.size}).\n It must have '
                         f'one entry for each of the {nOrientations} orientations!')
    H0, muxM, muyM, muzM = (_d(x) for x in F.ham(Sys, nargout=4))

    nStates = int(Sys.nStates)
    if computeNonEquiPops:
        initState = a2(Sys.initState[0])
        initStateBasis = Sys.initState[1]
        nElectronStates = int(np.prod(2 * vec(Sys.S) + 1))
        nNucStates = int(np.prod(vec(Sys.I) * 2 + 1))
        nSt = nElectronStates * nNucStates
        sz1, sz2 = initState.shape
        if sz1 == sz2:
            if nElectronStates != sz1 and nSt != sz1:
                raise ValueError(f'The density matrix in Sys.initState must have dimensions of nxn with n = '
                                 f'{nElectronStates} or {nSt}.')
            if initState.size == nElectronStates ** 2 and initState.size != nSt ** 2:
                initState = np.kron(initState, np.eye(nSt // nElectronStates))
            initState = initState / np.trace(initState)
        else:
            if initState.size != nElectronStates and initState.size != nSt:
                raise ValueError(f'The population vector in Sys.initState must have {nElectronStates} or {nSt} elements.')
            initState = vec(initState).reshape(-1, 1)
            if initState.size == nElectronStates and initState.size != nSt:
                initState = np.kron(initState, np.ones((nSt // nElectronStates, 1)))
            initState = initState / np.sum(initState)
            if initStateBasis == 'eigen':
                initState = np.diag(vec(initState))
        if initStateBasis == 'zerofield':
            ZFE, ZFStates = np.linalg.eigh(H0)
            idx = np.argsort(np.real(ZFE), kind='stable')
            ZFEnergies = np.real(ZFE)[idx]
            ZFStates = ZFStates[:, idx]
            if np.unique(ZFEnergies).size != ZFEnergies.size:
                raise ValueError('Degenerate energy levels detected at zero-field. This prevents unambiguous '
                                 'assignment of the provided sublevel populations to the zero-field states. '
                                 'Please provide the non-equilibriumstate using the full density matrix. See '
                                 'documentation for details.')
            if min(initState.shape) == 1:
                initState = ZFStates @ np.diag(vec(initState)) @ ZFStates.conj().T
            else:
                initState = ZFStates @ initState @ ZFStates.conj().T

    ComputeIntensities = nargout > 1 and bool(IntensitySwitch)
    ComputeVectors = ComputeIntensities
    ExciteWidth = float(np.asarray(Exp.ExciteWidth, dtype=float).ravel()[0])
    OrientationSelection = bool(np.isfinite(ExciteWidth) and ExciteWidth > 0)
    if OrientationSelection:
        F.logmsg(2, '  including excitation width %g MHz', ExciteWidth)
    Range = vec(np.asarray(Exp.Range, dtype=float)) if not isempty(Exp.Range) else np.zeros(0)

    if isempty(Opt.Transitions):
        F.logmsg(1, '  automatic transition selection')
        F.logmsg(2, '    (threshold %g, grid size %d, grid symmetry %s)', Opt.Threshold, Opt.TPSGridSize,
                 Opt.TPSGridSymm)
        grid = F.sphgrid(Opt.TPSGridSymm, Opt.TPSGridSize)
        phi = vec(grid.phi)
        theta = vec(grid.theta)
        TRWeights = vec(grid.weights)
        st, sp_, ct, cp = np.sin(theta), np.sin(phi), np.cos(theta), np.cos(phi)
        smuxM, smuyM, smuzM = (_d(x) for x in F.ham_nz(Sys, Opt.Nuclei, nargout=3))
        TransitionRates = np.zeros((nStates, nStates))
        maxE = np.zeros((nStates, nStates))
        minE = np.full((nStates, nStates), np.inf)
        for iOri in range(theta.size):
            E, Vs = np.linalg.eigh(H0 - Exp.Field * (st[iOri] * (cp[iOri] * muxM + sp_[iOri] * muyM) + ct[iOri] * muzM))
            smupM = cp[iOri] * smuxM + sp_[iOri] * smuyM
            TransitionRates = TransitionRates + TRWeights[iOri] * \
                (_sandwich(Vs, ct[iOri] * smupM - st[iOri] * smuzM) +
                 _sandwich(Vs, -sp_[iOri] * smuxM + cp[iOri] * smuyM)) / 2
            EE = np.abs(E.reshape(-1, 1) - E.reshape(1, -1))
            idx = EE > maxE
            maxE[idx] = EE[idx]
            idx = EE < minE
            minE[idx] = EE[idx]
        if Range.size and not np.any(np.isnan(Range)):
            TransitionRates[(maxE < Range[0]) | (minE > Range[1])] = 0
        nSStates = int(np.prod(2 * vec(Sys.S) + 1))
        idxElectronicTransitions = ~np.kron(np.eye(nSStates), np.ones((nStates // nSStates, nStates // nSStates))).astype(bool)
        idxLowerTriangle = np.tril(np.ones((nStates, nStates))).astype(bool)
        keepidx = ~(idxLowerTriangle | idxElectronicTransitions)
        kv = vec(keepidx)
        TR = vec(TransitionRates)[kv]
        lin = np.nonzero(kv)[0]
        Transitions = np.column_stack([lin % nStates + 1, lin // nStates + 1]).astype(float)
        nTransitions = int(np.sum(TR > Opt.Threshold * np.max(TR))) if TR.size else 0
        # MATLAB sort(-TR) (stable). Rates that are equal in exact arithmetic are ordered by eigensolver
        # rounding, so their order can differ from MATLAB's.
        idx = np.argsort(-TR, kind='stable')
        Transitions = Transitions[idx[:nTransitions], :]
    else:
        F.logmsg(1, '  using %d user-specified transitions', a2(Opt.Transitions).shape[0])
        Transitions = np.sort(a2(np.asarray(Opt.Transitions, dtype=float)), axis=1)
    u = Transitions[:, 0].astype(int) - 1
    v = Transitions[:, 1].astype(int) - 1
    nTransitions = u.size
    TRidx = (u, v)
    F.logmsg(1, '  %d transitions selected', nTransitions)
    if nTransitions == 0:
        import warnings
        warnings.warn('No transitions found. Decrease Opt.Threshold.')
    msg = '  computing peak positions'
    if ComputeIntensities:
        msg = msg + ', intensities'
    F.logmsg(1, msg)
    Pdat = np.full((nTransitions, nOrientations), np.nan)
    Idat = np.zeros((0, 0))
    if ComputeIntensities:
        Idat = np.zeros((nTransitions, nOrientations))
        DxM, DyM, DzM = (_d(x) for x in F.ham_nz(Sys, Opt.Nuclei, nargout=3))
        if EnhancementSwitch:
            DxMe, DyMe, DzMe = (_d(x) for x in F.ham_ez(Sys, nargout=3))
            DxM = DxM + DxMe
            DyM = DyM + DyMe
            DzM = DzM + DzMe
        if OrientationSelection:
            ExM, EyM, EzM = (_d(x) for x in F.ham_ez(Sys, 1.0, nargout=3))
            lwExcite2 = ExciteWidth ** 2
        if computeBoltzmann:
            with np.errstate(divide='ignore'):
                BoltzmannPreFactor = np.float64(-1e6 * F.planck() / F.boltzm()) / np.float64(_num(Exp.Temperature))
    if OrientationSelection:
        maxEPRfreq = -np.inf
        minEPRfreq = np.inf
        minExWidth = np.inf
    if UseOriWeights:
        F.logmsg(1, '  user-supplied orientation pre-selection: skipping %d of %d orientations',
                 int(np.sum(OriWeights < Opt.OriThreshold)), nOrientations)
    HS = vec(np.asarray(Sys.HStrain, dtype=float))
    if 'rf' not in Exp:
        Exp.rf = 0.0
    for iOri in range(nOrientations):
        if UseOriWeights and OriWeights[iOri] < Opt.OriThreshold:
            continue
        xLab, yLab, zLab = (vec(x) for x in F.erot(Orientations[iOri, :], 'rows', nargout=3))
        H = H0 - Exp.Field * (zLab[0] * muxM + zLab[1] * muyM + zLab[2] * muzM)
        if ComputeVectors:
            E0, Vs = np.linalg.eigh(H)
            E0 = np.sort(E0)
        else:
            E0 = np.sort(np.linalg.eigvalsh(H))
        Pdat[:, iOri] = E0[v] - E0[u]
        if ComputeIntensities:
            DyL = yLab[0] * DxM + yLab[1] * DyM + yLab[2] * DzM
            if AverageOverChi:
                DxL = xLab[0] * DxM + xLab[1] * DyM + xLab[2] * DzM
                EndorIntensity = (_sandwich(Vs, DxL) + _sandwich(Vs, DyL)) / 2
                if OrientationSelection:
                    ExL = xLab[0] * ExM + xLab[1] * EyM + xLab[2] * EzM
                    EyL = yLab[0] * ExM + yLab[1] * EyM + yLab[2] * EzM
                    EPRIntensity = (2 * np.pi) * (_sandwich(Vs, ExL) + _sandwich(Vs, EyL)) / 2
            else:
                EndorIntensity = _sandwich(Vs, DyL)
                if OrientationSelection:
                    ExL = xLab[0] * ExM + xLab[1] * EyM + xLab[2] * EzM
                    EPRIntensity = _sandwich(Vs, ExL)
            pulseENDOR = Exp.rf > 0
            if pulseENDOR:
                EndorIntensity = (1 - np.cos(np.sqrt(EndorIntensity) * Exp.rf)) / 2
            if OrientationSelection:
                lw2 = np.sum((HS * zLab) ** 2) + lwExcite2
                dE = np.tile(E0.reshape(-1, 1), (1, nStates))
                dE = dE.T - dE
                Populations = None
                if computeBoltzmann:
                    Populations = np.exp(BoltzmannPreFactor * (E0 - E0[0]))
                elif computeNonEquiPops:
                    if initStateBasis == 'eigen':
                        Populations = np.real_if_close(np.array([initState[i, i] for i in range(nStates)]))
                    else:
                        Populations = np.array([Vs[:, i].conj() @ initState @ Vs[:, i] for i in range(nStates)])
                if Populations is None:
                    Polarizations = np.ones(dE.shape)
                else:
                    P = np.tile(Populations.reshape(-1, 1), (1, nStates))
                    Polarizations = np.abs(P - P.T)
                EPRIntensity = np.sqrt(2 / np.pi) * Polarizations * EPRIntensity * \
                    np.exp(-2 / lw2 * (mwFreq - np.abs(dE)) ** 2)
                SumTP = np.sum(EPRIntensity, axis=0) - np.diag(EPRIntensity)
                EPRIntensity = SumTP[u] + SumTP[v] - 2 * EPRIntensity[TRidx]
                ExcitationFactor = EPRIntensity
                dEk = vec(dE)
                dEk = dEk[~(dEk < 0.3 * mwFreq)]
                minFreq = np.min(dEk) if dEk.size else np.zeros(0)
                maxFreq = np.max(dEk) if dEk.size else np.zeros(0)
                if np.size(minFreq) and minFreq < minEPRfreq:
                    minEPRfreq = minFreq
                if np.size(maxFreq) and maxFreq > maxEPRfreq:
                    maxEPRfreq = maxFreq
                if lw2 < minExWidth ** 2:
                    minExWidth = np.sqrt(lw2)
            else:
                ExcitationFactor = 1.0
            if computeBoltzmann:
                Populations = np.exp(BoltzmannPreFactor * (E0 - E0[0]))
                NuclearPolarization = (Populations[u] - Populations[v]) / np.sum(Populations)
            else:
                NuclearPolarization = 1.0
            Idat[:, iOri] = NuclearPolarization * ExcitationFactor * EndorIntensity[TRidx]
    if OrientationSelection:
        Selectivity = (maxEPRfreq - minEPRfreq) / minExWidth
        F.logmsg(2, '  limited excitation width, selectivity %g', Selectivity)
    else:
        Selectivity = 0.0
        F.logmsg(2, '  infinite excitation width')
    Info.Selectivity = Selectivity
    saltCall = _called_from('salt')
    if nSites > 1 and not saltCall:
        siz = (nTransitions * nSites, int(round(Pdat.size / nTransitions / nSites)))
        Pdat = reshape(Pdat, siz)
        if not isempty(Idat):
            Idat = reshape(Idat, siz)
    if nargout == 0:
        return Pdat
    return Pdat, Idat, Transitions, Info

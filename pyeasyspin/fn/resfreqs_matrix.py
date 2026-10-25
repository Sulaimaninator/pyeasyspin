"""Compute resonance frequencies for frequency-swept EPR (matrix diagonalization)."""
import sys
import numpy as np
import scipy.sparse as sps
from .._registry import F, nout
from .._struct import Struct
from .._mat import a2, vec, isempty, numel, reshape, isnumeric
from ..config import config

_rng = np.random.default_rng()


def _c(x):
    return x[0] if isinstance(x, tuple) else x


def _err(err):
    if err:
        raise ValueError(err)


def _any(x):
    return x is not None and bool(np.any(np.asarray(x.toarray() if sps.issparse(x) else x) != 0))


def _mif(x):
    a = np.asarray(x)
    return a.size > 0 and bool(np.all(a != 0))


def _num(x):
    return float(vec(np.asarray(x, dtype=float))[0])


def _d(x):
    return x.toarray() if sps.issparse(x) else np.asarray(x)


def _called_from(name):
    """MATLAB `d = dbstack; numel(d)>1 && strcmp(d(2).name, name)`."""
    f = sys._getframe(2)
    return f is not None and f.f_code.co_name == name


def _expect(Op, x):
    """x'*Op*x for a state vector x; Op may be a scalar (e.g. 0 from getdstrainops)."""
    if np.isscalar(Op) or np.ndim(Op) == 0:
        return complex(Op) * np.vdot(x, x)
    return np.vdot(x, Op @ x)


def _norm(v):
    return np.sqrt(np.sum(np.abs(v) ** 2))


@nout(4)
def resfreqs_matrix(Sys=None, Exp=None, Opt=None, nargout=4):
    """(Pdat, Idat, Wdat, Transitions): resonance frequencies (MHz) at Exp.Field, intensities
    (nargout > 1), widths (nargout > 2, if strains are present) and 1-based level pairs."""
    if Sys is None:
        return None
    if Exp is None or isempty(Exp):
        Exp = Struct()
    if isempty(Opt):
        Opt = Struct()
    if nargout < 0:
        raise ValueError('Not enough output arguments.')
    if nargout > 4:
        raise ValueError('Too many output arguments.')
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

    Sys, err = F.validatespinsys(Sys)
    _err(err)
    DefaultSystem = Struct(lw=0.0, A=0.0, HStrain=np.array([[0.0, 0.0, 0.0]]), gStrain=np.array([[0.0, 0.0, 0.0]]),
                           AStrain=np.array([[0.0, 0.0, 0.0]]), DStrain=0.0, gAStrainCorr=1.0)
    Sys = Struct(F.adddefaults(Sys, DefaultSystem))
    gc = Sys.gAStrainCorr
    if numel(gc) != 1 or not isnumeric(gc) or _num(gc) == 0 or not np.isfinite(_num(gc)):
        raise ValueError('Sys.gAStrainCorr must be a single number, either +1 or -1.')
    Sys.gAStrainCorr = float(np.sign(_num(gc)))
    nElectrons = int(Sys.nElectrons)
    nNuclei = int(Sys.nNuclei)
    if nElectrons > 1:
        if _any(Sys.gStrain) or _any(Sys.AStrain):
            raise ValueError('Cannot use D or g/A strain in spin system with more than one electron spin.')
    if _any(Sys.gStrain) or _any(Sys.AStrain):
        gFull = a2(Sys.g).shape[0] == 3 * numel(Sys.S)
        if gFull:
            raise ValueError('gStrain and AStrain are not allowed when full g matrices are given!')
        if _any(Sys.gFrame):
            raise ValueError('g strain cannot be used with tilted g tensors.')
        # EasySpin raises this also without A strain (gStrain + tilted A); only A strain needs it (DEVIATIONS.md)
        if nNuclei > 0 and _any(a2(Sys.AFrame)[0, :]) and _any(Sys.AStrain):
            raise ValueError('A strain cannot be used with a tilted A tensor.')
        if _any(Sys.DStrain):
            raise ValueError('D strain and g/A strain cannot be used at the same time.')
    if _any(Sys.DStrain) and _any(Sys.DFrame):
        raise ValueError('D stain cannot be used with tilted D tensors.')
    higherOrder = any(f[:2] == 'ZB' for f in Sys)

    DefaultExp = Struct(Range=np.nan, Field=np.nan, CenterSweep=np.nan, Temperature=np.nan, mwMode='perpendicular',
                        SampleFrame=np.zeros((1, 3)), CrystalSymmetry=1.0, MolFrame=np.zeros((1, 3)),
                        SampleRotation=np.zeros((0, 0)))
    Exp = Struct(F.adddefaults(Exp, DefaultExp))
    if 'CrystalOrientation' in Exp:
        raise ValueError('Exp.CrystalOrientation is no longer supported, use Exp.SampleFrame/Exp.MolFrame instead.')
    if 'Mode' in Exp:
        raise ValueError('Exp.Mode is no longer supported. Use Exp.mwMode instead.')
    if _mif(np.isnan(np.asarray(Exp.Field, dtype=float))):
        Exp.Field = 0.0
        F.logmsg(1, 'Exp.Field is missing, assuming 0.0 mT')
    if _mif(~np.isnan(np.asarray(Exp.CenterSweep, dtype=float))):
        if not isempty(Exp.Range) and _mif(~np.isnan(np.asarray(Exp.Range, dtype=float))):
            F.logmsg(1, 'Using Experiment.CenterSweep and ignoring Experiment.Range.')
        cs = vec(np.asarray(Exp.CenterSweep, dtype=float))
        Exp.Range = np.maximum(cs[0] + np.array([[-1.0, 1.0]]) * cs[1] / 2, 0)
    if not isempty(Exp.Range) and _mif(np.isnan(np.asarray(Exp.Range, dtype=float))):
        Exp.Range = np.zeros((0, 0))
    Rg = vec(np.asarray(Exp.Range))
    if np.iscomplexobj(Rg) or np.any(np.diff(Rg) <= 0) or np.any(~np.isfinite(Rg)) or np.any(Rg < 0):
        raise ValueError('Exp.Range is not valid!')
    Exp.Range = np.asarray(Exp.Range, dtype=float) * 1e3
    Range = vec(Exp.Range)
    xi1, xik, nB1_L, nk_L, nB0_L, mwmode = F.p_excitationgeometry(Exp.mwMode)[:6]
    nB1_L = vec(np.asarray(nB1_L, dtype=float))
    nk_L = vec(np.asarray(nk_L, dtype=float))
    nB0_L = vec(np.asarray(nB0_L, dtype=float))
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
    if 'Sites' not in Opt:
        Opt.Sites = np.zeros((0, 0))
    Orientations, nOrientations, nSites, averageOverChi = F.p_crystalorientations(Exp, Opt)[:4]
    Orientations = a2(np.asarray(Orientations, dtype=float))
    nOrientations, nSites = int(nOrientations), int(nSites)

    DefaultOptions = Struct(Transitions=np.zeros((0, 0)), Threshold=1e-4, Hybrid=0.0, HybridCoreNuclei=np.zeros((0, 0)),
                            FuzzLevel=1e-10, Intensity=1.0, Sparse=False, HybridHFIThreshold=0.02,
                            HybridIntThreshold=0.005, HybridOnlyHFI=0.0)
    Opt = Struct(F.adddefaults(Opt, DefaultOptions))
    if 'Method' in Opt:
        Opt.Hybrid = isinstance(Opt.Method, str) and Opt.Method == 'hybrid'
    t_ = np.asarray(Opt.Threshold)
    if np.iscomplexobj(t_) or t_.size > 2 or np.any(t_ < 0) or np.any(t_ >= 1):
        raise ValueError('Options.Threshold must be a number >=0 and <1.')
    preSelectionThreshold = vec(t_)[0]
    postSelectionThreshold = preSelectionThreshold if t_.size == 1 else vec(t_)[1]
    StrainsPresent = any(_any(Sys[f]) for f in ('HStrain', 'DStrain', 'gStrain', 'AStrain'))
    computeStrains = StrainsPresent and nargout > 2
    computeIntensities = nargout > 1 and _mif(Opt.Intensity)
    F.logmsg(1, '- Preparations')
    useSparse = _mif(Opt.Sparse)
    F.logmsg(1, '  using sparse matrices' if useSparse else '  using full matrices')

    nPerturbNuclei = 0
    if nNuclei >= 1 and _mif(Opt.Hybrid):
        if useSparse:
            raise ValueError('Cannot use sparse matrices in hybrid mode.')
        core = [int(v) for v in vec(Opt.HybridCoreNuclei)] if not isempty(Opt.HybridCoreNuclei) else []
        if any(v > nNuclei for v in core):
            raise ValueError(f'At least one entry in Opt.HybridCoreNuclei is incorrect. There are only {nNuclei} nuclei.')
        perturbNuclei = np.ones(nNuclei, dtype=bool)
        for v in core:
            perturbNuclei[v - 1] = False
        idxPerturbNuclei = np.nonzero(perturbNuclei)[0] + 1
        nPerturbNuclei = idxPerturbNuclei.size
        str_ = 'none' if nPerturbNuclei == 0 else ''.join(f'{i} ' for i in idxPerturbNuclei)
        F.logmsg(1, '  nuclei with first-order perturbation treatment: ' + str_)
        CoreSys = Struct(F.nucspinrmv(Sys, idxPerturbNuclei.astype(float).reshape(1, -1)))
        CoreSys.processed = False
        CoreSys.pop('lwpp', None)
        CoreSys, err = F.validatespinsys(CoreSys)
        _err(err)
        I_all = vec(np.asarray(Sys.I, dtype=float))
        SysA = a2(Sys.A).astype(float)
        Hhfi = [[None] * nPerturbNuclei for _ in range(nElectrons)]
        Hzeem = [None] * nPerturbNuclei
        Hquad = [None] * nPerturbNuclei
        for iiNuc in range(nPerturbNuclei - 1, -1, -1):
            iNuc = idxPerturbNuclei[iiNuc] - 1
            I = I_all[iNuc]
            Ix, Iy, Iz = (_d(x) for x in F.sop(I, 'x', 'y', 'z'))
            for iElectron in range(nElectrons):
                idxE = slice(3 * iElectron, 3 * iElectron + 3)
                if Sys.fullA:
                    Am = SysA[3 * iNuc:3 * iNuc + 3, idxE]
                else:
                    Am = SysA[iNuc, idxE]
                    R = np.eye(3)
                    if 'AFrame' in Sys:
                        R = np.asarray(F.erot(a2(Sys.AFrame)[iNuc, idxE])).T
                    Am = R @ np.diag(Am) @ R.T
                Hhfi[iElectron][iiNuc] = Struct(x=Am[0, 0] * Ix + Am[0, 1] * Iy + Am[0, 2] * Iz,
                                                y=Am[1, 0] * Ix + Am[1, 1] * Iy + Am[1, 2] * Iz,
                                                z=Am[2, 0] * Ix + Am[2, 1] * Iy + Am[2, 2] * Iz)
            if not _mif(Opt.HybridOnlyHFI):
                prefactor = -_c(F.nmagn()) / F.planck() / 1e9 * vec(Sys.gn)[iNuc]
                Hzeem[iiNuc] = Struct(x=prefactor * Ix, y=prefactor * Iy, z=prefactor * Iz)
                Hquad[iiNuc] = 0.0
                if I >= 1:
                    Q = np.zeros(3)
                    R = np.eye(3)
                    if 'Q' in Sys:
                        Q = a2(Sys.Q)[iNuc, :]
                    if 'QFrame' in Sys:
                        R = np.asarray(F.erot(a2(Sys.QFrame)[iNuc, :])).T
                    Q = R @ np.diag(Q) @ R.T
                    Ivec = [Ix, Iy, Iz]
                    for c1 in range(3):
                        for c2 in range(3):
                            Hquad[iiNuc] = Hquad[iiNuc] + Q[c1, c2] * Ivec[c1] @ Ivec[c2]
        Sop = [None] * nElectrons
        for iEl in range(nElectrons - 1, -1, -1):
            Sop[iEl] = Struct(x=_d(F.sop(CoreSys, np.array([[iEl + 1, 1.0]]))),
                              y=_d(F.sop(CoreSys, np.array([[iEl + 1, 2.0]]))),
                              z=_d(F.sop(CoreSys, np.array([[iEl + 1, 3.0]]))))
    else:
        CoreSys = Sys

    if higherOrder:
        nCore = int(F.hsdim(CoreSys))
        nLevels = nCore
        nFull = int(F.hsdim(Sys))
        nSHFNucStates = nFull / nCore
    else:
        if useSparse:
            kH0, kmuxM, kmuyM, kmuzM = F.ham(CoreSys, np.zeros((0, 0)), 'sparse', nargout=4)
        else:
            kH0, kmuxM, kmuyM, kmuzM = (_d(x) for x in F.ham(CoreSys, nargout=4))
        nLevels = max(kH0.shape)
        nCore = nLevels
        nFull = int(F.hsdim(Sys))
        nSHFNucStates = nFull / nCore

    computeNonEquiPops = 'initState' in Sys and not isempty(Sys.initState)
    Temperature = np.asarray(Exp.Temperature, dtype=float)
    if computeNonEquiPops:
        initState = a2(Sys.initState[0])
        initStateBasis = Sys.initState[1]
        nElectronStates = int(np.prod(2 * vec(Sys.S) + 1))
        sz1, sz2 = initState.shape
        if sz1 == sz2:
            if nElectronStates != sz1 and nCore != sz1:
                raise ValueError(f'The density matrix in Sys.initState must have dimensions of nxn with n = '
                                 f'{nElectronStates} or {nCore}.')
            if initState.size == nElectronStates ** 2 and initState.size != nCore ** 2:
                initState = np.kron(initState, np.eye(nCore // nElectronStates)) / (nCore / nElectronStates)
        else:
            if initState.size != nElectronStates and initState.size != nCore:
                raise ValueError(f'The population vector in Sys.initState must have {nElectronStates} or {nCore} elements.')
            initState = vec(initState).reshape(-1, 1)
            if initState.size == nElectronStates and initState.size != nCore:
                initState = np.kron(initState, np.ones((nCore // nElectronStates, 1))) / (nCore / nElectronStates)
        computeBoltzmannPopulations = False
    elif Temperature.size == 0:
        computeBoltzmannPopulations = False
    else:
        if Temperature.size != 1:
            raise ValueError('If given, Exp.Temperature must be a single number.')
        if np.isinf(Temperature.item()):
            raise ValueError('If given, Exp.Temperature must be a finite value.')
        computeBoltzmannPopulations = not np.isnan(Temperature.item())

    FuzzLevel = _num(Opt.FuzzLevel)
    if FuzzLevel > 0 and not higherOrder and (int(CoreSys.nNuclei) > 1 or int(CoreSys.nElectrons) > 1) \
            and not computeNonEquiPops:
        noise = 1 + FuzzLevel * (2 * _rng.random(kH0.shape) - 1)
        noise = (noise + noise.T) / 2
        if sps.issparse(kH0):
            kH0, kmuxM, kmuyM, kmuzM = (sps.csr_matrix(x.multiply(noise)) for x in (kH0, kmuxM, kmuyM, kmuzM))
        else:
            kH0, kmuxM, kmuyM, kmuzM = (x * noise for x in (kH0, kmuxM, kmuyM, kmuzM))
    if nPerturbNuclei > 0:
        F.logmsg(1, '  core system with %d spins and %d states', numel(F.spinvec(CoreSys)), nCore)
        F.logmsg(1, '  first-order perturbational nuclei with %d states', nSHFNucStates)
    elif int(CoreSys.nNuclei) > 0:
        F.logmsg(1, '  full treatment of all nuclei')

    if computeNonEquiPops and initStateBasis == 'zerofield':
        if higherOrder:
            ZFE, ZFStates = np.linalg.eig(_d(F.ham(CoreSys, np.zeros((1, 3)))))
        else:
            ZFE, ZFStates = np.linalg.eigh(_d(kH0))
        ZFE = np.real(ZFE)
        idx = np.argsort(ZFE, kind='stable')
        ZFEnergies = ZFE[idx]
        ZFStates = ZFStates[:, idx]
        if np.unique(ZFEnergies).size != ZFEnergies.size:
            raise ValueError('Degenerate energy levels detected at zero-field. This prevents unambiguous assignment of '
                             'the provided sublevel populations to the zero-field states. Please provide the '
                             'non-equilibriumstate using the full density matrix. See documentation for details.')
        if min(initState.shape) == 1:
            initState = ZFStates @ np.diag(vec(initState)) @ ZFStates.conj().T
        else:
            initState = ZFStates @ initState @ ZFStates.conj().T

    UserTransitions = not isempty(Opt.Transitions)
    if UserTransitions:
        if isinstance(Opt.Transitions, str):
            if Opt.Transitions == 'all':
                nSStates = int(np.prod(2 * vec(CoreSys.S) + 1) * np.prod(2 * vec(CoreSys.L) + 1)) \
                    if numel(CoreSys.L) else int(np.prod(2 * vec(CoreSys.S) + 1))
                F.logmsg(1, '  using all %d transitions', nSStates * (nSStates - 1) // 2)
                Transitions = _nchoosek2(nSStates)
            else:
                raise ValueError("Options.Transitions must be 'all' or a nx2 array of enery level indices.")
        else:
            T_ = a2(np.asarray(Opt.Transitions, dtype=float))
            F.logmsg(1, '  using %d user-specified transitions', T_.shape[0])
            if T_.shape[1] != 2:
                raise ValueError('Options.Transitions must be a nx2 array of energy level indices.')
            Transitions = np.sort(T_, axis=1)
    else:
        Transitions = _nchoosek2(nCore)
    if Transitions.size == 0:
        raise ValueError('No transitions selected! Decrease Opt.Threshold.')
    if np.any(Transitions > nCore):
        raise ValueError('Level index in Options.Transitions is out of range.')
    u = Transitions[:, 0].astype(int) - 1
    v = Transitions[:, 1].astype(int) - 1
    nTransitions = u.size
    F.logmsg(1, '  %d transitions pre-selected', nTransitions)
    F.logmsg(1, '- Broadenings')

    if computeStrains:
        F.logmsg(1, '  using strains')
        useDStrain, dHdD, dHdE = F.getdstrainops(CoreSys)[:3]
        useDStrain = _mif(useDStrain)
        if not isinstance(dHdD, list):
            dHdD, dHdE = [dHdD], [dHdE]
        dHdD = [_d(x) if not np.isscalar(x) else x for x in dHdD]
        dHdE = [_d(x) if not np.isscalar(x) else x for x in dHdE]
        usegStrain = _any(CoreSys.gStrain)
        simplegStrain = int(CoreSys.nElectrons) == 1
        nElC = int(CoreSys.nElectrons)
        gStrainMatrix = [None] * nElC
        cg = a2(CoreSys.gStrain)
        if usegStrain:
            F.logmsg(1, '  g strain present')
            for iEl in range(nElC - 1, -1, -1):
                gStrainMatrix[iEl] = np.diag(cg[iEl, :] / a2(CoreSys.g)[iEl, :])
                if _any(a2(CoreSys.gFrame)[iEl, :]):
                    R_g2M = np.asarray(F.erot(a2(CoreSys.gFrame)[iEl, :])).T
                    gStrainMatrix[iEl] = R_g2M @ gStrainMatrix[iEl] @ R_g2M.T
            if not simplegStrain:
                F.logmsg(1, '  multiple g strains present')
                for iEl in range(nElC - 1, -1, -1):
                    F.sop(CoreSys, np.array([[iEl + 1, 1.0]]))
                    F.sop(CoreSys, np.array([[iEl + 1, 2.0]]))
                    F.sop(CoreSys, np.array([[iEl + 1, 3.0]]))
        else:
            for iEl in range(nElC - 1, -1, -1):
                gStrainMatrix[iEl] = 0.0
        useAStrain = int(CoreSys.nNuclei) > 0 and _any(CoreSys.AStrain)
        if useAStrain:
            if 'AFrame' in CoreSys:
                R = np.asarray(F.erot(a2(CoreSys.AFrame)[0, :])).T
            else:
                R = np.eye(3)
            so = lambda s, c: _d(F.sop(CoreSys, np.array([[float(s), float(c)]])))
            Ix_ = R[0, 0] * so(2, 1) + R[1, 0] * so(2, 2) + R[2, 0] * so(2, 3)
            Iy_ = R[0, 1] * so(2, 1) + R[1, 1] * so(2, 2) + R[2, 1] * so(2, 3)
            Iz_ = R[0, 2] * so(2, 1) + R[1, 2] * so(2, 2) + R[2, 2] * so(2, 3)
            Sx_ = R[0, 0] * so(1, 1) + R[0, 1] * so(1, 2) + R[0, 2] * so(1, 3)
            Sy_ = R[1, 0] * so(1, 1) + R[1, 1] * so(1, 2) + R[1, 2] * so(1, 3)
            Sz_ = R[2, 0] * so(1, 1) + R[2, 1] * so(1, 2) + R[2, 2] * so(1, 3)
            AS = vec(np.asarray(CoreSys.AStrain, dtype=float))
            dHdAx = AS[0] * Ix_ @ Sx_
            dHdAy = AS[1] * Iy_ @ Sy_
            dHdAz = AS[2] * Iz_ @ Sz_
        if _any(CoreSys.HStrain):
            F.logmsg(2, '  ## using H strain')
        if usegStrain:
            F.logmsg(2, ' ## using g strain')
        if useAStrain:
            F.logmsg(2, '  ## using A strain')
        if useDStrain:
            F.logmsg(2, '  ## using D strain')
    else:
        F.logmsg(1, '  no strains specified', nTransitions)

    Pdat = np.zeros((nTransitions, nOrientations))
    Idat = np.zeros((nTransitions, nOrientations)) if computeIntensities else np.zeros((0, 0))
    Wdat = np.zeros((nTransitions, nOrientations)) if computeStrains else np.zeros((0, 0))
    TransitionRates = np.zeros(nTransitions)
    HS2 = vec(np.asarray(CoreSys.HStrain, dtype=float)) ** 2
    if nPerturbNuclei > 0:
        pPdatN = [None] * nPerturbNuclei
        pIdatN = [None] * nPerturbNuclei
    for iOri in range(nOrientations):
        xLab, yLab, zLab = (vec(x) for x in F.erot(Orientations[iOri, :], 'rows', nargout=3))
        if higherOrder:
            sp = 'sparse' if useSparse else ''
            Vs, E = F.gethamdata_hO(Exp.Field, zLab.reshape(-1, 1), CoreSys, Opt.Sparse, np.zeros((0, 0)), nLevels,
                                    nargout=2)[:2]
            g0e = F.ham_ez(CoreSys, np.zeros((0, 0)), sp, nargout=3)
            g0n = F.ham_nz(CoreSys, np.zeros((0, 0)), sp, nargout=3)
            g0o = F.ham_oz(CoreSys, np.zeros((0, 0)), sp, nargout=3)
            g0 = [_d(g0e[k]) + _d(g0n[k]) + _d(g0o[k]) for k in range(3)]
            g1 = F.ham_ezho(CoreSys, np.zeros((0, 0)), np.zeros((0, 0)), sp, 1.0)
            kGM = [g0[n] - _d(g1[0][n]) for n in range(3)]
            kmuzL = zLab[0] * kGM[0] + zLab[1] * kGM[1] + zLab[2] * kGM[2]
            kmuxL = xLab[0] * kGM[0] + xLab[1] * kGM[1] + xLab[2] * kGM[2]
            kmuyL = yLab[0] * kGM[0] + yLab[1] * kGM[1] + yLab[2] * kGM[2]
        else:
            kmuzL = zLab[0] * kmuxM + zLab[1] * kmuyM + zLab[2] * kmuzM
            kmuxL = xLab[0] * kmuxM + xLab[1] * kmuyM + xLab[2] * kmuzM
            kmuyL = yLab[0] * kmuxM + yLab[1] * kmuyM + yLab[2] * kmuzM
            Vs, E = F.gethamdata(Exp.Field, kH0, kmuzL, np.zeros((0, 0)), nLevels, nargout=2)[:2]
        Vs = _d(Vs)
        E = vec(np.asarray(E))
        kmuxL, kmuyL, kmuzL = _d(kmuxL), _d(kmuyL), _d(kmuzL)
        Pdat[:, iOri] = E[v] - E[u]
        if computeIntensities:
            if usePhotoSelection:
                k = Exp.lightBeam[0]
                alpha = Exp.lightBeam[1]
                ori = Orientations[iOri, 0:2] if averageOverChi else Orientations[iOri, 0:3]
                photoWeight = F.photoselect(Sys.tdm, ori.reshape(1, -1), k, alpha)
                photoWeight = (1 - Exp.lightScatter) * photoWeight + Exp.lightScatter
            else:
                photoWeight = 1
            for iTrans in range(nTransitions - 1, -1, -1):
                U = Vs[:, u[iTrans]]
                V = Vs[:, v[iTrans]]
                mu_L = np.array([np.vdot(V, kmuxL @ U), np.vdot(V, kmuyL @ U), np.vdot(V, kmuzL @ U)])
                TransitionRate = None
                if averageOverChi:
                    if mwmode.linearpolarizedMode:
                        TransitionRate = ((1 - xi1 ** 2) * _norm(mu_L) ** 2 + (3 * xi1 ** 2 - 1) * abs(nB0_L @ mu_L) ** 2) / 2
                    elif mwmode.unpolarizedMode:
                        TransitionRate = ((1 + xik ** 2) * _norm(mu_L) ** 2 - (3 * xik ** 2 - 1) * abs(nB0_L @ mu_L) ** 2) / 4
                    elif mwmode.circpolarizedMode:
                        TransitionRate = (((1 + xik ** 2) * _norm(mu_L) ** 2 - (3 * xik ** 2 - 1) * abs(nB0_L @ mu_L) ** 2) / 2
                                          - mwmode.circSense * xik * (nB0_L @ np.cross(1j * mu_L, np.conj(mu_L))))
                else:
                    if mwmode.linearpolarizedMode:
                        TransitionRate = abs(nB1_L @ mu_L) ** 2
                    elif mwmode.unpolarizedMode:
                        TransitionRate = (_norm(mu_L) ** 2 - abs(nk_L @ mu_L) ** 2) / 2
                    elif mwmode.circpolarizedMode:
                        TransitionRate = ((_norm(mu_L) ** 2 - abs(nk_L @ mu_L) ** 2)
                                          - mwmode.circSense * (nk_L @ np.cross(1j * mu_L, np.conj(mu_L))))
                if TransitionRate is None:
                    raise ValueError("Unrecognized function or variable 'TransitionRate'.")
                if abs(TransitionRate) < 1e-12:
                    TransitionRate = 0
                TransitionRates[iTrans] = abs(TransitionRate)
            if computeBoltzmannPopulations:
                Populations = np.ones(nCore)
                with np.errstate(divide='ignore', invalid='ignore', over='ignore'):   # T = 0 -> -Inf, as MATLAB
                    BoltzmannPreFactor = np.float64(-1e6 * F.planck() / F.boltzm()) / np.float64(Temperature.item())
                    for iState in range(nCore):
                        dE = E[iState] - E[0]
                        if dE < 1e-10:
                            dE = 0.0
                        Populations[iState] = np.exp(BoltzmannPreFactor * dE)
                Populations[np.isnan(Populations)] = 1
                Populations = Populations / np.sum(Populations)
                Polarization = Populations[u] - Populations[v]
                if nPerturbNuclei > 0:
                    Polarization = Polarization / np.prod(2 * vec(Sys.I) + 1)
            elif computeNonEquiPops:
                if initStateBasis == 'eigen':
                    Populations = np.array([initState[i, i] for i in range(nCore)])
                else:
                    Populations = np.array([np.vdot(Vs[:, i], initState @ Vs[:, i]) for i in range(nCore)])
                Polarization = np.real(Populations[u] - Populations[v])
                if nPerturbNuclei > 0:
                    Polarization = Polarization / np.prod(2 * vec(Sys.I) + 1)
            else:
                Polarization = 1 / np.prod(2 * vec(np.asarray(Sys.I, dtype=float)) + 1) if numel(Sys.I) else 1.0
            Idat[:, iOri] = TransitionRates * np.real(Polarization) * photoWeight
        if computeStrains:
            LineWidthSquared = HS2 @ zLab ** 2
            for iTrans in range(nTransitions):
                Vv = Vs[:, v[iTrans]]
                Vu = Vs[:, u[iTrans]]
                m = lambda Op: _expect(Op, Vv) - _expect(Op, Vu)
                LineWidth2 = LineWidthSquared
                if useDStrain:
                    for iEl in range(int(CoreSys.nElectrons)):
                        LineWidth2 = LineWidth2 + abs(m(dHdD[iEl])) ** 2
                        LineWidth2 = LineWidth2 + abs(m(dHdE[iEl])) ** 2
                if useAStrain:
                    LineWidth2 = LineWidth2 + abs(m(dHdAx)) ** 2
                    LineWidth2 = LineWidth2 + abs(m(dHdAy)) ** 2
                    LineWidth2 = LineWidth2 + abs(m(dHdAz)) ** 2
                if usegStrain:
                    dg2 = (m(kmuzL) * Exp.Field * (zLab @ gStrainMatrix[0] @ zLab)) ** 2
                    LineWidth2 = LineWidth2 + abs(dg2)
                Wdat[iTrans, iOri] = np.sqrt(np.real(LineWidth2))
        if nPerturbNuclei > 0:
            for iTrans in range(nTransitions):
                U = Vs[:, u[iTrans]]
                V = Vs[:, v[iTrans]]
                Su = np.zeros((3, nElectrons), dtype=complex)
                Sv = np.zeros((3, nElectrons), dtype=complex)
                for iEl in range(nElectrons - 1, -1, -1):
                    Su[:, iEl] = [np.vdot(U, Sop[iEl].x @ U), np.vdot(U, Sop[iEl].y @ U), np.vdot(U, Sop[iEl].z @ U)]
                    Sv[:, iEl] = [np.vdot(V, Sop[iEl].x @ V), np.vdot(V, Sop[iEl].y @ V), np.vdot(V, Sop[iEl].z @ V)]
                for iiNuc in range(nPerturbNuclei):
                    Hu = 0
                    Hv = 0
                    for iEl in range(nElectrons):
                        h = Hhfi[iEl][iiNuc]
                        Hu = Hu + Su[0, iEl] * h.x + Su[1, iEl] * h.y + Su[2, iEl] * h.z
                        Hv = Hv + Sv[0, iEl] * h.x + Sv[1, iEl] * h.y + Sv[2, iEl] * h.z
                    if not _mif(Opt.HybridOnlyHFI):
                        z = Hzeem[iiNuc]
                        Hc = Hquad[iiNuc] + Exp.Field * (zLab[0] * z.x + zLab[1] * z.y + zLab[2] * z.z)
                        Hu = Hu + Hc
                        Hv = Hv + Hc
                    Hu = (Hu + np.conj(Hu).T) / 2
                    Hv = (Hv + np.conj(Hv).T) / 2
                    dEi, Vii = np.linalg.eigh(Hu)
                    dEf, Vfi = np.linalg.eigh(Hv)
                    NucTransitionRates = np.abs(Vfi.conj().T @ Vii) ** 2
                    nn = NucTransitionRates.shape[0]
                    if pPdatN[iiNuc] is None:
                        pPdatN[iiNuc] = np.zeros((nTransitions, nOrientations, nn * nn))
                        pIdatN[iiNuc] = np.zeros((nTransitions, nOrientations, nn * nn))
                    # [vidx,uidx] = find(ones(n)): column-major (vidx fast)
                    vidx = np.tile(np.arange(nn), nn)
                    uidx = np.repeat(np.arange(nn), nn)
                    pPdatN[iiNuc][iTrans, iOri, :] = dEf[vidx] - dEi[uidx]
                    pIdatN[iiNuc][iTrans, iOri, :] = vec(NucTransitionRates)

    F.logmsg(2, '  ## %2d resonances total from %d level pairs', Pdat.shape[0], nTransitions)
    idxWeakResonances = np.zeros(0, dtype=int)
    if computeIntensities and not UserTransitions:
        maxIntensity = np.max(np.abs(Idat), axis=1) if Idat.shape[1] else np.zeros(nTransitions)
        absoluteMaxIntensity = np.max(maxIntensity) if maxIntensity.size else 0
        idxWeakResonances = np.nonzero(maxIntensity < absoluteMaxIntensity * postSelectionThreshold)[0]
        if idxWeakResonances.size:
            F.logmsg(2, '  ## %2d resonances below relative intensity threshold %f', idxWeakResonances.size,
                     postSelectionThreshold)
        else:
            F.logmsg(2, '  ## all resonances above relative intensity threshold %f', postSelectionThreshold)
    else:
        F.logmsg(2, '  ## no intensities computed, no intensity post-selection')
    if EasySpinLogLevel >= 2:
        partlyNaN = np.any(np.isnan(Pdat), axis=1)
        nChopped = int(np.sum(partlyNaN))
        if nChopped > 0:
            F.logmsg(2, '  ## %2d transitions partly out of range', nChopped)
        F.logmsg(2, '  ## %2d transitions fully in range', nTransitions - nChopped)
    if Range.size:
        Pdat[(Pdat < Range[0]) | (Pdat > Range[1])] = np.nan
    idxOutOfRangeResonances = np.nonzero(np.all(np.isnan(Pdat), axis=1))[0]
    if idxOutOfRangeResonances.size:
        F.logmsg(2, '  ## %2d resonances completely out of range', idxOutOfRangeResonances.size)
    idxRmv = np.concatenate([idxOutOfRangeResonances, idxWeakResonances]).astype(int)
    if idxRmv.size > 0:
        F.logmsg(2, '  ## removing %2d resonances (below threshold, out of range)', idxRmv.size)
        Pdat = np.delete(Pdat, idxRmv, axis=0)
        Transitions = np.delete(Transitions, idxRmv, axis=0)
        if computeIntensities:
            Idat = np.delete(Idat, idxRmv, axis=0)
        if computeStrains:
            Wdat = np.delete(Wdat, idxRmv, axis=0)
        if nPerturbNuclei > 0:
            for iiNuc in range(nPerturbNuclei):
                pPdatN[iiNuc] = np.delete(pPdatN[iiNuc], idxRmv, axis=0)
                pIdatN[iiNuc] = np.delete(pIdatN[iiNuc], idxRmv, axis=0)
    nTransitions = Pdat.shape[0]
    F.logmsg(2, '  ## %2d resonances left', nTransitions)
    F.logmsg(1, '  %d significant transitions with resonances in range', nTransitions)
    if nTransitions == 0:
        if Range.size:
            print('WARNING: No resonance frequencies at %g mT between %g and %g GHz.\n'
                  '         Check field value and spectrometer frequency range.' %
                  (Exp.Field, Range[0] / 1e3, Range[1] / 1e3))
        else:
            print('WARNING: No resonances, no frequency range!', end='')
        e = np.zeros((0, 0))
        return e, e, e, e
    if computeIntensities and not computeNonEquiPops:
        if np.any(TransitionRates < 0):
            F.logmsg(0, '*********** Negative intensity encountered in resfields!! Please report! **********')
    if np.any(Wdat < 0):
        F.logmsg(0, '*********** Negative width encountered in resfields!! Please report! **************')

    if nPerturbNuclei > 0:
        F.logmsg(1, '  transition post-selection on perturbation nuclei')
        nPertShifts = np.zeros(nPerturbNuclei, dtype=int)
        for iiNuc in range(nPerturbNuclei):
            TotalIntensity = np.sum(np.sum(pIdatN[iiNuc], axis=0), axis=0)
            rmv = TotalIntensity < np.max(TotalIntensity) * Opt.HybridIntThreshold
            pPdatN[iiNuc] = pPdatN[iiNuc][:, :, ~rmv]
            pIdatN[iiNuc] = pIdatN[iiNuc][:, :, ~rmv]
            nPertShifts[iiNuc] = pPdatN[iiNuc].shape[2]
        F.logmsg(1, '  combining shifts from perturbational nuclei')
        idxComb1 = [None] * nPerturbNuclei
        idxComb2 = [None] * nPerturbNuclei
        for iiNuc in range(1, nPerturbNuclei):
            n2, n1 = nPertShifts[iiNuc], int(np.prod(nPertShifts[:iiNuc]))
            idxComb2[iiNuc] = np.tile(np.arange(n2), n1)      # find(ones(n2,n1)): rows fast
            idxComb1[iiNuc] = np.repeat(np.arange(n1), n2)
        nSub = int(np.prod(nPertShifts))
        pPdat = np.zeros((nTransitions, nOrientations, nSub))
        pIdat = np.zeros((nTransitions, nOrientations, nSub))
        for iT in range(nTransitions - 1, -1, -1):
            for iO in range(nOrientations - 1, -1, -1):
                Pcombined = pPdatN[0][iT, iO, :]
                Icombined = pIdatN[0][iT, iO, :]
                for iiNuc in range(1, nPerturbNuclei):
                    Pcombined = Pcombined[idxComb1[iiNuc]] + pPdatN[iiNuc][iT, iO, idxComb2[iiNuc]]
                    Icombined = Icombined[idxComb1[iiNuc]] * pIdatN[iiNuc][iT, iO, idxComb2[iiNuc]]
                pPdat[iT, iO, :] = Pcombined
                pIdat[iT, iO, :] = Icombined
        nSubTransitions = pPdat.shape[2]

        def expand(x):
            return np.repeat(x[:, :, None], nSubTransitions, axis=2)

        def rearrange(x):
            # reshape(permute(x,[1 3 2]), nTransitions*nSubTransitions, [])
            return reshape(np.transpose(x, (0, 2, 1)), nTransitions * nSubTransitions, -1)
        Pdat = rearrange(expand(Pdat) + pPdat)
        Idat = rearrange(expand(Idat) * pIdat)
        if computeStrains and Wdat.size > 0:
            Wdat = rearrange(expand(Wdat))
        Transitions = rearrange(expand(Transitions))   # nTransitions keeps its pre-expansion value (as MATLAB)

    pepperCall = _called_from('pepper')
    if not pepperCall and nSites > 1:
        siz = (nTransitions * nSites, int(round(Pdat.size / nTransitions / nSites)))
        Pdat = reshape(Pdat, siz)
        if not isempty(Idat):
            Idat = reshape(Idat, siz)
        if not isempty(Wdat):
            Wdat = reshape(Wdat, siz)
    idx = np.lexsort(Transitions.T[::-1]) if Transitions.size else np.zeros(0, dtype=int)   # sortrows (stable)
    Transitions = Transitions[idx, :]
    if not pepperCall and nSites > 1:
        idx = vec(idx.reshape(-1, 1) + np.arange(nSites).reshape(1, -1) * nTransitions)
    Pdat = Pdat[idx, :]
    if not isempty(Idat):
        Idat = Idat[idx, :]
    if not isempty(Wdat):
        Wdat = Wdat[idx, :]
    return Pdat, Idat, Wdat, Transitions


def _nchoosek2(n):
    """MATLAB nchoosek(1:n,2) as an (n(n-1)/2) x 2 float array."""
    if n < 2:
        return np.zeros((0, 2))
    i, j = np.triu_indices(n, 1)
    return np.column_stack([i + 1, j + 1]).astype(float)

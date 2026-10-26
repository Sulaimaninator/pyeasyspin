"""Compute resonance fields for field-swept EPR (matrix diagonalization, adaptive segmentation)."""
import itertools

import numpy as np

from .._registry import F, nout
from .._parallel import chunked
from .._struct import Struct
from .._mat import a2, vec, isempty, numel, isnumeric
from .._matlab import fix
from ..config import config
from .resfreqs_matrix import _c, _d, _err, _any, _mif, _num, _called_from

_rng = np.random.default_rng()

# Hybrid (first-order) nuclei: EasySpin's NucTransitionRates = abs(Vu'*Vv).^2 has u states as rows,
# but the shifts dEv(vidx) - dEu(uidx) take the row as the v state, so |<u_i|v_j>|^2 sits on the
# line dEv(i) - dEu(j). The port builds abs(Vv'*Vu).^2, as EasySpin's resfreqs_matrix does
# (abs(Vf'*Vi).^2); positions and their order are unchanged, and for I = 1/2 (symmetric 2x2 rates)
# so are the intensities (docs/DEVIATIONS.md). True restores EasySpin's numbers.
EASYSPIN_HYBRID_PAIRING = False
_M = np.array([[2.0, -2.0, 1.0, 1.0], [-3.0, 3.0, -2.0, -1.0], [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]])


def _t(x):
    """MATLAB truth value of a scalar option."""
    a = np.asarray(x)
    return a.size > 0 and bool(np.all(a != 0))


def _m(Op, U, V):
    """real((V-U)'*Op*(V+U)); Op may be the scalar 0 from getdstrainops."""
    if np.ndim(Op) == 0:
        return float(np.real(complex(Op) * np.vdot(V - U, V + U)))
    return float(np.real(np.vdot(V - U, _d(Op) @ (V + U))))


def _eigvec(H):
    """MATLAB [V,E] = eig(H) for a Hermitian H: ascending energies, vectors as columns."""
    E, V = np.linalg.eigh(H)
    return V, E


def _merge_orientation_chunks(chunks, nTransitions, nOrientations):
    """Combine the results of the orientation loop run on consecutive chunks of orientations.

    Each chunk holds idxTr (row -> transition, without the inf sentinel) and arrays with one
    column per orientation of the chunk. In the serial loop, row k of transition t holds the k-th
    resonance of t at every orientation; rows k > 1 are inserted when the first orientation has
    a k-th resonance. Rows are therefore keyed by (t, k), ordered by t and then k, and a missing
    entry takes the value the serial loop would leave there: the initial value of the array for
    k = 1 rows (NaN, but 0 for Gdat and the perturbation-nucleus arrays) and the inserted value
    for k > 1 rows (NaN, but 0 for the perturbation-nucleus arrays)."""
    kmax = np.ones(nTransitions + 1, dtype=int)
    keys_c = []
    for c in chunks:
        t = np.asarray(c['idxTr'], dtype=int)
        k = np.zeros(t.size, dtype=int)
        seen = {}
        for r, tt in enumerate(t):
            seen[tt] = seen.get(tt, 0) + 1
            k[r] = seen[tt]
            kmax[tt] = max(kmax[tt], k[r])
        keys_c.append((t, k))
    row_of = {}
    idxTr = []
    for tt in range(1, nTransitions + 1):
        for kk in range(1, kmax[tt] + 1):
            row_of[(tt, kk)] = len(idxTr)
            idxTr.append(tt)
    nRows = len(idxTr)
    first = np.array([kk == 1 for tt in range(1, nTransitions + 1) for kk in range(1, kmax[tt] + 1)])

    def alloc(name, shape_tail, init_first, init_inserted, dtype):
        a = np.empty((nRows, nOrientations) + shape_tail, dtype=dtype)
        a[first] = init_first
        a[~first] = init_inserted
        return a

    out = {'idxTr': idxTr}
    specs = [('Pdat', np.nan, np.nan), ('Idat', np.nan, np.nan), ('Wdat', np.nan, np.nan), ('Gdat', 0.0, np.nan)]
    col0 = 0
    cols = []
    for c in chunks:
        cols.append(slice(col0, col0 + c['ncols']))
        col0 += c['ncols']
    for name, f1, f2 in specs:
        if chunks[0][name] is None:
            out[name] = None
            continue
        dtype = complex if any(np.iscomplexobj(c[name]) for c in chunks) else float
        a = alloc(name, (), f1, f2, dtype)
        for c, (t, k), cs in zip(chunks, keys_c, cols):
            rows = [row_of[(tt, kk)] for tt, kk in zip(t, k)]
            a[rows, cs] = c[name]
        out[name] = a
    for name in ('pPdatN', 'pIdatN'):
        lists = [c[name] for c in chunks]
        if not lists[0]:
            out[name] = []
            continue
        merged = []
        for i in range(len(lists[0])):
            tail = lists[0][i].shape[2:]
            a = alloc(name, tail, 0.0, 0.0, float)
            for c, (t, k), cs in zip(chunks, keys_c, cols):
                rows = [row_of[(tt, kk)] for tt, kk in zip(t, k)]
                a[rows, cs] = c[name][i]
            merged.append(a)
        out[name] = merged
    for name in ('nDiagonalizations', 'nRediags', 'nMaxSegmentsReached'):
        out[name] = sum(c[name] for c in chunks)
    return out


@nout(5)
def resfields(Sys=None, Exp=None, Opt=None, nargout=4):
    """(Pdat, Idat, Wdat, Transitions, Gdat): resonance fields (mT), intensities, strain widths
    (mT), 1-based level pairs and gradients. nargout controls what is computed, as in MATLAB:
    intensities for nargout > 1, widths for nargout > 2 (if strains are present), gradients for
    nargout > 4 (or with strains). Outputs not computed are empty."""
    pepperCall = _called_from('pepper')   # MATLAB: d = dbstack; strcmp(d(2).name,'pepper')
    if Sys is None or Exp is None:
        raise ValueError('Incorrect number of inputs!')
    if nargout < 0:
        raise ValueError('Not enough output arguments.')
    if nargout > 5:
        raise ValueError('Too many output arguments.')
    if Opt is None or isempty(Opt):
        Opt = Struct()
    if not isinstance(Sys, dict) or not isinstance(Exp, dict) or not isinstance(Opt, dict):
        raise ValueError('SpinSystem, Parameters and Options must be structures!')
    Opt = Struct(Opt)
    if 'Verbosity' not in Opt:
        Opt.Verbosity = 0.0
    config.LogLevel = Opt.Verbosity
    LogLevel = _num(Opt.Verbosity)

    # ---------------------------------------------------------------- spin system
    Sys, err = F.validatespinsys(Sys)
    _err(err)
    DefaultSys = Struct(lw=0.0, HStrain=np.zeros((1, 3)), gStrain=np.zeros((1, 3)), AStrain=np.zeros((1, 3)),
                        DStrain=0.0, gAStrainCorr=1.0)
    Sys = Struct(F.adddefaults(Sys, DefaultSys))
    gc = Sys.gAStrainCorr
    if numel(gc) != 1 or not isnumeric(gc) or _num(gc) == 0 or not np.isfinite(_num(gc)):
        raise ValueError('Sys.gAStrainCorr must be a single number, either +1 or -1.')
    Sys.gAStrainCorr = float(np.sign(_num(gc)))
    nEl = int(Sys.nElectrons)
    nNuc = int(Sys.nNuclei)
    if nEl > 1 and _any(Sys.AStrain):
        raise ValueError('AStrain is not supported in spin systems with more than one electron spin.')
    if _any(Sys.gStrain) or _any(Sys.AStrain):
        if a2(Sys.g).shape[0] == 3 * numel(Sys.S):
            raise ValueError('gStrain and AStrain are not supported when full g matrices are given!')
        if _any(Sys.gFrame):
            raise ValueError('g strain cannot be used with tilted g tensors.')
        # EasySpin raises this also for g strain alone with a tilted A tensor; only A strain needs it
        # (DEVIATIONS.md, T6; same fix as T7 D2)
        if nNuc > 0 and _any(a2(Sys.AFrame)[0, :]) and _any(Sys.AStrain):
            raise ValueError('A strain cannot be used with a tilted A tensor.')
        if _any(Sys.DStrain):
            raise ValueError('D strain and g/A strain cannot be used at the same time.')
    higherOrder = any(str(k)[:3] == 'Ham' for k in Sys)

    # ---------------------------------------------------------------- experiment
    DefaultExp = Struct(mwFreq=np.nan, Range=np.nan, CenterSweep=np.nan, Temperature=np.nan, mwMode='perpendicular',
                        SampleFrame=np.zeros((1, 3)), CrystalSymmetry=1.0, MolFrame=np.zeros((1, 3)),
                        SampleRotation=np.zeros((0, 0)))
    Exp = Struct(F.adddefaults(Exp, DefaultExp))
    if 'CrystalOrientation' in Exp:
        raise ValueError('Exp.CrystalOrientation is no longer supported, use Exp.SampleFrame/Exp.MolFrame instead.')
    if 'Mode' in Exp:
        raise ValueError('Exp.Mode is no longer supported. Use Exp.mwMode instead.')
    if np.isnan(_num(Exp.mwFreq)):
        raise ValueError('Exp.mwFreq is missing!')
    mwFreq = _num(Exp.mwFreq) * 1e3
    if _mif(~np.isnan(np.asarray(Exp.CenterSweep, dtype=float))):
        cs = vec(np.asarray(Exp.CenterSweep, dtype=float))
        Exp.Range = cs[0] + np.array([[-1.0, 1.0]]) * cs[1] / 2
        if vec(Exp.Range)[0] < 0:
            raise ValueError('Lower field limit from Exp.CenterSweep cannt be negative.')
    if _mif(np.isnan(np.asarray(Exp.Range, dtype=float))):
        raise ValueError('Exp.Range/Exp.CenterSweep is missing!')
    Rg = vec(np.asarray(Exp.Range))
    if np.iscomplexobj(Rg) or np.any(np.diff(Rg) <= 0) or np.any(~np.isfinite(Rg)):
        raise ValueError('Exp.Range is not valid!')
    Range = Rg.astype(float)
    if np.any(Range < 0):
        raise ValueError('Negative magnetic fields in Exp.Range are not possible.')
    xi1, xik, nB1_L, nk_L, nB0_L, mwmode = F.p_excitationgeometry(Exp.mwMode)[:6]
    xi1, xik = _num(xi1), _num(xik)
    nB1_L, nk_L, nB0_L = (vec(np.asarray(x, dtype=float)) for x in (nB1_L, nk_L, nB0_L))
    if 'Sites' not in Opt:
        Opt.Sites = np.zeros((0, 0))
    if 'lightBeam' not in Exp:
        Exp.lightBeam = ''
    if 'lightScatter' not in Exp:
        Exp.lightScatter = 0.0
    usePhotoSelection = not isempty(Exp.lightBeam) and _num(Exp.lightScatter) < 1
    if usePhotoSelection:
        if 'tdm' not in Sys or isempty(Sys.tdm):
            raise ValueError('To include photoselection weights, Sys.tdm must be given.')
        if isinstance(Exp.lightBeam, str):
            alphas = {'perpendicular': -np.pi / 2, 'parallel': np.pi, 'unpolarized': np.nan}
            if Exp.lightBeam not in alphas:
                raise ValueError("Unknown string in Exp.lightBeam. Use '', 'perpendicular', 'parallel' or 'unpolarized'.")
            Exp.lightBeam = [np.array([[0.0], [1.0], [0.0]]), alphas[Exp.lightBeam]]
        elif not isinstance(Exp.lightBeam, (list, tuple)) or len(Exp.lightBeam) != 2:
            raise ValueError('Exp.lightBeam should be a 2-element cell {k alpha}.')
    angles_M2L, nOrientations, nSites, averageOverChi = F.p_crystalorientations(Exp, Opt)[:4]
    angles_M2L = a2(np.asarray(angles_M2L, dtype=float))
    nOrientations, nSites, averageOverChi = int(nOrientations), int(nSites), _t(averageOverChi)

    # ---------------------------------------------------------------- options
    if 'Perturb' in Opt:
        raise ValueError("Options.Perturb is obsolete. Use Opt.Method='perturb' or Opt.Method='hybrid' instead.")
    for fld in ('Intensity', 'Gradient'):
        if fld in Opt and isinstance(Opt[fld], str):
            raise ValueError(f'Options.{fld} is obsolete. Please remove from code!')
    DefaultOptions = Struct(Transitions=np.zeros((0, 0)), Threshold=1e-4, Hybrid=0.0, HybridCoreNuclei=np.zeros((0, 0)),
                            TPSGridSize=4.0, TPSGridSymm='D2h', FuzzLevel=1e-10, Freq2Field=1.0, maxSegments=2000.0,
                            ModellingAccuracy=2e-6, RediagLimit=0.95, Sparse=0.0, nLevels=np.zeros((0, 0)),
                            Gradient=1.0, Intensity=1.0, HybridHFIThreshold=0.02, HybridIntThreshold=0.005,
                            HybridOnlyHFI=0.0)
    Opt = Struct(F.adddefaults(Opt, DefaultOptions))
    if 'Method' in Opt:
        Opt.Hybrid = isinstance(Opt.Method, str) and Opt.Method == 'hybrid'
    t_ = vec(np.asarray(Opt.Threshold))
    if np.iscomplexobj(t_) or t_.size > 2 or np.any(t_ < 0) or np.any(t_ >= 1):
        raise ValueError('Options.Threshold must be a number >=0 and <1.')
    preSelectionThreshold = float(t_[0])
    postSelectionThreshold = preSelectionThreshold if t_.size == 1 else float(t_[1])
    IntensitySwitch, GradientSwitch = _t(Opt.Intensity), _t(Opt.Gradient)
    if _num(Opt.Freq2Field) not in (0.0, 1.0):
        raise ValueError('Options.Freq2Field incorrect!')
    computeFreq2Field = bool(_num(Opt.Freq2Field))
    Sparse = _t(Opt.Sparse)
    StrainsPresent = any(_any(np.asarray(Sys[f], dtype=float)) for f in ('HStrain', 'DStrain', 'gStrain', 'AStrain'))
    computeStrains = StrainsPresent and nargout > 2
    computeGradient = (computeStrains or nargout > 4) and GradientSwitch
    computeIntensities = (nargout > 1 and IntensitySwitch) or computeGradient
    computeEigenPairs = computeIntensities or computeGradient or computeStrains
    F.logmsg(1, '- Preparations')
    F.logmsg(1, '  using sparse matrices' if Sparse else '  using full matrices')

    # ---------------------------------------------------------------- core system, hybrid nuclei
    CoreSys = Sys
    HFIStrength = np.zeros(1)
    if nNuc > 0:
        A_ = a2(np.asarray(Sys.A, dtype=float))
        if _t(Sys.fullA):
            Alin = vec(A_)   # EasySpin: Sys.A((iNuc-1)*3+(1:3)), linear indexing (kept)
            maxHF = np.array([np.max(Alin[3 * i:3 * i + 3]) for i in range(nNuc)])
        else:
            maxHF = np.max(np.abs(A_), axis=1)
        HFIStrength = maxHF * (0.5 + vec(np.asarray(F.nucspin(Sys.Nucs), dtype=float))) / mwFreq
    nPerturbNuclei = 0
    if int(CoreSys.nNuclei) >= 1 and _t(Opt.Hybrid):
        hcn = vec(np.asarray(Opt.HybridCoreNuclei, dtype=float))
        if np.any(hcn > int(CoreSys.nNuclei)):
            raise ValueError('Opt.HybridCoreNuclei is incorrect!')
        perturb = np.ones(int(CoreSys.nNuclei), dtype=bool)
        perturb[hcn.astype(int) - 1] = False
        pidx = np.flatnonzero(perturb)
        nPerturbNuclei = pidx.size
        str1 = ''.join(f'{i + 1} ' for i in pidx) or 'none'
        F.logmsg(1, '  nuclei with first-order treatment: ' + str1)
        CoreSys = Struct(F.nucspinrmv(CoreSys, (pidx + 1).astype(float)))
        CoreSys.processed = 0.0
        CoreSys.pop('lwpp', None)
        CoreSys, err = F.validatespinsys(CoreSys)
        _err(err)
        Ivals = vec(np.asarray(Sys.I, dtype=float))
        Afull = a2(np.asarray(Sys.A, dtype=float))
        Hhfi, Hzeem, Hquad = {}, {}, {}
        nPerturbTransitions = np.zeros(nPerturbNuclei, dtype=int)
        for iiNuc in range(nPerturbNuclei - 1, -1, -1):
            iNuc = pidx[iiNuc]
            I = Ivals[iNuc]
            Ix, Iy, Iz = (_d(x) for x in F.sop(I, 'x', 'y', 'z'))
            nPerturbTransitions[iiNuc] = int(round((2 * I + 1) ** 2))
            for iEl in range(nEl):
                idxE = slice(3 * iEl, 3 * iEl + 3)
                if _t(Sys.fullA):
                    A = Afull[3 * iNuc:3 * iNuc + 3, idxE]
                else:
                    R_A2M = np.eye(3)
                    if 'AFrame' in Sys:
                        R_A2M = np.asarray(F.erot(a2(Sys.AFrame)[iNuc, idxE])).T
                    A = R_A2M @ np.diag(Afull[iNuc, idxE]) @ R_A2M.T
                Hhfi[(iEl, iiNuc)] = Struct(x=A[0, 0] * Ix + A[0, 1] * Iy + A[0, 2] * Iz,
                                            y=A[1, 0] * Ix + A[1, 1] * Iy + A[1, 2] * Iz,
                                            z=A[2, 0] * Ix + A[2, 1] * Iy + A[2, 2] * Iz)
            if not _t(Opt.HybridOnlyHFI):
                prefactor = -_num(F.nmagn()) / _num(F.planck()) / 1e9 * vec(np.asarray(Sys.gn, dtype=float))[iNuc]
                Hzeem[iiNuc] = Struct(x=prefactor * Ix, y=prefactor * Iy, z=prefactor * Iz)
                Hquad[iiNuc] = 0.0
                if I >= 1:
                    Q = np.zeros(3)
                    R_Q2M = np.eye(3)
                    if 'Q' in Sys:
                        Q = a2(np.asarray(Sys.Q, dtype=float))[iNuc, :]
                    if 'QFrame' in Sys:
                        R_Q2M = np.asarray(F.erot(a2(Sys.QFrame)[iNuc, :])).T
                    Q = R_Q2M @ np.diag(Q) @ R_Q2M.T
                    Ivec = (Ix, Iy, Iz)
                    for c1 in range(3):
                        for c2 in range(3):
                            Hquad[iiNuc] = Hquad[iiNuc] + Q[c1, c2] * (Ivec[c1] @ Ivec[c2])
        Sops = [Struct(x=_d(F.sop(CoreSys, np.array([[iEl + 1, 1.0]]))),
                       y=_d(F.sop(CoreSys, np.array([[iEl + 1, 2.0]]))),
                       z=_d(F.sop(CoreSys, np.array([[iEl + 1, 3.0]])))) for iEl in range(nEl)]

    if higherOrder:
        nCore = int(_num(F.hsdim(CoreSys)))
        nLevels = nCore
    else:
        if Sparse:
            kH0, kmuxM, kmuyM, kmuzM = F.ham(CoreSys, np.zeros((0, 0)), 'sparse', nargout=4)[:4]
        else:
            kH0, kmuxM, kmuyM, kmuzM = F.ham(CoreSys, nargout=4)[:4]
        kH0, kmuxM, kmuyM, kmuzM = (_d(x) for x in (kH0, kmuxM, kmuyM, kmuzM))   # dense from here on
        nLevels = kH0.shape[0]
        nCore = nLevels
    nFull = int(_num(F.hsdim(Sys)))
    nSHFNucStates = nFull / nCore

    # ---------------------------------------------------------------- populations
    computeNonEquiPops = 'initState' in Sys and not isempty(Sys.initState)
    computeBoltzmannPopulations = False
    if computeNonEquiPops:
        initState = np.asarray(Sys.initState[0])
        initStateBasis = Sys.initState[1]
        nElectronStates = int(np.prod(2 * vec(np.asarray(Sys.S, dtype=float)) + 1))
        if initState.ndim == 2 and initState.shape[0] == initState.shape[1] and initState.shape[0] > 1:
            sz1 = initState.shape[0]
            if sz1 != nElectronStates and sz1 != nCore:
                raise ValueError(f'The density matrix in Sys.initState must have dimensions of nxn with n = '
                                 f'{nElectronStates} or {nCore}.')
            if initState.size == nElectronStates ** 2 and nCore > nElectronStates:
                r = nCore // nElectronStates
                initState = np.kron(initState, np.eye(r)) / r
        elif initState.ndim <= 1 or min(initState.shape) == 1:
            if initState.size != nElectronStates and initState.size != nCore:
                raise ValueError(f'The population vector in Sys.initState must have {nElectronStates} or '
                                 f'{nCore} elements.')
            initState = vec(initState)
            if initState.size == nElectronStates and initState.size != nCore:
                r = nCore // nElectronStates
                initState = np.kron(initState, np.ones(r)) / r
            initState = np.diag(initState)
        else:
            raise ValueError('Sys.initState must contain either a population vector or a density matrix.')
    elif isempty(Exp.Temperature):
        computeBoltzmannPopulations = False
    else:
        if numel(Exp.Temperature) != 1:
            raise ValueError('If given, Exp.Temperature must be a single number.')
        if np.isinf(_num(Exp.Temperature)):
            raise ValueError('If given, Exp.Temperature must be a finite value.')
        computeBoltzmannPopulations = not np.isnan(_num(Exp.Temperature))

    if (_num(Opt.FuzzLevel) > 0 and not higherOrder and (int(CoreSys.nNuclei) > 1 or int(CoreSys.nElectrons) > 1)
            and not computeNonEquiPops):
        noise = 1 + _num(Opt.FuzzLevel) * (2 * _rng.random(kH0.shape) - 1)   # MATLAB rand: not reproducible
        noise = (noise + noise.T) / 2
        kH0, kmuxM, kmuyM, kmuzM = kH0 * noise, kmuxM * noise, kmuyM * noise, kmuzM * noise
    if nPerturbNuclei > 0:
        F.logmsg(1, '  core system with %d spins and %d states', numel(F.spinvec(CoreSys)), nCore)
        F.logmsg(1, '  first-order perturbational nuclei with %d states', nSHFNucStates)
    elif int(CoreSys.nNuclei) > 0:
        F.logmsg(1, '  full treatment of all nuclei')

    if computeNonEquiPops and initStateBasis == 'zerofield':
        if higherOrder:
            ZFEnergies, ZFStates = np.linalg.eig(_d(F.ham(CoreSys, np.zeros((1, 3)), nargout=1)))
        else:
            ZFEnergies, ZFStates = np.linalg.eigh(kH0)
        o = np.argsort(np.real(ZFEnergies), kind='stable')
        ZFEnergies, ZFStates = np.real(ZFEnergies)[o], ZFStates[:, o]
        if np.unique(ZFEnergies).size != ZFEnergies.size:
            raise ValueError('Degenerate energy levels detected at zero-field. This prevents unambiguous assignment of '
                             'the provided sublevel populations to the zero-field states. Please provide the '
                             'non-equilibrium state using the full density matrix. See documentation for details.')
        initState = ZFStates @ initState @ ZFStates.conj().T
    else:
        if higherOrder:
            ZFEnergies = np.sort(np.real(np.linalg.eigvals(_d(F.ham(CoreSys, np.zeros((1, 3)), nargout=1)))))
        else:
            ZFEnergies = np.sort(np.linalg.eigvalsh(kH0))
    maxZeroFieldSplit = ZFEnergies[-1] - ZFEnergies[0]
    if maxZeroFieldSplit > 1e3:
        F.logmsg(2, '  ## maximum splitting at zero field: %g GHz', maxZeroFieldSplit / 1e3)
    else:
        F.logmsg(2, '  ## maximum splitting at zero field: %g MHz', maxZeroFieldSplit)
    LoopFields = maxZeroFieldSplit / mwFreq > 1 - 1e-6
    F.logmsg(1, '  looping transitions possible' if LoopFields else '  no looping transitions possible')

    # ---------------------------------------------------------------- transition pre-selection
    F.logmsg(1, '- Transition pre-selection')
    CoreS = vec(np.asarray(CoreSys.S, dtype=float))
    CoreL = vec(np.asarray(CoreSys.get('L', np.zeros(0)), dtype=float))
    if not isempty(Opt.Transitions):
        if isinstance(Opt.Transitions, str):
            if Opt.Transitions != 'all':
                raise ValueError("Options.Transitions must be 'all' or a nx2 array of enery level indices.")
            if isempty(Opt.nLevels):
                nStates_ = int(np.prod(2 * CoreS + 1) * np.prod(2 * vec(np.asarray(CoreSys.get('I', np.zeros(0)), dtype=float)) + 1))
                if 'L' in CoreSys:
                    nStates_ *= int(np.prod(2 * CoreL + 1))
            else:
                nStates_ = int(_num(Opt.nLevels))
            F.logmsg(1, '  using all %d transitions', nStates_ * (nStates_ - 1) / 2)
            Transitions = np.array(list(itertools.combinations(range(1, nStates_ + 1), 2)), dtype=float).reshape(-1, 2)
            postSelectionThreshold = 0.0
        else:
            T_ = a2(np.asarray(Opt.Transitions, dtype=float))
            F.logmsg(1, '  using %d user-specified transitions', T_.shape[0])
            if T_.shape[1] != 2:
                raise ValueError('Options.Transitions must be a nx2 array of energy level indices.')
            Transitions = np.sort(T_, axis=1)
            Transitions = Transitions[~np.any(Transitions > nLevels, axis=1), :]
    else:
        if preSelectionThreshold == 0:
            F.logmsg(1, '  selection threshold is zero -> using all transitions')
            maxTransitionRates = np.ones((nCore, nCore))
        elif int(CoreSys.nElectrons) == 1 and int(CoreSys.nNuclei) == 0 and not np.any(CoreL != 0):
            F.logmsg(1, '  one electron spin and no nuclei -> using all transitions')
            maxTransitionRates = np.ones((nCore, nCore))
        else:
            if nOrientations > 1:
                F.logmsg(1, '  selection threshold %g', preSelectionThreshold)
                F.logmsg(2, '  ## (selection threshold %g, grid size %d, grid symmetry %s)', preSelectionThreshold,
                         _num(Opt.TPSGridSize), Opt.TPSGridSymm)
                TPSgrid = _c(F.sphgrid(Opt.TPSGridSymm, Opt.TPSGridSize))
                phi, theta = vec(np.asarray(TPSgrid.phi, dtype=float)), vec(np.asarray(TPSgrid.theta, dtype=float))
            else:
                phi, theta = vec(angles_M2L)[0:1], vec(angles_M2L)[1:2]   # angles_M2L(1), angles_M2L(2)
            if higherOrder:
                sp = 'sparse' if Sparse else ''
                if Sparse:
                    g1 = F.ham_ezho(CoreSys, np.zeros((0, 0)), sp, 1.0)
                else:
                    g1 = F.ham_ezho(CoreSys, np.zeros((0, 0)), np.zeros((0, 0)), sp, 1.0)
                g0 = list(F.ham_ez(CoreSys, np.zeros((0, 0)), sp, nargout=3))
                if nNuc > 0:
                    mu0n = F.ham_nz(CoreSys, np.zeros((0, 0)), sp, nargout=3)
                    g0 = [g0[k] - mu0n[k] for k in range(3)]
                ExM, EyM, EzM = (_d(g1[0][k] + g0[k]) for k in range(3))
            else:
                ExM, EyM, EzM = -kmuxM, -kmuyM, -kmuzM
            st, ct, sp_, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
            centerB = np.mean(Range)
            maxTransitionRates = np.zeros((nCore, nCore))
            for iOri in range(theta.size):
                if higherOrder:
                    Vecs = _d(F.gethamdata_hO(centerB, np.array([[st[iOri] / np.sqrt(2), st[iOri] / np.sqrt(2), ct[iOri]]]),
                                              CoreSys, float(Sparse), np.zeros((0, 0)), float(nLevels), nargout=2)[0])
                else:
                    kmuzL = st[iOri] * (cp[iOri] * kmuxM + sp_[iOri] * kmuyM) + ct[iOri] * kmuzM
                    Vecs = _eigvec(kH0 - centerB * kmuzL)[0]
                ExyM = cp[iOri] * ExM + sp_[iOri] * EyM
                if _t(mwmode.parallelMode):
                    EzL = st[iOri] * ExyM + ct[iOri] * EzM
                    TransRate_ = np.abs(Vecs.conj().T @ EzL @ Vecs) ** 2
                else:
                    EyL = -sp_[iOri] * ExM + cp[iOri] * EyM
                    ExL = ct[iOri] * ExyM - st[iOri] * EzM
                    TransRate_ = (np.abs(Vecs.conj().T @ ExL @ Vecs) ** 2 + np.abs(Vecs.conj().T @ EyL @ Vecs) ** 2) / 2
                maxTransitionRates = np.maximum(maxTransitionRates, TransRate_)
        keepidx = np.triu(np.ones((nCore, nCore), dtype=bool), 1)
        if np.max(HFIStrength) < 0.5 and preSelectionThreshold > 0:
            nElStates_ = int(np.prod(2 * CoreS + 1) * np.prod(2 * CoreL + 1))
            keepidx &= ~np.kron(np.eye(nElStates_), np.ones((nCore // nElStates_, nCore // nElStates_))).astype(bool)
        lin = np.flatnonzero(keepidx.ravel(order='F'))
        rates = maxTransitionRates.ravel(order='F')[lin]
        Transitions = np.column_stack([lin % nCore + 1, lin // nCore + 1]).astype(float)
        order = np.argsort(-rates, kind='stable')       # MATLAB sort(...,'descend') is stable
        nTr_ = int(np.sum(rates > preSelectionThreshold * np.max(rates))) if rates.size else 0
        Transitions = Transitions[order[:nTr_], :]
    if Transitions.size == 0:
        raise ValueError('No transitions selected! Decrease Opt.Threshold.')
    if np.any(Transitions > nCore):
        raise ValueError('Level index in Options.Transitions is out of range.')
    u = Transitions[:, 0].astype(int)    # 1-based
    v = Transitions[:, 1].astype(int)
    nTransitions = u.size
    Trans = (u + (v - 1) * nCore).astype(float)    # 1-based linear indices, as MATLAB passes them
    F.logmsg(1, '  %d transitions pre-selected', nTransitions)

    # ---------------------------------------------------------------- broadenings
    F.logmsg(1, '- Broadenings')
    simplegStrain, usegStrain, useAStrain, useDStrain = True, False, False, False
    if computeStrains:
        F.logmsg(1, '  using strains')
        HStrain2 = vec(np.asarray(CoreSys.HStrain, dtype=float)) ** 2
        useDStrain, dHdD, dHdE = F.getdstrainops(CoreSys)[:3]
        useDStrain = _t(useDStrain)
        usegStrain = _any(np.asarray(CoreSys.gStrain, dtype=float))
        nElC = int(CoreSys.nElectrons)
        gStrainMatrix = [np.zeros((3, 3)) for _ in range(nElC)]
        if usegStrain:
            F.logmsg(1, '  g strain present')
            simplegStrain = nElC == 1
            gS, gV, gF = (a2(np.asarray(CoreSys[f], dtype=float)) for f in ('gStrain', 'g', 'gFrame'))
            for iEl in range(nElC - 1, -1, -1):
                gStrainMatrix[iEl] = np.diag(gS[iEl, :] / gV[iEl, :]) * mwFreq
                if np.any(gF[iEl, :]):
                    R_g2M = np.asarray(F.erot(gF[iEl, :])).T
                    gStrainMatrix[iEl] = R_g2M @ gStrainMatrix[iEl] @ R_g2M.T
            if not simplegStrain:
                F.logmsg(1, '  multiple g strains present')
                kSxM, kSyM, kSzM = ({e: _d(F.sop(CoreSys, np.array([[e + 1, float(c)]]))) for e in range(nElC)}
                                    for c in (1, 2, 3))
        useAStrain = int(CoreSys.nNuclei) > 0 and _any(np.asarray(CoreSys.AStrain, dtype=float))
        if useAStrain:
            AStrainMatrix = np.diag(vec(np.asarray(CoreSys.AStrain, dtype=float)))
            if 'AFrame' in CoreSys:
                R_A2M = np.asarray(F.erot(a2(CoreSys.AFrame)[0, :])).T
                AStrainMatrix = R_A2M @ AStrainMatrix @ R_A2M.T
            centerB = np.mean(Range)
            Vecs = _eigvec(kH0 - centerB * kmuzM)[0]
            mI = np.real(np.diag(Vecs.conj().T @ _d(F.sop(CoreSys, np.array([[2.0, 3.0]]))) @ Vecs))
            mITr = np.mean(mI[Transitions.astype(int) - 1], axis=1)   # MATLAB errors for one transition
            AS3 = mITr[None, None, :] * AStrainMatrix[:, :, None]
            corr = Sys.gAStrainCorr
            gAslw2 = [(gStrainMatrix[e][:, :, None] + corr * AS3) ** 2 for e in range(nEl)]
        else:
            gAslw2 = [np.repeat((gStrainMatrix[e] ** 2)[:, :, None], nTransitions, axis=2) for e in range(nEl)]
        if np.any(HStrain2):
            F.logmsg(2, '  ## using H strain')
        if usegStrain or useAStrain:
            F.logmsg(2, '  ## using g/A strain')
        if useDStrain:
            F.logmsg(2, '  ## using D strain')
    else:
        F.logmsg(1, '  no strains specified', nTransitions)

    # ---------------------------------------------------------------- resonance data
    msg = 'positions'
    if computeIntensities:
        if computeBoltzmannPopulations:
            BoltzmannPreFactor = 1e6 * _num(F.planck()) / _num(F.boltzm()) / _num(Exp.Temperature)
        msg += ', intensities'
    if computeStrains:
        msg += ', strain widths'
    if computeGradient:
        msg += ', gradients'
    F.logmsg(1, '- Resonance data: computing %s', msg)
    levelAccuracy = mwFreq * _num(Opt.ModellingAccuracy)
    if higherOrder:
        maxSlope = 0.0
        for iOri in range(nOrientations):
            zLab_M = vec(F.erot(angles_M2L[iOri, :], 'rows', nargout=3)[2])
            der = F.gethamdata_hO(Range[1], zLab_M, CoreSys, float(Sparse), np.zeros((0, 0)), float(nLevels), nargout=3)[2]
            maxSlope = max(maxSlope, float(np.max(der)))
    else:
        eigs_ = [np.linalg.eigvalsh(-M_) for M_ in (kmuxM, kmuyM, kmuzM)]
        if Sparse:   # eigs(-mu,1): largest-magnitude eigenvalue of each; then abs(max(...))
            maxSlope = abs(max(e[np.argmax(np.abs(e))] for e in eigs_))
        else:
            maxSlope = max(float(np.max(e)) for e in eigs_)
    minStateStability = 2
    F.logmsg(2, '  ## modelling accuracy %0.3f MHz, max slope %g MHz/mT', levelAccuracy, maxSlope)
    maxSegments = _num(Opt.maxSegments)
    RediagLimit = _num(Opt.RediagLimit)
    HybridOnlyHFI = _t(Opt.HybridOnlyHFI)
    Ivals_all = vec(np.asarray(Sys.get('I', np.zeros(0)), dtype=float))
    nucPolDiv = float(np.prod(2 * Ivals_all + 1))

    def _orientations(oris):
        """The orientation loop for the orientations oris (one column each). Run on all of them,
        or on chunks in parallel and merged by _merge_orientation_chunks."""
        nC = len(oris)
        Pdat = np.full((nTransitions, nC), np.nan)
        pPdatN, pIdatN = [], []
        if nPerturbNuclei > 0:
            pPdatN = [np.zeros((nTransitions, nC, nPerturbTransitions[i])) for i in range(nPerturbNuclei)]
        Idat = Wdat = Gdat = None
        if computeIntensities:
            Idat = np.full((nTransitions, nC), np.nan)
            if nPerturbNuclei > 0:
                pIdatN = [np.zeros((nTransitions, nC, nPerturbTransitions[i])) for i in range(nPerturbNuclei)]
        if computeStrains:
            Wdat = np.full((nTransitions, nC), np.nan)
        if computeGradient:
            Gdat = np.zeros((nTransitions, nC))
        idxTr = list(range(1, nTransitions + 1)) + [np.inf]
        nDiagonalizations = nRediags = nMaxSegmentsReached = 0
        for col, iOri in enumerate(oris):
            xLab_M, yLab_M, zLab_M = (vec(x) for x in F.erot(angles_M2L[iOri, :], 'rows', nargout=3))
            if not higherOrder:
                kmuzL = zLab_M[0] * kmuxM + zLab_M[1] * kmuyM + zLab_M[2] * kmuzM
                kmuxL = xLab_M[0] * kmuxM + xLab_M[1] * kmuyM + xLab_M[2] * kmuzM
                kmuyL = yLab_M[0] * kmuxM + yLab_M[1] * kmuyM + yLab_M[2] * kmuzM
                if usegStrain and not simplegStrain:
                    kSzL = {e: zLab_M[0] * kSxM[e] + zLab_M[1] * kSyM[e] + zLab_M[2] * kSzM[e] for e in range(nEl)}
            if computeStrains:
                LineWidthSquared = float(HStrain2 @ zLab_M ** 2)
            if usePhotoSelection:
                k_, alpha_ = Exp.lightBeam[0], Exp.lightBeam[1]
                ori = angles_M2L[iOri, 0:2] if averageOverChi else angles_M2L[iOri, 0:3]
                photoWeight = _num(F.photoselect(Sys.tdm, ori, k_, alpha_))
                photoWeight = (1 - _num(Exp.lightScatter)) * photoWeight + _num(Exp.lightScatter)
            else:
                photoWeight = 1.0

            # adaptive segmentation of the field range
            nSegments = 1
            Bknots = [Range[0], Range[1]]       # linspace(Range(1), Range(2), 2)
            vectors, E, dEdB, deltaE = [], [], [], []
            for s in range(nSegments + 1):
                if higherOrder:
                    d_ = F.gethamdata_hO(Bknots[s], zLab_M, CoreSys, float(Sparse), Trans, float(nLevels), nargout=4)
                else:
                    d_ = F.gethamdata(Bknots[s], kH0, kmuzL, Trans, float(nLevels), nargout=4)
                vectors.append(_d(d_[0])); E.append(vec(d_[1])); dEdB.append(vec(d_[2])); deltaE.append(vec(d_[3]))
                nDiagonalizations += 1
            converged = [False] * nSegments
            while not all(converged) and nSegments < maxSegments:
                s = converged.index(False)
                dB = Bknots[s + 1] - Bknots[s]
                if E[s + 1][-1] - E[s + 1][0] > mwFreq:
                    if LoopFields:
                        tp = np.abs((deltaE[s] + deltaE[s + 1]) / 2 - mwFreq) <= maxSlope * dB
                    else:
                        tp = (deltaE[s] - mwFreq) * (deltaE[s + 1] - mwFreq) <= 0
                else:
                    tp = np.zeros(nTransitions, dtype=bool)
                if not np.any(tp):
                    converged[s] = True
                    continue
                newB = (Bknots[s] + Bknots[s + 1]) / 2
                if higherOrder:
                    d_ = F.gethamdata_hO(newB, zLab_M, CoreSys, float(Sparse), Trans, float(nLevels), nargout=4)
                else:
                    d_ = F.gethamdata(newB, kH0, kmuzL, Trans, float(nLevels), nargout=4)
                Ve, En, dEdBn, dEn = _d(d_[0]), vec(d_[1]), vec(d_[2]), vec(d_[3])
                nDiagonalizations += 1
                include = np.zeros(nCore, dtype=bool)
                include[u[tp] - 1] = True
                include[v[tp] - 1] = True
                deviation = 2 * ((E[s] + E[s + 1]) / 2 + dB / 8 * (dEdB[s] - dEdB[s + 1]) - En)
                converged_ = bool(np.max(np.abs(deviation[include])) <= levelAccuracy)
                Bknots.insert(s + 1, newB)
                E.insert(s + 1, En)
                vectors.insert(s + 1, Ve)
                dEdB.insert(s + 1, dEdBn)
                deltaE.insert(s + 1, dEn)
                nSegments += 1
                converged = converged[:s] + [converged_, converged_] + converged[s + 1:]
            if nSegments >= maxSegments:
                nMaxSegmentsReached += 1
                F.logmsg(2, '   segmentation terminated, %d segments, max number of segment reached', nSegments)
            else:
                F.logmsg(2, '   segmentation converged, %d segments', nSegments)
            StateStability = np.column_stack([np.abs(np.sum(np.conj(vectors[s + 1]) * vectors[s], axis=0))
                                              for s in range(nSegments)])
            dBs = np.diff(Bknots)
            SplineModelCoeffs = [_M @ np.vstack([E[s], E[s + 1], dBs[s] * dEdB[s], dBs[s] * dEdB[s + 1]])
                                 for s in range(nSegments)]

            # resonances of each transition
            ii = 0
            for iTrans in range(1, nTransitions + 1):
                while idxTr[ii] < iTrans:
                    ii += 1
                for s in range(nSegments):
                    coeffs = SplineModelCoeffs[s]
                    dEpoly = coeffs[:, v[iTrans - 1] - 1] - coeffs[:, u[iTrans - 1] - 1]
                    dEpoly[3] -= mwFreq
                    cubicZeros = vec(F.cubicsolve(dEpoly, float(LoopFields))[0])
                    if cubicZeros.size == 0:
                        continue
                    ResonanceFields = Bknots[s] + dBs[s] * cubicZeros
                    for iReson in range(ResonanceFields.size):
                        RF = ResonanceFields[iReson]
                        if idxTr[ii] > iTrans:
                            Pdat = np.insert(Pdat, ii, np.nan, axis=0)
                            if computeIntensities:
                                Idat = np.insert(Idat, ii, np.nan, axis=0)
                            if computeStrains:
                                Wdat = np.insert(Wdat, ii, np.nan, axis=0)
                            if computeGradient:
                                Gdat = np.insert(Gdat, ii, np.nan, axis=0)
                            idxTr.insert(ii, iTrans)
                            if nPerturbNuclei > 0:
                                for k_ in range(nPerturbNuclei - 1, -1, -1):
                                    pPdatN[k_] = np.insert(pPdatN[k_], ii, 0.0, axis=0)
                                    if pIdatN:
                                        pIdatN[k_] = np.insert(pIdatN[k_], ii, 0.0, axis=0)
                        Pdat[ii, col] = RF
                        if computeEigenPairs or nPerturbNuclei > 0:
                            uv = Transitions[int(idxTr[ii]) - 1, :].astype(int)
                            if np.any(StateStability[uv - 1, s] < RediagLimit):
                                nRediags += 1
                                if higherOrder:
                                    d_ = F.gethamdata_hO(RF, zLab_M, CoreSys, float(Sparse), Trans, float(nLevels), nargout=2)
                                    Vectors_, Energies = _d(d_[0]), np.asarray(d_[1])
                                    if Sparse:
                                        Energies = np.diag(Energies) if Energies.ndim == 2 else Energies
                                        o = np.argsort(Energies, kind='stable')
                                        Energies, Vectors_ = Energies[o], Vectors_[:, o]
                                else:
                                    Vectors_, Energies = _eigvec(kH0 - RF * kmuzL)
                                U = Vectors_[:, uv[0] - 1]
                                V = Vectors_[:, uv[1] - 1]
                                F.logmsg(3, '   %d-%d: stabilities %f and %f ===> rediagonalization' %
                                         (uv[0], uv[1], StateStability[uv[0] - 1, s], StateStability[uv[1] - 1, s]))
                            else:
                                z = cubicZeros[iReson]
                                Ua, Ub = vectors[s][:, uv[0] - 1], vectors[s + 1][:, uv[0] - 1]
                                k_ = int(np.argmax(np.abs(Ua)))
                                phase = Ua[k_] / Ub[k_]
                                U = Ua * (1 - z) + z * phase / abs(phase) * Ub
                                U = U / np.linalg.norm(U)
                                Va, Vb = vectors[s][:, uv[1] - 1], vectors[s + 1][:, uv[1] - 1]
                                k_ = int(np.argmax(np.abs(Va)))
                                phase = Va[k_] / Vb[k_]
                                V = Va * (1 - z) + z * phase / abs(phase) * Vb
                                V = V / np.linalg.norm(V)
                                if computeBoltzmannPopulations:
                                    t = z
                                    Energies = np.array([t ** 3, t ** 2, t, 1.0]) @ SplineModelCoeffs[s]
                            if higherOrder:
                                sp = 'sparse' if Sparse else ''
                                g1 = F.ham_ezho(CoreSys, np.zeros((0, 0)), np.zeros((0, 0)), sp, 1.0)
                                g0 = list(F.ham_ez(CoreSys, np.zeros((0, 0)), sp, nargout=3))
                                if nNuc > 0:
                                    mu0n = F.ham_nz(CoreSys, np.zeros((0, 0)), sp, nargout=3)
                                    g0 = [g0[k] - mu0n[k] for k in range(3)]
                                kmuM = [-_d(g1[0][n_] + g0[n_]) for n_ in range(3)]
                                kmuzL = zLab_M[0] * kmuM[0] + zLab_M[1] * kmuM[1] + zLab_M[2] * kmuM[2]
                                kmuxL = xLab_M[0] * kmuM[0] + xLab_M[1] * kmuM[1] + xLab_M[2] * kmuM[2]
                                kmuyL = yLab_M[0] * kmuM[0] + yLab_M[1] * kmuM[1] + yLab_M[2] * kmuM[2]
                            if computeFreq2Field:
                                dBdE = 1 / abs(np.real(np.vdot(V - U, (-kmuzL) @ (V + U))))
                                if dBdE > 1e5:
                                    raise ValueError('1/g factor diverges because d(Ev-Eu)/dB is almost zero for transition '
                                                     f'between levels u={uv[0]} and v={uv[1]}.')
                            else:
                                dBdE = 1.0
                        if computeIntensities:
                            mu_L = np.array([np.vdot(V, kmuxL @ U), np.vdot(V, kmuyL @ U), np.vdot(V, kmuzL @ U)])
                            nmu2 = float(np.sum(np.abs(mu_L) ** 2))
                            if averageOverChi:
                                if _t(mwmode.linearpolarizedMode):
                                    TransitionRate = ((1 - xi1 ** 2) * nmu2 + (3 * xi1 ** 2 - 1) * abs(nB0_L @ mu_L) ** 2) / 2
                                elif _t(mwmode.unpolarizedMode):
                                    TransitionRate = ((1 + xik ** 2) * nmu2 + (1 - 3 * xik ** 2) * abs(nB0_L @ mu_L) ** 2) / 4
                                elif _t(mwmode.circpolarizedMode):
                                    TransitionRate = (((1 + xik ** 2) * nmu2 + (1 - 3 * xik ** 2) * abs(nB0_L @ mu_L) ** 2) / 2
                                                      - _num(mwmode.circSense) * xik * (nB0_L @ np.cross(1j * mu_L, np.conj(mu_L))))
                            else:
                                if _t(mwmode.linearpolarizedMode):
                                    TransitionRate = abs(nB1_L @ mu_L) ** 2
                                elif _t(mwmode.unpolarizedMode):
                                    TransitionRate = (nmu2 - abs(nk_L @ mu_L) ** 2) / 2
                                elif _t(mwmode.circpolarizedMode):
                                    TransitionRate = ((nmu2 - abs(nk_L @ mu_L) ** 2)
                                                      - _num(mwmode.circSense) * (nk_L @ np.cross(1j * mu_L, np.conj(mu_L))))
                            if abs(TransitionRate) < 1e-10:
                                TransitionRate = 0.0
                            if computeBoltzmannPopulations:
                                Energies = vec(np.real(Energies))
                                Populations = np.exp(-BoltzmannPreFactor * (Energies - Energies[0]))
                                Populations = Populations / np.sum(Populations)
                                Polarization = Populations[u[iTrans - 1] - 1] - Populations[v[iTrans - 1] - 1]
                                if Polarization < 0:
                                    raise ValueError(f'Negative thermal polarization for transition {u[iTrans - 1]}<->'
                                                     f'{v[iTrans - 1]}: {Polarization:f}')
                                if nPerturbNuclei > 0:
                                    Polarization = Polarization / nucPolDiv
                            elif computeNonEquiPops:
                                if initStateBasis == 'eigen':
                                    PopulationU, PopulationV = initState[uv[0] - 1, uv[0] - 1], initState[uv[1] - 1, uv[1] - 1]
                                else:
                                    PopulationU, PopulationV = np.vdot(U, initState @ U), np.vdot(V, initState @ V)
                                Polarization = float(np.real(PopulationU - PopulationV))
                                if nPerturbNuclei > 0:
                                    Polarization = Polarization / nucPolDiv
                            else:
                                Polarization = 1.0 / nucPolDiv
                            val = dBdE * TransitionRate * Polarization * photoWeight
                            if np.iscomplexobj(val) and not np.iscomplexobj(Idat):
                                Idat = Idat.astype(complex)
                            Idat[ii, col] = val
                        if computeGradient:
                            Gradient2 = (np.real(np.vdot(V - U, kmuxL @ (V + U))) ** 2
                                         + np.real(np.vdot(V - U, kmuyL @ (V + U))) ** 2)
                            Gdat[ii, col] = dBdE * RF * np.sqrt(Gradient2)
                        if computeStrains:
                            LineWidth2 = LineWidthSquared
                            if useDStrain:
                                for iEl in range(int(CoreSys.nElectrons)):
                                    LineWidth2 = LineWidth2 + abs(_m(dHdD[iEl], U, V)) ** 2
                                    LineWidth2 = LineWidth2 + abs(_m(dHdE[iEl], U, V)) ** 2
                            if usegStrain or useAStrain:
                                if simplegStrain:
                                    gA2 = gAslw2[0][:, :, iTrans - 1]
                                else:
                                    gA2 = 0.0
                                    for iEl in range(nEl):
                                        gA2 = gA2 + abs(_m(kSzL[iEl], U, V)) * gAslw2[iEl][:, :, iTrans - 1]
                                LineWidth2 = LineWidth2 + float(zLab_M @ gA2 @ zLab_M)
                            Wdat[ii, col] = dBdE * np.sqrt(LineWidth2)
                        if nPerturbNuclei > 0:
                            Su = np.array([[np.vdot(U, Sops[e][c] @ U) for e in range(nEl)] for c in 'xyz'])
                            Sv = np.array([[np.vdot(V, Sops[e][c] @ V) for e in range(nEl)] for c in 'xyz'])
                            for iiNuc in range(nPerturbNuclei):
                                Hu = 0.0
                                Hv = 0.0
                                for iEl in range(nEl):
                                    h = Hhfi[(iEl, iiNuc)]
                                    Hu = Hu + Su[0, iEl] * h.x + Su[1, iEl] * h.y + Su[2, iEl] * h.z
                                    Hv = Hv + Sv[0, iEl] * h.x + Sv[1, iEl] * h.y + Sv[2, iEl] * h.z
                                if not HybridOnlyHFI:
                                    hz = Hzeem[iiNuc]
                                    Hc = Hquad[iiNuc] + RF * (zLab_M[0] * hz.x + zLab_M[1] * hz.y + zLab_M[2] * hz.z)
                                    Hu = Hu + Hc
                                    Hv = Hv + Hc
                                Hu = (Hu + np.conj(Hu).T) / 2
                                Hv = (Hv + np.conj(Hv).T) / 2
                                dEu, Vu = np.linalg.eigh(Hu)
                                dEv, Vv = np.linalg.eigh(Hv)
                                if EASYSPIN_HYBRID_PAIRING:   # EasySpin: abs(Vu'*Vv).^2, rows are u states
                                    NTR = np.abs(Vu.conj().T @ Vv) ** 2
                                else:                         # rows v, columns u, as resfreqs_matrix's Vf'*Vi
                                    NTR = np.abs(Vv.conj().T @ Vu) ** 2
                                nn = NTR.shape[0]
                                vidx = np.tile(np.arange(nn), NTR.shape[1])     # [vidx,uidx] = find(ones(size(NTR)))
                                uidx = np.repeat(np.arange(NTR.shape[1]), nn)
                                pPdatN[iiNuc][ii, col, :] = -dBdE * (dEv[vidx] - dEu[uidx])
                                if pIdatN:
                                    pIdatN[iiNuc][ii, col, :] = NTR.ravel(order='F')
                        ii += 1
                    if not LoopFields:
                        break
        return dict(idxTr=idxTr[:-1], ncols=nC, Pdat=Pdat, Idat=Idat, Wdat=Wdat, Gdat=Gdat, pPdatN=pPdatN,
                    pIdatN=pIdatN, nDiagonalizations=nDiagonalizations, nRediags=nRediags,
                    nMaxSegmentsReached=nMaxSegmentsReached)

    # Orientations are independent: large problems run in chunks on the worker pool
    # (pyeasyspin/_parallel.py); the merged result equals the serial loop's.
    R = _merge_orientation_chunks(chunked(_orientations, nOrientations), nTransitions, nOrientations)
    idxTr = R['idxTr'] + [np.inf]
    Pdat, pPdatN, pIdatN = R['Pdat'], R['pPdatN'], R['pIdatN']
    Idat = R['Idat'] if computeIntensities else np.zeros((0, 0))
    Wdat = R['Wdat'] if computeStrains else np.zeros((0, 0))
    Gdat = R['Gdat'] if computeGradient else np.zeros((0, 0))
    nDiagonalizations, nRediags, nMaxSegmentsReached = (R['nDiagonalizations'], R['nRediags'],
                                                        R['nMaxSegmentsReached'])

    # ---------------------------------------------------------------- post-selection
    idxTr = idxTr[:-1]
    F.logmsg(2, '  ## %2d resonances total from %d level pairs', Pdat.shape[0], nTransitions)
    idxWeak = np.zeros(0, dtype=int)
    if computeIntensities:
        with np.errstate(invalid='ignore'):
            A_I = np.abs(Idat)
            maxIntensity = np.array([np.nanmax(r) if np.any(~np.isnan(r)) else np.nan for r in A_I])
            absoluteMax = np.nanmax(maxIntensity) if np.any(~np.isnan(maxIntensity)) else np.nan
            idxWeak = np.flatnonzero(maxIntensity < absoluteMax * postSelectionThreshold)
        if idxWeak.size:
            F.logmsg(2, '  ## %2d resonances below relative intensity threshold %f', idxWeak.size, postSelectionThreshold)
        else:
            F.logmsg(2, '  ## all resonances above relative intensity threshold %f', postSelectionThreshold)
    else:
        F.logmsg(2, '  ## no intensities computed, no intensity post-selection')
    idxOut = np.flatnonzero(np.all(np.isnan(Pdat), axis=1))
    if idxOut.size:
        F.logmsg(2, '  ## %2d resonances completely out of range', idxOut.size)
    idxRmv = np.unique(np.concatenate([idxOut, idxWeak]))
    if idxRmv.size:
        F.logmsg(2, '  ## removing %2d resonances (below threshold, out of range)', np.concatenate([idxOut, idxWeak]).size)
        idxTr = [x for i, x in enumerate(idxTr) if i not in set(idxRmv.tolist())]
        Pdat = np.delete(Pdat, idxRmv, axis=0)
        if computeIntensities:
            Idat = np.delete(Idat, idxRmv, axis=0)
        if computeStrains:
            Wdat = np.delete(Wdat, idxRmv, axis=0)
        if computeGradient:
            Gdat = np.delete(Gdat, idxRmv, axis=0)
        if nPerturbNuclei > 0:
            pPdatN = [np.delete(p, idxRmv, axis=0) for p in pPdatN]
            if computeIntensities:
                pIdatN = [np.delete(p, idxRmv, axis=0) for p in pIdatN]
    Transitions = Transitions[np.asarray(idxTr, dtype=int) - 1, :]
    nTransitions = Pdat.shape[0]
    F.logmsg(2, '  ## %2d resonances left', nTransitions)
    if LogLevel >= 2 and nTransitions > 0:
        partlyNaN = np.any(np.isnan(Pdat), axis=1)
        x = np.zeros((int(Transitions[:, 0].max()), int(Transitions[:, 1].max())))
        np.add.at(x, (Transitions[:, 0].astype(int) - 1, Transitions[:, 1].astype(int) - 1), partlyNaN.astype(float))
        nLoopPairs = int(np.sum(fix(x / 2)))
        nChopped = int(np.sum(partlyNaN)) - nLoopPairs * 2
        if nLoopPairs > 0:
            F.logmsg(2, '  ## %2d transitions form %d looping pairs', nLoopPairs * 2, nLoopPairs)
        if nChopped > 0:
            F.logmsg(2, '  ## %2d transitions partly out of range', nChopped)
        F.logmsg(2, '  ## %2d transitions fully in range', nTransitions - int(np.sum(partlyNaN)))
    F.logmsg(1, '  %d significant transitions with resonances in range', nTransitions)
    if nTransitions == 0:
        print('WARNING: No resonance fields at %g GHz between %g and %g mT.\n'
              '         Check field range and spectrometer frequency.' % (_num(Exp.mwFreq), Range[0], Range[1]))
        e = np.zeros((0, 0))
        return e, e, e, e, e
    if not computeNonEquiPops and np.any(np.real(Idat) < 0):
        F.logmsg(-np.inf, '*********** Negative intensity encountered in resfields!! Please report! **********')
    if np.any(np.asarray(Wdat) < 0):
        F.logmsg(-np.inf, '*********** Negative width encountered in resfields!! Please report! **************')
    if nMaxSegmentsReached > 0:
        F.logmsg(1, '  *** Maximum number of segments (%d) reached for %d orientations!', maxSegments, nMaxSegmentsReached)

    # ---------------------------------------------------------------- perturbational nuclei
    if nPerturbNuclei > 0:
        F.logmsg(1, '  transition post-selection on perturbation nuclei')
        nPertShifts = np.zeros(nPerturbNuclei, dtype=int)
        for iiNuc in range(nPerturbNuclei - 1, -1, -1):
            TotalIntensity = np.sum(pIdatN[iiNuc], axis=(0, 1))
            rmv = TotalIntensity < np.max(TotalIntensity) * _num(Opt.HybridIntThreshold)
            pPdatN[iiNuc] = pPdatN[iiNuc][:, :, ~rmv]
            pIdatN[iiNuc] = pIdatN[iiNuc][:, :, ~rmv]
            nPertShifts[iiNuc] = pPdatN[iiNuc].shape[2]
        F.logmsg(1, '  combining shifts from perturbational nuclei')
        idxComb1, idxComb2 = {}, {}
        for iiNuc in range(nPerturbNuclei - 1, 0, -1):
            nr, nc = nPertShifts[iiNuc], int(np.prod(nPertShifts[:iiNuc]))
            idxComb2[iiNuc] = np.tile(np.arange(nr), nc)      # find(ones(nr, nc)): rows, then columns
            idxComb1[iiNuc] = np.repeat(np.arange(nc), nr)
        nSub = int(np.prod(nPertShifts))
        pPdat = np.zeros((nTransitions, nOrientations, nSub))
        pIdat = np.zeros((nTransitions, nOrientations, nSub))
        for iT in range(nTransitions):
            for iO in range(nOrientations):
                Pc = pPdatN[0][iT, iO, :]
                Ic = pIdatN[0][iT, iO, :]
                for iiNuc in range(1, nPerturbNuclei):
                    Pc = Pc[idxComb1[iiNuc]] + pPdatN[iiNuc][iT, iO, idxComb2[iiNuc]]
                    Ic = Ic[idxComb1[iiNuc]] * pIdatN[iiNuc][iT, iO, idxComb2[iiNuc]]
                pPdat[iT, iO, :] = Pc
                pIdat[iT, iO, :] = Ic
        nSubTransitions = pPdat.shape[2]

        def expand(x):
            return np.repeat(np.asarray(x)[:, :, None], nSubTransitions, axis=2)

        def rearrange(x):
            return x.transpose(0, 2, 1).reshape((x.shape[0] * nSubTransitions, -1), order='F')

        Pdat = rearrange(expand(Pdat) + pPdat)
        Idat = rearrange(expand(Idat) * pIdat)
        if Wdat.size > 0:
            Wdat = rearrange(expand(Wdat))
        if Gdat.size > 0:
            Gdat = rearrange(expand(Gdat))
        Transitions = rearrange(expand(Transitions))

    nResonances = int(np.count_nonzero(~np.isnan(Pdat)))
    F.logmsg(2, '  ## diags %d  resonances %d  oris %d', nDiagonalizations, nResonances, nOrientations)
    if nResonances:
        F.logmsg(1, '  ## diags/field %2.4g  diags/ori %2.4g  fields/ori %2.4g', nDiagonalizations / nResonances,
                 nDiagonalizations / nOrientations, nResonances / nOrientations)
        F.logmsg(2, '  ## Hamiltonian rediagonalised for %d resonances (%0.3f%%)', nRediags, nRediags / nResonances * 100)
    F.logmsg(2, '  ## minimum state stability encountered %f', minStateStability)

    # ---------------------------------------------------------------- multiple sites, sorting
    if not pepperCall and nSites > 1:
        siz = (nTransitions * nSites, Pdat.size // nTransitions // nSites)   # nTransitions as in EasySpin
        Pdat = Pdat.reshape(siz, order='F')
        if Idat.size:
            Idat = Idat.reshape(siz, order='F')
        if Wdat.size:
            Wdat = Wdat.reshape(siz, order='F')
        if Gdat.size:
            Gdat = Gdat.reshape(siz, order='F')
    order = np.lexsort((Transitions[:, 1], Transitions[:, 0]))   # sortrows (stable)
    Transitions = Transitions[order, :]
    idx = order
    if not pepperCall and nSites > 1:
        idx = (order[:, None] + np.arange(nSites)[None, :] * nTransitions).ravel(order='F')
    Pdat = Pdat[idx, :]
    if Idat.size:
        Idat = Idat[idx, :]
    if Wdat.size:
        Wdat = Wdat[idx, :]
    if Gdat.size:   # EasySpin leaves Gdat unsorted, so its rows no longer match Transitions (DEVIATIONS.md, T6)
        Gdat = Gdat[idx, :]
    return Pdat, Idat, Wdat, Transitions, Gdat

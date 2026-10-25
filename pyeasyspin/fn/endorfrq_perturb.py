"""ENDOR frequencies and intensities from 1st/2nd order perturbation theory (1 electron spin)."""
import numpy as np
from .._registry import F, nout
from .._struct import Struct
from .._mat import a2, vec, isempty, numel
from ..config import config


def _c(x):
    return x[0] if isinstance(x, tuple) else x


def _err(err):
    if err:
        raise ValueError(err)


def _any(x):
    return x is not None and bool(np.any(np.asarray(x) != 0))


def _norm(v):
    return np.sqrt(np.sum(np.abs(v) ** 2))


@nout(4)
def endorfrq_perturb(Sys=None, Exp=None, Opt=None):
    """(EndorFreqs, EndorIntensities, Transitions, Info): nuclear transition frequencies (MHz),
    one row per (nucleus, mI pair, mS) and one column per orientation."""
    if Sys is None or Exp is None:
        raise ValueError('Incorrect number of inputs!')
    Opt = Struct() if isempty(Opt) else Struct(Opt)
    if 'Verbosity' not in Opt:
        Opt.Verbosity = 0.0
    config.LogLevel = Opt.Verbosity

    Sys, err = F.validatespinsys(Sys)
    _err(err)
    Sys = Struct(Sys)
    if int(Sys.nElectrons) != 1:
        err = 'Perturbation theory available only for systems with 1 electron.'
    if _any(Sys.AStrain):
        err = 'A strain not supported with perturbation theory. Use matrix diagonalization.'
    if _any(Sys.DStrain):
        err = 'D strain not supported with perturbation theory. Use matrix diagonalization.'
    if 'nn' in Sys and _any(Sys.nn):
        err = 'Nuclear-nuclear couplings not supported with perturbation theory. Use matrix diagonalization.'
    _err(err)

    S = float(vec(np.asarray(Sys.S, dtype=float))[0])
    highSpin = S > 0.5
    SS1 = S * (S + 1)
    n2S1 = int(round(2 * S)) + 1
    mS = -S + np.arange(n2S1)                                   # -S:1:S
    mS2 = mS ** 2
    if Sys.fullg:
        g = a2(Sys.g).astype(float)
    else:
        R_g2M = np.asarray(F.erot(Sys.gFrame)).T
        g = R_g2M @ np.diag(vec(Sys.g)) @ R_g2M.T
    if highSpin:
        if 'D' in Sys:
            R_D2M = np.asarray(F.erot(Sys.DFrame)).T
            D = np.diag(vec(Sys.D))
            D = R_D2M @ D @ R_D2M.T

    I = vec(np.asarray(Sys.I, dtype=float)) if numel(Sys.I) else np.zeros(0)
    nNuclei = int(Sys.nNuclei)
    gn = vec(np.asarray(Sys.gn, dtype=float)) if numel(Sys.gn) else np.zeros(0)
    Field = Exp['Field'] if 'Field' in Exp else None
    nuI = -gn * _c(F.nmagn()) * Field / 1e3 / F.planck() / 1e6
    SysA = a2(Sys.A).astype(float) if nNuclei > 0 else None
    SysQ = a2(Sys.Q).astype(float) if nNuclei > 0 else None
    mI = [None] * nNuclei
    A = [None] * nNuclei
    P = [None] * nNuclei
    for iNuc in range(nNuclei - 1, -1, -1):
        mI[iNuc] = -I[iNuc] + np.arange(int(round(2 * I[iNuc])) + 1)
        if Sys.fullA:
            A[iNuc] = SysA[3 * iNuc:3 * iNuc + 3, :]
        else:
            R_A2M = np.asarray(F.erot(a2(Sys.AFrame)[iNuc, :])).T
            A[iNuc] = R_A2M @ np.diag(SysA[iNuc, :]) @ R_A2M.T
        if Sys.fullQ:
            P[iNuc] = SysQ[3 * iNuc:3 * iNuc + 3, :]
        else:
            R_Q2M = np.asarray(F.erot(a2(Sys.QFrame)[iNuc, :])).T
            P[iNuc] = R_Q2M @ np.diag(SysQ[iNuc, :]) @ R_Q2M.T
        P[iNuc] = P[iNuc] - np.eye(3) * np.trace(P[iNuc]) / 3

    DefaultExp = Struct(SampleFrame=np.zeros((1, 3)), CrystalSymmetry=1.0, MolFrame=np.zeros((1, 3)),
                        SampleRotation=np.zeros((0, 0)))
    if 'Field' not in Exp:
        raise ValueError('Exp.Field is missing.')
    Exp = Struct(F.adddefaults(Exp, DefaultExp))
    err = ''
    if 'Detection' in Exp:
        raise ValueError('Exp.Detection is obsolete. Use Exp.mwMode instead.')
    if 'mwMode' in Exp:
        if isinstance(Exp.mwMode, str) and Exp.mwMode == 'parallel':
            err = 'Parallel mode ENDOR not supported with perturbation theory. Use matrix diagonalization.'
    if 'Temperature' in Exp:
        if np.all(np.asarray(Exp.Temperature, dtype=float) < np.inf) and numel(Exp.Temperature) > 0:
            err = 'ENDOR temperature effects are not supported with perturbation theory. Use matrix diagonalization.'
    _err(err)
    if 'ExciteWidth' not in Exp:
        Exp.ExciteWidth = np.inf
    OrientationSelection = not np.isinf(Exp.ExciteWidth)
    lwExcite2 = Exp.ExciteWidth ** 2
    if 'Sites' not in Opt:
        Opt.Sites = np.zeros((0, 0))
    out = F.p_crystalorientations(Exp, Opt)
    Orientations, nOrientations, AverageOverChi = out[0], out[1], out[3]
    Orientations = a2(np.asarray(Orientations, dtype=float))
    nOrientations = int(nOrientations)
    if 'Transitions' in Opt and not isempty(Opt.Transitions):
        raise ValueError('ENDOR perturbation theory does not support Opt.Transitions.')
    if 'PerturbOrder' not in Opt:
        Opt.PerturbOrder = 2.0
    secondOrder = Opt.PerturbOrder == 2
    F.logmsg(1, '  2nd order perturbation theory' if secondOrder else '  1st order perturbation theory')
    if Opt.PerturbOrder != 1 and Opt.PerturbOrder != 2:
        raise ValueError('Opt.PerturbOrder must be either 1 or 2.')
    if 'Enhancement' not in Opt:
        Opt.Enhancement = 0.0
    if np.all(np.asarray(Opt.Enhancement) != 0) if not isinstance(Opt.Enhancement, str) else bool(Opt.Enhancement):
        F.logmsg(1, '  skipping hyperfine enhancement.')
    F.logmsg(1, '  hyperfine enhancement: off.')
    if 'Nuclei' not in Opt:
        Opt.Nuclei = np.zeros((0, 0))
    if isempty(Opt.Nuclei):
        Opt.Nuclei = np.arange(1, nNuclei + 1, dtype=float).reshape(1, -1)
    Nuclei = [int(v) for v in vec(Opt.Nuclei)]
    if any(v < 1 for v in Nuclei) or any(v > nNuclei for v in Nuclei):
        raise ValueError('Opt.Nuclei is out of range.')

    hasQuadrupole = I > 0.5
    detA = np.zeros(nNuclei)
    invA = [None] * nNuclei
    trAA = np.zeros(nNuclei)
    for iNuc in range(nNuclei - 1, -1, -1):
        A_ = A[iNuc]
        detA[iNuc] = np.linalg.det(A_)
        invA[iNuc] = np.linalg.inv(A_)
        trAA[iNuc] = np.trace(A_.conj().T @ A_)
    II1 = I * (I + 1)
    if highSpin:
        trDD = np.trace(D @ D)
    states = np.rint(a2(F.allcombinations(*mI, 'i'))).astype(int) - 1      # 0-based
    if OrientationSelection:
        maxEPRfreq = -np.inf
        minEPRfreq = np.inf
        minExWidth = np.inf
    HS = vec(np.asarray(Sys.HStrain, dtype=float))

    EndorFrequencies = [np.zeros((nOrientations, n2S1 * (mI[i].size - 1))) for i in range(nNuclei)]
    NuclearEnergyShifts = [np.zeros((n2S1, mI[i].size)) for i in range(nNuclei)]
    totalWeight = [np.zeros((nOrientations, mI[i].size - 1)) for i in range(nNuclei)]
    for iOri in range(nOrientations - 1, -1, -1):
        h = vec(F.erot(Orientations[iOri, :], 'rows', nargout=3)[2])
        geff = _norm(g @ h)
        E0 = geff * _c(F.bmagn()) * Field * 1e-3 / F.planck() / 1e6
        u = g @ h / geff
        if highSpin:
            Du = D @ u
            uDu = u @ Du
            E1D_ = -uDu / 2 * (SS1 - 3 * mS2)
            if secondOrder:
                uDDu = Du @ Du
                D1sq = uDDu - uDu ** 2
                D2sq = 2 * trDD + uDu ** 2 - 4 * uDDu
                E2D_ = D1sq * mS * (4 * SS1 - 8 * mS2 - 1) - D2sq / 4 * mS * (2 * SS1 - 2 * mS2 - 1)
                E2D_ = -E2D_ / (2 * E0)
            else:
                E2D_ = np.zeros(mS.size)
        else:
            E1D_ = np.array([0.0, 0.0])
            E2D_ = np.array([0.0, 0.0])
        ElectronEnergies = mS * E0 + E1D_ + E2D_
        for iNuc in range(nNuclei):
            mI_ = mI[iNuc]
            Ah = A[iNuc] @ g / geff @ h
            for imS in range(n2S1):
                mS_ = mS[imS]
                HyperfineField = mS_ * Ah
                LarmorField = nuI[iNuc] * h
                Kh = HyperfineField + LarmorField
                normKh = _norm(Kh)
                k = Kh / normKh
                E1A_ = mI_ * normKh
                strongCouplingCase = _norm(HyperfineField) > _norm(LarmorField)
                if strongCouplingCase and mS_ < 0:
                    E1A_ = -E1A_
                if hasQuadrupole[iNuc]:
                    kPk = k @ P[iNuc] @ k
                    E1Q_ = -kPk / 2 * (II1[iNuc] - 3 * mI_ ** 2)
                else:
                    E1Q_ = 0.0
                if secondOrder:
                    Ak = A[iNuc].T @ k
                    kAu = Ak @ u
                    kAAk = Ak @ Ak
                    A1sq = kAAk - kAu ** 2
                    A2 = detA[iNuc] * (u @ invA[iNuc] @ k)
                    A3 = trAA[iNuc] - _norm(A[iNuc] @ u) ** 2 - kAAk + kAu ** 2
                    E2A_ = (A1sq * mS_ * mI_ ** 2
                            - A2 * (SS1 - mS_ ** 2) * mI_
                            + A3 / 2 * mS_ * (II1[iNuc] - mI_ ** 2))
                    E2A_ = E2A_ / (2 * E0)
                    if strongCouplingCase and mS_ < 0:
                        E2A_ = -E2A_
                    if highSpin:
                        DA = Du @ Ak - uDu * kAu
                        E2DA = DA * (SS1 - 3 * mS_ ** 2) * mI_
                        E2DA = -E2DA / E0
                    else:
                        E2DA = 0.0
                    if hasQuadrupole[iNuc]:
                        kPPk = k @ (P[iNuc] @ P[iNuc]) @ k
                        P1sq = kPPk - kPk ** 2
                        # EasySpin: 2*trace(P)^2, which is 0 for the traceless P; the D analogue
                        # (D2sq) uses trace(D^2) (DEVIATIONS.md)
                        P2sq = 2 * np.trace(P[iNuc] @ P[iNuc]) + kPk ** 2 - 4 * kPPk
                        E2Q_ = (P1sq * mI_ * (4 * II1[iNuc] - 8 * mI_ ** 2 - 1)
                                - P2sq / 4 * mI_ * (2 * II1[iNuc] - 2 * mI_ ** 2 - 1))
                        E2Q_ = -E2Q_ / (2 * normKh)
                    else:
                        E2Q_ = 0.0
                    NuclearEnergyShifts[iNuc][imS, :] = E1Q_ + E1A_ + E2A_ + E2DA + E2Q_
                else:
                    NuclearEnergyShifts[iNuc][imS, :] = E1Q_ + E1A_
        for iNuc in range(nNuclei):
            dE_ = np.abs(np.diff(NuclearEnergyShifts[iNuc], axis=1))
            EndorFrequencies[iNuc][iOri, :] = vec(dE_)
        if OrientationSelection:
            nSt = states.shape[0]
            EnergyShifts = np.zeros((nSt, nNuclei))
            TotalEnergy = np.zeros((nSt, n2S1))
            for imS in range(n2S1):
                for iNuc in range(nNuclei):
                    EnergyShifts[:, iNuc] = NuclearEnergyShifts[iNuc][imS, states[:, iNuc]]
                TotalEnergy[:, imS] = ElectronEnergies[imS] + np.sum(EnergyShifts, axis=1)
            EPRfreq = np.diff(TotalEnergy, axis=1)
            lw2 = np.sum((HS * h) ** 2) + lwExcite2
            offsetFreq = EPRfreq - Exp.mwFreq * 1e3
            exciteFactor = np.exp(-2 / lw2 * offsetFreq ** 2)
            if np.min(EPRfreq) < minEPRfreq:
                minEPRfreq = np.min(EPRfreq)
            if np.max(EPRfreq) > maxEPRfreq:
                maxEPRfreq = np.max(EPRfreq)
            if lw2 < minExWidth ** 2:
                minExWidth = np.sqrt(lw2)
            ef = vec(exciteFactor)
            for iNuc in range(nNuclei):
                for imI in range(mI[iNuc].size - 1):
                    idx = (states[:, iNuc] == imI) | (states[:, iNuc] == imI + 1)
                    # MATLAB logical indexing with a mask shorter than the array: linear, first nSt elements
                    totalWeight[iNuc][iOri, imI] = np.sum(ef[:idx.size][idx])

    if OrientationSelection:
        Selectivity = (maxEPRfreq - minEPRfreq) / minExWidth
        F.logmsg(2, '  limited excitation width, selectivity %g', Selectivity)
    else:
        Selectivity = 0.0
        F.logmsg(2, '  infinite excitation width')
    Info = Struct(Selectivity=Selectivity)
    EndorFreqs = [EndorFrequencies[i - 1].T for i in Nuclei]
    EndorFreqs = np.vstack(EndorFreqs) if EndorFreqs else np.zeros((0, 0))
    EndorIntensities = []
    nNucStates = 2 * I + 1
    for iNuc in Nuclei:
        i = iNuc - 1
        n2I = int(round(2 * I[i]))
        matrixelement = np.sqrt(np.arange(1, n2I + 1) * (2 * I[i] - np.arange(n2I))) / 2
        pre = gn[i] * _c(F.nmagn()) / F.planck() / 1e6 / 1e3
        pre = (pre * matrixelement) ** 2
        newIntensities = pre.reshape(-1, 1) / nNucStates[i]
        newIntensities = np.tile(newIntensities, (1, nOrientations))
        if OrientationSelection:
            newIntensities = newIntensities * totalWeight[i].T
        idx = np.repeat(np.arange(n2I), n2S1)
        newIntensities = newIntensities[idx, :]
        EndorIntensities.append(newIntensities)
    EndorIntensities = np.vstack(EndorIntensities) if EndorIntensities else np.zeros((0, 0))
    EndorIntensities = EndorIntensities * np.prod(2 * I + 1)
    Transitions = np.zeros((0, 0))
    return EndorFreqs, EndorIntensities, Transitions, Info

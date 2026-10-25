"""Computation of magnetometry data (magnetic moment, susceptibility)"""
import numpy as np
from .._registry import F, nout
from .._struct import Struct, isfield
from .._mat import a2, vec, isempty
from ..config import config

_KEYWORDS = ('mu', 'mumol', 'muBM', 'mueff', 'muvec', 'chi', 'chimol', 'chimolT', '1/chimol')


def _c(name):
    v = getattr(F, name)()
    return v[0] if isinstance(v, tuple) else v


def _dense(M):
    return M.toarray() if hasattr(M, 'toarray') else np.asarray(M)


@nout(-1)
def curry(Sys=None, Exp=None, Opt=None, nargout=None):
    """curry(Sys, Exp, Opt) -> outputs named in Opt.Output (one value, or a tuple in keyword order).
    Without Opt.Output, nargout selects 'muBM' (1) or 'muBM chimol' (2, the default here).
    Each output is nFields x nTemperatures ('muvec': 3*nFields x nTemperatures). SI units unless
    Opt.Units = 'CGS'. No plotting (MATLAB's nargout=0 plot is not ported; nargout=0 returns None)."""
    if nargout is None:
        nargout = len(str(Opt['Output']).split()) if isinstance(Opt, dict) and 'Output' in Opt else 2
    if Exp is None:
        Exp = Struct()
    if Opt is None:
        Opt = Struct()
    if not isinstance(Opt, dict):
        raise ValueError('Opt (third input argument) must be a structure or a string.')
    Exp = Struct(Exp) if isinstance(Exp, dict) else Exp
    Opt = Struct(Opt)
    if not isfield(Opt, 'Verbosity'):
        Opt.Verbosity = 0
    config.LogLevel = Opt.Verbosity
    F.logmsg(1, '=begin=curry======' + __import__('datetime').datetime.now().strftime('%d-%b-%Y %H:%M:%S')
             + '=================')

    doPlot = nargout == 0

    F.logmsg(1, 'Spin system')
    if isinstance(Sys, list):
        raise ValueError('curry does not support calculations with multiple components.')
    Sys = Struct(Sys)
    if not isfield(Sys, 'TIP'):
        Sys.TIP = 0

    F.logmsg(1, 'Experimental parameters')
    DefaultExp = Struct()
    DefaultExp.SampleFrame = np.zeros((0, 0))
    DefaultExp.CrystalSymmetry = np.zeros((0, 0))
    DefaultExp.MolFrame = np.zeros((0, 0))
    DefaultExp.SampleRotation = np.zeros((0, 0))
    DefaultExp.Ordering = np.zeros((0, 0))
    Exp = F.adddefaults(Exp, DefaultExp)

    if isfield(Exp, 'Orientations'):
        raise ValueError('Exp.Orientations is no longer supported, use Exp.SampleFrame instead.')
    if isfield(Exp, 'CrystalOrientation'):
        raise ValueError('Exp.CrystalOrientation is no longer supported, use Exp.SampleFrame instead.')
    if isfield(Exp, 'lightBeam') and not isempty(Exp.lightBeam):
        raise ValueError('Photoselection (via Exp.lightBeam) is not supported.')

    if not isfield(Exp, 'Field'):
        Exp.Field = 0
        print('Exp.Field is missing, assuming zero field.')
    B = vec(np.asarray(Exp.Field, dtype=float)) / 1e3
    nFields = B.size
    F.logmsg(1, '  number of field values: %d', nFields)

    if not isfield(Exp, 'Temperature'):
        raise ValueError('Exp.Temperature is missing.')
    T = vec(np.asarray(Exp.Temperature, dtype=float)).reshape(1, -1)
    nTemperatures = T.size
    F.logmsg(1, '  number of temperature values: %d', nTemperatures)
    if np.any(T < 0):
        raise ValueError('Negative temperatures are not possible.')
    zeroTemp = bool(np.any(T == 0))

    Exp, Opt = F.p_sampletype(Exp, Opt)
    Exp, Opt = Struct(Exp), Struct(Opt)
    disorderedSample = bool(Opt.disorderedSample)
    crystalSample = bool(Opt.crystalSample)
    if bool(Opt.partiallyOrderedSample):
        raise ValueError('Partially ordered samples are not supported in curry().')
    F.logmsg(1, '  powder calculation' if disorderedSample else '  crystal calculation')

    F.logmsg(1, 'Options')
    if isfield(Opt, 'nKnots'):
        raise ValueError('Options.nKnots is obsolete. Use Options.GridSize instead, e.g. Options.GridSize = 91.')
    if not isfield(Opt, 'GridSize'):
        Opt.GridSize = 10
    F.logmsg(1, '  grid size: %d', np.ravel(Opt.GridSize)[0])
    if not isfield(Opt, 'GridSymmetry'):
        Opt.GridSymmetry = np.zeros((0, 0))
    if not isfield(Opt, 'GridFrame'):
        Opt.GridFrame = np.zeros((0, 0))
    if not isfield(Opt, 'deltaB'):
        Opt.deltaB = 1
    dB = Opt.deltaB * 1e-3

    if not isfield(Opt, 'Output'):
        if nargout == 0:
            Opt.Output = ''
        elif nargout == 1:
            Opt.Output = 'muBM'
        elif nargout == 2:
            Opt.Output = 'muBM chimol'
        else:
            raise ValueError('Incorrect number of outputs (1 or 2 expected if Opt.Output is not given.')
    F.logmsg(1, '  output: %s', Opt.Output)

    calculateMu = nargout == 0
    calculateChi = nargout == 0 or nargout > 1
    calculateMuVec = False
    keywords = Opt.Output.split() if not isempty(Opt.Output) else []
    for k in keywords:
        if k in ('mu', 'mumol', 'muBM'):
            calculateMu = True
        elif k == 'mueff':
            calculateChi = True
        elif k == 'muvec':
            calculateMu = True
            calculateMuVec = True
        elif k in ('chi', 'chimol', 'chimolT', '1/chimol'):
            calculateChi = True
        else:
            raise ValueError("'%s' keyword in Opt.Output is not known." % k)
    F.logmsg(1, '  number of outputs: %d', len(keywords))
    if calculateMuVec and doPlot:
        raise ValueError('Cannot plot results when calculating full magentic moment vector.')

    if not isfield(Opt, 'Units'):
        Opt.Units = 'SI'
    units = Opt.Units.upper() if isinstance(Opt.Units, str) else Opt.Units
    if units == 'SI':
        useCGSunits = False
    elif units == 'CGS':
        useCGSunits = True
    else:
        raise ValueError("Unknown units '%s' in Opt.Units. Use either 'SI' or 'CGS'." % (Opt.Units,))
    F.logmsg(1, ' units: %s', units)

    if not isfield(Opt, 'Method'):
        Opt.Method = 'operator'
    method = Opt.Method.lower() if isinstance(Opt.Method, str) else Opt.Method
    if method == 'operator':
        useOperatorMethod = True
    elif method == 'partitionfunction':
        useOperatorMethod = False
    else:
        raise ValueError("Opt.Method can be 'operator' or 'partitionfunction'!")
    F.logmsg(1, '  calculation method: %s', Opt.Method)

    if not isfield(Opt, 'Spins'):
        Opt.Spins = np.zeros((0, 0))
    elif not isempty(Opt.Spins):
        nS = np.size(Sys.S)
        sp = np.asarray(Opt.Spins)
        if np.any(sp < 1) or np.any(sp > nS):
            raise ValueError('Opt.Spins must contain electron spin indices between 1 and %d.' % nS)
        if not useOperatorMethod:
            raise ValueError("To use Opt.Spins, you must use Opt.Method = 'operator'.")

    F.logmsg(1, '-Hamiltonian-------------------------------------------')
    H0, muOpxM, muOpyM, muOpzM = F.ham(Sys, nargout=4)
    if not isempty(Opt.Spins):
        muOpxM, muOpyM, muOpzM = F.ham_ez(Sys, Opt.Spins, nargout=3)
    H0 = _dense(H0) * 1e6 * _c('planck')
    c = 1e6 * 1e3 * _c('planck')
    muOpxM = _dense(muOpxM) * c
    muOpyM = _dense(muOpyM) * c
    muOpzM = _dense(muOpzM) * c

    Exp, Opt, _ = F.p_gridsetup(Sys, Exp, Opt)
    Exp, Opt = Struct(Exp), Struct(Opt)
    Orientations, nOrientations, nSites, _ = F.p_crystalorientations(Exp, Opt)
    Orientations = a2(np.asarray(Orientations, dtype=float))
    nOrientations = int(nOrientations)
    nSites = int(nSites)
    OriWeights = vec(np.asarray(Exp.OriWeights, dtype=float)) / 4 / np.pi
    Exp.OriWeights = OriWeights

    with np.errstate(divide='ignore'):
        beta = 1 / T / _c('boltzm')        # 1 x nT (Inf at T = 0, as in MATLAB)
    muz = np.zeros((nFields, nTemperatures))
    if calculateMuVec:
        mux = np.zeros((nFields, nTemperatures))
        muy = np.zeros((nFields, nTemperatures))
    if calculateChi:
        chizz = np.zeros((nFields, nTemperatures))

    def calcmu(Vecs, muOp, pop):
        d = np.real(np.einsum('ij,ik,kj->j', Vecs.conj(), muOp, Vecs))
        return (d @ pop) / np.sum(pop, axis=0)

    def getmuproj(nM):
        nM = vec(nM)
        return nM[0] * muOpxM + nM[1] * muOpyM + nM[2] * muOpzM

    def eigh(M):
        return np.linalg.eigh(M)

    F.logmsg(1, '-orientation loop--------------------------------------')
    if crystalSample:
        F.logmsg(1, '  looping over %d orientations and %d sites', nOrientations / nSites, nSites)
    else:
        F.logmsg(1, '  looping over %d orientations', nOrientations)

    for iOri in range(nOrientations):
        xL_M, yL_M, zL_M = F.erot(Orientations[iOri, :].reshape(1, -1), 'rows', nargout=3)
        muOpzL = getmuproj(zL_M)
        if calculateMuVec:
            muOpxL = getmuproj(xL_M)
            muOpyL = getmuproj(yL_M)
        for iB in range(nFields):
            if useOperatorMethod:
                E, V = eigh(H0 - B[iB] * muOpzL)
                with np.errstate(invalid='ignore', over='ignore'):
                    populations = np.exp(-(E - E[0]).reshape(-1, 1) * beta)
                if zeroTemp:
                    populations[0, :] = 1    # EasySpin: populations(1) = 1 (first column only; DEVIATIONS)
                muz_ = calcmu(V, muOpzL, populations)
                if calculateMuVec:
                    mux_ = calcmu(V, muOpxL, populations)
                    muy_ = calcmu(V, muOpyL, populations)
                if calculateChi:
                    B_ = B[iB] + dB
                    E, V = eigh(H0 - B_ * muOpzL)
                    with np.errstate(invalid='ignore', over='ignore'):
                        populations = np.exp(-(E - E[0]).reshape(-1, 1) * beta)
                    if zeroTemp:
                        populations[0, :] = 1    # EasySpin: populations(1) = 1 (first column only; DEVIATIONS)
                    muz2_ = calcmu(V, muOpzL, populations)
                    chizz_ = (muz2_ - muz_) / dB
            else:
                def lnZ(E, Emin):
                    with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
                        return np.log(np.sum(np.exp(-(E - Emin).reshape(-1, 1) * beta), axis=0))
                E1 = eigh(H0 - (B[iB] - dB) * muOpzL)[0]
                E3 = eigh(H0 - (B[iB] + dB) * muOpzL)[0]
                Emin = min(np.min(E1), np.min(E3))
                lnZ1 = lnZ(E1, Emin)
                lnZ3 = lnZ(E3, Emin)
                if calculateChi:
                    E2 = eigh(H0 - B[iB] * muOpzL)[0]
                    lnZ2 = lnZ(E2, Emin)
                if calculateMu:
                    muz_ = (lnZ3 - lnZ1) / (2 * dB) / beta.ravel()
                if calculateChi:
                    chizz_ = (lnZ3 - 2 * lnZ2 + lnZ1) / (dB ** 2) / beta.ravel()
            if crystalSample:
                w = 1 / nSites
            else:
                w = OriWeights[iOri]
            if calculateMu:
                muz[iB, :] = muz[iB, :] + w * muz_
            if calculateMuVec:
                mux[iB, :] = mux[iB, :] + w * mux_
                muy[iB, :] = muy[iB, :] + w * muy_
            if calculateChi:
                chizz[iB, :] = chizz[iB, :] + w * chizz_

    F.logmsg(1, '  performing unit conversions')
    if calculateMu:
        muz_SI = muz
        if useCGSunits:
            muz_CGS = muz_SI / 1e-3
    if calculateMuVec:
        mux_SI = mux
        muy_SI = muy
        if useCGSunits:
            mux_CGS = mux_SI / 1e-3
            muy_CGS = muy_SI / 1e-3
    if calculateChi:
        chizz_SI = chizz * _c('mu0') + Sys.TIP / _c('avogadro')
        if useCGSunits:
            chizz_CGS = chizz_SI / (4 * np.pi * 1e-6)

    if useCGSunits:
        if calculateMu:
            muz = muz_CGS
        if calculateMuVec:
            mux, muy = mux_CGS, muy_CGS
        if calculateChi:
            chizz = chizz_CGS
        muB = _c('bmagn') / 1e-3
        kB = _c('boltzm') / 1e-7
        cc = 3 * kB / muB ** 2
    else:
        if calculateMu:
            muz = muz_SI
        if calculateMuVec:
            mux, muy = mux_SI, muy_SI
        if calculateChi:
            chizz = chizz_SI
        muB = _c('bmagn')
        kB = _c('boltzm')
        cc = 3 * kB / muB ** 2 / _c('mu0')

    varargout = [None] * len(keywords)
    for n in range(len(keywords) - 1, -1, -1):
        k = keywords[n]
        if k == 'mu':
            outval = muz
        elif k == 'mumol':
            outval = muz * _c('avogadro')
        elif k == 'muBM':
            outval = muz / muB
        elif k == 'muvec':
            outval = np.vstack([mux, muy, muz])
        elif k == 'chi':
            outval = chizz
        elif k == 'chimol':
            outval = chizz * _c('avogadro')
        elif k == 'chimolT':
            outval = chizz * _c('avogadro') * np.tile(T, (nFields, 1))
        elif k == '1/chimol':
            with np.errstate(divide='ignore'):
                outval = 1 / (chizz * _c('avogadro'))
        elif k == 'mueff':
            outval = np.sqrt(chizz * np.tile(T, (nFields, 1)) * cc)
        else:
            raise ValueError('Keyword %s in Opt.Output is unknown.' % k)
        varargout[n] = outval

    F.logmsg(1, '=end=curry========%s=================\n',
             __import__('datetime').datetime.now().strftime('%d-%b-%Y %H:%M:%S'))
    config.LogLevel = None
    if not varargout:
        return None
    return varargout[0] if len(varargout) == 1 else tuple(varargout)

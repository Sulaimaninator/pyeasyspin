"""Least-squares fitting for EPR and other data (headless: the esfit GUI is not ported)."""
import inspect
import numbers
import time

import numpy as np
from scipy.special import erfcinv

from .._registry import F, nout
from .._struct import Struct
from .._mat import isempty, vec
from .._parallel import ModelCache

_ALGNAMES = ['Nelder-Mead simplex', 'Levenberg-Marquardt', 'Monte Carlo', 'genetic algorithm',
             'grid search', 'particle swarm', 'lsqnonlin']
_ALGABBREV = ['simplex', 'levmar', 'montecarlo', 'genetic', 'grid', 'swarm', 'lsqnonlin']
_TARGETNAMES = ['data as is', 'integral', 'double integral', 'derivative', 'Fourier transform']
_TARGETABBREV = ['fcn', 'int', 'dint', 'diff', 'fft']
_BLSETTINGS = [-1, 0, 1, 2, 3]
_BLSTRINGS = ['none', 'offset', 'linear', 'quadratic', 'cubic']

# Keep the finite-difference stencil of the uncertainty Jacobian inside [lb, ub] (DEVIATIONS.md).
# False reproduces EasySpin's unbounded stencil exactly (used by the parity tests).
BOUNDED_FD = True


def _isnum(v):
    return isinstance(v, (numbers.Number, np.number)) and not isinstance(v, (bool, np.bool_))


def _iscell(v):
    """A list/tuple is a MATLAB cell unless it is a plain numeric (nested) list."""
    if not isinstance(v, (list, tuple)):
        return False
    if len(v) == 0:
        return True

    def numeric(x):
        return _isnum(x) or (isinstance(x, (list, tuple)) and len(x) > 0 and all(numeric(y) for y in x))
    return not all(numeric(x) for x in v)


def _lsq(A, b):
    """MATLAB A\\b for a tall matrix (least squares)."""
    return np.linalg.lstsq(A, b, rcond=None)[0]


def _accepts_nargout(f):
    try:
        return 'nargout' in inspect.signature(f).parameters
    except (TypeError, ValueError):
        return False


@nout(1)
def esfit(data=None, fcn=None, p0=None, *varargin, rng=None):
    """result = esfit(data, fcn, p0, pvary[, Opt]) or esfit(data, fcn, p0, lb, ub[, Opt]).

    data: vector or list of vectors (global fit). fcn: model function; with p0 a list such as
    [Sys, Exp] or [Sys, Exp, Opt] it is called as fcn(*args) (EasySpin functions: pepper, garlic,
    chili, salt, recognised by __name__), with p0 a numeric vector as fcn(p) with p 1-D.
    fcn may return the data, a list of datasets, or a tuple of outputs (the last one is used,
    as MATLAB uses the last declared output; for EasySpin functions the spectrum).
    rng: numpy Generator/RandomState for the stochastic algorithms and RandomStart (default the
    global numpy stream, so np.random.seed(s) reproduces MATLAB rng(s) wherever only rand is
    used). Headless: no GUI, no plot. Returns the result Struct of EasySpin's esfit."""
    nargin = 3 + len(varargin)
    if nargin == 4:
        varyProvided = True
        pvary = varargin[0]
        Opt = Struct()
    elif nargin == 5:
        if isinstance(varargin[1], dict):
            varyProvided = True
            pvary = varargin[0]
            Opt = varargin[1]
        else:
            varyProvided = False
            lb, ub = varargin[0], varargin[1]
            Opt = Struct()
    elif nargin == 6:
        varyProvided = False
        lb, ub, Opt = varargin
    else:
        raise ValueError(('You provided %d inputs, but esfit requires 4, 5, or 6.\nExamples:\n'
                          '   esfit(data, fcn, p0, pvary)\n   esfit(data, fcn, p0, pvary, Opt)\n'
                          '   esfit(data, fcn, p0, lb, ub)\n   esfit(data, fcn, p0, lb, ub, Opt)') % nargin)
    if isempty(Opt):
        Opt = Struct()
    elif not isinstance(Opt, dict):
        raise ValueError('Opt (last input argument) must be a structure.')
    Opt = Struct(Opt)

    S = Struct()                     # replaces MATLAB's global esfitdata
    S.currFitSet = None
    S.UserCommand = 0
    argspar = F.esfit_argsparams()
    if not _iscell(data):
        data = [data]
    data_vec = []
    datasize = []
    for d in data:
        a = np.asarray(d)
        if a.dtype.kind not in 'fiuc' or a.size == 0 or (a.ndim > 1 and sorted(a.shape)[-2] != 1):
            raise ValueError('First input must be numeric experimental data in the form of a vector '
                             'or a cell array of vectors.')
        if np.any(np.isnan(a)):
            raise ValueError('The experimental data must not contain NaN.')
        data_vec.append(vec(a))
        datasize.append(a.size)
    data_vec = np.concatenate(data_vec)
    if data_vec.dtype.kind in 'iu':
        data_vec = data_vec.astype(float)
    idx_ = np.concatenate(([0], np.cumsum(datasize)))
    idx = []
    for i in range(len(data)):
        m = np.zeros(data_vec.size, dtype=bool)
        m[idx_[i]:idx_[i + 1]] = True
        idx.append(m)
    S.data = data_vec
    S.idx = idx
    S.nDataSets = len(data)
    S.datasize = datasize
    if not callable(fcn):
        s = 'The simulation/model function (2nd input) must be a function handle.'
        if isinstance(fcn, str):
            raise ValueError("%s\nUse esfit(data,@%s,...) instead of esfit(data,'%s',...)." % (s, fcn, fcn))
        raise ValueError('%s\nFor example, to use the function pepper(...), use esfit(data,@pepper,...).' % s)
    S.fcn = fcn
    S.fcnName = getattr(fcn, '__name__', type(fcn).__name__)
    S.lastSetID = 0
    EasySpinFunction = S.fcnName in ('pepper', 'garlic', 'chili', 'salt')
    cellInputs = _iscell(p0)
    S.cellInputs = cellInputs
    if isinstance(p0, dict):
        raise ValueError('The third input must be a cell array such as {Sys0,Exp} (when using an '
                         'EasySpin function) or an array of initial parameter values (when using a custom function).')
    if cellInputs:
        p0 = list(p0)
        argspar.validargs(p0)
        S.nSystems = len(p0[0]) if _iscell(p0[0]) else 1
        if varyProvided:
            pinfo = argspar.getparaminfo(pvary)
            argspar.checkparcompatibility(pinfo, p0)
            pvec_0 = argspar.getparamvalues(p0, pinfo)
            pvec_vary = argspar.getparamvalues(pvary, pinfo)
            pvec_lb = pvec_0 - pvec_vary
            pvec_ub = pvec_0 + pvec_vary
            nonnegFieldNames = ['lw', 'lwpp', 'HStrain', 'gStrain', 'DStrain', 'AStrain', 'weight', 'initState']
            ind = np.array([p.FieldName in nonnegFieldNames for p in pinfo], dtype=bool) & (pvec_lb < 0)
            pvec_lb[ind] = 0
        else:
            pinfo = argspar.getparaminfo(lb)
            argspar.checkparcompatibility(pinfo, p0)
            argspar.checkparcompatibility(pinfo, ub)
            pvec_0 = argspar.getparamvalues(p0, pinfo)
            pvec_lb = argspar.getparamvalues(lb, pinfo)
            pvec_ub = argspar.getparamvalues(ub, pinfo)
    else:
        if varyProvided:
            pvec_0 = vec(np.asarray(p0, dtype=float))
            pvec_vary = vec(np.asarray(pvary, dtype=float))
            pvec_lb = pvec_0 - pvec_vary
            pvec_ub = pvec_0 + pvec_vary
        else:
            pvec_0 = np.asarray(p0, dtype=float)
            pvec_lb = np.asarray(lb, dtype=float)
            pvec_ub = np.asarray(ub, dtype=float)
        pinfo = [Struct(Name='p(%d)' % (k + 1)) for k in range(np.size(p0))]
    pvec_0 = vec(np.asarray(pvec_0, dtype=float))
    pvec_lb = vec(np.asarray(pvec_lb, dtype=float))
    pvec_ub = vec(np.asarray(pvec_ub, dtype=float))
    nParams = pvec_0.size
    if pvec_lb.size != nParams:
        raise ValueError('Vector of lower bounds has %d elements, but %d are expected.' % (pvec_lb.size, nParams))
    if pvec_ub.size != nParams:
        raise ValueError('Vector of upper bounds has %d elements, but %d are expected.' % (pvec_lb.size, nParams))
    for test, msg in ((pvec_lb > pvec_ub, 'upper bound cannot be smaller than lower bound.'),
                      (pvec_0 < pvec_lb, 'start value is smaller than lower bound.'),
                      (pvec_0 > pvec_ub, 'start value is larger than upper bound.')):
        if np.any(test):
            raise ValueError('Parameter #%d: %s' % (int(np.flatnonzero(test)[0]) + 1, msg))
    keep = pvec_lb != pvec_ub
    pinfo = [p for p, k in zip(pinfo, keep) if k]
    pvec_0 = pvec_0[keep]
    pvec_lb = pvec_lb[keep]
    pvec_ub = pvec_ub[keep]
    nParameters = pvec_0.size
    if nParameters == 0:
        raise ValueError('No variable parameters to fit.')
    S.args = p0
    S.pinfo = pinfo
    S.pvec_0 = pvec_0
    S.p_start = pvec_0
    S.pvec_lb = pvec_lb
    S.pvec_ub = pvec_ub
    S.nParameters = nParameters
    S.fixedParams = np.zeros(nParameters, dtype=bool)
    if EasySpinFunction:
        if len(data) > 1:
            raise ValueError('Cannot use EasySpin functions for global fitting directly. Write a custom model function.')
        if not cellInputs or len(p0) < 2:
            raise ValueError('The third input must contain the initial parameters, e.g. {Sys0,Exp} or {Sys0,Exp,Opt}.')
        Exp = Struct(p0[1])
        if 'nPoints' in Exp:
            if Exp.nPoints != np.size(data[0]):
                raise ValueError('Exp.nPoints is %d, but the data vector has %d elements.'
                                 % (Exp.nPoints, np.size(data[0])))
        else:
            Exp.nPoints = np.size(data[0])
        p0[1] = Exp
        if not any(f in Exp for f in ('Range', 'CenterSweep', 'mwRange', 'mwCenterSweep')):
            raise ValueError('Please specify field or frequency range, in Exp.Range/Exp.mwRange or in '
                             'Exp.CenterSweep/Exp.mwCenterSweep.')
        if 'mwRange' in Exp or 'mwCenterSweep' in Exp:
            rangefield, centersweepfield = 'mwRange', 'mwCenterSweep'
        else:
            rangefield, centersweepfield = 'Range', 'CenterSweep'
        if centersweepfield in Exp and np.all(~np.isnan(vec(Exp[centersweepfield]))):
            cs = vec(np.asarray(Exp[centersweepfield], dtype=float))
            range_ = cs[0] + np.array([-1, 1]) / 2 * cs[1]
        else:
            range_ = vec(np.asarray(Exp[rangefield]))
        if (np.any(range_ < 0) or np.diff(range_.real)[0] <= 0 or np.any(~np.isfinite(range_))
                or np.iscomplexobj(range_)):
            raise ValueError('Invalid sweep range! Check Exp.(mw)CenterSweep or Exp.(mw)Range.')
        Opt.x = np.linspace(range_[0], range_[1], int(Exp.nPoints))
    if cellInputs:
        p0_fixed = p0
        S.p2args = lambda pars: argspar.setparamvalues(p0_fixed, pinfo, pars)
    if 'Verbosity' not in Opt:
        Opt.Verbosity = 1
    if 'Scaling' in Opt:
        raise ValueError('Fitting option Opt.Scaling has been replaced by Opt.AutoScale.')
    if 'OutArg' not in Opt:
        S.nOutArguments = None          # MATLAB abs(nargout(fcn)): taken from what fcn returns
        S.OutArgument = None
    else:
        oa = vec(np.asarray(Opt.OutArg))
        if oa.size != 2:
            raise ValueError('Opt.OutArg must contain two values [nOut iOut].')
        if oa[1] > oa[0]:
            raise ValueError('Opt.OutArg: second number cannot be larger than first one.')
        S.nOutArguments = int(oa[0])
        S.OutArgument = int(oa[1])
    S.EasySpinFunction = EasySpinFunction
    S.nargout_kw = None
    if _accepts_nargout(fcn):
        S.nargout_kw = S.nOutArguments if S.nOutArguments is not None else (1 if EasySpinFunction else None)
    if 'Method' not in Opt:
        Opt.Method = 'simplex fcn'
    if EasySpinFunction:
        if 'Harmonic' in p0[1] and p0[1].Harmonic > 0:
            Opt.TargetID = 2
        elif S.fcnName in ('pepper', 'garlic'):
            Opt.TargetID = 2
    for kw in Opt.Method.split(' '):
        if kw in _ALGABBREV:
            Opt.AlgorithmID = _ALGABBREV.index(kw) + 1
        elif kw in _TARGETABBREV:
            Opt.TargetID = _TARGETABBREV.index(kw) + 1
        else:
            raise ValueError("Unknown '%s' in Opt.Method." % kw)
    if 'TargetID' not in Opt:
        Opt.TargetID = 1
    S.AlgorithmNames = _ALGNAMES
    S.AlgorithmAbbrev = _ALGABBREV
    S.TargetNames = _TARGETNAMES
    S.TargetAbbrev = _TARGETABBREV
    if 'Mask' not in Opt:
        Opt.Mask = np.ones(data_vec.size, dtype=bool)
    else:
        if _iscell(Opt.Mask):
            Opt.Mask = np.concatenate([vec(np.asarray(m)) for m in Opt.Mask])
        Opt.Mask = vec(np.asarray(Opt.Mask)).astype(bool)
        if Opt.Mask.size != data_vec.size:
            raise ValueError('Opt.Mask has %d elements, but the data has %d elements.' % (Opt.Mask.size, len(data)))
    Opt.useMask = True
    if 'AutoScale' not in Opt:
        Opt.AutoScale = 'lsq' if (EasySpinFunction or S.nDataSets > 1) else 'none'
    if Opt.AutoScale not in ('none', 'lsq', 'maxabs'):
        raise ValueError("Unknown setting for Opt.AutoScale - possible values are 'lsq', 'maxabs' and 'none'.")
    Opt.AutoScaleID = ('none', 'lsq', 'maxabs').index(Opt.AutoScale)
    S.AutoScaleSettings = [0, 1, 2]
    S.AutoScaleStrings = ['none', 'lsq', 'maxabs']
    if 'BaseLine' not in Opt:
        Opt.BaseLine = None
    if isempty(Opt.BaseLine):
        Opt.BaseLine = -np.ones(S.nDataSets)
    elif np.size(Opt.BaseLine) == 1 and S.nDataSets != 1:
        Opt.BaseLine = float(np.asarray(Opt.BaseLine).reshape(-1)[0]) * np.ones(S.nDataSets)
    else:
        Opt.BaseLine = vec(np.asarray(Opt.BaseLine, dtype=float))
    S.BaseLine = Opt.BaseLine
    if Opt.BaseLine.size != S.nDataSets:
        raise ValueError('The number of entries in Opt.BaseLine must be equal to the number of datasets (%d).'
                         % S.nDataSets)
    S.BaseLineSettings = _BLSETTINGS
    S.BaseLineStrings = _BLSTRINGS
    if 'CalculateUncertainties' not in Opt:
        Opt.CalculateUncertainties = 1
    if 'weight' not in Opt:
        Opt.weight = np.ones(S.nDataSets)
    wds = vec(np.asarray(Opt.weight, dtype=float))
    if wds.size != S.nDataSets:
        raise ValueError('The number of elements in Opt.weight must be equal to the number of datasets.')
    weight = np.zeros(int(sum(datasize)))
    for i in range(len(data)):
        weight[idx[i]] = wds[i]
    S.weight = weight
    if 'x' in Opt:
        if _iscell(Opt.x):
            Opt.x = np.concatenate([vec(np.asarray(x)) for x in Opt.x])
        if np.size(Opt.x) != data_vec.size:
            raise ValueError('The size of Opt.x must match the size of the experimental data.')
    else:
        x = np.zeros(data_vec.size)
        for i in range(S.nDataSets):
            x[idx[i]] = np.arange(1, datasize[i] + 1)
        Opt.x = x
    Opt.x = vec(np.asarray(Opt.x))
    S.rmsdhistory = []
    S.besthistory = Struct(rmsd=[], par=[])
    if 'PlotStretchFactor' not in Opt:
        Opt.PlotStretchFactor = 0.05
    if 'maxParameters' not in Opt:
        Opt.maxParameters = 30
    if S.nParameters > Opt.maxParameters:
        raise ValueError('Cannot fit more than %d parameters simultaneously.' % Opt.maxParameters)
    Opt.IterationPrintFunction = _iterationprint
    Opt.InfoPrintFunction = _infoprint
    S.Opts = Opt
    if S.Opts.Verbosity >= 1:
        print('-- esfit ------------------------------------------------')
        print('Number of datasets:       %d' % S.nDataSets)
        for i in range(S.nDataSets):
            print('Data set %d:           %d points' % (i + 1, S.datasize[i]))
        print('Model function name:      %s' % S.fcnName)
        print('Number of fit parameters: %d' % S.nParameters)
        print('Minimization algorithm:   %s' % S.AlgorithmNames[S.Opts.AlgorithmID - 1])
        print('Residuals computed from:  %s' % S.TargetNames[S.Opts.TargetID - 1])
        print('Autoscaling:              %s' % S.Opts.AutoScale)
        print('---------------------------------------------------------')
    return _runFitting(S, rng)


def _iterationprint(str_):
    print(str_.strip())


def _infoprint(str_):
    if isinstance(str_, (list, tuple)):
        for s in str_:
            print(s)
    else:
        print(str_)


def _runFitting(S, rng):
    data_ = S.data
    fixedParams = S.fixedParams
    activeParams = ~fixedParams
    Verbosity = S.Opts.Verbosity
    S.besthistory.rmsd = []
    S.besthistory.par = []

    def modelErrorHandler(ME):
        raise ValueError('\nThe model simulation function raised the following error:\n  %s\n' % ME) from ME
    S.modelErrorHandler = modelErrorHandler
    p_start = S.p_start
    lb = S.pvec_lb
    ub = S.pvec_ub
    S.best = Struct(rmsd=np.inf, rmsdtarget=np.inf)
    # Model outputs that the algorithms ask for ahead of use are computed in parallel
    # (pyeasyspin/_parallel.py); the fit itself consumes them in its own order.
    S.modelcache = ModelCache(S.fcn, _model_kw(S))
    fitOpt = Struct(S.Opts)
    nActiveParams = int(np.sum(activeParams))
    if nActiveParams > 0:
        if Verbosity >= 1:
            print('Running optimization algorithm with %d active parameters...' % nActiveParams)
        fitOpt.track = True
        iterupdate = False
        residualfun = lambda x: _residuals(S, x, fitOpt, iterupdate)
        rmsdfun = lambda x: _rmsd(S, x, fitOpt, iterupdate)
        residualfun.prefetch = rmsdfun.prefetch = lambda X: _prefetch(S, X)
        p0_active = p_start[activeParams]
        lb_active = lb[activeParams]
        ub_active = ub[activeParams]
        alg = fitOpt.AlgorithmID
        if alg == 1:
            pfit_active = F.esfit_simplex(rmsdfun, p0_active, lb_active, ub_active, fitOpt, rng=rng)
        elif alg == 2:
            pfit_active = F.esfit_levmar(residualfun, p0_active, lb_active, ub_active, fitOpt, rng=rng)
        elif alg == 3:
            pfit_active = F.esfit_montecarlo(rmsdfun, lb_active, ub_active, fitOpt, rng=rng)
        elif alg == 4:
            pfit_active = F.esfit_genetic(rmsdfun, lb_active, ub_active, fitOpt, rng=rng)
        elif alg == 5:
            pfit_active = F.esfit_grid(rmsdfun, lb_active, ub_active, fitOpt, rng=rng)
        elif alg == 6:
            pfit_active = F.esfit_swarm(rmsdfun, lb_active, ub_active, fitOpt, rng=rng)
        elif alg == 7:
            # MATLAB lsqnonlin (Optimization Toolbox, default trust-region-reflective) is
            # replaced by scipy's least_squares with the same method (DEVIATIONS.md).
            from scipy.optimize import least_squares
            sol = least_squares(lambda x: residualfun(x)[0], p0_active, bounds=(lb_active, ub_active),
                                method='trf')
            pfit_active = sol.x
        if isinstance(pfit_active, tuple):
            pfit_active = pfit_active[0]
        pfit_active = vec(np.asarray(pfit_active, dtype=float))
        pfit = p_start.copy()
        pfit[activeParams] = pfit_active
    else:
        if Verbosity >= 1:
            print('No active parameters; skipping optimization')
        pfit = p_start.copy()
        pfit_active = pfit[activeParams]
    if S.get('modelEvalError', False):
        return None
    argsfit = S.p2args(pfit) if S.cellInputs else None
    fit = S.best.fit
    scale = S.best.scale
    fitraw = np.zeros(fit.shape, dtype=fit.dtype)
    for i in range(S.nDataSets):
        ix = S.idx[i]
        fitraw[ix] = fit[ix] / scale[i]
    baseline = S.best.baseline
    baselinetype = S.best.baselinetype
    rmsd0 = S.best.rmsd
    residuals0 = S.best.residuals
    ssr0 = float(np.sum(np.abs(residuals0)**2))
    noise_std = float(np.std(residuals0, ddof=1))
    N = data_.size
    noise_der = 1.482602 / np.sqrt(6) * np.median(np.abs(2.0 * data_[2:N - 2] - data_[0:N - 4] - data_[4:N]))
    with np.errstate(divide='ignore', invalid='ignore'):
        red_chisquare = _redchisquare(residuals0, N, noise_std, noise_der)
    if noise_std > noise_der:
        chisquareinfo = '(using noise der estimate)'
    else:
        chisquareinfo = '(using noise std estimate; upper limit)'
    calculateUncertainties = bool(S.Opts.CalculateUncertainties) and S.UserCommand == 0 and nActiveParams > 0
    pstd = ci95 = covmatrix = corrmatrix = None
    if calculateUncertainties:
        if Verbosity >= 1:
            print('Calculating parameter uncertainties...')
            print('  Estimating Jacobian...')
        fitOpt2 = Struct(fitOpt)
        fitOpt2.track = False
        residualfun = lambda x: _residuals(S, x, fitOpt2, False)
        residualfun.prefetch = lambda X: _prefetch(S, X)
        # lb/ub keep the finite-difference stencil inside the bounds (DEVIATIONS.md)
        if BOUNDED_FD:
            J = F.jacobianest(residualfun, pfit_active, lb=lb[activeParams], ub=ub[activeParams])
        else:
            J = F.jacobianest(residualfun, pfit_active)
        if isinstance(J, tuple):
            J = J[0]
        J = np.asarray(J)
        S.modelcache = None
        if J.size > 0 and not np.any(np.isnan(J)):
            if Verbosity >= 1:
                print('  Calculating parameter covariance matrix...')
            residuals, _ = _calculateResiduals(vec(fit), vec(data_), S.Opts.TargetID, S.idx)
            residuals = residuals.reshape(1, -1)
            covmatrix = np.asarray(F.hccm(J, residuals, 'HC1'))
            pstd = np.sqrt(np.diag(covmatrix))
            norm_icdf = lambda p: -np.sqrt(2) * erfcinv(2 * p)
            pctl = 0.95
            ci = norm_icdf(1 / 2 + pctl / 2) * np.sqrt(np.diag(covmatrix))
            ci95 = pfit_active[:, None] + ci[:, None] * np.array([[-1, 1]])
            if Verbosity >= 1:
                print('  Calculating parameter correlation matrix...')
            with np.errstate(divide='ignore', invalid='ignore'):
                Q = np.diag(np.diag(covmatrix)**(-1 / 2))
                corrmatrix = Q @ covmatrix @ Q
            if S.Opts.Verbosity >= 1:
                msg = ['']
                if S.Opts.AutoScaleID != 0:
                    msg += [' ', 'Fitted scale:      %s\n' % ' '.join('%g' % s for s in scale)]
                msg += ['Parameters:', _printparlist(pfit_active, S.pinfo, pstd, ci95), ' ']
                if corrmatrix.size and pfit_active.size > 1:
                    Sigma = corrmatrix
                    msg.append('Correlation matrix:')
                    msg.append(''.join('    ' + ''.join('%f  ' % v for v in Sigma[:, c]) + '\n'
                                       for c in range(Sigma.shape[1])))
                    triuCorr = np.triu(np.abs(Sigma), 1)
                    msg.append('Strongest correlations:')
                    order = np.argsort(-vec(triuCorr), kind='stable')
                    i1, i2 = np.unravel_index(order, Sigma.shape, order='F')
                    np_ = pfit_active.size
                    parind = np.arange(1, activeParams.size + 1)[activeParams]
                    for k in range(min(5, (np_ - 1) * np_ // 2)):
                        msg.append('    p(%d)-p(%d):    %g' % (parind[i1[k]], parind[i2[k]], Sigma[i1[k], i2[k]]))
                    if np.any(triuCorr > 0.8):
                        msg.append('    WARNING! Strong correlations between parameters.')
                msg += [' ', 'Goodness of fit:',
                        '   ssr             %g' % ssr0,
                        '   rmsd            %g' % rmsd0,
                        '   noise std       %g (estimated from residuals; assumes excellent fit)' % noise_std,
                        '   noise der       %g (estimated using der_snr algorithm)' % noise_der,
                        '   red chi-square  %g %s' % (np.real(red_chisquare), chisquareinfo)]
                print('-' * 110)
                for m in msg:
                    print(m)
                print('-' * 110)
        else:
            if Verbosity >= 1:
                print('  Jacobian estimation interrupted by user, cannot calculate parameter uncertainties.'
                      if J.size == 0 else '  NaN elements in Jacobian, cannot calculate parameter uncertainties.')
            pstd = ci95 = covmatrix = corrmatrix = None
    else:
        if Verbosity >= 1:
            print('Fitting stopped by user. Skipping uncertainty quantification.'
                  if S.Opts.CalculateUncertainties else 'Skipping uncertainty quantification as requested.')
    result = Struct()
    result.algorithm = S.AlgorithmAbbrev[S.Opts.AlgorithmID - 1]
    result.target = S.TargetAbbrev[S.Opts.TargetID - 1]
    if S.nDataSets > 1:
        result.fit, result.fitraw, result.mask, result.residuals, result.baseline = [], [], [], [], []
        for k in range(S.nDataSets):
            ix = S.idx[k]
            result.fit.append(fit[ix])
            result.fitraw.append(fitraw[ix])
            result.mask.append(S.Opts.Mask[ix])
            result.residuals.append(residuals0[ix])
            result.baseline.append(baseline[ix])
    else:
        result.fit = fit.copy()
        result.fitraw = fitraw
        result.mask = S.Opts.Mask
        result.residuals = residuals0.copy()
        result.baseline = baseline.copy()
    result.baselinetype = list(baselinetype)
    result.scale = scale
    result.bestfithistory = Struct()
    result.bestfithistory.rmsd = np.array(S.besthistory.rmsd)
    result.bestfithistory.pfit = (np.column_stack(S.besthistory.par) if S.besthistory.par
                                  else np.zeros((0, 0)))
    if S.cellInputs:
        result.bestfithistory.pfit2structs = S.p2args
    result.pnames = [p.Name for p, a in zip(S.pinfo, activeParams) if a]
    result.p_start = p_start
    result.p_fixed = fixedParams
    result.pfit = pfit_active
    result.pfit_full = pfit
    result.argsfit = argsfit
    result.pstd = pstd
    result.ci95 = ci95
    result.cov = covmatrix
    result.corr = corrmatrix
    result.ssr = ssr0
    result.rmsd = rmsd0
    result.redchisquare = red_chisquare
    S.best.fit = fit
    for fn in ('pstd', 'ci95', 'pfit'):
        S.best[fn] = result[fn]
    return result


def _redchisquare(residuals0, N, noise_std, noise_der):
    d = noise_der if noise_std > noise_der else noise_std
    r = (1 / N) * np.sum(residuals0**2) / d**2
    return complex(r) if np.iscomplexobj(r) else float(r)


def _rmsd(S, x, Opt, iterupdate):
    _, rmsd, userstop = _residuals(S, x, Opt, iterupdate)
    return rmsd, userstop


def _model_kw(S):
    return {} if S.nargout_kw is None else {'nargout': S.nargout_kw}


def _model_args(S, par):
    return tuple(S.p2args(par)) if S.cellInputs else (par,)


def _par(S, x):
    par = S.p_start.copy()
    par[~S.fixedParams] = vec(np.asarray(x, dtype=float))
    return par


def _prefetch(S, X):
    """Compute the model for the parameter points X (active parameters) ahead of use."""
    pars = [_par(S, x) for x in X]
    S.modelcache.compute([p.tobytes() for p in pars], [_model_args(S, p) for p in pars])


def _model_output(S, par):
    cache = S.get('modelcache')
    out = cache.take(par.tobytes()) if cache is not None else None
    if out is None:
        t = time.perf_counter()
        out = S.fcn(*_model_args(S, par), **_model_kw(S))
        if cache is not None:
            cache.timing(time.perf_counter() - t)
    outs = out if isinstance(out, tuple) else (out,)
    if S.OutArgument is not None:
        return outs[S.OutArgument - 1]
    if S.EasySpinFunction and len(outs) > 1:
        return outs[1]              # [x, spc, ...] = pepper(...): the spectrum
    return outs[-1]


def _residuals(S, x, Opt, iterupdate):
    userstop = S.UserCommand != 0
    expdata = S.data
    mask = Opt.Mask if S.Opts.useMask else np.ones(Opt.Mask.shape, dtype=bool)
    par = _par(S, x)
    try:
        simdata = _model_output(S, par)
        S.modelEvalError = False
    except Exception as ME:
        S.modelEvalError = True
        S.modelErrorHandler(ME)
    if not _iscell(simdata):
        simdata = [simdata]
    if len(simdata) != S.nDataSets:
        raise ValueError('\n  Experimental and model data have unequal number of datasets:\n'
                         '    experimental: %d\n    model: %d\n' % (S.nDataSets, len(simdata)))
    simdata_vec = np.concatenate([vec(np.asarray(s)) for s in simdata])
    if simdata_vec.dtype.kind in 'iub':
        simdata_vec = simdata_vec.astype(float)
    if simdata_vec.size != expdata.size:
        raise ValueError('\n  Experimental data and model have unequal total number of points:\n'
                         '    experimental: %d\n    model: %d\n' % (expdata.size, simdata_vec.size))
    if np.iscomplexobj(expdata) and not np.iscomplexobj(simdata_vec):
        simdata_vec = simdata_vec.astype(complex)
    order = Opt.BaseLine
    baseline = np.zeros(simdata_vec.shape, dtype=simdata_vec.dtype)
    simscale = np.ones(S.nDataSets)
    baselinetype = []
    for k in range(S.nDataSets):
        ok = order[k]
        baselinetype.append(S.BaseLineStrings[S.BaseLineSettings.index(ok)])
        ix = S.idx[k]
        mi = mask & ix
        if ok != -1:
            N = S.datasize[k]
            xx = np.arange(1, N + 1)[:, None] / N
            D = xx**np.arange(int(ok) + 1)[None, :]
            if Opt.AutoScaleID == 0:
                coeffs = _lsq(D[mask[ix], :], expdata[mi] - simdata_vec[mi])
                baseline[ix] = D @ coeffs
                simdata_vec[ix] = simdata_vec[ix] + baseline[ix]
            elif Opt.AutoScaleID == 1:
                D = np.column_stack([simdata_vec[ix], D])
                coeffs = _lsq(D[mask[ix], :], expdata[mi])
                coeffs[0] = abs(coeffs[0])
                baseline[ix] = D[:, 1:] @ coeffs[1:]
                simdata_vec[ix] = D @ coeffs
                simscale[k] = np.real(coeffs[0])
            elif Opt.AutoScaleID == 2:
                D = np.column_stack([simdata_vec[ix], D])
                coeffs = _lsq(D[mask[ix], :], expdata[mi])
                baseline[ix] = D[:, 1:] @ coeffs[1:]
                coeff = np.max(np.abs(expdata[mi] - baseline[mi])) / np.max(np.abs(simdata_vec[mi]))
                simdata_vec[ix] = coeff * simdata_vec[ix] + baseline[ix]
                simscale[k] = coeff
        else:
            if Opt.AutoScaleID == 1:
                coeffs = _lsq(simdata_vec[mi][:, None], expdata[mi])
                c = abs(coeffs[0])
                simdata_vec[ix] = simdata_vec[ix] * c
                simscale[k] = c
            elif Opt.AutoScaleID == 2:
                coeff = np.max(np.abs(expdata[mi])) / np.max(np.abs(simdata_vec[mi]))
                simdata_vec[ix] = simdata_vec[ix] * coeff
                simscale[k] = coeff
    residuals, residuals0 = _calculateResiduals(simdata_vec, expdata, Opt.TargetID, S.idx, vec(mask))
    rmsd = float(np.sqrt(np.mean(np.abs(residuals)**2 * S.weight)))
    rmsd0 = float(np.sqrt(np.mean(np.abs(residuals0)**2)))
    S.curr = Struct(rmsd=rmsd0, sim=simdata_vec, par=par, scale=simscale, baseline=baseline,
                    baselinetype=baselinetype)
    if Opt.track:
        S.rmsdhistory.append(rmsd0)
        isNewBest = rmsd < S.best.rmsdtarget
        if isNewBest:
            S.best.residuals = residuals0
            S.best.rmsdtarget = rmsd
            S.best.rmsd = rmsd0
            S.best.fit = simdata_vec
            S.best.scale = simscale
            S.best.par = par
            S.best.baseline = baseline
            S.best.baselinetype = baselinetype
            S.besthistory.rmsd.append(rmsd0)
            S.besthistory.par.append(par)
    return residuals, rmsd, userstop


def _calculateResiduals(A, B, mode, idx, includemask=None):
    """Residuals A-B per dataset, transformed for the fit target. mode 4 ('diff') is the
    derivative and mode 5 ('fft') the |FFT|; EasySpin has the two swapped (DEVIATIONS.md)."""
    A = np.array(A, copy=True)
    B = np.array(B, copy=True)
    if includemask is not None:
        A[~includemask] = 0
        B[~includemask] = 0
    idxNaN = np.isnan(A) | np.isnan(B)
    A[idxNaN] = 0
    B[idxNaN] = 0
    res0, res = [], []
    for ix in idx:
        r0 = A[ix] - B[ix]
        res0.append(r0)
        if mode == 1:
            res.append(r0)
        elif mode == 2:
            res.append(np.cumsum(r0))
        elif mode == 3:
            res.append(np.cumsum(np.cumsum(r0)))
        elif mode == 4:
            res.append(vec(F.deriv(r0)))
        elif mode == 5:
            res.append(np.abs(np.fft.fft(r0)))
    return np.concatenate(res), np.concatenate(res0)


def _printparlist(par, pinfo, pstd=None, pci95=None):
    maxNameLength = max(len(p.Name) for p in pinfo)
    indent = '   '
    if pstd is not None and np.size(pstd) > 0:
        s = indent + '    name%svalue        standard deviation        95%% confidence interval' % (
            ' ' * (max(maxNameLength - 4, 0) + 2))
        for p in range(par.size):
            pname = pinfo[p].Name.ljust(maxNameLength)
            s += '\n' + indent + '%2d  %s  %-#12.7g %-#12.7g (%6.3f %%)   %-#12.7g - %-#12.7g' % (
                p + 1, pname, par[p], pstd[p], pstd[p] / par[p] * 100, pci95[p, 0], pci95[p, 1])
    else:
        s = indent + '    name%svalue' % (' ' * (max(maxNameLength - 4, 0) + 2))
        for p in range(par.size):
            s += '\n' + indent + '%2d  %s  %-#12.7g' % (p + 1, pinfo[p].Name.ljust(maxNameLength), par[p])
    return s

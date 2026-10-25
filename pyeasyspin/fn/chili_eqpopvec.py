"""the vector representation of sqrt(Peq) in a LMK or LMKjK basis

Also holds _integral, _integral2 and _integral3: reimplementations of MATLAB R2024b's adaptive
quadratures (integral: vectorized Gauss-Kronrod 7-15 with the [a,b] -> [-1,1] cubic transform and
10 initial subintervals; integral2 'tiled': Shampine's TwoD with Gauss-Kronrod 3-7 tensor rules on
the cosine-transformed square; integral3: integral over x of integral2 over (y,z)). EasySpin's
results depend on where these adaptive rules stop at AbsTol/RelTol 1e-4, so a different quadrature
would not reproduce them; the same subdivision rules do."""
import math
import warnings
import numpy as np
import scipy.sparse as sp
from .._registry import F, nout
from .._struct import Struct

# ---------------------------------------------------------------- MATLAB integral (integralCalc)
_GK15_NODES = np.array([
    -0.9914553711208126, -0.9491079123427585, -0.8648644233597691,
    -0.7415311855993944, -0.5860872354676911, -0.4058451513773972,
    -0.2077849550078985, 0, 0.2077849550078985,
    0.4058451513773972, 0.5860872354676911, 0.7415311855993944,
    0.8648644233597691, 0.9491079123427585, 0.9914553711208126])
_GK15_WT = np.array([
    0.02293532201052922, 0.06309209262997855, 0.1047900103222502,
    0.1406532597155259, 0.1690047266392679, 0.1903505780647854,
    0.2044329400752989, 0.2094821410847278, 0.2044329400752989,
    0.1903505780647854, 0.1690047266392679, 0.1406532597155259,
    0.1047900103222502, 0.06309209262997855, 0.02293532201052922])
_GK15_LOW = np.array([
    0, 0.1294849661688697, 0,
    0.2797053914892767, 0, 0.3818300505051189,
    0, 0.4179591836734694, 0,
    0.3818300505051189, 0, 0.2797053914892767,
    0, 0.1294849661688697, 0])
_GK15_EWT = _GK15_WT - _GK15_LOW
_EPS = np.finfo(float).eps


class _Unsuccessful(Exception):
    """MATLAB:integral:unsuccessful / MATLAB:integral2:unsuccessful (ThrowOnFail)."""


def _split(x, minsubs):
    x = list(x)
    absdx = [abs(x[i + 1] - x[i]) for i in range(len(x) - 1)]
    pathlen = x[-1] - x[0]
    if minsubs > 1:
        if pathlen > 0:
            udelta = minsubs / pathlen
            nnew = [math.ceil(d * udelta) - 1 for d in absdx]
            idxnew = [k for k, n in enumerate(nnew) if n > 0]
            for k in reversed(idxnew):
                nnj = nnew[k]
                newpts = [x[k] + (j / (nnj + 1)) * (x[k + 1] - x[k]) for j in range(1, nnj + 1)]
                x = x[:k + 1] + newpts + x[k + 1:]
        x = [x[i] for i in range(len(x)) if not (i < len(x) - 1 and abs(x[i + 1] - x[i]) == 0)]
        if len(x) == 1:
            x = [x[0], x[0]]
    return np.array(x, dtype=float), pathlen


def _AtoB(t, A, B):
    x = 0.25 * (B - A) * t * (3 - t ** 2) + 0.5 * (B + A)
    w = 0.75 * (B - A) * (1 - t ** 2)
    tShift = np.abs(t) - 1
    xOffset = 0.25 * (B - A) * tShift ** 2 * (tShift + 3)
    bd = 1 / 4
    lo, hi = t < -bd, t > bd
    x[lo] = A + xOffset[lo]
    x[hi] = B - xOffset[hi]
    wB = -0.75 * (B - A) * tShift * (tShift + 2)
    big = np.abs(t) > bd
    w[big] = wB[big]
    return x, w


def _checkspacing(x, throw):
    ax = np.abs(x)
    close = np.abs(np.diff(x)) <= 100 * _EPS * np.maximum(ax[:-1], ax[1:])
    if np.any(close):
        warnings.warn('integral: minimum step size reached; singularity possible.')
        if throw:
            raise _Unsuccessful('integral')
        return True
    return False


def _integral(fun, a, b, AbsTol=1e-10, RelTol=1e-6, InitialIntervalCount=10, Persistence=1,
              ThrowOnFail=False):
    """MATLAB integral(fun, a, b, 'AbsTol', AbsTol, 'RelTol', RelTol) for real finite limits and a
    scalar-valued vectorized fun (integralCalc, iterateScalarValued)."""
    a, b = float(a), float(b)
    if not (math.isfinite(a) and math.isfinite(b)):
        raise NotImplementedError('integral: only finite limits are ported')
    reverse = b < a
    A, B = (b, a) if reverse else (a, b)
    if A == B:
        x = (A + B) / 2
        q = (B - A) * fun(np.array([x]))[0]
        return -q if reverse else q
    MAXINTERVALCOUNT = math.ceil(16384 * Persistence)
    tinterval, pathlen = _split([-1.0, 1.0], InitialIntervalCount)
    MAXINTERVALCOUNT = max(MAXINTERVALCOUNT, 2 * (tinterval.size - 1))
    NODES, WT, EWT = _GK15_NODES, _GK15_WT, _GK15_EWT
    ATOL, RTOL = AbsTol, RelTol
    subs = np.vstack([tinterval[:-1], tinterval[1:]])
    q_ok = 0.0
    err_ok = 0.0
    first = True
    q = 0.0
    while True:
        midpt = subs.sum(axis=0) / 2
        halfh = np.diff(subs, axis=0)[0] / 2
        x = (NODES[:, None] * halfh[None, :] + midpt[None, :]).ravel(order='F')
        t, w = _AtoB(x, A, B)
        if first:
            fx = np.asarray(fun(t))
            if fx.shape != t.shape:
                fx = fx.reshape(t.shape)
            if not (ATOL > 0 and RTOL == 0) and RTOL < 100 * _EPS:
                RTOL = 100 * _EPS
            fx = fx * w
            first = False
        else:
            if _checkspacing(t, ThrowOnFail):
                break
            fx = np.asarray(fun(t)).reshape(t.shape) * w
        fx = fx.reshape(WT.size, -1, order='F')
        qsubs = (WT @ fx) * halfh
        errsubs = (EWT @ fx) * halfh
        q = np.sum(qsubs) + q_ok
        tol = max(ATOL, RTOL * abs(q))
        ok = np.abs(errsubs) <= (2 * tol / pathlen) * np.abs(halfh)
        err_ok = err_ok + np.sum(errsubs[ok])
        errbnd = abs(err_ok) + np.sum(np.abs(errsubs[~ok]))
        if not (np.isfinite(q) and np.isfinite(errbnd)):
            warnings.warn('integral: infinite or NaN value encountered.')
            if ThrowOnFail:
                raise _Unsuccessful('integral')
            break
        if errbnd <= tol:
            break
        q_ok = q_ok + np.sum(qsubs[ok])
        subs = subs[:, ~ok]
        midpt = midpt[~ok]
        if subs.shape[1] == 0:
            break
        nsubs = 2 * subs.shape[1]
        if nsubs > MAXINTERVALCOUNT:
            warnings.warn('integral: maximum number of intervals reached.')
            if ThrowOnFail:
                raise _Unsuccessful('integral')
            break
        subs = np.vstack([subs[0], midpt, midpt, subs[1]]).reshape(2, -1, order='F')
    return -q if reverse else q


# ---------------------------------------------------------------- MATLAB integral2 (tiled)
_GK7_NODES = np.array([-0.9604912687080202, -0.7745966692414834, -0.4342437493468026, 0,
                       0.4342437493468026, 0.7745966692414834, 0.9604912687080202])
_GK7_LOW = np.array([0, 5 / 9, 0, 8 / 9, 0, 5 / 9, 0])
_GK7_HIGH = np.array([0.1046562260264672, 0.2684880898683334, 0.4013974147759622,
                      0.4509165386584744, 0.4013974147759622, 0.2684880898683334, 0.1046562260264672])


def _integral2(FUN, XMIN, XMAX, YMIN, YMAX, AbsTol=1e-10, RelTol=1e-6, Persistence=1, ThrowOnFail=False):
    """MATLAB integral2(FUN, XMIN, XMAX, YMIN, YMAX, 'AbsTol', ., 'RelTol', .) with finite limits
    (method 'tiled', integral2t). YMIN/YMAX are numbers or callables of x; FUN(X, Y) is vectorized."""
    XMIN, XMAX = float(XMIN), float(XMAX)
    ymin = YMIN if callable(YMIN) else (lambda x, v=float(YMIN): v * np.ones(np.shape(x)))
    ymax = YMAX if callable(YMAX) else (lambda x, v=float(YMAX): v * np.ones(np.shape(x)))
    ATOL, RTOL = AbsTol, RelTol
    maxFunEvals = math.ceil(10000 * Persistence)
    thetaL, thetaR, phiB, phiT = 0.0, math.pi, 0.0, math.pi
    AREA = (thetaR - thetaL) * (phiT - phiB)
    nodes = _GK7_NODES
    NNODES = nodes.size
    NARRAY = np.concatenate([nodes + 1, nodes + 3]) / 4
    WT3, WT7 = _GK7_LOW, _GK7_HIGH
    VTSTIDX = (1 + np.arange(1, 7) * (NNODES * NNODES // 2)).reshape(2, 3, order='F') - 1   # 0-based linear
    st = {'first': True, 'nfe': 0}

    def tensor(thetaL, thetaR, phiB, phiT, Qsub, esub):
        dtheta = thetaR - thetaL
        theta = thetaL + NARRAY * dtheta
        x = 0.5 * (XMAX + XMIN) + 0.5 * (XMAX - XMIN) * np.cos(theta)
        if not st['first'] and (x[0] == XMAX or x[-1] == XMIN):
            return Qsub, esub
        X = np.tile(x, (2 * NNODES, 1))
        bottom = np.asarray(ymin(x), dtype=float).reshape(x.shape)
        top = np.asarray(ymax(x), dtype=float).reshape(x.shape)
        dydt = top - bottom
        dphi = phiT - phiB
        phi = phiB + NARRAY * dphi
        Y = np.tile(bottom, (2 * NNODES, 1)) + np.outer(0.5 + 0.5 * np.cos(phi), dydt)
        if not st['first'] and np.any((Y[0, :] == top) | (Y[-1, :] == bottom)):
            return Qsub, esub
        Z = np.asarray(FUN(X, Y))
        st['nfe'] += 1
        if st['first']:
            Xf, Yf = X.ravel(order='F'), Y.ravel(order='F')
            FUN(Xf[VTSTIDX], Yf[VTSTIDX])   # vectorization test evaluation (no effect on the result)
            st['nfe'] += 1
            st['first'] = False
        temp = np.outer(0.25 * (XMAX - XMIN) * np.sin(phi), dydt * np.sin(theta))
        Z = Z * temp
        Z = np.hstack([Z[:NNODES, :], Z[NNODES:, :]])
        r = (dtheta / 4) * (dphi / 4)
        Qs = (WT7 @ (WT7 @ Z).reshape(NNODES, 4, order='F')) * r
        es = np.abs((WT3 @ (WT3 @ Z).reshape(NNODES, 4, order='F')) * r - Qs)
        return Qs, es

    # rectangle lists (MATLAB XREFLIST/ADJERRLIST sorted ascending by adjusted error)
    ADJ, XREF, RECT, QSUB = [], [], [], []
    state = {'ERR_OK': 0.0}

    def add(q, e, L, R, B, T, adjerr):
        n = len(ADJ) + 1
        idx = next((i for i in range(n - 1) if adjerr < ADJ[i]), n - 1)
        ADJ.insert(idx, adjerr)
        XREF.insert(idx, n - 1)
        RECT.append((e, L, R, B, T))
        QSUB.append(q)

    def save(Qsub, esub, thetaL, thetaR, phiB, phiT, TOL, ADJUST):
        dthetad2 = (thetaR - thetaL) / 2
        thetaM = thetaL + dthetad2
        dphid2 = (phiT - phiB) / 2
        phiM = phiB + dphid2
        localtol = TOL * dthetad2 * dphid2 / AREA
        localtol = max(abs(localtol), EPS100 * abs(np.sum(Qsub)))
        adjer = ADJUST * esub
        rects = ((thetaL, thetaM, phiB, phiM), (thetaM, thetaR, phiB, phiM),
                 (thetaL, thetaM, phiM, phiT), (thetaM, thetaR, phiM, phiT))
        for k in range(4):
            if adjer[k] > localtol:
                add(Qsub[k], esub[k], *rects[k], adjer[k])
            else:
                state['ERR_OK'] = state['ERR_OK'] + adjer[k]
        return state['ERR_OK'] + sum(ADJ)

    def nextentry():
        n = len(ADJ)
        smallestFirst = n > 2000
        pos = 0 if smallestFirst else n - 1
        idx = XREF[pos]
        adjerr = ADJ[pos]
        e, L, R, B, T = RECT[idx]
        q = QSUB[idx]
        last = n - 1
        if idx != last:
            RECT[idx] = RECT[last]
            QSUB[idx] = QSUB[last]
            XREF[XREF.index(last)] = idx
        del XREF[pos]
        del ADJ[pos]
        RECT.pop()
        QSUB.pop()
        return q, e, L, R, B, T, adjerr

    Qsub, esub = tensor(thetaL, thetaR, phiB, phiT, None, None)
    Q = np.sum(Qsub)
    EPS100 = 100 * _EPS
    if RTOL < EPS100:
        RTOL = EPS100
    rtold8 = max(RTOL / 8, EPS100)
    atold8 = ATOL / 8
    TOL = EPS100 * abs(Q)
    ADJUST = 1
    minRectWarn = maxNFEWarn = False
    ERRBND = save(Qsub, esub, thetaL, thetaR, phiB, phiT, TOL, ADJUST)
    while len(ADJ) > 0 and ERRBND > TOL:
        q, e, tL, tR, pB, pT, adjerr = nextentry()
        Qsub, esub = tensor(tL, tR, pB, pT, q, adjerr)
        if np.isscalar(Qsub) or np.ndim(Qsub) == 0:
            minRectWarn = True
            state['ERR_OK'] = state['ERR_OK'] + adjerr
            ERRBND = state['ERR_OK'] + sum(ADJ)
        else:
            Newq = np.sum(Qsub)
            ADJUST = min(1, abs(q - Newq) / e)
            Q = Q + (Newq - q)
            TOL = max(atold8, rtold8 * abs(Q))
            ERRBND = save(Qsub, esub, tL, tR, pB, pT, TOL, ADJUST)
            if st['nfe'] >= maxFunEvals:
                maxNFEWarn = True
                break
    fail = False
    if not (np.isfinite(Q) and np.isfinite(ERRBND)):
        warnings.warn('integral2: non-finite result.')
        fail = True
    elif maxNFEWarn:
        warnings.warn('integral2: maximum number of function evaluations reached.')
        fail = ERRBND > max(ATOL, RTOL * abs(Q))
    elif minRectWarn:
        warnings.warn('integral2: minimum rectangle size reached.')
        fail = ERRBND > max(ATOL, RTOL * abs(Q))
    if fail and ThrowOnFail:
        raise _Unsuccessful('integral2')
    return Q


def _integral3(fun, xmin, xmax, ymin, ymax, zmin, zmax, AbsTol=1e-10, RelTol=1e-6):
    """MATLAB integral3 with constant finite limits (outer integralCalc with 3 initial intervals,
    inner integral2 'tiled')."""
    def inner(x):
        Q = np.zeros(np.shape(x), dtype=complex)
        for i, xi in enumerate(np.ravel(x)):
            Q.flat[i] = _integral2(lambda y, z, xi=xi: fun(xi * np.ones(np.shape(z)), y, z), ymin, ymax,
                                   zmin, zmax, AbsTol=AbsTol, RelTol=RelTol, ThrowOnFail=True)
        return Q if np.any(Q.imag) else Q.real
    try:
        return _integral(inner, xmin, xmax, AbsTol=AbsTol, RelTol=RelTol, InitialIntervalCount=3)
    except _Unsuccessful:
        warnings.warn('integral3: unsuccessful.')
        return np.nan


# ---------------------------------------------------------------- chili_eqpopvec
def _col(x):
    return np.asarray(x).ravel(order='F')


@nout(2)
def chili_eqpopvec(basis=None, Potential=None, Opt=None):
    """chili_eqpopvec(basis, Potential (struct L, M, K, lambda or Nx4 array), Opt) ->
    (sqrtPeq, nIntegrals): expansion coefficients of sqrt(Peq) over the orientational basis and the
    number of 1D/2D/3D quadratures used."""
    Opt = Struct() if Opt is None else Struct(Opt)
    if 'useSelectionRules' not in Opt:
        Opt['useSelectionRules'] = True
    if 'PeqTolerances' not in Opt or np.size(Opt['PeqTolerances']) == 0:
        Opt['PeqTolerances'] = np.array([1e-6, 1e-4, 1e-4])
    useSelectionRules = bool(Opt['useSelectionRules'])
    PeqTolerances = _col(Opt['PeqTolerances']).astype(float)
    PeqIntThreshold, PeqIntAbsTol, PeqIntRelTol = PeqTolerances[0], PeqTolerances[1], PeqTolerances[2]
    if isinstance(Potential, dict):
        Lp, Mp, Kp = _col(Potential['L']), _col(Potential['M']), _col(Potential['K'])
        lam = _col(Potential['lambda'])
    else:
        P = np.atleast_2d(np.asarray(Potential))
        if P.shape[1] != 4:
            raise ValueError('Potential must have four columns (L, M, K, lambda)')
        Lp, Mp, Kp, lam = P[:, 0].real, P[:, 1].real, P[:, 2].real, P[:, 3]
    if np.any(Kp < 0):
        raise ValueError('Only potential terms with nonnegative values of K are allowed. Terms with negative K '
                         'required to render the potential real-valued are supplemented automatically.')
    if np.any(Mp[Kp == 0] < 0):
        raise ValueError('For potential terms with K=0, M must be nonnegative. Terms with negative M required to '
                         'render the potential real-valued are supplemented automatically.')
    zeroMK = (Mp == 0) & (Kp == 0)
    if np.any(np.imag(lam[zeroMK]) != 0):
        raise ValueError('Potential coefficients for M=K=0 must be real-valued.')
    rmv = (lam == 0) | ((Lp == 0) & (Mp == 0) & (Kp == 0))
    if np.any(rmv):
        lam, Lp, Mp, Kp = lam[~rmv], Lp[~rmv], Mp[~rmv], Kp[~rmv]
    nIntegrals = np.zeros((1, 3))
    jKbasis = 'jK' in basis and np.size(basis['jK']) > 0 and bool(np.any(basis['jK']))
    L = _col(basis['L'])
    M = _col(basis['M'])
    K = _col(basis['K'])
    if jKbasis:
        jK = _col(basis['jK'])
    nOriBasis = L.size
    if not np.any(lam):
        idx0 = np.flatnonzero((L == 0) & (M == 0) & (K == 0))
        if idx0.size != 1:
            raise ValueError('Exactly one orientational basis function with L=M=K=0 is allowed.')
        sqrtPeq = np.zeros((nOriBasis, 1))
        sqrtPeq[idx0[0], 0] = 1
        return sqrtPeq, nIntegrals

    zeroMp = bool(np.all(Mp == 0))
    zeroKp = bool(np.all(Kp == 0))
    evenLp = bool(np.all(np.mod(Lp, 2) == 0))
    evenMp = bool(np.all(np.mod(Mp, 2) == 0))
    evenKp = bool(np.all(np.mod(Kp, 2) == 0))
    tol = dict(AbsTol=PeqIntAbsTol, RelTol=PeqIntRelTol)
    twopi = 2 * math.pi

    def int_b(f):
        return _integral(f, 0, math.pi, **tol)

    def int_ab(f):
        return _integral2(f, 0, twopi, 0, math.pi, **tol)

    def int_bc(f):
        return _integral2(f, 0, math.pi, 0, twopi, **tol)

    def int_abc(f):
        return _integral3(f, 0, twopi, 0, math.pi, 0, twopi, **tol)

    terms = [(float(Lp[p]), float(Mp[p]), float(Kp[p]), lam[p]) for p in range(lam.size)]

    def U(a, b, c):
        u = 0
        for Lq, Mq, Kq, lq in terms:
            if lq == 0:
                continue
            if Kq == 0 and Mq == 0:
                u = u - F.wignerd([Lq, Mq, Kq], b) * np.real(lq)
            else:
                u = u - 2 * np.real(F.wignerd([Lq, Mq, Kq], a, b, c) * lq)
        return u

    if zeroMp and zeroKp:
        Z = twopi ** 2 * int_b(lambda b: np.exp(-U(0, b, 0)) * np.sin(b))
        nIntegrals = nIntegrals + [1, 0, 0]
    elif zeroMp and not zeroKp:
        Z = twopi * int_bc(lambda b, c: np.exp(-U(0, b, c)) * np.sin(b))
        nIntegrals = nIntegrals + [0, 1, 0]
    elif not zeroMp and zeroKp:
        Z = twopi * int_ab(lambda a, b: np.exp(-U(a, b, 0)) * np.sin(b))
        nIntegrals = nIntegrals + [0, 1, 0]
    else:
        Z = int_abc(lambda a, b, c: np.exp(-U(a, b, c)) * np.sin(b))
        nIntegrals = nIntegrals + [0, 0, 1]
    sqrtZ = np.sqrt(Z)
    sqrtPeq = np.zeros(nOriBasis, dtype=complex)
    for b_ in range(nOriBasis):
        L_, M_, K_ = float(L[b_]), float(M[b_]), float(K[b_])
        if jKbasis:
            jK_ = jK[b_]
        if useSelectionRules:
            if zeroMp:
                if M_ != 0:
                    continue
                if evenLp and L_ % 2 != 0:
                    continue
                if evenKp and K_ % 2 != 0:
                    continue
                if jKbasis and jK_ != 1:
                    continue
                if zeroKp:
                    if K_ != 0:
                        continue
                    f = lambda b, L_=L_: F.wignerd([L_, 0, 0], b) * np.exp(-U(0, b, 0) / 2) / sqrtZ * np.sin(b)
                    Int = twopi ** 2 * int_b(f)
                    nIntegrals = nIntegrals + [1, 0, 0]
                else:
                    f = (lambda b, c, L_=L_, K_=K_: np.cos(K_ * c) * F.wignerd([L_, 0, K_], b)
                         * np.exp(-U(0, b, c) / 2) / sqrtZ * np.sin(b))
                    Int = twopi * int_bc(f)
                    nIntegrals = nIntegrals + [0, 1, 0]
            elif zeroKp:
                if K_ != 0:
                    continue
                if evenLp and L_ % 2 != 0:
                    continue
                if evenMp and M_ % 2 != 0:
                    continue
                f = (lambda a, b, L_=L_, M_=M_: np.cos(M_ * a) * F.wignerd([L_, M_, 0], b)
                     * np.exp(-U(a, b, 0) / 2) / sqrtZ * np.sin(b))
                Int = twopi * int_ab(f)
                nIntegrals = nIntegrals + [0, 1, 0]
            else:
                f = (lambda a, b, c, L_=L_, M_=M_, K_=K_: np.conj(F.wignerd([L_, M_, K_], a, b, c))
                     * np.exp(-U(a, b, c) / 2) / sqrtZ * np.sin(b))
                Int = int_abc(f)
                nIntegrals = nIntegrals + [0, 0, 1]
        else:
            f = (lambda a, b, c, L_=L_, M_=M_, K_=K_: np.conj(F.wignerd([L_, M_, K_], a, b, c))
                 * np.exp(-U(a, b, c) / 2) / sqrtZ * np.sin(b))
            Int = int_abc(f)
            nIntegrals = nIntegrals + [0, 0, 1]
        Int = complex(Int)
        with np.errstate(divide='ignore', invalid='ignore'):
            if abs(np.float64(Int.imag) / np.float64(Int.real)) < 1e-5:
                Int = Int.real
        Int = math.sqrt((2 * L_ + 1) / (8 * math.pi ** 2)) * Int
        if jKbasis:
            Int = math.sqrt(2 / (1 + (K_ == 0))) * Int
        if abs(Int) >= PeqIntThreshold:
            sqrtPeq[b_] = Int
    if not np.any(sqrtPeq.imag):
        sqrtPeq = sqrtPeq.real
    return sp.csc_matrix(sqrtPeq.reshape(-1, 1)), nIntegrals

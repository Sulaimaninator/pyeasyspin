"""Higher-order (multipole) Zeeman interaction Hamiltonian (Sys.Ham<lB><lS><l>)."""
from math import factorial
import numpy as np
import scipy.sparse as sps
from scipy.special import lpmv
from .._registry import F, nout
from .._mat import vec, isempty, numel
from .._ham import spzeros, dense, perms

_ALPHAPM1 = 1 / np.sqrt(np.array([1, 1, 3 / 2, 5 / 2, 35 / 8, 63 / 8, 231 / 16, 429, 16, 6435 / 128]))  # sic
_ALM = np.zeros((8, 9))
_ALM[7, :] = [24 * np.sqrt(1430), 2 * np.sqrt(1430), 4 * np.sqrt(143 / 7), 2 * np.sqrt(78 / 7), 4 * np.sqrt(130 / 7),
              2 * np.sqrt(10 / 7), 4 * np.sqrt(15), 2 * np.sqrt(2), 8 * np.sqrt(2)]
_ALM[6, :8] = [4 * np.sqrt(429), 8 * np.sqrt(429 / 7), 4 * np.sqrt(286 / 7), 8 * np.sqrt(143 / 7), 4 * np.sqrt(13 / 7),
               8 * np.sqrt(13 / 7), 4 * np.sqrt(2 / 7), 8]
_ALM[5, :7] = [4 * np.sqrt(231), 2 * np.sqrt(11), 4 * np.sqrt(22 / 5), 2 * np.sqrt(22 / 5), 4 * np.sqrt(11 / 3),
               2 * np.sqrt(2 / 3), 4 * np.sqrt(2)]
_ALM[4, :6] = [6 * np.sqrt(14), 2 * np.sqrt(42 / 5), np.sqrt(6 / 5), 12 / np.sqrt(5), 2 * np.sqrt(2 / 5), 4]
_ALM[3, :5] = [2 * np.sqrt(70), np.sqrt(7), np.sqrt(14), 1, 2 * np.sqrt(2)]
_ALM[2, :4] = [np.sqrt(10), 2 * np.sqrt(5 / 3), np.sqrt(2 / 3), 2]
_ALM[1, :3] = [np.sqrt(6), 1 / np.sqrt(2), np.sqrt(2)]
_ALM[0, :2] = [1, 1]


def _numeric(SpinSystem, Field, Spins, opt, lBlist):
    if not isinstance(opt, str):
        raise ValueError("Last input must be a string, 'sparse'.")
    Sys, err = F.validatespinsys(SpinSystem)
    if err:
        raise ValueError(err)
    SpinVec = Sys.Spins
    nE = int(Sys.nElectrons)
    Sp = list(range(1, nE + 1)) if isempty(Spins) else [int(s) for s in vec(Spins)]
    if any(s < 1 or s > nE for s in Sp):
        raise ValueError('Spin indices (2nd input argument) invalid!')
    Field = vec(Field).astype(float)
    rB = np.linalg.norm(Field)
    phiB, theta = F.vec2ang(Field)
    ctheta = np.cos(theta)
    H = np.zeros((int(Sys.nStates),) * 2, dtype=complex)
    fields = list(Sys.keys())
    lStemp = []

    def stev(l, m, i):
        return dense(F.stev(SpinVec, np.array([l, m, i], float), opt))

    for iSpin in Sp:
        for lB in [int(v) for v in vec(lBlist)]:
            strlB = f'Ham{lB}'
            paramtext = [f for f in fields if f[:4] == strlB]
            if not paramtext:
                continue
            for n in range(len(paramtext), 0, -1):
                v = float(paramtext[n - 1][4])
                if n > len(lStemp):
                    lStemp += [0.0] * (n - len(lStemp))
                lStemp[n - 1] = v
            lStemp = sorted(set(lStemp))
            for lS in [int(s) for s in lStemp if (lB + s) % 2 == 0]:
                mB = np.arange(lB, -lB - 1, -1)
                amB = np.abs(mB)
                LlB = np.array([lpmv(m, lB, ctheta) for m in range(lB + 1)])
                hs = (np.sign(-mB) + 1) / 2
                pre = np.sqrt(np.array([factorial(lB - a) / factorial(lB + a) for a in amB])) * (-1.0) ** (hs * mB)
                TlBmB = _ALPHAPM1[lB] * pre * LlB[amB] * np.exp(1j * mB * phiB)
                Glb = rB ** lB / np.sqrt(2)
                minl, maxl = abs(lB - lS), lB + lS
                pt = [f for f in fields if f[:5] == f'Ham{lB}{lS}']
                l_ = [float(p[5:]) for p in pt]
                l_ = [int(l) for l in l_ if minl <= l <= maxl and l % 2 == 0]
                for l in l_:
                    Z = vec(np.asarray(Sys[f'Ham{lB}{lS}{l}'])[iSpin - 1, :]) if np.ndim(Sys[f'Ham{lB}{lS}{l}']) == 2 \
                        else vec(Sys[f'Ham{lB}{lS}{l}'])
                    if not np.any(Z):
                        continue
                    a = np.zeros(2 * l + 1, dtype=complex)
                    for m in range(l, 0, -1):
                        lm, lp = l + m, l - m          # 0-based
                        a[lp] = Glb * (-1.0) ** m * (Z[lp] - 1j * Z[lm])
                        a[lm] = Glb * (Z[lp] + 1j * Z[lm])
                    a[l] = Glb * np.sqrt(2) * Z[l]
                    for iq in np.nonzero(a != 0)[0]:
                        m = l - iq
                        pf = (-1.0) ** m * np.sqrt(2 * l + 1)
                        for mb in range(lB, -lB - 1, -1):
                            if abs(TlBmB[lB - mb]) < 1e-10:
                                continue
                            mS = m - mb
                            if -lS <= mS <= lS:
                                if mS == 0:
                                    T = stev(lS, mS, iSpin) / _ALM[lS - 1, 0]
                                elif mS > 0:
                                    T = (-1.0) ** mS / (_ALM[lS - 1, mS] * np.sqrt(2)) * \
                                        (stev(lS, mS, iSpin) + 1j * stev(lS, -mS, iSpin))
                                else:
                                    T = 1 / (_ALM[lS - 1, -mS] * np.sqrt(2)) * \
                                        (stev(lS, -mS, iSpin) - 1j * stev(lS, mS, iSpin))
                                H = H + a[iq] * pf * F.wigner3j(lB, lS, l, mb, mS, -m) * TlBmB[lB - mb] * T
    return sps.csr_matrix(H)   # MATLAB's `if ~sparseResult` on a string never densifies


@nout(-1)
def ham_ezho(SpinSystem=None, *varargin, nargout=None):
    """ham_ezho(Sys, B0[, Spins[, opt[, lB]]]) -> H (MHz, sparse), or tensor form
    ham_ezho(Sys[, Spins[, opt[, lB]]]) -> one tensor per order in B0 (cell lists, column-major)."""
    nargin = 1 + len(varargin)
    if nargout is None:
        nargout = 1
    if nargin > 5:
        raise ValueError('Wrong number of input arguments!')
    if nargout > 1 or nargin < 2:
        tensor = True
    elif nargin > 2 and isempty(varargin[0]):
        tensor = True
    else:
        tensor = False
    if not tensor:
        Field = varargin[0] if nargin >= 2 else np.zeros(3)
        Spins = varargin[1] if nargin >= 3 else None
        opt = varargin[2] if nargin >= 4 else ''
        lBlist = varargin[3] if nargin >= 5 else np.arange(9)
        return _numeric(SpinSystem, Field, Spins, opt, lBlist)

    if nargout > 4:
        raise ValueError('Incorrect number of outputs! Tensor output implemented only up to 3th order in B0')
    v = list(varargin)
    if nargout == 1:
        Spins = v[1] if nargin >= 3 else None
        opt = v[2] if nargin >= 4 else ''
        lb, selection = (v[3], True) if nargin >= 5 else (None, False)
    else:
        Spins = v[0] if nargin >= 2 else None
        opt = v[1] if nargin >= 3 else ''
        lb, selection = (v[2], True) if nargin >= 4 else (None, False)
    highest = None
    for n in range(4):
        if any(f[:4] == f'Ham{n}' for f in SpinSystem):
            highest = n
    if selection:
        lb = [int(x) for x in vec(lb)]
        if any(x > highest for x in lb):
            raise ValueError('Requested order in B0 is higher than that of the SpinSystem!')
    else:
        lb = list(range(highest + 1))
    if nargout != len(lb) and nargout > 2:
        raise ValueError('Incorrect number of outputs!')
    h = lambda Fld, order: ham_ezho(SpinSystem, np.array(Fld, float), Spins, opt, order)
    G0 = G1 = G2 = G3 = None
    if 0 in lb:
        G0 = h([0, 0, 0], 0)
    if 1 in lb:
        G1 = [None] * 3
        for n in (3, 2, 1):
            Fld = np.zeros(3); Fld[n - 1] = 1
            G1[n - 1] = h(Fld, 1)
    if 2 in lb:
        G = {}
        for n in (3, 2, 1):
            Fld = np.zeros(3); Fld[n - 1] = 1
            G[(n, n)] = h(Fld, 2)
        for n in (1, 2, 3):
            Fld = np.ones(3); Fld[n - 1] = 0
            mt = [x for x in (1, 2, 3) if x != n]
            G[(mt[0], mt[1])] = 0.5 * (h(Fld, 2) - G[(mt[0], mt[0])] - G[(mt[1], mt[1])])
            G[(mt[1], mt[0])] = G[(mt[0], mt[1])]
        G2 = [G[(a, b)] for b in (1, 2, 3) for a in (1, 2, 3)]
    if 3 in lb:
        G = {}
        for n in (3, 2, 1):
            Fld = np.zeros(3); Fld[n - 1] = 1
            G[(n, n, n)] = h(Fld, 3)
        for n in (1, 2, 3):
            Fld = np.ones(3); Fld[n - 1] = 0
            lp = h(Fld, 3)
            mt = perms([x for x in (1, 2, 3) if x != n])
            for m in range(len(mt), 0, -1):
                Fld[mt[m - 1][1] - 1] = -1
                lm = h(Fld, 3)
                me = (lp + lm - 2 * G[(mt[m - 1][0],) * 3]) / 6
                for ind in perms([mt[m - 1][0], mt[m - 1][1], mt[m - 1][1]]):
                    G[tuple(ind)] = me
                Fld[mt[m - 1][1] - 1] = 1
        dif = None
        for c in (1, 2, 3):
            for b in (1, 2, 3):
                for a in (1, 2, 3):
                    if (a, b, c) in G:
                        dif = G[(a, b, c)] if dif is None else dif + G[(a, b, c)]
        me = (h(np.ones(3), 3) - dif) / 6
        for ind in perms([1, 2, 3]):
            G[tuple(ind)] = me
        G3 = [G.get((a, b, c), np.zeros((0, 0))) for c in (1, 2, 3) for b in (1, 2, 3) for a in (1, 2, 3)]
    out = [{0: G0, 1: G1, 2: G2, 3: G3}[x] for x in lb]
    return out if nargout == 1 else tuple(out)

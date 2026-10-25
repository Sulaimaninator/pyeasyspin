"""Calculate the maximum-likelihood parameters for a multivariate Gaussian hidden Markov model with
periodic boundaries using the expectation-maximization (Baum-Welch) algorithm"""
import numpy as np
from .._registry import F, nout

EPS = np.finfo(float).eps

# EasySpin resets muUpdater for every trajectory and builds Sigma from the last trajectory only,
# while dividing by the state weights summed over all trajectories: for nTraj > 1, mu converges to
# the last trajectory's means and Sigma is about nTraj times too small. The port accumulates both
# over all trajectories (Baum-Welch for several sequences). True restores EasySpin (DEVIATIONS).
EASYSPIN_LAST_TRAJECTORY = False


def normalise(A):
    """EasySpin/BNT normalise(A) (whole array): (A/s, z) with z = sum(A(:)), s = z + (z==0)."""
    z = np.sum(np.ravel(A, order='F'))
    s = z + (z == 0)
    return A / s, z


def msmtransitionmatrix(c, maxiteration=1000):
    """Reversible maximum-likelihood transition matrix from a count matrix (L-BFGS-B on the
    symmetric flux matrix x, x0 = c + c'). Returns (t, pi_i, x)."""
    c = np.asarray(c.toarray() if hasattr(c, 'toarray') else c, dtype=float)
    nstate = c.shape[0]
    c_sym = c + c.T
    x = c_sym
    c_i = np.sum(c, axis=1, keepdims=True)

    def fcn(xx):
        X = np.asarray(xx, dtype=float).reshape(nstate, nstate, order='F')
        f, g = _myfunc_matrix(X, c, c_i)
        return f, g.reshape(-1, 1, order='F')

    from .._struct import Struct
    opts = Struct(x0=x.reshape(-1, 1, order='F'), maxIts=maxiteration, maxTotalIts=50000)
    res = F.mdhmm_lbfgsb(fcn, np.zeros((nstate * nstate, 1)), np.full((nstate * nstate, 1), np.inf), opts)
    x = np.asarray(res[0], dtype=float).reshape(nstate, nstate, order='F')
    x_i = np.sum(x, axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        t = x / x_i
    t[np.isnan(t)] = 0
    pi_i = x_i / np.sum(x_i)
    return t, pi_i, x


def _myfunc_matrix(x, c, c_i):
    x_i = np.sum(x, axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        tmp = c * np.log(x / x_i)
        index = x > (10 * EPS)
        f = -np.sum(tmp.ravel(order='F')[index.ravel(order='F')])
        t = c_i / x_i
        g = (c / x) + (c.T / x.T) - (t + t.T)
    g[np.ravel(x_i < (10 * EPS)), :] = 0
    index = (x > (10 * EPS)) & (x.T > (10 * EPS))
    g[~index] = 0
    g[np.isnan(g)] = 0
    return f, -g


def _dist_pbc(dist, W):
    w = W / 2
    dist = dist.copy()
    dist[dist > w] -= W
    dist[dist < -w] += W
    return dist


def _gaussian_prob_pbc(x, m, C, W):
    m = np.asarray(m, dtype=float).reshape(-1, 1)
    if m.size == 1:
        x = np.ravel(x, order='F').reshape(1, -1)
    d, N = x.shape
    M = m @ np.ones((1, N))
    denom = (2 * np.pi) ** (d / 2) * np.sqrt(abs(np.linalg.det(C)))
    dist = _dist_pbc(x - M, W)
    mahal = np.sum((dist.T @ np.linalg.inv(C)) * dist.T, axis=1)
    if np.any(mahal < 0):
        import warnings
        warnings.warn('mahal < 0 => C is not psd')
    return np.exp(-0.5 * mahal) / (denom + EPS)


def _fwdback(prior, transmat, obslik):
    nStates, nSteps = obslik.shape
    scale = np.ones(nSteps)
    alpha = np.zeros((nStates, nSteps))
    gamma = np.zeros((nStates, nSteps))
    xi_summed = np.zeros((nStates, nStates))
    alpha[:, 0], scale[0] = normalise(np.ravel(prior) * obslik[:, 0])
    tT = transmat.T
    for iStep in range(1, nSteps):
        alpha[:, iStep], scale[iStep] = normalise((tT @ alpha[:, iStep - 1]) * obslik[:, iStep])
    if np.any(scale == 0):
        loglik = -np.inf
    else:
        loglik = np.sum(np.log(scale))
    beta = np.zeros((nStates, nSteps))
    beta[:, -1] = 1
    gamma[:, -1] = normalise(alpha[:, -1] * beta[:, -1])[0]
    for iStep in range(nSteps - 2, -1, -1):
        b = beta[:, iStep + 1] * obslik[:, iStep + 1]
        beta[:, iStep] = normalise(transmat @ b)[0]
        gamma[:, iStep] = normalise(alpha[:, iStep] * beta[:, iStep])[0]
        xi_summed = xi_summed + normalise(transmat * np.outer(alpha[:, iStep], b))[0]
    return alpha, beta, gamma, loglik, xi_summed


def _em_converged(loglik, previous_loglik, threshold, verbosity):
    if loglik - previous_loglik < -1e-3:
        if verbosity:
            print('      likelihood decreased from %6.4f to %6.4f!' % (previous_loglik, loglik))
        return False
    delta_loglik = abs(loglik - previous_loglik)
    avg_loglik = (abs(loglik) + abs(previous_loglik) + EPS) / 2
    with np.errstate(invalid='ignore'):
        return bool((delta_loglik / avg_loglik) < threshold)


@nout(5)
def mdhmm_em(data=None, initDistr=None, TransProb=None, mu=None, Sigma=None, verbosity=None):
    """mdhmm_em(data, initDistr, TransProb, mu, Sigma, verbosity) -> (logLikIter, eqDistr, TransProb, mu,
    Sigma). data is nDims x nSteps[ x nTraj] (or a list of trajectories); angles are periodic (2*pi)."""
    iterMax = 200
    thresh = 1e-4
    previous_loglik = -np.inf
    converged = False
    it = 1
    logLikIter = []
    if isinstance(data, (list, tuple)):
        data = [np.atleast_2d(np.asarray(d, dtype=float)) for d in data]
    else:
        a = np.asarray(data, dtype=float)
        a = a.reshape(a.shape + (1,) * (3 - a.ndim)) if a.ndim < 3 else a
        data = [a[:, :, k] for k in range(a.shape[2])]
    nTraj = len(data)
    initDistr = np.asarray(initDistr, dtype=float).reshape(-1, 1)
    nStates = initDistr.size
    TransProb = np.asarray(TransProb, dtype=float)
    mu = np.array(mu, dtype=float).reshape(-1, nStates, order='F')
    Sigma = np.asarray(Sigma, dtype=float)
    Sigma = Sigma.reshape(Sigma.shape + (1,) * (3 - Sigma.ndim)) if Sigma.ndim < 3 else Sigma
    eqDistr = None
    while it <= iterMax and not converged:
        transCounts = np.zeros((nStates, nStates))
        expNumVisits1 = np.zeros((nStates, 1))
        weightsSummed = np.zeros((nStates, 1))
        logLik = 0
        gammas = []
        for iTraj in range(nTraj):
            obs = data[iTraj]
            nDims, nSteps = obs.shape
            if iTraj == 0 or EASYSPIN_LAST_TRAJECTORY:
                muUpdater = np.zeros((nDims, nStates))     # EasySpin resets it for every trajectory
            B = np.zeros((nStates, nSteps))
            for iState in range(nStates):
                B[iState, :] = _gaussian_prob_pbc(obs, mu[:, iState], Sigma[:, :, iState], 2 * np.pi)
            _, _, gamma, current_loglik, xi_summed = _fwdback(initDistr, TransProb, B)
            logLik = logLik + current_loglik
            transCounts = transCounts + xi_summed
            expNumVisits1 = expNumVisits1 + gamma[:, 0:1]
            gammas.append(gamma)
            weightsSummed = weightsSummed + np.sum(gamma, axis=1, keepdims=True)
            for iState in range(nStates):
                weights = gamma[iState, :]
                distances = _dist_pbc(obs - mu[:, iState:iState + 1], 2 * np.pi)
                muUpdater[:, iState] = muUpdater[:, iState] + np.sum(weights * distances, axis=1)
        TransProb, eqDistr, _ = msmtransitionmatrix(transCounts, 1000)
        initDistr = normalise(expNumVisits1)[0]
        cov_prior = np.repeat((0.01 * np.eye(nDims))[:, :, None], nStates, axis=2)
        weightsSummed = weightsSummed + (weightsSummed == 0)
        for iState in range(nStates):
            mu[:, iState] = mu[:, iState] + muUpdater[:, iState] / weightsSummed[iState, 0]
        mu[mu > np.pi] -= 2 * np.pi
        mu[mu < -np.pi] += 2 * np.pi
        Sigma = np.zeros((nDims, nDims, nStates))
        trajs = [nTraj - 1] if EASYSPIN_LAST_TRAJECTORY else range(nTraj)   # EasySpin: last trajectory only
        for iState in range(nStates):
            SigmaUpdater = 0
            for jTraj in trajs:
                weights = gammas[jTraj][iState, :]
                distances = _dist_pbc(data[jTraj] - mu[:, iState:iState + 1], 2 * np.pi)
                SigmaUpdater = SigmaUpdater + (weights * distances) @ distances.T
            Sigma[:, :, iState] = SigmaUpdater / weightsSummed[iState, 0]
        Sigma = Sigma + cov_prior
        if verbosity:
            print('    iteration %d, loglik = %f' % (it, logLik))
        it += 1
        converged = _em_converged(logLik, previous_loglik, thresh, verbosity)
        previous_loglik = logLik
        logLikIter.append(logLik)
    return np.array(logLikIter, dtype=float).reshape(1, -1), eqDistr, TransProb, mu, Sigma

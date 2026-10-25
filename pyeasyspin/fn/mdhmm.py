"""Build hidden Markov model (HMM) from MD trajectory of dihedrals"""
import warnings
import numpy as np
import scipy.sparse as sps
from .._registry import F
from .._struct import Struct
from .mdhmm_em import msmtransitionmatrix

EPS = np.finfo(float).eps


def _dist_pbc(x1, x2):
    dist = x1 - x2
    i1 = dist > np.pi
    i2 = dist < -np.pi
    dist = dist.copy()
    dist[i1] -= 2 * np.pi
    dist[i2] += 2 * np.pi
    return dist


def _cov_pbc(x, mu):
    nPoints = x.shape[0]
    xc = _dist_pbc(x, mu)
    return (xc.T @ xc) / nPoints


def _initializehmm(dihedrals, chiStart, nStates, nRepeats, verbosity, rng):
    nDihedrals, nSteps, nTraj = dihedrals.shape
    d2 = np.transpose(dihedrals, (1, 2, 0)).reshape(nSteps * nTraj, nDihedrals, order='F')
    stateTraj, centroids = F.mdhmm_kmeans(d2, nStates, nRepeats, chiStart, verbosity, rng=rng)
    stateTraj = np.asarray(stateTraj, dtype=float).ravel(order='F')
    mu0 = np.asarray(centroids, dtype=float).reshape(-1, nDihedrals).T
    Sigma0 = np.zeros((nDihedrals, nDihedrals, nStates))
    for iState in range(nStates):
        idx = stateTraj == iState + 1
        Sigma0[:, :, iState] = _cov_pbc(d2[idx, :], mu0[:, iState].reshape(1, -1))
    return stateTraj.reshape(nSteps, nTraj, order='F'), mu0, Sigma0


def _estimatemarkovparameters(stateTraj, tau=1):
    stateTraj = np.asarray(stateTraj, dtype=float)
    stateTraj = stateTraj.reshape(-1, 1) if stateTraj.ndim == 1 else stateTraj
    nStates = int(np.max(stateTraj))
    N = sps.csc_matrix((nStates, nStates))
    for iTraj in range(stateTraj.shape[1]):
        s1 = stateTraj[:-tau, iTraj].astype(int) - 1
        s2 = stateTraj[tau:, iTraj].astype(int) - 1
        N = N + sps.csc_matrix((np.ones(s1.size), (s1, s2)), shape=(nStates, nStates))
    tpmat, distr, _ = msmtransitionmatrix(N.toarray(), 1000)
    return tpmat, distr


def _gaussian_prob_circular(x, m, C):
    m = np.asarray(m, dtype=float).reshape(-1, 1)
    if m.size == 1:
        x = np.ravel(x, order='F').reshape(1, -1)
    d, N = x.shape
    M = m @ np.ones((1, N))
    denom = (2 * np.pi) ** (d / 2) * np.sqrt(abs(np.linalg.det(C)))
    dx = _dist_pbc(x, M).T
    mahal = np.sum(np.linalg.solve(C.T, dx.T).T * dx, axis=1)       # (dx/C).*dx
    if np.any(mahal < 0):
        warnings.warn('mahal < 0 => C is not positive semidefinite.')
    return np.exp(-0.5 * mahal) / (denom + EPS)


def _viterbi_path(prior, transmat, obslik):
    T = obslik.shape[1]
    prior = np.ravel(prior)
    Q = prior.size
    delta = np.zeros((Q, T))
    psi = np.zeros((Q, T), dtype=int)
    path = np.zeros(T, dtype=int)
    delta[:, 0] = prior * obslik[:, 0]
    n = np.sum(delta[:, 0])
    if n != 0:
        delta[:, 0] = delta[:, 0] / n
    for t in range(1, T):
        P = delta[:, t - 1][:, None] * transmat          # P(i, j) = delta(i,t-1)*transmat(i,j)
        psi[:, t] = np.argmax(P, axis=0)
        delta[:, t] = P[psi[:, t], np.arange(Q)] * obslik[:, t]
        n = np.sum(delta[:, t])
        if n != 0:
            delta[:, t] = delta[:, t] / n
    path[T - 1] = int(np.argmax(delta[:, T - 1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[path[t + 1], t + 1]
    return path + 1


def _viterbitrajectory(dihedrals, transmat, eqdistr, mu, Sigma):
    nStates = transmat.shape[0]
    _, nSteps, nTraj = dihedrals.shape
    vTraj = np.zeros((nSteps, nTraj))
    for iTraj in range(nTraj):
        obs = dihedrals[:, :, iTraj]
        obslik = np.zeros((nStates, nSteps))
        for iState in range(nStates):
            obslik[iState, :] = _gaussian_prob_circular(obs, mu[:, iState], Sigma[:, :, iState])
        vTraj[:, iTraj] = _viterbi_path(eqdistr, transmat, obslik)
    return vTraj


def mdhmm(dihedrals=None, dt=None, nStates=None, nLag=None, Opt=None, rng=None):
    """mdhmm(dihedrals, dt, nStates, nLag, Opt=None) -> HMM struct (eqDistr, TransProb, mu, Sigma, logLik,
    viterbiTraj, nStates, nLag, dt, tLag, tauRelax). dihedrals is nDihedrals x nTraj x nSteps (radians).
    Random: the k-means seeds come from rng (numpy Generator/RandomState, default the global numpy
    stream); the transition matrices use SciPy's L-BFGS-B (mdhmm_lbfgsb)."""
    if dihedrals is None:
        return None
    if dt is None:
        raise ValueError('dt (2nd input) must be given.')
    if nStates is None:
        raise ValueError('nStates (3rd input) must be given.')
    if nLag is None:
        raise ValueError('nLag (4th input) must be given.')
    Opt = Struct() if Opt is None else Struct(Opt)
    if 'Verbosity' not in Opt:
        Opt.Verbosity = 1
    if np.size(nStates) != 1 or float(nStates) % 1 != 0 or nStates < 1:
        raise ValueError('nStates must be a positive integer.')
    if np.size(nLag) != 1 or float(nLag) % 1 != 0 or nLag < 1:
        raise ValueError('nLag must be a positive integer.')
    nStates, nLag = int(nStates), int(nLag)
    dihedrals = np.asarray(dihedrals, dtype=float)
    if np.any(np.abs(dihedrals) > np.pi):
        raise ValueError('Some dihedral angles are outside the interval [-pi,pi].')
    dihedrals = dihedrals.reshape(dihedrals.shape + (1,) * (3 - dihedrals.ndim)) if dihedrals.ndim < 3 else dihedrals
    dihedrals = np.ascontiguousarray(np.transpose(dihedrals, (0, 2, 1)))
    from ..config import config
    config.LogLevel = Opt.Verbosity
    if 'isSeeded' not in Opt:
        Opt.isSeeded = False
    if 'nTrials' not in Opt:
        Opt.nTrials = 1 if Opt.isSeeded else 10
    F.logmsg(1, '-- HMM model building ----------------------------------')
    nDihedrals, nSteps, nTraj = dihedrals.shape
    F.logmsg(1, '  %d dihedrals; %d time steps; %d trajectories', nDihedrals, nSteps, nTraj)
    F.logmsg(1, '  clustering into %d clusters using k-means (%d repeats)', nStates, Opt.nTrials)
    if Opt.isSeeded:
        F.logmsg(1, '    using rotamer seeds')
        if nDihedrals == 5:
            chi = [[-60, 60, 180], [-60, 60, 180], [-90, 90], [-60, 60, 180], [-90, 90]]
            grids = np.meshgrid(*[np.arange(len(a)) for a in chi], indexing='ij')
            chiStart = np.zeros((grids[0].size, 5))
            for k in range(4, -1, -1):
                chiStart[:, k] = np.asarray(chi[k], dtype=float)[grids[k].ravel(order='F')]
            chiStart = chiStart * np.pi / 180
            if nStates != chiStart.shape[0]:
                warnings.warn(f'Changing the number of states from {nStates} to {chiStart.shape[0]}.')
                nStates = chiStart.shape[0]
            if Opt.nTrials != 1:
                warnings.warn(f'Changing the number of trials from {Opt.nTrials} to 1.')
                Opt.nTrials = 1
        else:
            raise ValueError('No systematic seed available for spin labels other than R1.')
    else:
        F.logmsg(1, '    using random seeds')
        chiStart = np.zeros((0, 0))
    stateTraj, mu0, Sigma0 = _initializehmm(dihedrals, chiStart, nStates, Opt.nTrials, Opt.Verbosity, rng)
    if Opt.Verbosity >= 1:
        print('    cluster  population  max(stddev)/deg  mu0/deg')
        for k in range(nStates):
            pop = np.sum(stateTraj == k + 1) / stateTraj.size
            stddev = np.sqrt(np.max(np.linalg.eigvals(Sigma0[:, :, k]).real))
            print('     %3d       %0.4f       %6.1f        (' % (k + 1, pop, stddev * 180 / np.pi)
                  + ''.join('%4.0f ' % (mu0[d, k] * 180 / np.pi) for d in range(mu0.shape[0])) + ')')
    offset = 1
    idx = np.arange(offset - 1, nSteps, nLag)
    dihedrals = dihedrals[:, idx, :]
    stateTraj = stateTraj[idx, :]
    F.logmsg(1, '  downsampling: offset %d, nLag %d -> %d time steps', offset, nLag, stateTraj.shape[0])
    F.logmsg(1, '  estimation of transition probability matrix and initial distribution')
    TransProb0, eqDistr0 = _estimatemarkovparameters(stateTraj)
    initDistr0 = eqDistr0
    F.logmsg(1, '  HMM optimization using EM algorithm')
    HMM = Struct()
    logLik, HMM.eqDistr, HMM.TransProb, HMM.mu, HMM.Sigma = F.mdhmm_em(dihedrals, initDistr0, TransProb0, mu0,
                                                                       Sigma0, Opt.Verbosity)
    HMM.logLik = float(np.ravel(logLik)[-1])
    F.logmsg(1, '  Viterbi state trajectories calculation')
    mu = np.asarray(HMM.mu, dtype=float).reshape(nDihedrals, -1, order='F')
    Sig = np.asarray(HMM.Sigma, dtype=float)
    Sig = Sig.reshape(Sig.shape + (1,) * (3 - Sig.ndim)) if Sig.ndim < 3 else Sig
    HMM.viterbiTraj = _viterbitrajectory(dihedrals, np.atleast_2d(np.asarray(HMM.TransProb, dtype=float)),
                                         HMM.eqDistr, mu, Sig)
    visited = np.zeros(nStates, dtype=bool)
    visited[np.unique(HMM.viterbiTraj).astype(int) - 1] = True
    if np.any(~visited):
        F.logmsg(1, '  Eliminating %d unvisited states from model', int(np.sum(~visited)))
        newStateNumbers = np.cumsum(visited).astype(float)
        HMM.viterbiTraj = newStateNumbers[HMM.viterbiTraj.astype(int) - 1]
        HMM.mu = mu[:, visited]
        HMM.Sigma = Sig[:, :, visited]
        HMM.TransProb, HMM.eqDistr = _estimatemarkovparameters(HMM.viterbiTraj)
    HMM.nStates = float(np.size(HMM.eqDistr))
    HMM.nLag = float(nLag)
    HMM.dt = dt
    HMM.tLag = nLag * dt
    F.logmsg(1, '  Calculate relaxation times')
    lam = np.linalg.eigvals(np.asarray(HMM.TransProb, dtype=float))
    lam = np.sort(np.real(lam))[::-1]
    lam = lam[lam > 0]
    HMM.tauRelax = (-nLag * dt / np.log(lam[1:])).reshape(1, -1)
    F.logmsg(1, '    max: %0.3g s, min: %0.3g s', np.max(HMM.tauRelax), np.min(HMM.tauRelax))
    F.logmsg(1, '-- done ------------------------------------------------')
    return HMM

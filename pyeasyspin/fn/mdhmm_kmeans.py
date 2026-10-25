"""Perform k-means clustering on periodic (angle) data using random seeds (k-means++)"""
import warnings
import numpy as np
from .._registry import nout


def _dist_pbc(x1, x2):
    return np.pi - np.abs(np.pi - np.abs(x1 - x2))


def _calcdistances(data, centroids):
    nPoints = data.shape[0]
    nClusters = centroids.shape[0]
    d2 = np.zeros((nPoints, nClusters))
    for k in range(nClusters):
        differences = np.pi - np.abs(np.pi - np.abs(data - centroids[k, :]))
        d2[:, k] = np.sum(differences ** 2, axis=1)
    return d2


def _calc_centroids_pbc(x):
    if x.shape[0] == 0:
        raise IndexError('Index in position 1 exceeds array bounds.')
    refPoint = x[0, :]
    distances = x - refPoint
    distances[distances > np.pi] -= 2 * np.pi
    distances[distances < -np.pi] += 2 * np.pi
    c = np.mean(refPoint + distances, axis=0)
    c[c > np.pi] -= 2 * np.pi
    c[c < -np.pi] += 2 * np.pi
    return c


def _kppseed(data, nSeeds, rng):
    nPoints, nDims = data.shape
    D = np.full(nPoints, np.inf)
    c0 = np.zeros((nSeeds, nDims))
    c0[0, :] = data[int(np.ceil(nPoints * rng.random())) - 1, :]
    for k in range(1, nSeeds):
        dist = _dist_pbc(data, c0[k - 1, :])
        D = np.minimum(D, np.sum(dist ** 2, axis=1))
        D = D / np.sum(D)
        c0[k, :] = data[np.flatnonzero(rng.random() < np.cumsum(D))[0], :]
    return c0


def _rand_matrix(rng, m, n):
    """MATLAB rand(m, n) fills column-major."""
    return rng.random(m * n).reshape((m, n), order='F')


@nout(2)
def mdhmm_kmeans(data=None, nClusters=None, nRepeats=None, centroids0=None, verbosity=None, rng=None):
    """mdhmm_kmeans(data, nClusters, nRepeats=1, centroids0=[], verbosity=0) -> (idxBest, centroidsBest).
    data is nPoints x nDims (angles, periodic in 2*pi). Random: k-means++ seeds, randperm and the
    replacement of empty clusters come from rng (numpy Generator/RandomState, default the global
    numpy stream). With nClusters rows of centroids0 the result is deterministic."""
    rng = np.random if rng is None else rng
    nRepeats = 1 if nRepeats is None else int(nRepeats)
    centroids0 = np.zeros((0, 0)) if centroids0 is None else np.asarray(centroids0, dtype=float)
    verbosity = 0 if verbosity is None else verbosity
    maxIterations = 500
    centroidsChangeThreshold = 1e-4
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    nPoints, nDims = data.shape
    nClusters = int(nClusters)
    if centroids0.size:
        centroids0 = centroids0.reshape(-1, nDims) if centroids0.ndim < 2 else centroids0
        if centroids0.shape[1] != nDims:
            raise ValueError(f'centroids0 must have exactly {nDims} columns.')
        if centroids0.shape[0] > nClusters:
            raise ValueError(f'centroids0 must have at most {nClusters} rows.')
        nRepeats = 1
    idxAll, centAll = [None] * nRepeats, [None] * nRepeats
    sumdist2 = np.zeros(nRepeats)
    for iRepeat in range(nRepeats):
        centroids = np.zeros((nClusters, nDims))
        if centroids0.size:
            n0 = centroids0.shape[0]
            centroids[:n0, :] = centroids0
            if n0 < nClusters:
                randidx = rng.permutation(nPoints)[:nClusters - n0]
                centroids[n0:, :] = data[randidx, :]
        else:
            centroids = _kppseed(data, nClusters, rng)
        for iIter in range(1, maxIterations + 1):
            centroidsLast = centroids.copy()
            if iIter == 1:
                clustersAssigned = np.zeros(nClusters, dtype=bool)
                while True:
                    distances2 = _calcdistances(data, centroids)
                    idxClusters = np.argmin(distances2, axis=1)
                    minDistances2 = distances2[np.arange(nPoints), idxClusters]
                    clustersAssigned[np.unique(idxClusters)] = True
                    if np.all(clustersAssigned):
                        break
                    newC = np.pi * (2 * _rand_matrix(rng, int(np.sum(~clustersAssigned)), nDims) - 1)
                    centroids[~clustersAssigned, :] = newC
            else:
                distances2 = _calcdistances(data, centroids)
                idxClusters = np.argmin(distances2, axis=1)
                minDistances2 = distances2[np.arange(nPoints), idxClusters]
            for iCluster in range(nClusters):
                centroids[iCluster, :] = _calc_centroids_pbc(data[idxClusters == iCluster, :])
            centroidsChange = _dist_pbc(centroids, centroidsLast)
            if iIter > 1 and np.max(np.abs(centroidsChange)) < centroidsChangeThreshold:
                break
        if iIter == maxIterations:
            warnings.warn(f'Centroid changes did not converge within {maxIterations} iterations.')
        centroids = centroidsLast
        sumdist2[iRepeat] = np.sum(minDistances2)
        if verbosity:
            print('    repeat %d: best sum of squared distances is %0.3f after %d iterations'
                  % (iRepeat + 1, sumdist2[iRepeat], iIter))
        idxAll[iRepeat] = (idxClusters + 1).astype(float).reshape(-1, 1)
        centAll[iRepeat] = centroids
    iBest = int(np.argmin(sumdist2))
    return idxAll[iBest], centAll[iBest]

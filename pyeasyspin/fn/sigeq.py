"""Thermal equilibrium spin density"""
import numpy as np
import scipy.linalg
from .._registry import F


def sigeq(Ham=None, Temp=None, *varargin):
    """sigeq(Ham, Temp[, 'pol']) -> sigma: Boltzmann density matrix for Hamiltonian Ham (MHz) at
    temperature Temp (K), trace-normalized; 'pol' subtracts the identity part."""
    if Ham is None:
        return None
    nargin = 2 + len(varargin) if Temp is not None else 1
    if nargin < 2 or nargin > 3:
        raise ValueError('Wrong number of input arguments!')
    if np.iscomplexobj(Temp) or np.size(Temp) != 1 or float(np.ravel(Temp)[0]) <= 0:
        raise ValueError('Temperature (in K) must be a positive scalar!')
    Temp = float(np.ravel(Temp)[0])
    if hasattr(Ham, 'toarray'):
        Ham = Ham.toarray()
    Ham = np.atleast_2d(np.asarray(Ham))
    sigma = scipy.linalg.expm(-1e6 * F.planck() / F.boltzm() / Temp * Ham)
    sigma = sigma / np.trace(sigma)
    sigma = (sigma + sigma.conj().T) / 2
    if nargin == 3:
        if isinstance(varargin[0], str) and varargin[0].lower() == 'pol':
            N = max(sigma.shape)
            sigma = sigma - np.eye(N) / N
        else:
            raise ValueError('Invalid third argument!')
    if np.iscomplexobj(sigma) and not np.any(sigma.imag):
        sigma = sigma.real
    return sigma

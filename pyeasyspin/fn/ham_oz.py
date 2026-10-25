"""Orbital Zeeman interaction Hamiltonian."""
import numpy as np
import scipy.sparse as sps
from .._registry import F, nout
from .._mat import isempty, numel, vec
from .._ham import spzeros, finish, idxlist, rotate


@nout(-1)
def ham_oz(SpinSystem=None, *varargin, nargout=None):
    """ham_oz(Sys, B0[, idx[, 'sparse']]) -> H (MHz), or with nargout=3
    ham_oz(Sys[, idx[, 'sparse']]) -> (muxM, muyM, muzM) (MHz/mT). Indices are 1-based.
    Without nargout: one output if the first extra argument is a 3-element field vector."""
    if nargout is None:
        nargout = 1 if (varargin and numel(varargin[0]) == 3) else 3
    if nargout not in (1, 3):
        raise ValueError('Wrong number of output arguments!')
    nargin = 1 + len(varargin)
    if nargout == 1:
        if nargin > 4:
            raise ValueError('Wrong number of input arguments!')
        if nargin < 2:
            raise ValueError('Field vector (second input, in mT) is missing.')
        B0 = varargin[0]
        idx = varargin[1] if nargin >= 3 else None
        opt = varargin[2] if nargin >= 4 else ''
    else:
        B0 = None
        idx = varargin[0] if nargin >= 2 else None
        opt = varargin[1] if nargin >= 3 else ''
    if not isinstance(opt, str):
        raise ValueError("Last input must be a string, 'sparse'.")
    useSparse = opt == 'sparse'
    Sys, err = F.validatespinsys(SpinSystem)
    if err:
        raise ValueError(err)
    spins = Sys.Spins
    nElectrons, nNuclei = int(Sys.nElectrons), int(Sys.nNuclei)
    idx = idxlist(idx, range(1, int(Sys.nL) + 1))
    if not isempty(B0) and numel(B0) != 3:
        raise ValueError('Magnetic field vector (2nd input) must be a 3-element array.')
    if any(i < 1 or i > nElectrons for i in idx):
        raise ValueError('Orbital angular momentum indices (2nd input argument) invalid!')
    nStates = Sys.nStates
    mux, muy, muz = spzeros(nStates), spzeros(nStates), spzeros(nStates)
    pre = -F.bmagn()[0] / F.planck() * vec(Sys.gL) / 1e9
    L = vec(Sys.L)
    for i in idx:
        if L[i - 1] == 0:
            continue
        iSpin = nElectrons + nNuclei + i
        mux = mux + pre[i - 1] * F.sop(spins, np.array([[iSpin, 1]], float), 'sparse')
        muy = muy + pre[i - 1] * F.sop(spins, np.array([[iSpin, 2]], float), 'sparse')
        muz = muz + pre[i - 1] * F.sop(spins, np.array([[iSpin, 3]], float), 'sparse')
    if isempty(B0):
        return finish(mux, useSparse), finish(muy, useSparse), finish(muz, useSparse)
    B0 = vec(B0)
    H = -(mux * B0[0] + muy * B0[1] + muz * B0[2])
    return finish(H, useSparse)

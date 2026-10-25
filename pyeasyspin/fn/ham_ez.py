"""Electron Zeeman interaction Hamiltonian."""
import numpy as np
import scipy.sparse as sps
from .._registry import F, nout
from .._mat import isempty, numel, vec
from .._ham import spzeros, finish, idxlist, rotate


@nout(-1)
def ham_ez(SpinSystem=None, *varargin, nargout=None):
    """ham_ez(Sys, B0[, idx[, 'sparse']]) -> H (MHz), or with nargout=3
    ham_ez(Sys[, idx[, 'sparse']]) -> (muxM, muyM, muzM) (MHz/mT). Indices are 1-based.
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
    nElectrons = int(Sys.nElectrons)
    idx = idxlist(idx, range(1, nElectrons + 1))
    if not isempty(B0) and numel(B0) != 3:
        raise ValueError('Magnetic field vector (2nd input) must be a 3-element array.')
    if any(i < 1 or i > nElectrons for i in idx):
        raise ValueError('Electron spin indices (2nd input argument) invalid!')
    nStates = Sys.nStates
    mux, muy, muz = spzeros(nStates), spzeros(nStates), spzeros(nStates)
    pre = -F.bmagn()[0] / F.planck() * np.asarray(Sys.g) / 1e9
    for i in idx:
        g = pre[(i - 1) * 3:(i - 1) * 3 + 3, :] if Sys.fullg else np.diag(pre[i - 1, :])
        ang = Sys.gFrame[i - 1, :]
        if np.any(ang):
            g = rotate(g, ang, F)
        for k in range(1, 4):
            Sk = F.sop(spins, np.array([[i, k]], float), 'sparse')
            mux = mux + g[0, k - 1] * Sk
            muy = muy + g[1, k - 1] * Sk
            muz = muz + g[2, k - 1] * Sk
    if isempty(B0):
        return finish(mux, useSparse), finish(muy, useSparse), finish(muz, useSparse)
    B0 = vec(B0)
    H = -(mux * B0[0] + muy * B0[1] + muz * B0[2])
    return finish(H, useSparse)

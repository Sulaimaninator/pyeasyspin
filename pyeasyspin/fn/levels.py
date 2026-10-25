"""Energy levels of a spin system"""
import numpy as np
from .._registry import F, nout
from .._mat import a2


def _first(x):
    return x[0] if isinstance(x, tuple) else x


def _squeeze(a):
    """MATLAB squeeze: drop singleton dimensions of an N-D array, keeping at least 2 dims."""
    if a.ndim <= 2:
        return a
    s = [d for d in a.shape if d != 1]
    if len(s) == 0:
        s = [1, 1]
    elif len(s) == 1:
        s = [s[0], 1]
    return a.reshape(s)


@nout(-1)
def levels(*varargin, nargout=None):
    """levels(Sys, Ori, B) or levels(Sys, phi, theta, B) -> Energies (MHz);
    nargout=2 -> (Energies, Vectors). Ori: 'x','y','z','xy',..., Nx2 [phi theta] or Nx3 [phi theta chi];
    B: field values (mT), or a 2-element range expanded to 101 points."""
    nargin = len(varargin)
    if nargin == 0:
        return None
    if nargin == 1:
        raise ValueError('At least three input arguments (Sys, Ori, B0) expected.')
    if nargin == 2:
        raise ValueError('Third input argument (magnetic field range) is missing.')
    if nargin == 3:
        OriList = True
        SpinSystem, Ori, MagnField = varargin
        if isinstance(Ori, str):
            zL = F.letter2vec(Ori)
            Ori = np.asarray(F.vec2ang(zL, nargout=1)).T
    elif nargin == 4:
        OriList = False
        SpinSystem, phi, theta, MagnField = varargin
    else:
        raise ValueError('Wrong number of input arguments!')
    nFieldPoints = 101
    MagnField = np.asarray(MagnField, dtype=float).ravel(order='F')
    if MagnField.size == 2:
        MagnField = np.linspace(MagnField[0], MagnField[1], nFieldPoints)
    if nargout is None:
        nargout = 1
    if nargout in (0, 1):
        computeVectors = False
    elif nargout == 2:
        computeVectors = True
    else:
        raise ValueError('Wrong number of output arguments!')
    Sys, err = F.validatespinsys(SpinSystem)
    if err:
        raise ValueError(err)
    H0, muxM, muyM, muzM = (np.asarray(x.toarray() if hasattr(x, 'toarray') else x)
                            for x in F.ham(Sys, nargout=4))
    n = H0.shape[0]
    nB = MagnField.size

    def solve(H):
        if computeVectors:
            E_, V_ = np.linalg.eigh(H)
            return E_, V_
        return np.sort(np.linalg.eigvalsh(H)), None

    if OriList:
        Ori = a2(np.asarray(Ori, dtype=float))
        if Ori.shape[1] == 2:
            Ori = np.hstack([Ori, np.zeros((Ori.shape[0], 1))])
        elif Ori.shape[1] == 3:
            pass
        else:
            raise ValueError("Ori must be a string ('x','y','z','xy','xz','yz','xyz'), a Nx2 array "
                             "([phi theta]) or a Nx3 array [phi theta chi].")
        nOri = Ori.shape[0]
        Energies = np.zeros((nOri, nB, n))
        Vectors = np.zeros((nOri, nB, n, n), dtype=complex) if computeVectors else None
        zL_M = np.asarray(F.ang2vec(Ori[:, 0], Ori[:, 1]))
        for iOri in range(nOri):
            muzL = zL_M[0, iOri] * muxM + zL_M[1, iOri] * muyM + zL_M[2, iOri] * muzM
            for iField in range(nB):
                E_, V_ = solve(H0 - MagnField[iField] * muzL)
                if computeVectors:
                    Vectors[iOri, iField] = V_
                Energies[iOri, iField, :] = E_
    else:
        phi = np.asarray(phi, dtype=float).ravel(order='F')
        theta = np.asarray(theta, dtype=float).ravel(order='F')
        cosphi, sinphi = np.cos(phi), np.sin(phi)
        costheta, sintheta = np.cos(theta), np.sin(theta)
        Energies = np.zeros((phi.size, theta.size, nB, n))
        Vectors = np.zeros((phi.size, theta.size, nB, n, n), dtype=complex) if computeVectors else None
        for iphi in range(phi.size):
            muxyM = cosphi[iphi] * muxM + sinphi[iphi] * muyM
            for itheta in range(theta.size):
                muzL = sintheta[itheta] * muxyM + costheta[itheta] * muzM
                for iField in range(nB):
                    E_, V_ = solve(H0 - MagnField[iField] * muzL)
                    if computeVectors:
                        Vectors[iphi, itheta, iField] = V_
                    Energies[iphi, itheta, iField, :] = E_
    Energies = _squeeze(Energies)
    if computeVectors:
        return Energies, _squeeze(Vectors)
    return Energies

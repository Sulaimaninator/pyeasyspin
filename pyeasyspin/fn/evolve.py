"""Time-domain evolution of density matrix"""
import numpy as np
import scipy.linalg
from .._mat import vec


def _dense(A):
    return A.toarray() if hasattr(A, 'toarray') else np.asarray(A)


def _eig(H):
    """MATLAB [V,E] = eig(H): Hermitian solver for Hermitian input, general solver otherwise."""
    if np.array_equal(H, H.conj().T):
        E, V = scipy.linalg.eigh(H)
    else:
        E, V = np.linalg.eig(H)
    return V, E


def _cvec(A):
    return np.asarray(A).ravel(order='F')


def evolve(Sig=None, Det=None, Ham=None, n=None, dt=None, IncScheme=None, Mix=None):
    """Signal = evolve(Sig, Det, Ham, n, dt, IncScheme=[1], Mix=[]): free-evolution signal of a
    density matrix under one Hamiltonian (or two, a list, for 2D schemes), with mixing
    propagators between the evolution periods. Several detectors (list) give a list of signals."""
    nargin = sum(x is not None for x in (Sig, Det, Ham, n, dt)) + (IncScheme is not None) + (Mix is not None)
    if nargin < 5 or nargin > 7:
        raise ValueError('Wrong number of input arguments!')
    MixGiven = Mix is not None
    if IncScheme is None:
        IncScheme = 1
    if Mix is None:
        Mix = []                      # Mix = {}: no mixing blocks
    nn = np.asarray(n, dtype=float).ravel()
    if np.any(np.mod(nn, 1) != 0) or np.any(nn <= 0):
        raise ValueError('n, the number of points (4th argument), must be a positive integer.')
    Inc = np.asarray(IncScheme)
    if Inc.size > 1 and not MixGiven:
        raise ValueError('The requested IncScheme requires mixing propagators, but none are provided!')
    if (Inc.dtype.kind not in 'iuf') or (Inc.ndim > 2 or (Inc.ndim == 2 and min(Inc.shape) != 1)) or Inc.size == 0:
        raise ValueError('IncScheme (6th input) must be a vector containing 1, -1, 2, and -2.')
    Inc = [int(x) for x in vec(Inc)]
    if any(abs(x) != 1 and abs(x) != 2 for x in Inc):
        raise ValueError('IncScheme (6th input) can contain only 1, -1, 2, and -2.')
    nEvolutionPeriods = len(Inc)
    nDimensions = max(abs(x) for x in Inc)

    Sig = _dense(Sig)
    if not isinstance(Det, (list, tuple)):
        Det = [Det]
    Det = [_dense(d) for d in Det]
    nDetectors = len(Det)
    if not isinstance(Mix, (list, tuple)):
        Mix = [Mix]                   # a single matrix (even []) is one mixing block
    Mix = [_dense(m) for m in Mix]
    nMixingBlocks = len(Mix)
    if nMixingBlocks != nEvolutionPeriods - 1:
        raise ValueError('Number of mixing propagators not correct! %d are needed.' % (nEvolutionPeriods - 1))

    N = Sig.shape[0]
    dt = vec(np.asarray(dt, dtype=float))
    if nDimensions == 1:
        n = int(nn[0])
        Signal = [np.zeros(n, dtype=complex) for _ in range(nDetectors)]
        if isinstance(Ham, (list, tuple)):
            Ham = Ham[0]
    else:
        if dt.size == 1:
            dt = np.array([dt[0], dt[0]])
        if nn.size == 1:
            nn = np.array([nn[0], nn[0]])
        n = [int(nn[0]), int(nn[1])]
        Signal = [np.zeros((n[0], n[1]), dtype=complex) for _ in range(nDetectors)]

    if not isinstance(Ham, (list, tuple)):
        Ham = _dense(Ham)
        if np.count_nonzero(Ham) == np.count_nonzero(np.diag(Ham)):
            E = np.diag(Ham)
            Density = Sig
            Detector = list(Det)
        else:
            Vecs, E = _eig(Ham)
            E = np.real(E)
            Density = Vecs.conj().T @ Sig @ Vecs
            Mix = [Vecs.conj().T @ m @ Vecs for m in Mix]
            Detector = [Vecs.conj().T @ d @ Vecs for d in Det]
        if nDimensions == 1:
            diagU = np.exp(-2j * np.pi * dt[0] * E)
        else:
            diagUX = np.exp(-2j * np.pi * dt[0] * E)
            diagUY = np.exp(-2j * np.pi * dt[1] * E)
    else:
        H1, H2 = _dense(Ham[0]), _dense(Ham[1])
        if (np.count_nonzero(H1) == np.count_nonzero(np.diag(H1))
                and np.count_nonzero(H2) == np.count_nonzero(np.diag(H2))):
            Ex = H1
            Ey = H2
            Density = Sig
            Detector = list(Det)
        else:
            V1, e1 = _eig(H1)
            V2, e2 = _eig(H2)
            Vecs = {1: V1, 2: V2}
            Ex, Ey = np.diag(e1), np.diag(e2)
            d = [abs(x) for x in Inc]
            Density = Vecs[d[0]].conj().T @ Sig @ Vecs[d[0]]
            Mix = [Vecs[d[i + 1]].conj().T @ Mix[i] @ Vecs[d[i]] for i in range(nMixingBlocks)]
            Detector = [Vecs[d[-1]].conj().T @ dd @ Vecs[d[-1]] for dd in Det]
        diagUX = np.exp((-2j * np.pi * dt[0]) * np.real(np.diag(Ex)))
        diagUY = np.exp((-2j * np.pi * dt[1]) * np.real(np.diag(Ey)))

    # reshape(Detector.',1,N^2): row-major flattening of the detector
    Detector = [np.asarray(d).ravel(order='C') for d in Detector]

    def store(ix, iy, rho_vec):
        for iDet in range(nDetectors):
            v = Detector[iDet] @ rho_vec
            if iy is None:
                Signal[iDet][ix] = v
            else:
                Signal[iDet][ix, iy] = v

    def outer(a, b):
        return np.outer(a, b)

    if Inc == [1]:
        FinalDensity = _cvec(Density)
        U_ = _cvec(outer(diagU, diagU.conj()))
        for ix in range(n):
            store(ix, None, FinalDensity)
            FinalDensity = U_ * FinalDensity
    elif Inc == [1, 1]:
        UU_ = outer(diagU, diagU)
        Mix1 = Mix[0]
        for ix in range(n):
            FinalDensity = Mix1 @ Density @ Mix1.conj().T
            store(ix, None, _cvec(FinalDensity))
            Mix1 = UU_ * Mix1
    elif Inc == [1, -1]:
        MixX = np.diag(diagU ** n) @ Mix[0]
        UtU_ = outer(diagU.conj(), diagU)
        for ix in range(n):
            FinalDensity = MixX @ Density @ MixX.conj().T
            store(ix, None, _cvec(FinalDensity))
            MixX = UtU_ * MixX
    elif Inc == [1, 2]:
        UX_ = outer(diagUX, diagUX.conj())
        UY_ = _cvec(outer(diagUY, diagUY.conj()))
        Mix1 = Mix[0]
        for ix in range(n[0]):
            FinalDensity = _cvec(Mix1 @ Density @ Mix1.conj().T)
            for iy in range(n[1]):
                store(ix, iy, FinalDensity)
                FinalDensity = UY_ * FinalDensity
            Density = UX_ * Density
    elif Inc == [1, 1, 2]:
        UUX_ = outer(diagUX, diagUX)
        UY_ = outer(diagUY, diagUY.conj())
        Mix1, Mix2 = Mix[0], Mix[1]
        for ix in range(n[0]):
            M = Mix2 @ Mix1
            FinalDensity = M @ Density @ M.conj().T
            for iy in range(n[1]):
                store(ix, iy, _cvec(FinalDensity))
                FinalDensity = UY_ * FinalDensity
            Mix1 = UUX_ * Mix1
    elif Inc == [1, -1, 2]:
        UtUX_ = outer(diagUX.conj(), diagUX)
        UY_ = outer(diagUY, diagUY.conj())
        MixX = np.diag(diagUX ** n[0]) @ Mix[0]
        Mix2 = Mix[1]
        for ix in range(n[0]):
            M = Mix2 @ MixX
            FinalDensity = M @ Density @ M.conj().T
            for iy in range(n[1]):
                store(ix, iy, _cvec(FinalDensity))
                FinalDensity = UY_ * FinalDensity
            MixX = UtUX_ * MixX
    elif Inc == [1, 2, 1]:
        Mix1, Mix2 = Mix[0], Mix[1]
        UUX_ = outer(diagUX, diagUX)
        UY = np.diag(diagUY)
        for iy in range(n[1]):
            MixY = Mix2 @ Mix1
            for ix in range(n[0]):
                FinalDensity = MixY @ Density @ MixY.conj().T   # EasySpin: stale MixYadj (DEVIATIONS)
                store(ix, iy, _cvec(FinalDensity))
                MixY = UUX_ * MixY
            Mix1 = UY @ Mix1
    elif Inc == [1, 1, 2, 2]:
        UUX_ = outer(diagUX, diagUX)
        UUY_ = outer(diagUY, diagUY)
        Mix2 = Mix[1]
        Uy3y = Mix[2]
        for iy in range(n[1]):
            Uy3y2 = Uy3y @ Mix2
            Ux1x = Mix[0]
            for ix in range(n[0]):
                P = Uy3y2 @ Ux1x
                FinalDensity = P @ Density @ P.conj().T
                store(ix, iy, _cvec(FinalDensity))
                Ux1x = UUX_ * Ux1x
            Uy3y = UUY_ * Uy3y
    elif Inc == [1, 2, 2, 1]:
        Mix1, Mix2, Mix3 = Mix[0], Mix[1], Mix[2]
        UUX_ = outer(diagUX, diagUX)
        UUY_ = outer(diagUY, diagUY)
        for iy in range(n[1]):
            MixY = Mix3 @ Mix2 @ Mix1
            for ix in range(n[0]):
                FinalDensity = MixY @ Density @ MixY.conj().T   # EasySpin: stale MixYadj (DEVIATIONS)
                store(ix, iy, _cvec(FinalDensity))
                MixY = UUX_ * MixY
            Mix2 = UUY_ * Mix2
    elif Inc == [1, 2, -2, 1]:
        Mix1 = Mix[0]
        Mix2 = np.diag(diagUY ** n[1]) @ Mix[1]
        Mix3 = Mix[2]
        UUX_ = outer(diagUX, diagUX)
        UtUY_ = outer(diagUY.conj(), diagUY)
        for iy in range(n[1]):
            MixY = Mix3 @ Mix2 @ Mix1
            for ix in range(n[0]):
                FinalDensity = MixY @ Density @ MixY.conj().T   # EasySpin: stale MixYadj (DEVIATIONS)
                store(ix, iy, _cvec(FinalDensity))
                MixY = UUX_ * MixY
            Mix2 = UtUY_ * Mix2
    elif Inc == [1, -1, 1, -1]:
        Mix1X = np.diag(diagU ** n) @ Mix[0]
        Mix2 = Mix[1]
        Mix3X = np.diag(diagU ** n) @ Mix[2]
        UtU_ = outer(diagU.conj(), diagU)
        for ix in range(n):
            MixX = Mix3X @ Mix2 @ Mix1X
            FinalDensity = MixX @ Density @ MixX.conj().T
            store(ix, None, _cvec(FinalDensity))
            Mix1X = UtU_ * Mix1X
            Mix3X = UtU_ * Mix3X
    elif Inc == [1, 1, -1, -1]:
        Mix1 = Mix[0]
        Mix2X = np.diag(diagU ** n) @ Mix[1]
        Mix3X = np.diag(diagU ** n) @ Mix[2]
        UU1_ = outer(diagU, diagU)
        UU3_ = outer(diagU, diagU).conj()
        for ix in range(n):
            MixX = Mix3X @ Mix2X @ Mix1
            FinalDensity = MixX @ Density @ MixX.conj().T
            store(ix, None, _cvec(FinalDensity))
            Mix1 = UU1_ * Mix1
            Mix3X = UU3_ * Mix3X
    elif Inc == [1, -1, -1, 1]:
        Mix1X = np.diag(diagU ** n) @ Mix[0]
        Mix2 = Mix[1]
        Mix3X = Mix[2] @ np.diag(diagU ** n)
        UtU1_ = outer(diagU.conj(), diagU)
        UtU3_ = outer(diagU, diagU.conj())
        for ix in range(n):
            MixX = Mix3X @ Mix2 @ Mix1X
            FinalDensity = MixX @ Density @ MixX.conj().T
            store(ix, None, _cvec(FinalDensity))
            Mix1X = UtU1_ * Mix1X
            Mix3X = UtU3_ * Mix3X
    elif Inc == [1, 1, -1, -1, 2]:
        Mix1 = Mix[0]
        Mix2X = np.diag(diagUX ** n[0]) @ Mix[1]
        Mix3X = np.diag(diagUX ** n[0]) @ Mix[2]
        Mix4 = Mix[3]
        UU1_ = outer(diagUX, diagUX)
        UU3_ = outer(diagUX, diagUX).conj()
        UY_ = outer(diagUY, diagUY.conj())
        for ix in range(n[0]):
            MixX = Mix4 @ Mix3X @ Mix2X @ Mix1
            FinalDensity = MixX @ Density @ MixX.conj().T
            for iy in range(n[1]):
                store(ix, iy, _cvec(FinalDensity))
                FinalDensity = UY_ * FinalDensity
            Mix1 = UU1_ * Mix1
            Mix3X = UU3_ * Mix3X
    elif Inc == [1, -1, -1, 1, 2]:
        Mix1X = np.diag(diagUX ** n[0]) @ Mix[0]
        Mix2 = Mix[1]
        Mix3X = Mix[2] @ np.diag(diagUX ** n[0])
        Mix4 = Mix[3]
        UtU1_ = outer(diagUX.conj(), diagUX)
        UtU3_ = outer(diagUX, diagUX.conj())
        UY_ = outer(diagUY, diagUY.conj())
        for ix in range(n[0]):
            MixX = Mix4 @ Mix3X @ Mix2 @ Mix1X
            FinalDensity = MixX @ Density @ MixX.conj().T
            for iy in range(n[1]):
                store(ix, iy, _cvec(FinalDensity))
                FinalDensity = UY_ * FinalDensity
            Mix1X = UtU1_ * Mix1X
            Mix3X = UtU3_ * Mix3X
    else:
        s = ''.join('%d ' % x for x in Inc)
        raise ValueError('The incrementation scheme\n  [%s]\ngiven in IncScheme is not supported.' % s)

    if nDimensions == 1:
        Signal = [s.reshape(-1, 1) for s in Signal]
    return Signal[0] if nDetectors == 1 else Signal

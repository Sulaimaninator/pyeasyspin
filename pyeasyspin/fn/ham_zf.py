"""Electronic zero-field interaction Hamiltonian."""
import numpy as np
from .._registry import F
from .._ham import spzeros, finish, idxlist, rotate, herm, add


def _removenoise(A):
    A = np.asarray(A, dtype=complex)
    re, im = A.real.copy(), A.imag.copy()
    re[(np.abs(re) < 1e-14) & (re != 0)] = 0
    im[(np.abs(im) < 1e-14) & (im != 0)] = 0
    return re + 1j * im


def ham_zf(SpinSystem=None, idxElectrons=None, opt=None):
    """Zero-field splitting Hamiltonian (MHz) from Sys.D/DFrame and Stevens terms Sys.B1..B12."""
    Sys, err = F.validatespinsys(SpinSystem)
    if err:
        raise ValueError(err)
    if opt is None:
        opt = ''
    if not isinstance(opt, str):
        raise ValueError("Third input must be a string, 'sparse'.")
    sparseResult = opt == 'sparse'
    idx = idxlist(idxElectrons, range(1, int(Sys.nElectrons) + 1))
    if any(i > int(Sys.nElectrons) or i < 1 for i in idx):
        raise ValueError('Electron spin index/indices (2nd argument) out of range!')
    H = spzeros(Sys.nStates)
    Spins = Sys.Spins
    Dall = np.asarray(Sys.D)
    D = None
    for i in idx:
        D = Dall[3 * (i - 1):3 * i, :] if Sys.fullD else np.diag(Dall[i - 1, :])
        if not np.any(D):
            continue
        ang = Sys.DFrame[i - 1, :]
        if np.any(ang):
            D = rotate(D, ang, F)
        S = {c: F.sop(Spins, np.array([[i, c]], float), 'sparse') for c in (3, 2, 1)}
        for c1 in range(1, 4):
            for c2 in range(1, 4):
                H = H + D[c1 - 1, c2 - 1] * (S[c1] @ S[c2])
    if 'aF' in Sys:
        raise ValueError('Sys.aF is no longer supported. Use Sys.B4 and Sys.B4Frame instead.')
    B = Sys.B if isinstance(Sys.B, list) else []
    for k in range(1, len(B) + 1):
        Bk = np.asarray(B[k - 1])
        if Bk.size == 0:
            continue
        for i in idx:
            D_present = 'D' in Sys and D is not None and np.size(D) > 0 and np.any(D != 0)
            if D_present and k == 2:
                continue
            if np.all(Bk[i - 1, :] == 0):
                continue
            tilt = np.asarray(Sys.BFrame[k - 1])[i - 1, :]
            if np.any(tilt):
                Dk = np.asarray(F.wignerd(k, tilt[0], tilt[1], tilt[2]))
                Ck = np.asarray(F.isto2stev(k))
                DB = np.linalg.solve(Ck.T, (Ck @ Dk.T).T).T
                DB = _removenoise(DB)
                BkM = Bk[i - 1, :] @ DB
            else:
                BkM = Bk[i - 1, :]
            q = list(range(k, -k - 1, -1))
            for iq in np.nonzero(BkM != 0)[0]:
                H = add(H, BkM[iq] * F.stev(Spins, np.array([k, q[iq], i], float), 'sparse'))
    return finish(herm(H), sparseResult)

"""Propagate the density matrix of a spin system along orientational trajectories (cardamom)."""
import numpy as np
from .._registry import F
from .._struct import Struct
from .._matlab import expm_eig_batch as _expm_eig_batch


def _mdrop(a):
    """MATLAB drops trailing singleton dimensions beyond the second."""
    while a.ndim > 2 and a.shape[-1] == 1:
        a = a[..., 0]
    return a


def _nd4(a):
    a = np.asarray(a)
    return a.reshape(a.shape + (1,) * (4 - a.ndim)) if a.ndim < 4 else a


def _wigD(q):
    q = np.asarray(q)
    if q.ndim == 2:
        q = q[:, :, None]
    A = q[0] - 1j * q[3]
    B = -q[2] - 1j * q[1]
    Ast = q[0] + 1j * q[3]
    Bst = -q[2] + 1j * q[1]
    Z = A * Ast - B * Bst
    ASq, BSq, AstSq, BstSq = A * A, B * B, Ast * Ast, Bst * Bst
    s6 = np.sqrt(6)
    D2 = np.zeros((5, 5) + A.shape, dtype=complex)
    D2[0, 0] = ASq * ASq
    D2[0, 1] = 2 * A ** 3 * B
    D2[0, 2] = s6 * ASq * BSq
    D2[0, 3] = 2 * A * B ** 3
    D2[0, 4] = BSq * BSq
    D2[1, 0] = -2 * A ** 3 * Bst
    D2[1, 1] = ASq * (2 * Z - 1)
    D2[1, 2] = s6 * A * B * Z
    D2[1, 3] = BSq * (2 * Z + 1)
    D2[1, 4] = 2 * Ast * B ** 3
    D2[2, 0] = s6 * ASq * BstSq
    D2[2, 1] = -s6 * A * Bst * Z
    D2[2, 2] = 1 / 2 * (3 * Z ** 2 - 1)
    D2[2, 3] = s6 * Ast * B * Z
    D2[2, 4] = s6 * AstSq * BSq
    D2[3, 0] = -2 * A * Bst ** 3
    D2[3, 1] = BstSq * (2 * Z + 1)
    D2[3, 2] = -s6 * Ast * Bst * Z
    D2[3, 3] = AstSq * (2 * Z - 1)
    D2[3, 4] = 2 * Ast ** 3 * B
    D2[4, 0] = BstSq * BstSq
    D2[4, 1] = -2 * Ast * Bst ** 3
    D2[4, 2] = s6 * AstSq * BstSq
    D2[4, 3] = -2 * Ast ** 3 * Bst
    D2[4, 4] = AstSq * AstSq
    return D2


def _block_average(X, BlockLength):
    nBlocks = X.shape[2] // BlockLength
    out = np.zeros(X.shape[:2] + (nBlocks, X.shape[3]), dtype=X.dtype)
    for iBlock in range(nBlocks):
        out[:, :, iBlock, :] = np.mean(X[:, :, iBlock * BlockLength:(iBlock + 1) * BlockLength, :], axis=2)
    return out


def _sliding(X, nSteps, nTraj, lag):
    out = np.zeros(X.shape[:2] + (nSteps, nTraj), dtype=X.dtype)
    idx = np.arange(nSteps)
    for iTraj in range(nTraj):
        out[:, :, :, iTraj] = X[:, :, idx, 0]
        idx = idx + lag
    return out


def _propagate(rho, U, nSteps, Method):
    if Method == 'fast':
        if rho.shape[0] == 1:
            for iStep in range(1, nSteps):
                rho[0, 0, iStep, :] = U[0, 0, iStep - 1, :] ** 2 * rho[0, 0, iStep - 1, :]
        else:
            for iStep in range(1, nSteps):
                Ui = _mdrop(U[:, :, iStep - 1:iStep, :])
                ri = _mdrop(rho[:, :, iStep - 1:iStep, :])
                rho[:, :, iStep, :] = np.asarray(F.multimatmult(Ui, F.multimatmult(ri, Ui))).reshape(rho.shape[0], rho.shape[1], -1)
    elif Method == 'ISTOs':
        Uadj = np.conj(np.transpose(U, (1, 0, 2, 3)))
        for iStep in range(1, nSteps):
            Ui = _mdrop(U[:, :, iStep - 1:iStep, :])
            ri = _mdrop(rho[:, :, iStep - 1:iStep, :])
            Ua = _mdrop(Uadj[:, :, iStep - 1:iStep, :])
            rho[:, :, iStep, :] = np.asarray(F.multimatmult(Ui, F.multimatmult(ri, Ua))).reshape(rho.shape[0], rho.shape[1], -1)
    return rho


def cardamom_propagatedm(Sys=None, Par=None, Opt=None, MD=None, omega=None, CenterField=None, cache=None):
    """cardamom_propagatedm(Sys, Par, Opt, MD, omega, CenterField) -> Sprho (density matrix trajectory
    averaged over trajectories; 1x1xnSteps or 3x3xnSteps for 'fast', nStates x nStates x nSteps for
    'ISTOs'). EasySpin's persistent caches (spin tensors, MD Wigner trajectory, HMM state tensors)
    live in the optional dict `cache` (cardamom passes one per run; without it they are recomputed,
    which gives the same values). The per-step expm_(.., 'eig') calls are batched."""
    if cache is None:
        cache = {}
    Opt = Struct(Opt)
    Par = Struct(Par)
    Sys = Struct(Sys)
    if 'Verbosity' not in Opt:
        Opt.Verbosity = 0
    from ..config import config
    config.LogLevel = Opt.Verbosity
    PropagationMethod = Opt.Method
    if PropagationMethod == 'fast':
        if 'RTraj' not in Par:
            raise ValueError('Par.RTraj must be provided.')
    elif PropagationMethod == 'ISTOs':
        if 'qTraj' not in Par:
            raise ValueError('Par.qTraj must be provided.')
    else:
        raise ValueError(f"Unknown propagation method '{PropagationMethod}'.")
    useMD = MD is not None and not (isinstance(MD, np.ndarray) and MD.size == 0) and not (isinstance(MD, (list, dict)) and len(MD) == 0)
    if useMD:
        isHMMfromMD = Par.Model == 'MD-HMM'
        isMDdirect = Par.Model == 'MD-direct'
    else:
        isHMMfromMD = False
        isMDdirect = False
    dtSpin = Par.dtSpin
    nSteps = int(Par.nSteps)
    nTraj = int(Par.nTraj)
    doBlockAveraging = Par.BlockLength > 1
    BlockLength = int(Par.BlockLength)
    doSlidingWindowProcessing = isMDdirect and Par.lag > 0
    lag = int(Par.lag)
    if PropagationMethod == 'fast':
        if 'g' not in Sys:
            raise ValueError('A g-tensor is required for the fast method.')
        g = np.asarray(Sys.g, dtype=float)
        if g.shape != (1, 3):
            raise ValueError('g-tensor must be a 3-vector.')
        includeHF = 'A' in Sys
        if includeHF:
            A = np.asarray(Sys.A, dtype=float)
            if A.shape != (1, 3):
                raise ValueError('A-tensor must be a 3-vector.')
        RTraj = np.asarray(Par.RTraj)
        RTrajInv = np.transpose(_nd4(RTraj), (1, 0, 2, 3)).reshape(RTraj.shape)
        if not isHMMfromMD:
            F.logmsg(2, '  calculating tensor trajectories from orientational trajectories')
            gTensor = F.cardamom_tensortraj(g, RTraj, RTrajInv)
            if includeHF:
                ATensor = F.cardamom_tensortraj(A, RTraj, RTrajInv) * 1e6 * 2 * np.pi
        else:
            if 'gTensorState' not in cache:
                F.logmsg(2, '  calculating tensor trajectories from orientational trajectories')
                gTensorMD = _nd4(F.cardamom_tensortraj(g, RTraj, RTrajInv))
                if includeHF:
                    ATensorMD = _nd4(F.cardamom_tensortraj(A, RTraj, RTrajInv) * 1e6 * 2 * np.pi)
                F.logmsg(1, '  calculating state averages of tensors')
                vit = np.atleast_2d(np.asarray(MD.viterbiTraj))
                nVitTraj = vit.shape[0]
                nStatesMD = int(MD.nStates)
                gS = np.zeros((3, 3, nStatesMD, nVitTraj))
                AS = np.zeros((3, 3, nStatesMD, nVitTraj)) if includeHF else None
                import warnings
                with warnings.catch_warnings(), np.errstate(invalid='ignore', divide='ignore'):
                    warnings.simplefilter('ignore', RuntimeWarning)
                    for iState in range(1, nStatesMD + 1):
                        for iTraj in range(nVitTraj):
                            idxState = vit[iTraj, :] == iState
                            gS[:, :, iState - 1, iTraj] = np.mean(gTensorMD[:, :, idxState, iTraj], axis=2)
                            if includeHF:
                                AS[:, :, iState - 1, iTraj] = np.mean(ATensorMD[:, :, idxState, iTraj], axis=2)
                cache['gTensorState'] = np.nanmean(gS, axis=3) if nVitTraj > 1 else gS[:, :, :, 0]
                if includeHF:
                    cache['ATensorState'] = np.nanmean(AS, axis=3) if nVitTraj > 1 else AS[:, :, :, 0]
            gTensorState = cache['gTensorState']
            ATensorState = cache.get('ATensorState')
            F.logmsg(2, '  calculating tensor trajectories from state trajectories')
            gTensor = np.zeros((3, 3, nSteps, nTraj))
            if includeHF:
                ATensor = np.zeros((3, 3, nSteps, nTraj))
            stateTraj = np.asarray(Par.stateTraj).reshape(nSteps, -1, order='F') if np.ndim(Par.stateTraj) < 2 else np.asarray(Par.stateTraj)
            for iState in range(1, int(MD.nStates) + 1):
                for iTraj in range(nTraj):
                    idxState = stateTraj[:, iTraj] == iState
                    gTensor[:, :, idxState, iTraj] = gTensorState[:, :, iState - 1][:, :, None]
                    if includeHF:
                        ATensor[:, :, idxState, iTraj] = ATensorState[:, :, iState - 1][:, :, None]
        gTensor = _nd4(gTensor)
        if includeHF:
            ATensor = _nd4(ATensor)
        if doBlockAveraging:
            F.logmsg(2, '  coarse-graining with tensor averaging, block length %d frames', BlockLength)
            gTensorBlock = _block_average(gTensor, BlockLength)
            if includeHF:
                ATensorBlock = _block_average(ATensor, BlockLength)
        else:
            F.logmsg(2, '  no coarse-graining tensor averaging')
            gTensorBlock = gTensor
            if includeHF:
                ATensorBlock = ATensor
        if doSlidingWindowProcessing:
            F.logmsg(2, '  sliding window processing, increment %d', lag)
            gTensor = _sliding(gTensorBlock, nSteps, nTraj, lag)
            if includeHF:
                ATensor = _sliding(ATensorBlock, nSteps, nTraj, lag)
        else:
            F.logmsg(2, '  no sliding window processing')
            gTensor = gTensorBlock
            if includeHF:
                ATensor = ATensorBlock
        F.logmsg(2, '  combine local with global dynamics')
        if 'RLab' not in Par:
            raise ValueError("Unrecognized field name \"RLab\".")
        if np.size(Par.RLab):
            RLab = np.asarray(Par.RLab)
            RLabInv = np.transpose(_nd4(RLab), (1, 0, 2, 3)).reshape(RLab.shape)
            gT = gTensor.reshape(RLab.shape) if gTensor.size == RLab.size else gTensor
            gTensor = _nd4(F.multimatmult(RLab, F.multimatmult(_mdrop(gT), RLabInv)))
            if includeHF:
                aT = ATensor.reshape(RLab.shape) if ATensor.size == RLab.size else ATensor
                ATensor = _nd4(F.multimatmult(RLab, F.multimatmult(_mdrop(aT), RLabInv)))
        gIso = np.sum(g) / 3
        ge = F.gfree()
        ge = ge[0] if isinstance(ge, tuple) else ge
        GpTensor = (gTensor - gIso) / ge
        Gp_zz = GpTensor[2:3, 2:3, :, :]
        if includeHF:
            a = np.sqrt(ATensor[0:1, 2:3] * ATensor[0:1, 2:3] + ATensor[1:2, 2:3] * ATensor[1:2, 2:3]
                        + ATensor[2:3, 2:3] * ATensor[2:3, 2:3])
            theta = dtSpin * 0.5 * a[0, 0]
            nx = (ATensor[0:1, 2:3] / a)[0, 0]
            ny = (ATensor[1:2, 2:3] / a)[0, 0]
            nz = (ATensor[2:3, 2:3] / a)[0, 0]
            ct = np.cos(theta) - 1
            st = -np.sin(theta)
            r2 = np.sqrt(0.5)
            E = np.zeros((3, 3) + Gp_zz.shape[2:], dtype=complex)
            E[0, 0] = 1 + ct * (nz * nz + 0.5 * (nx * nx + ny * ny)) + 1j * st * nz
            E[0, 1] = r2 * (st * ny + ct * nz * nx) + 1j * r2 * (st * nx - ct * nz * ny)
            E[0, 2] = 0.5 * ct * (nx * nx - ny * ny) - 1j * ct * nx * ny
            E[1, 0] = r2 * (-st * ny + ct * nz * nx) + 1j * r2 * (st * nx + ct * nz * ny)
            E[1, 1] = 1 + ct * (nx * nx + ny * ny)
            E[1, 2] = r2 * (st * ny - ct * nz * nx) + 1j * r2 * (st * nx + ct * nz * ny)
            E[2, 0] = 0.5 * ct * (nx * nx - ny * ny) + 1j * ct * nx * ny
            E[2, 1] = r2 * (-st * ny - ct * nz * nx) + 1j * r2 * (st * nx - ct * nz * ny)
            E[2, 2] = 1 + ct * (nz * nz + 0.5 * (nx * nx + ny * ny)) - 1j * st * nz
            U = np.exp(-1j * dtSpin * 0.5 * omega * Gp_zz) * E
        else:
            U = np.exp(-1j * dtSpin * 0.5 * omega * Gp_zz)
        if includeHF:
            rho = np.zeros((3, 3, nSteps, nTraj), dtype=complex)
            rho[:, :, 0, :] = 0.5 * np.eye(3)[:, :, None]
        else:
            rho = np.zeros((1, 1, nSteps, nTraj), dtype=complex)
            rho[:, :, 0, :] = 0.5
        F.logmsg(2, '  propagate density matrix')
        Sprho = _propagate(rho, U, nSteps, PropagationMethod)
        Sprho = np.mean(Sprho, axis=3)
    else:   # ISTOs
        if 'tensors' not in cache:
            Spins = Sys.Spins
            nSp = np.size(Spins)
            SpinOps = [None] * (nSp * 3)
            for iSpin in range(1, nSp + 1):
                for k in (1, 2, 3):
                    SpinOps[(iSpin - 1) + (k - 1) * nSp] = F.sop(Spins, np.array([[iSpin, k]], dtype=float))
            if 'DiffFrame' not in Sys:
                Sys.DiffFrame = np.array([[0.0, 0.0, 0.0]])
            T, Fm, _, _, _ = F.magint(Sys, SpinOps, CenterField, False, False, False)
            F0 = np.ravel(np.asarray(Fm['F0']), order='F') * 2 * np.pi
            F2 = np.atleast_2d(np.asarray(Fm['F2'])) * 2 * np.pi
            T0 = T['T0']
            T2 = T['T2']
            nInt = F2.shape[0]
            c = {}
            Q0 = 0
            for k in range(F0.size):
                Q0 = Q0 + np.conj(F0[k]) * _dense(T0[k])
            c['Q0'] = Q0
            c['H0'] = np.conj(F0[0]) * _dense(T0[0])
            Q2 = {}
            for mp in range(5):
                for m in range(5):
                    acc = np.zeros(_dense(T2[0]).shape)
                    for iInt in range(nInt):
                        acc = acc + np.conj(F2[iInt, mp]) * _dense(T2[iInt + m * nInt])
                    Q2[(mp, m)] = acc
            c['Q2'] = Q2
            cache['tensors'] = c
        c = cache['tensors']
        if 'D2Traj' in cache:
            D2Traj = cache['D2Traj']
        else:
            D2Traj = _wigD(Par.qTraj)
            if doBlockAveraging:
                D2BlockAvg = _block_average(D2Traj, BlockLength)
            else:
                D2BlockAvg = D2Traj
            if doSlidingWindowProcessing:
                D2Traj = _sliding(D2BlockAvg, nSteps, nTraj, lag)
            else:
                D2Traj = D2BlockAvg
            if useMD:
                cache['D2Traj'] = D2Traj
        if np.size(Par.qLab):
            D2Lab = _wigD(Par.qLab)
            shp = _mdrop(D2Lab).shape
            D2Traj = _nd4(F.multimatmult(_mdrop(D2Lab), _mdrop(D2Traj).reshape(shp)))
        H0 = c['H0']
        Q0 = c['Q0']
        nS = Q0.shape[0]
        H = np.broadcast_to(Q0[:, :, None, None], (nS, nS, nSteps, nTraj)).astype(complex)
        for mp in range(5):
            for m in range(5):
                H = H + D2Traj[m, mp, :nSteps, :][None, None] * c['Q2'][(mp, m)][:, :, None, None]
        nStates = (2 * np.ravel(np.asarray(Sys.Spins, dtype=float)) + 1).astype(int)
        blk = int(np.prod(nStates) // nStates[0])
        SzFilter = np.zeros((nStates[0] * blk, nStates[0] * blk))
        for b in range(nStates[0]):
            SzFilter[b * blk:(b + 1) * blk, b * blk:(b + 1) * blk] = 1
        U0 = F.expm_(-1j * dtSpin * SzFilter * H0, 'eig')
        Hs = np.moveaxis(1j * dtSpin * SzFilter[:, :, None, None] * H, (2, 3), (0, 1))   # nSteps x nTraj x n x n
        Us = U0[None, None] @ _expm_eig_batch(np.ascontiguousarray(Hs))
        U = np.ascontiguousarray(np.moveaxis(Us, (0, 1), (2, 3)))
        nSt = int(Sys.nStates)
        rho = np.zeros((nSt, nSt, nSteps, nTraj), dtype=complex)
        rho0 = _dense(F.sop(Sys.Spins, 'x1'))
        rho[:, :, 0, :] = rho0[:, :, None]
        rho = _propagate(rho, U, nSteps, PropagationMethod)
        rho = np.mean(rho, axis=3)
        projector = _dense(F.sop(Sys.Spins, '+1'))
        Sprho = F.multimatmult(np.ascontiguousarray(np.broadcast_to(projector[:, :, None], projector.shape + (nSteps,))), _mdrop(rho))
    return _mdrop(np.asarray(Sprho))


def _dense(x):
    return x.toarray() if hasattr(x, 'toarray') else np.asarray(x)

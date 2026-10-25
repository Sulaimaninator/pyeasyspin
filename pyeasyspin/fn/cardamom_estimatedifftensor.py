"""Estimate the rotational diffusion tensor from a trajectory of rotation matrices"""
import numpy as np
from .._registry import F, nout
from .._matlab import round_half_away, cumtrapz_rows as _cumtrapz_rows

# EasySpin builds the step-to-step rotations R(k+1)*R(k)' and then treats them as an orientation
# trajectory (q2wp differentiates them), which underestimates Diff 50-1000x. The port uses the
# orientations relative to the first frame, R(k+1)*R(1)' (what the prepended identity quaternion
# implies), with a continuous quaternion sign instead of q0 >= 0. True restores EasySpin's numbers
# (DEVIATIONS).
EASYSPIN_INCREMENTS = False


def _q2wp(qTraj, dt):
    dq = np.diff(qTraj, 1, axis=1) / dt
    q0, q1, q2, q3 = (qTraj[k, :-1] for k in range(4))
    dq0, dq1, dq2, dq3 = (dq[k, :] for k in range(4))
    wxp = 2 * (-q1 * dq0 + q0 * dq1 + q3 * dq2 - q2 * dq3)
    wyp = 2 * (-q2 * dq0 - q3 * dq1 + q0 * dq2 + q1 * dq3)
    wzp = 2 * (-q3 * dq0 + q2 * dq1 - q1 * dq2 + q0 * dq3)
    return np.vstack([wxp, wyp, wzp])


def _msd_fft(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x.T
    nComps = x.shape[0]
    N = max(x.shape)
    D = np.zeros((nComps, N + 1))
    D[:, 1:] = x ** 2
    S2 = F.autocorrfft(x, 2, 0, 0, 0)
    Q = 2 * np.sum(D, axis=1)
    S1 = np.zeros((nComps, N))
    for m in range(1, N + 1):
        Q = Q - D[:, m - 1] - D[:, N + 1 - m - 1]
        S1[:, m - 1] = Q / ((N + 1) - m)
    return S1 - 2 * S2


@nout(3)
def cardamom_estimatedifftensor(RTraj=None, dt=None, stopFitT=None):
    """cardamom_estimatedifftensor(RTraj, dt, stopFitT) -> (Diff, msadp, tLag): principal rotational
    diffusion rates (rad^2/s, 1x3) from linear fits of the mean square angular displacements (body
    frame) up to stopFitT; msadp is 3 x nLag, tLag in ns. Corrected orientation trajectory, see
    EASYSPIN_INCREMENTS."""
    RTraj = np.asarray(RTraj, dtype=float)
    nSteps = max(RTraj.shape)
    RAlign = RTraj[:, :, 0]
    RTrajp = np.zeros((3, 3, nSteps))
    for iStep in range(nSteps):
        RTrajp[:, :, iStep] = RTraj[:, :, iStep] @ RAlign.T
    RRot = np.zeros((3, 3, nSteps - 1))
    for iStep in range(nSteps - 1):
        if EASYSPIN_INCREMENTS:
            RRot[:, :, iStep] = RTrajp[:, :, iStep + 1] @ RTrajp[:, :, iStep].T
        else:
            RRot[:, :, iStep] = RTrajp[:, :, iStep + 1]
    qRot = np.asarray(F.rotmat2quat(RRot)).reshape(4, -1, order='F')
    qRot = np.concatenate([np.array([[1.0], [0], [0], [0]]), qRot], axis=1)
    if EASYSPIN_INCREMENTS:
        idx = qRot[0, :] < 0
        qRot[:, idx] = -qRot[:, idx]
    else:      # continuous sign along the orientation trajectory (q and -q are the same rotation)
        for k in range(1, qRot.shape[1]):
            if np.dot(qRot[:, k], qRot[:, k - 1]) < 0:
                qRot[:, k] = -qRot[:, k]
    wp = _q2wp(qRot, dt)
    t = dt * np.arange(max(wp.shape))
    Deltawp = _cumtrapz_rows(t, wp)
    msadp = _msd_fft(Deltawp)
    msadp = msadp[:, :int(round_half_away(msadp.shape[1] / 2))]
    tLag = t[:max(msadp.shape)] / 1e-9
    stopFitN = int(np.floor(stopFitT / dt))
    p = [np.polyfit(tLag[:stopFitN], msadp[k, :stopFitN], 1) for k in range(3)]
    Diff = np.array([[p[0][0], p[1][0], p[2][0]]]) / 2 * 1e9
    return Diff, msadp, tLag.reshape(1, -1)

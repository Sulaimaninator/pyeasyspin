"""Plot orientational potential"""
import numpy as np
from .._registry import F
from .._struct import Struct


def _LMKsum(a, b, c, lam, Lp, Mp, Kp):
    y = 0
    for p in range(lam.size):
        if lam[p] == 0:
            continue
        if Mp[p] == 0 and Kp[p] == 0:
            y = y + lam[p] * F.wignerd([Lp[p], 0, 0], a, b, c)
        else:
            ph = (-1.0) ** int(Kp[p] - Mp[p])
            y = y + lam[p] * F.wignerd([Lp[p], Mp[p], Kp[p]], a, b, c) + \
                ph * np.conj(lam[p]) * F.wignerd([Lp[p], -Mp[p], -Kp[p]], a, b, c)
    return y


def _normalization(Pop, allMzero, tol=1e-10):
    """Z = int Pop(a,b,c) sin(b) da db dc (or 2*pi*int Pop(0,b,c) sin(b) db dc). MATLAB uses
    integral3/integral2 (AbsTol 1e-6); here a periodic trapezoid rule in a, c and Gauss-Legendre
    in b, refined until it converges (the integrand is a smooth trigonometric polynomial
    exponential)."""
    Zold = None
    for n in (16, 32, 64, 128):
        xb, wb = np.polynomial.legendre.leggauss(n)
        b = (xb + 1) * np.pi / 2
        wb = wb * np.pi / 2
        t = np.arange(n) * 2 * np.pi / n
        wt = 2 * np.pi / n
        if allMzero:
            A, Bb, C = np.meshgrid(t, b, t, indexing='ij')
            W = wt * wb[None, :, None] * wt
            f = Pop(A.ravel(), Bb.ravel(), C.ravel()).reshape(A.shape)
            Z = float(np.real(np.sum(W * f * np.sin(Bb))))
        else:
            Bb, C = np.meshgrid(b, t, indexing='ij')
            f = Pop(np.zeros(Bb.size), Bb.ravel(), C.ravel()).reshape(Bb.shape)
            Z = 2 * np.pi * float(np.real(np.sum(wb[:, None] * wt * f * np.sin(Bb))))
        if Zold is not None and abs(Z - Zold) <= tol * abs(Z):
            break
        Zold = Z
    return Z


def oripotentialplot(Potential=None, population=False, alphabeta=True, plot=True, ax=None):
    """Orientational potential U(alpha,beta,gamma)/kT = -sum lambda_LMK D^L_MK (Nx4 rows
    [L M K lambda]) on a sphere: (alpha,beta,0) or, with alphabeta=False, (0,beta,gamma).
    population=True shows the normalized equilibrium population exp(-U/kT)/Z instead. These
    two keywords replace the figure's toggle buttons.

    Returns a Struct with vecs (3xN unit vectors of the sphgrid 'C1' grid), tri (Nt x 3,
    1-based), alpha/beta or gamma/beta angles, P (plotted values on the vertices), title,
    coeffs (the coefficient text), Z (population normalization, population=True only) and
    axes (plot=True only)."""
    if Potential is None:
        raise ValueError('Potential must be a Nx4 array')
    Potential = np.atleast_2d(np.asarray(Potential))
    if Potential.dtype.kind not in 'iufc' or Potential.shape[1] != 4:
        raise ValueError('Potential must be a Nx4 array')
    Potential = Potential[Potential[:, 3] != 0, :]
    Lp = np.real(Potential[:, 0]).astype(int)
    Mp = np.real(Potential[:, 1]).astype(int)
    Kp = np.real(Potential[:, 2]).astype(int)
    lam = Potential[:, 3]
    if lam.size:
        if np.any(Lp < 0):
            raise ValueError('L values of potential coefficients must be nonnegative.')
        if np.any(np.abs(Kp) > Lp):
            raise ValueError('L and K values of potential coefficients do not satisfy -L<=K<=L.')
        if np.any(np.abs(Mp) > Lp):
            raise ValueError('L and M values of potential coefficients do not satisfy -L<=M<=L.')
        if np.any(Kp < 0):
            raise ValueError('Only nonnegative values of K are allowed.')
        Mzero = Mp == 0
        if np.any(Kp[Mzero] < 0):
            raise ValueError('For potential terms with M=0, K must be nonnegative.')
        if np.any(np.imag(lam[(Kp == 0) & (Mp == 0)]) != 0):
            raise ValueError('Potential coefficients for M=K=0 must be real-valued.')
    else:
        Mzero = np.zeros(0, dtype=bool)
    kT = 2
    Pot = lambda a, b, c: -kT * _LMKsum(a, b, c, lam, Lp, Mp, Kp)
    Pop0 = lambda a, b, c: np.exp(-Pot(a, b, c) / kT)
    out = Struct()
    if population:
        Z = _normalization(Pop0, bool(Mzero.size) and bool(np.all(Mzero)))
        out.Z = Z
        Pop = lambda a, b, c: Pop0(a, b, c) / Z
    s = 'Potential coefficients (L,M,K):\n'
    for p in range(lam.size):
        s += ' (%d,%d,%d):  %+0.4f  %+0.4f\n' % (Lp[p], Mp[p], Kp[p], np.real(lam[p]), np.imag(lam[p]))
    out.coeffs = s

    GridSize = max(40, int(np.max(Lp)) * 2 if Lp.size else 0)
    Grid, tri = F.sphgrid('C1', GridSize, nargout=2)
    Vectors = np.asarray(Grid.vecs)
    ang = F.vec2ang(Vectors)
    a1, beta = np.asarray(ang[0]).ravel(), np.asarray(ang[1]).ravel()
    z0 = np.zeros(a1.size)
    if alphabeta:
        if population:
            P = Pop(a1, beta, z0)
            title = 'Equilibrium population density (α,β,0)'
        else:
            P = Pot(a1, beta, z0) / kT
            title = 'Potential energy (α,β,0)'
        out.alpha = a1
    else:
        if population:
            P = Pop(z0, beta, a1)
            title = 'Equilibrium population density (0,β,γ)'
        else:
            P = Pot(z0, beta, a1) / kT
            title = 'Potential energy (0,β,γ)'
        out.gamma = a1
    out.beta = beta
    vec = np.asarray(F.ang2vec(a1, beta))
    P = np.broadcast_to(np.asarray(P), a1.shape).copy() if np.ndim(P) == 0 else np.asarray(P).ravel()
    out.vecs = vec
    out.tri = np.asarray(tri.idx)
    out.P = P
    out.title = title
    if plot:
        out.axes = _draw(out, population, ax)
    return out


def _draw(out, population, ax):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib
    if ax is None:
        fig = plt.gcf()
        fig.clf()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
    else:
        fig = ax.figure
    v = out.vecs
    tri = out.tri - 1
    Pr = np.real(out.P)
    cmap = matplotlib.colormaps['gray_r' if population else 'viridis']
    lo = 0.0 if population else float(np.min(Pr))
    hi = float(np.max(Pr))
    norm = matplotlib.colors.Normalize(lo, hi if hi > lo else lo + 1)
    fc = cmap(norm(Pr[tri].mean(axis=1)))
    fc[:, 3] = 0.85
    pc = Poly3DCollection([v[:, t].T for t in tri], facecolors=fc, edgecolors='none')
    ax.add_collection3d(pc)
    w, d = 1.2, 0.1
    for seg in (([0, 0], [0, 0], [-w, w]), ([0, 0], [-w, w], [0, 0]), ([-w, w], [0, 0], [0, 0])):
        ax.plot(*seg, color='k', linewidth=0.8)
    phi = np.linspace(0, 2 * np.pi, 100)
    xc, yc, zc = np.cos(phi) * 0.99, np.sin(phi) * 0.99, np.zeros(phi.size)
    for xs, ys, zs in ((xc, yc, zc), (yc, zc, xc), (zc, xc, yc)):
        ax.plot(xs, ys, zs, color='k', linewidth=0.5)
    for pos, lab in (((w + d, 0, 0), '+x'), ((0, w + d, 0), '+y'), ((0, 0, w + d), '+z'),
                     ((-w - d, 0, 0), '-x'), ((0, -w - d, 0), '-y'), ((0, 0, -w - d), '-z')):
        ax.text(*pos, lab, ha='center', va='center')
    ax.set_box_aspect((1, 1, 1))
    lim = (-w - d, w + d)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_zlim(lim)
    ax.set_axis_off()
    ax.set_title(out.title)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax)
    cb.set_label('equilibrium population density (rad$^{-3}$)' if population else 'potential energy ($k_BT$)')
    fig.text(0.02, 0.98, out.coeffs, va='top', fontsize=9)
    return ax

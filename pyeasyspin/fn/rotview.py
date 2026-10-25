"""Display rotation between two frames"""
import numpy as np
from .._registry import F
from .._struct import Struct

_LABELS = ['general (A->B)', 'Sys.gFrame (M->g)', 'Sys.AFrame (M->A)', 'Sys.DFrame (M->D)',
           'Sys.QFrame (M->Q)', 'Sys.eeFrame (M->ee)', 'Sys.DiffFrame (M->Diff)',
           'Exp.MolFrame (C->M)', 'Exp.SampleFrame (L->C)']
_NAMES = [('A', 'B'), ('M', 'g'), ('M', 'A'), ('M', 'D'), ('M', 'Q'), ('M', 'ee'), ('M', 'Diff'),
          ('C', 'M'), ('L', 'C')]
_CTENSOR = np.array([0.2, 0.3, 0.4])


def _rows(angles_deg):
    return [np.asarray(v, dtype=float).ravel() for v in F.erot(np.asarray(angles_deg) * np.pi / 180, 'rows', nargout=3)]


def calculateframes(alpha, beta, gamma):
    """Frame vectors (rows of the rotation matrices, 3-vectors) for angles in degrees: A frame,
    line of nodes (x1, y1), x after the second rotation (x2), B frame; R = erot(angles);
    and the three 101-point circles (z-z plane, xy planes of A and B)."""
    xA, yA, zA = _rows([0, 0, 0])
    x1, y1, _ = _rows([alpha, 0, 0])
    x2, _, _ = _rows([alpha, beta, 0])
    xB, yB, zB = _rows([alpha, beta, gamma])
    R = np.column_stack([xB, yB, zB]).T
    phi = np.linspace(0, 2 * np.pi, 101)
    zzplane = np.outer(zA, np.cos(phi)) + np.outer(x1, np.sin(phi))
    xyAplane = np.outer(xA, np.cos(phi)) + np.outer(yA, np.sin(phi))
    xyBplane = np.outer(xB, np.cos(phi)) + np.outer(yB, np.sin(phi))
    return Struct(xA=xA, yA=yA, zA=zA, x1=x1, y1=y1, x2=x2, xB=xB, yB=yB, zB=zB, R=R,
                  zzplane=zzplane, xyAplane=xyAplane, xyBplane=xyBplane)


def _tensor(R, n=100):
    # MATLAB [x,y,z] = sphere(n): (n+1)x(n+1), azimuth along columns, elevation along rows
    theta = np.arange(-n, n + 1, 2) / n * np.pi
    phi = np.arange(-n, n + 1, 2).reshape(-1, 1) / n * np.pi / 2
    xs = np.cos(phi) * np.cos(theta)
    ys = np.cos(phi) * np.sin(theta)
    zs = np.sin(phi) * np.ones(theta.size)
    pts = R @ np.vstack([_CTENSOR[0] * xs.ravel(order='F'), _CTENSOR[1] * ys.ravel(order='F'),
                         _CTENSOR[2] * zs.ravel(order='F')])
    sz = xs.shape
    return tuple(pts[i].reshape(sz, order='F') for i in range(3))


def rotview(*varargin, label=1, showTensor=True, plot=True, sliders=True):
    """rotview(R) / rotview([alpha beta gamma]) / rotview(alpha, beta, gamma) (radians; default
    30, 20, 40 degrees): 3D view of the rotation from frame A (dashed axes) to frame B (solid
    axes), the line of nodes, the frames' xy planes, an ellipsoid fixed in B and the rotation
    matrix. label (1-9) picks the frame names (general, Sys.gFrame, ..., Exp.SampleFrame);
    sliders=True adds matplotlib sliders for the three angles (the MATLAB edit boxes and menu
    are keywords here).

    Returns a Struct with alpha, beta, gamma (degrees, clamped to [-360, 360]), the frame data
    of calculateframes (R, xA..zB, planes), the matrix text, labels, the ellipsoid surface
    (tensor) and fig/axes (plot=True only)."""
    nargin = len(varargin)
    if nargin == 1:
        R = np.asarray(varargin[0], dtype=float)
        if R.shape == (3, 3):
            alpha, beta, gamma = F.eulang(R, nargout=3)
        elif R.size == 3:
            alpha, beta, gamma = R.ravel()
        else:
            raise ValueError('For one input argument, it must be a 3x3 rotation matrix or a 3-element vector of Euler angles.')
    elif nargin == 3:
        alpha, beta, gamma = (float(np.asarray(v)) for v in varargin)
    else:
        alpha, beta, gamma = 30 * np.pi / 180, 20 * np.pi / 180, 40 * np.pi / 180
    ang = np.clip(np.array([alpha, beta, gamma], dtype=float) * 180 / np.pi, -360, 360)
    alpha, beta, gamma = (float(a) for a in ang)
    out = Struct(alpha=alpha, beta=beta, gamma=gamma)
    out.update(calculateframes(alpha, beta, gamma))
    out.matrix = [['%5.4f' % out.R[r, c] for c in range(3)] for r in range(3)]
    An, Bn = _NAMES[label - 1]
    out.labels = ['x_' + An, 'y_' + An, 'z_' + An, 'x_' + Bn, 'y_' + Bn, 'z_' + Bn]
    out.tensor = _tensor(out.R)
    if plot:
        _draw(out, label, showTensor, sliders)
    return out


def _draw(out, label, showTensor, sliders):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(443123, figsize=(10, 6))
    fig.clf()
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title('Rotation viewer')
    ax = fig.add_axes([0.2, 0, 0.8, 1], projection='3d')
    state = {'a': [out.alpha, out.beta, out.gamma]}

    def render(a, b, g):
        ax.cla()
        fr = calculateframes(a, b, g)
        An, Bn = _NAMES[label - 1]
        cols = ['r', (0, 0.6, 0), 'b']
        for k, v in enumerate((fr.xA, fr.yA, fr.zA)):
            ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color=cols[k], linewidth=2, linestyle='--')
        for k, v in enumerate((fr.xB, fr.yB, fr.zB)):
            ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color=cols[k], linewidth=2)
        for v in (fr.y1, -fr.y1, fr.x1, fr.x2):
            ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color='k')
        names = ['x', 'y', 'z']
        for k, v in enumerate((fr.xA, fr.yA, fr.zA, fr.xB, fr.yB, fr.zB)):
            fn = An if k < 3 else Bn
            ax.text(*(1.1 * v), '$x_{%s}$'.replace('x', names[k % 3]) % fn)
        for P, c in ((fr.zzplane, 'none'), (fr.xyAplane, 'r'), (fr.xyBplane, 'g')):
            ax.add_collection3d(Poly3DCollection([P.T], facecolors=c, edgecolors='k', alpha=0.1))
        if showTensor:
            tx, ty, tz = _tensor(fr.R, 40)
            ax.plot_surface(tx, ty, tz, color=(0.6, 0.7, 0.9), linewidth=0, alpha=0.8)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        ax.view_init(elev=16, azim=167 - 90)
        txt = '\n'.join('  '.join('%7.4f' % fr.R[r, c] for c in range(3)) for r in range(3))
        ax.text2D(-0.2, 0.25, 'rows %s, columns %s\n%s' % ('x/y/z_' + Bn, 'x/y/z_' + An, txt),
                  transform=ax.transAxes, family='monospace', fontsize=8)
        fig.canvas.draw_idle()

    render(*state['a'])
    if sliders:
        from matplotlib.widgets import Slider
        sl = []
        for k, (nm, val) in enumerate(zip(('alpha (deg)', 'beta (deg)', 'gamma (deg)'), state['a'])):
            sax = fig.add_axes([0.08, 0.02 + 0.05 * (2 - k), 0.18, 0.03])
            s = Slider(sax, nm, -360, 360, valinit=val, valstep=1)
            sl.append(s)

        def update(_):
            render(*(s.val for s in sl))
        for s in sl:
            s.on_changed(update)
        fig._rotview_sliders = sl      # keep references alive
    out.fig = fig
    out.axes = ax
    return fig

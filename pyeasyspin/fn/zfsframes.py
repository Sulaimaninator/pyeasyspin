"""Print a D tensor in the various principal-axis-system conventions."""
import numpy as np
from .._mat import vec
from .._ham import perms


def zfsframes(*varargin):
    """zfsframes(D) or zfsframes(D1, D2, D3): prints D and E in all six axis assignments."""
    if len(varargin) == 1:
        D = vec(varargin[0]).astype(float)
    elif len(varargin) == 3:
        D = np.array([float(v) for v in varargin])
    else:
        raise ValueError('Please provide three principal values of D tensor.')
    p = print
    p('\n==================================================================')
    p('D tensor in various axis system conventions')
    p('==================================================================')
    p(f'Principal values (PVs) of D tensor in input order:\n  D1 = {D[0]:+g}     D2 = {D[1]:+g}     D3 = {D[2]:+g}')
    if np.sum(D) != 0:
        meanD = np.mean(D)
        D = D - meanD
        p(f"PVs don't sum up to 0, subtracting mean ({meanD:f}).")
        p(f'PVs of traceless D tensor in input order:\n  D1 = {D[0]:+g}     D2 = {D[1]:+g}     D3 = {D[2]:+g}')
    else:
        p('This D tensor is traceless (sum of PVs is zero).')
    if np.all(D == 0):
        p('All D values are zero.')
        return
    if D[0] == D[1] or D[0] == D[2] or D[1] == D[2]:
        p('This D tensor is exactly axial (two PVs are equal).')
    elif np.any(D == 0):
        p('This D tensor is exactly orthorhombic (one PV is zero).')
    else:
        p('This D tensor is neither exactly axial nor exactly orthorhombic.')
    p('Principal axis system conventions:')
    p('  Poole 1974:    |Dz|>=|Dx|>=|Dy|    (used for organic triplets)')
    p('  Blumberg 1967: |Dz|>=|Dy|>=|Dx|    (used for transition metals)')
    p('  (for references see documentation)')
    p('Computation of D and E:')
    p('  D = 3/2*Dz      E = (Dx-Dy)/2')
    p('  Dx = -D/3+E     Dy = -D/3-E        Dz = 2/3*D')
    ax = perms([3, 2, 1])
    D_, E_, name = [0.0] * 6, [0.0] * 6, [''] * 6
    for k in range(5, -1, -1):
        Dx, Dy, Dz = D[ax[k][0] - 1], D[ax[k][1] - 1], D[ax[k][2] - 1]
        D_[k] = 1.5 * Dz
        E_[k] = (Dx - Dy) / 2
        if abs(Dz) >= abs(Dx) >= abs(Dy):
            name[k] = 'Poole '
        if abs(Dz) >= abs(Dy) >= abs(Dx):
            name[k] += 'Blumberg '
    with np.errstate(divide='ignore', invalid='ignore'):
        ED = np.array(E_) / np.array(D_)
    p('D and E in all possible principal axis systems:')
    for k in range(6):
        a = ax[k]
        p(f'  {a[0]}{a[1]}{a[2]}->xyz:  D = {D_[k]:<+5.3f}    E = {E_[k]:<+5.3f}    E/D = {ED[k]:<+5.2f}    {name[k]}')
    p(' ')

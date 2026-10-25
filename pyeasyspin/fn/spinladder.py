"""Spin-ladder (strong-exchange) manifold Hamiltonians of a two-electron-spin system."""
import numpy as np
from .._registry import F, nout
from .._struct import Struct
from .._mat import a2, vec, numel


@nout(2)
def spinladder(Sys=None, Temperature=None, nargout=2):
    """(CoupledSystems, Energies): one spin system per total spin S of the exchange-coupled pair,
    sorted by energy (MHz); with a finite Temperature (K) each gets a Boltzmann .weight."""
    if Temperature is None:
        Temperature = np.inf
    if not isinstance(Sys, dict):
        raise ValueError('First input must be a spin system structure.')
    if numel(Sys['S']) != 2:
        raise ValueError('Only systems with two electron spins are supported.')
    Sys_, err = F.validatespinsys(Sys)
    if err:
        raise ValueError(err)
    nNuclei = int(Sys_.nNuclei)
    if 'ee2' in Sys_ and np.any(vec(Sys_.ee2) != 0):
        raise ValueError('Cannot compute spin ladder with Sys.ee2 present.')
    Sys = Struct(Sys)
    Sa, Sb = vec(Sys.S)[0], vec(Sys.S)[1]
    S = np.arange(abs(Sa - Sb), Sa + Sb + 1e-9, 1.0)
    Sa2, Sb2 = Sa * (Sa + 1), Sb * (Sb + 1)
    S2 = S * (S + 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        c = (Sa2 - Sb2) / S2
        cp = (3 * (Sa2 - Sb2) ** 2 + S2 * (3 * S2 - 3 - 2 * Sa2 - 2 * Sb2)) / ((2 * S + 3) * (2 * S - 1) * S2)
        cm = (4 * S2 * (Sa2 - Sb2) - 3 * (Sa2 - Sb2)) / ((2 * S + 3) * (2 * S - 1) * S2)
    c[S == 0] = 0; cp[S == 0] = 0; cm[S == 0] = 0
    c1, c2 = (1 + c) / 2, (1 - c) / 2
    d1, d2, d12 = (cp + cm) / 2, (cp - cm) / 2, (1 - cp) / 2
    full_ee = False
    if 'ee' in Sys:
        ee = a2(Sys.ee)
        if ee.size == 1:
            J = ee.item()
        elif ee.size == 3:
            J = np.mean(ee)
        elif ee.size == 9:
            full_ee = True
            J = np.mean(np.diag(ee))
        else:
            raise ValueError('Sys.ee must have 1 or 3 elements.')
    elif 'J' in Sys:
        J = Sys.J
    Energies = J / 2 * (S2 - Sa2 - Sb2)
    if 'g' in Sys:
        g = a2(Sys.g)
        if g.size == 2:
            g1, g2 = vec(g)[0], vec(g)[1]
        else:
            g1, g2 = g[0, :], g[1, :]
    else:
        g1 = g2 = 2
    if nNuclei > 0:
        A = a2(Sys.A)
        if A.size == 4:
            A1, A2 = A[:, 0], A[:, 1]
        else:
            A1, A2 = A[:, 0:3], A[:, 3:6]
    if 'D' not in Sys:
        Sys.D = np.zeros((2, 3))
    D = a2(Sys.D)
    if D.size == 2:
        Dv = vec(D)
        Sys.D = np.vstack([Dv[0] * np.array([-1, -1, 2]) / 3, Dv[1] * np.array([-1, -1, 2]) / 3])
    elif D.size == 4:
        Sys.D = np.vstack([np.array([-1 / 3, -1 / 3, 2 / 3]) * D[0, 0] + np.array([1, -1, 0]) * D[0, 1],
                           np.array([-1 / 3, -1 / 3, 2 / 3]) * D[1, 0] + np.array([1, -1, 0]) * D[1, 1]])
    D1, D2 = a2(Sys.D)[0, :], a2(Sys.D)[1, :]
    if full_ee:
        D12 = a2(Sys.ee) - J * np.eye(3)
        D1, D2 = np.diag(D1), np.diag(D2)
    else:
        D12 = a2(Sys.ee) - J if 'ee' in Sys else 0
    CS = [None] * S.size
    for iS in range(S.size - 1, -1, -1):
        s = Struct(Sys)
        s.S = S[iS]
        s.g = c1[iS] * g1 + c2[iS] * g2
        if nNuclei > 0:
            s.A = c1[iS] * A1 + c2[iS] * A2
        if s.S > 0.5:
            s.D = d1[iS] * D1 + d2[iS] * D2 + d12[iS] * D12
        elif 'D' in s:
            del s['D']
        s.pop('ee', None)
        s.pop('J', None)
        CS[iS] = s
    if not np.isinf(Temperature):
        P = np.exp(-(Energies - Energies.min()) * 1e6 * F.planck() / F.boltzm() / Temperature)
        P = P * (2 * S + 1)
        P = P / P.sum()
        for iS in range(S.size - 1, -1, -1):
            CS[iS].weight = P[iS]
    idx = np.argsort(Energies, kind='stable')
    Energies = Energies[idx]
    CS = [CS[i] for i in idx]
    if nargout == 0:
        print(f'S1 = {Sa:g}, S2 = {Sb:g} (total {int(np.sum(2 * vec(Sys.S) + 1))} electronic states)')
        print(f'{S.size} manifolds (highest to lowest energy):')
        for iS in range(len(Energies) - 1, -1, -1):
            pop = ''
            if not np.isinf(Temperature):
                pop = (f'\n     population {P[idx][iS] / (2 * CS[iS].S + 1):0.3e}/state, {P[idx][iS]:0.3e} total')
            print(f'  S = {CS[iS].S:g} ({int(2 * CS[iS].S + 1)} electronic states)\n     energy '
                  f'{Energies[iS] / 1e3:+g} GHz{pop}')
        return None
    return CS, Energies

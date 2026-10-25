"""Electron coherence transfer pathways that refocus into an echo at the detection point."""
import itertools

import numpy as np

from .._mat import vec


def sf_pathways(t=None, inc=None, nargout=None):
    """Refocusing pathways (one row per pathway, codes 1=alpha, 2=beta, 3=+1, 4=-1 coherence; last
    column 4 = detection) for delays t (us) and incrementation scheme inc (0 = fixed, +-d = dimension
    d). A single interval gives 4. nargout=0 (or no input) prints the pathway table instead."""
    nPoints = 256
    dt = 0.08
    Display = nargout == 0
    if t is None:
        Display = True
        tau1, tau2, T0 = 30, 100, 20
        t = np.array([tau1, tau1, T0, tau2, tau2]) * 1e-3
        inc = np.array([0, 0, +1, 0, 0])
    t = vec(np.asarray(t, dtype=float))
    inc = vec(np.asarray(inc, dtype=float))
    if inc.size != t.size:
        raise ValueError('inc (first input) and t (second input) must have the same number of entries.')
    nIntervals = t.size
    if nIntervals > 1:
        N = nIntervals - 1
        # [idx{1:N}] = ndgrid(1:4); fliplr(reshape(cat(...),[],N)): first column varies slowest
        Pathways = np.array(list(itertools.product(range(1, 5), repeat=N)), dtype=float).reshape(-1, N)
        Pathways = np.concatenate([Pathways, 4 * np.ones((Pathways.shape[0], 1))], axis=1)
        nPathways = 4 ** N
        nEchoPathways = 4 ** N - 3 ** N
        coherenceOrder = np.array([0, 0, +1, -1], dtype=float)
        eco = coherenceOrder[Pathways.astype(int) - 1]
        maxinc = int(np.max(np.abs(inc)))
        nFlips = np.zeros(nPathways)
        flipping = np.zeros(nPathways, dtype=bool)
        t0 = np.zeros(nPathways)
        incdec = np.zeros((nPathways, max(maxinc, 0)))
        refocusing = np.zeros(nPathways, dtype=bool)
        crossing = np.zeros(nPathways, dtype=bool)
        for p in range(nPathways):
            co = eco[p, :]
            co = co[co != 0]
            nFlips[p] = np.sum(np.abs(np.diff(co)) == 2)
            flipping[p] = nFlips[p] > 0
            s = 0.0
            for a, b in zip(eco[p, :], t):          # row*column product, summed left to right
                s += a * b
            t0[p] = s
            if maxinc > 0:
                for d in range(1, maxinc + 1):
                    incdec[p, d - 1] = np.sum(eco[p, np.abs(inc) == d])
                refocusing[p] = flipping[p] and t0[p] == 0 and np.all(incdec[p, :] == 0)
                crossing[p] = flipping[p] and np.any((t0[p] + nPoints * dt * incdec[p, :]) * t0[p] <= 0)
            else:
                refocusing[p] = flipping[p] and t0[p] == 0
                crossing[p] = flipping[p] and t0[p] == 0
        refocusingPathways = Pathways[refocusing, :]
        if Display:
            print('%d pulses\n%d eCTPs\n%d eCTPs give at least one echo' % (nIntervals, nPathways, nEchoPathways))
            print('%d eCTPs lead to an echo always in the detection window' % np.sum(refocusing))
            print('%d crossing echoes' % (np.sum(crossing) - np.sum(refocusing)))
            Str = 'ab+-'
            print('CTP   #echoes  time  type        walk directions')
            for iCTP in range(nPathways):
                if nFlips[iCTP] > 0 and crossing[iCTP]:
                    line = '%s    %d    %+5.5g' % (''.join(Str[int(k) - 1] for k in Pathways[iCTP, :]),
                                                  nFlips[iCTP], t0[iCTP])
                    if refocusing[iCTP]:
                        line += '  fixed       '
                    elif crossing[iCTP]:
                        line += '  crossing    '
                    for iDim in range(maxinc):
                        line += '   %+g' % incdec[iCTP, iDim]
                    print(line)
    else:
        refocusingPathways = 4.0
        if Display:
            print('Single pulse: no echo')
    return None if nargout == 0 else refocusingPathways

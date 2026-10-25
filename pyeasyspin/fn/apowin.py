"""Apodization windows"""
import numpy as np


def apowin(Type=None, n=None, alpha=None):
    """apowin(Type, n[, alpha]) -> n x 1 window, max-normalized. Types ham, kai, gau, exp, han, bla,
    bar, con, cos, wel; a 4th character '+' or '-' gives the one-sided half."""
    if not isinstance(Type, str) or len(Type) < 3 or len(Type) > 4:
        raise ValueError('The first argument must be a 3- or 4-character string!')
    if len(Type) == 4:
        if Type[3] == '+':
            xmin, xmax = 0, 1
        elif Type[3] == '-':
            xmin, xmax = -1, 0
        else:
            raise ValueError('Wrong 4th character in Type. Should be + or -.')
    else:
        xmin, xmax = -1, 1
    if np.iscomplexobj(n) or not (n > 0) or np.mod(n, 1):
        raise ValueError('n must be a positive integer!')
    n = int(n)
    nargin = 2 if alpha is None else 3
    x = np.linspace(xmin, xmax, n)
    t = Type[:3]
    two = {'ham': lambda: 0.54 + 0.46 * np.cos(np.pi * x),
           'han': lambda: 0.5 + 0.5 * np.cos(np.pi * x),
           'bla': lambda: 0.42 + 0.5 * np.cos(np.pi * x) + 0.08 * np.cos(2 * np.pi * x),
           'bar': lambda: 1 - np.abs(x),
           'con': lambda: (1 - x ** 2) ** 2,
           'cos': lambda: np.cos(np.pi * x / 2),
           'wel': lambda: 1 - x ** 2}
    three = ('kai', 'gau', 'exp')
    if t in two:
        n_arg = 2
    elif t in three:
        n_arg = 3
    else:
        raise ValueError('Unknown apodization window specified!')
    if nargin != n_arg:
        raise ValueError('Wrong number of input arguments')
    if t in two:
        w = two[t]()
    elif t == 'kai':
        from scipy.special import iv
        w = iv(0, alpha * np.sqrt(1 - x ** 2)) / iv(0, alpha)
    elif t == 'gau':
        w = np.exp(-2 * x ** 2 / alpha ** 2)
    else:
        w = np.exp(-alpha * np.abs(x))
    if xmax - xmin == 2:
        w = (w + w[::-1]) / 2
    w = w / np.max(w)
    return w.reshape(-1, 1)

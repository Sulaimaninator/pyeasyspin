"""All combinations of elements from several groups."""
import numpy as np
from .._mat import a2, vec


def allcombinations(*varargin):
    """allcombinations(g1, g2, ..., mode): mode 'c' (default, combinations as rows), '+' (sums),
    '*' (products), 'i' (index combinations, 1-based)."""
    if varargin and isinstance(varargin[-1], str):
        Elements, mode = list(varargin[:-1]), varargin[-1]
    else:
        Elements, mode = list(varargin), 'c'
    if mode == 'i':
        Elements = [np.arange(1, np.size(x) + 1, dtype=float) for x in Elements]
    Elements = [a2(x) for x in Elements]
    nEl = [x.size for x in Elements]
    n1 = int(np.prod(nEl)) if nEl else 1
    n2 = 1
    Result = 0.0 if mode == '+' else 1.0 if mode == '*' else np.zeros((n1, 0))
    for g, E in enumerate(Elements):
        n1 //= nEl[g]
        R = vec(np.tile(E, (n1, n2))).reshape(-1, 1)
        if mode == '+':
            Result = Result + R
        elif mode == '*':
            Result = Result * R
        else:
            Result = np.hstack([Result, R])
        n2 *= nEl[g]
    return Result

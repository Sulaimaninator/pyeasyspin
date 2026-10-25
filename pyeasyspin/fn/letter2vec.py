"""a string shorthand for an direction to the corresponding unit vector."""
import numpy as np

_V = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1], 'xy': [1, 1, 0], 'xz': [1, 0, 1],
      'yz': [0, 1, 1], 'xyz': [1, 1, 1]}


def letter2vec(s=None):
    """letter2vec('x'|'-xy'|...) -> 3x1 unit vector."""
    if not isinstance(s, str):
        raise ValueError('Input argument must be a string or character array.')
    sign, key = 1.0, s
    if s.startswith('-'):
        sign, key = -1.0, s[1:]
    if key not in _V:
        raise ValueError(f"Unknown value '{s}' for direction.")
    n = sign * np.array(_V[key], dtype=float).reshape(3, 1)
    return n / np.linalg.norm(n)

"""Spin operator matrices."""
import re
import numpy as np
import scipy.sparse as sps
from .._registry import F
from .._mat import vec, a2, isnumeric

_NUM2CHAR = {0: 'e', 1: 'x', 2: 'y', 3: 'z', 4: '+', 5: '-'}


def sop(SpinSystem=None, *varargin, nargout=None):
    """Spin operator matrix for a spin vector or spin system and an operator specification:
    'xe', 'x1z2', 'x(1|2)', 'z,e,x', or a numeric Nx2 array [spin, component]. Several
    specifications give a tuple of matrices. A trailing 'sparse' returns scipy.sparse (CSR)."""
    SpinVec = F.spinvec(SpinSystem) if isinstance(SpinSystem, dict) else SpinSystem
    if not isnumeric(SpinVec) or (np.ndim(SpinVec) > 1 and min(np.shape(SpinVec)) != 1) or np.size(SpinVec) == 0:
        raise ValueError('First input must be a 1D array of spin quantum numbers.')
    SpinVec = vec(SpinVec)
    if np.any(SpinVec < 0) or np.any(np.mod(SpinVec, 0.5)) or np.iscomplexobj(SpinVec):
        raise ValueError('First input must contain valid spin quantum numbers (0, 1/2, 1, 3/2, etc).')
    SpinVec = SpinVec.astype(float)
    nSpins = SpinVec.size
    if len(varargin) == 0:
        raise ValueError('Not enough input arguments - at least 2 are needed!')
    useSparse = isinstance(varargin[-1], str) and varargin[-1].lower() == 'sparse'
    OperatorSpec = list(varargin[:-1]) if useSparse else list(varargin)
    Transitions = None
    if not isinstance(OperatorSpec[0], str):
        if len(OperatorSpec) != 1:
            raise ValueError('Incorrect number of input arguments.')
        a = a2(OperatorSpec[0])
        if a.shape[1] != 2:
            raise ValueError('For numeric input, the array specifying spins and components must be Nx2.')
        Spins = a[:, 0].astype(int)
        Coords = [_NUM2CHAR.get(int(c), '?') for c in a[:, 1]]
    else:
        if nargout is not None and nargout != len(OperatorSpec) and nargout > 0:
            raise ValueError(f'sop: Number of output arguments ({nargout}) and number of requested operator '
                             f'matrices ({len(OperatorSpec)}) do not match.')
        if len(OperatorSpec) > 1:
            extra = ('sparse',) if useSparse else ()
            return tuple(sop(SpinVec, o, *extra) for o in OperatorSpec)
        spec = OperatorSpec[0]
        isS1 = re.fullmatch('[exyzab+-]{%d}' % nSpins, spec) is not None
        isS2 = isS3 = False
        if not isS1:
            isS2 = re.fullmatch(r'([exyzab+-]\d+){1,%d}' % nSpins, spec) is not None
            if not isS2:
                parts = spec.split(',')
                matched = [re.fullmatch(r'[exyzab+-](\(\d+(\|\d+)?\))?(\d+)?', p) for p in parts]
                isS3 = all(m is not None for m in matched)
                if not isS3:
                    raise ValueError(f"sop: Cannot parse '{spec}' for the given spin system.")
        if isS1:
            Coords = list(spec)
            Spins = np.arange(1, nSpins + 1)
        elif isS2:
            toks = re.findall(r'([exyzab+-])(\d+)', spec)
            Spins = np.array([int(t[1]) for t in toks])
            if np.any(Spins > nSpins):
                raise ValueError(f"sop: The spin system only contains {nSpins} spins, but you requested '{spec}'.")
            if len(set(Spins.tolist())) < Spins.size:
                raise ValueError(f"sop: Duplicate spin index in '{spec}'.")
            Coords = [t[0] for t in toks]
        else:
            Coords, Spins = [], []
            Transitions = [None] * max(nSpins, len(parts))
            present = []
            for it, m in enumerate(matched):
                s = m.group(0)
                if '(' not in s:
                    tk = re.match(r'([exyzab+-])(\d?)', s).groups()
                    Coords.append(tk[0])
                    Transitions[it] = None
                    present.append(tk[1] != '')
                    Spins.append(it + 1 if tk[1] == '' else int(tk[1]))
                else:
                    tk = re.match(r'([exyzab+-])\((\d+)\|?(\d+)?\)(\d?)', s).groups('')
                    Coords.append(tk[0])
                    if tk[0] in 'ab':
                        raise ValueError(f"sop: 'a' or 'b' cannot be used in conjunction with levels in '{spec}'")
                    present.append(tk[3] != '')
                    Spins.append(it + 1 if tk[3] == '' else int(tk[3]))
                    if tk[1] != '':
                        if tk[2] != '':
                            tr = [int(tk[1]), int(tk[2])]
                            if tk[0] in 'xyz+-' and tr[0] == tr[1]:
                                raise ValueError(f"sop: The component '{s}' of your spin operator connects identical levels.")
                            elif tk[0] == 'e' and tr[0] != tr[1]:
                                raise ValueError(f"sop: The component '{s}' can not connect two different levels. "
                                                 f"Use e(L1) instead of e(L1|L2).")
                            elif tr[1] - tr[0] < 0:
                                raise ValueError(f"sop: The two levels (L1|L2) in '{s}' must be ascending order such "
                                                 f"that L1<=L2.")
                            Transitions[it] = tr
                        else:
                            Transitions[it] = [int(tk[1])]
                            if tk[0] in 'xyz+-':
                                raise ValueError(f"sop: The component '{s}' of your operator must connect two "
                                                 f"(different) levels.")
            if any(present) and not all(present):
                raise ValueError(f"sop: Please give spin indices for all components, or omit them all. "
                                 f"'{spec}' is inconsistent.")
            if not any(present) and len(matched) != nSpins:
                raise ValueError(f"sop: Could not determine what '{spec}' is for the given spin system.")
            elif not any(present):
                Spins = list(range(1, nSpins + 1))
            Spins = np.array(Spins)
            if len(set(Spins.tolist())) < Spins.size:
                raise ValueError(f"sop: Repeated spin index in '{spec}'.")
        for c, s in zip(Coords, Spins):
            if c in 'ab' and SpinVec[s - 1] != 0.5:
                raise ValueError("'a' and 'b' work only for spin-1/2.")
    Components = ['e'] * nSpins
    for c, s in zip(Coords, Spins):
        Components[int(s) - 1] = c
    Op = sps.csr_matrix(np.ones((1, 1)))
    for i in range(nSpins):
        I = SpinVec[i]
        if I <= 0:
            continue
        n = int(round(2 * I + 1))
        T = Transitions[i] if Transitions is not None and i < len(Transitions) else None
        comp = Components[i]
        if comp == 'x':
            if T:
                r, c, val = [T[0], T[1]], [T[1], T[0]], [0.5, 0.5]
            else:
                m = np.arange(1, n)
                dia = 0.5 * np.sqrt(m * m[::-1])
                r, c, val = np.r_[m, m + 1], np.r_[m + 1, m], np.r_[dia, dia]
        elif comp == 'y':
            if T:
                r, c, val = [T[0], T[1]], [T[1], T[0]], [-0.5j, 0.5j]
            else:
                m = np.arange(1, n)
                dia = -0.5j * np.sqrt(m * m[::-1])
                r, c, val = np.r_[m, m + 1], np.r_[m + 1, m], np.r_[dia, -dia]
        elif comp == 'z':
            if T:
                r, c, val = [T[0], T[1]], [T[0], T[1]], [0.5, -0.5]
            else:
                m = np.arange(1, n + 1)
                r, c, val = m, m, I + 1 - m
        elif comp == '+':
            if T:
                r, c, val = [T[0]], [T[1]], [1.0]
            else:
                m = np.arange(1, n)
                r, c, val = m, m + 1, np.sqrt(m * m[::-1])
        elif comp == '-':
            if T:
                r, c, val = [T[1]], [T[0]], [1.0]
            else:
                m = np.arange(1, n)
                r, c, val = m + 1, m, np.sqrt(m * m[::-1])
        elif comp == 'a':
            r, c, val = [1], [1], [1.0]
        elif comp == 'b':
            r, c, val = [2], [2], [1.0]
        elif comp == 'e':
            if T:
                r, c, val = [T[0]], [T[0]], [1.0]
            else:
                m = np.arange(1, n + 1)
                r, c, val = m, m, np.ones(n)
        else:
            raise ValueError('Unknown operator specification.')
        M = sps.csr_matrix((np.asarray(val), (np.asarray(r) - 1, np.asarray(c) - 1)), shape=(n, n))
        Op = sps.kron(Op, M, format='csr')
    if not useSparse:
        Op = Op.toarray()
    return Op

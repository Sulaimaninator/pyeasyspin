"""Full spin Hamiltonian."""
import numpy as np
import scipy.sparse as sps
from .._registry import F, nout
from .._mat import isempty, isnumeric, numel, vec
from .._ham import add, dense


def _conv(x, sparse):
    """Make every matrix in a (nested list of) ham_ezho output(s) sparse or dense."""
    if isinstance(x, list):
        return [_conv(v, sparse) for v in x]
    if x is None or (isinstance(x, np.ndarray) and x.size == 0):
        return x
    return sps.csr_matrix(x) if sparse else dense(x)


@nout(-1)
def ham(SpinSystem=None, B0=None, opt=None, nargout=None):
    """ham(Sys, B0) -> H (MHz) at field vector B0 (mT); ham(Sys, B0, nargout=2) -> (H0, muzL);
    ham(Sys, nargout=4) -> (H0, muxM, muyM, muzM). opt='sparse' gives scipy.sparse matrices.
    With nargout=1 and no field, a nested list is returned as in EasySpin."""
    if nargout is None:
        nargout = 1 if not isempty(B0) else 4
    if opt is None:
        opt = ''
    if not isinstance(opt, str):
        raise ValueError("Third argument must be a string, 'sparse' or ''.")
    sparse = opt == 'sparse'
    sp = 'sparse' if sparse else 'full'
    Sys, err = F.validatespinsys(SpinSystem)
    if err:
        raise ValueError(err)
    if not isempty(B0):
        if not isnumeric(B0) or numel(B0) != 3 or np.iscomplexobj(B0):
            raise ValueError('Magnetic field vector must be 3-element array!')
    H0 = F.ham_zf(Sys, None, sp)
    for t in (F.ham_ee(Sys, None, sp), F.ham_hf(Sys, None, None, sp), F.ham_nq(Sys, None, sp),
              F.ham_nn(Sys, None, sp), F.ham_so(Sys, None, sp), F.ham_cf(Sys, None, sp)):
        H0 = add(H0, t)
    mux, muy, muz = F.ham_ez(Sys, None, sp, nargout=3)
    if Sys.nNuclei > 0:
        a, b, c = F.ham_nz(Sys, None, sp, nargout=3)
        mux, muy, muz = add(mux, a), add(muy, b), add(muz, c)
    if Sys.nL > 0:
        a, b, c = F.ham_oz(Sys, None, sp, nargout=3)
        mux, muy, muz = add(mux, a), add(muy, b), add(muz, c)
    highest = 0
    ho = False
    for f in Sys:
        if f[:3] == 'Ham' and np.any(np.asarray(Sys[f]) != 0):
            ho = True
            order = int(f[3])
            if highest < order < 4:
                highest = order

    if isempty(B0):
        if ho:
            zHo = _conv(F.ham_ezho(Sys, None, opt), sparse)
            if highest == 0:
                # MATLAB adds a cell to a matrix here (an error); the single order-0 term is used
                out = (add(H0, zHo[0]), mux, muy, muz)
            elif highest == 1:
                out = (add(H0, zHo[0]), add(mux, zHo[1][0]), add(muy, zHo[1][1]), add(muz, zHo[1][2]))
            else:
                zHo[0] = add(zHo[0], H0)
                Gn = [mux, muy, muz]
                for k in (2, 1, 0):
                    zHo[1][k] = add(zHo[1][k], Gn[k])
                if nargout == highest + 1:
                    return tuple(zHo)
                elif nargout == 1:
                    return zHo
                raise ValueError('Wrong number of output arguments!')
            if nargout == 4:
                return out
            elif nargout == 1:
                return list(out)
            raise ValueError('1 or 4 output arguments expected!')
        if nargout == 4:
            return H0, mux, muy, muz
        elif nargout == 1:
            return [[H0, mux, muy, muz]]
        raise ValueError('1 or 4 output arguments expected!')

    B0 = vec(B0).astype(float)
    nrm = np.linalg.norm(B0)
    nB0 = B0 / nrm if nrm > 0 else np.zeros(3)
    muzL = add(add(nB0[0] * mux, nB0[1] * muy), nB0[2] * muz)
    if ho:
        if nargout in (0, 1):
            return add(add(H0, -nrm * muzL), _conv(F.ham_ezho(Sys, B0, None, opt), sparse))
        elif nargout == 2 and highest < 2:
            zHo = _conv(F.ham_ezho(Sys, None, opt), sparse)
            if highest == 1:
                muzL = add(add(add(muzL, nB0[0] * zHo[1][0]), nB0[1] * zHo[1][1]), nB0[2] * zHo[1][2])
            H0 = add(H0, zHo[0])
            return H0, muzL
        raise ValueError('Wrong number of output arguments!')
    if nargout in (0, 1):
        return add(H0, -nrm * muzL)
    elif nargout == 2:
        return H0, muzL
    raise ValueError('Wrong number of output arguments!')

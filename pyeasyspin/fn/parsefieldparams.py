"""Convert string-valued parameter fields to numbers/booleans where possible."""
import numpy as np
from .._struct import Struct
from .._io import sscanf_f


def parsefieldparams(ParamsIn=None):
    """Fields 'true'/'false' -> bool; fields that sscanf('%e') parses completely -> float (one
    value) or 1xN row; everything else (and values starting with a letter) stays a string."""
    if not isinstance(ParamsIn, dict):
        return ParamsIn
    Pout = Struct(ParamsIn)
    for k, v in list(Pout.items()):
        if not isinstance(v, str) or len(v) == 0:
            continue
        if v.lower() == 'true':
            Pout[k] = True
            continue
        if v.lower() == 'false':
            Pout[k] = False
            continue
        if v[0].isalpha():
            continue
        vals, err = sscanf_f(v)
        if err or not vals:
            continue
        Pout[k] = vals[0] if len(vals) == 1 else np.array(vals, dtype=float).reshape(1, -1)
    return Pout

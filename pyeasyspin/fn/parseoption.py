"""Parse a string option field against a list of allowed values."""
from .._registry import nout
from .._struct import isfield


@nout(2)
def parseoption(Stru=None, Field=None, Values=None, nargout=1):
    """Return the 1-based index of Stru.(Field) in Values (case-insensitive). With nargout<2 an
    error is raised on failure; with nargout=2 returns (index, errmsg) with [] / None on failure."""
    err = []
    out = []
    if isinstance(Values, str):
        Values = [Values]
    if isfield(Stru, Field):
        this = Stru[Field]
        found = [isinstance(this, str) and isinstance(v, str) and this.lower() == v.lower() for v in Values]
        n = sum(found)
        if n == 0:
            err = f"Unknown value '{this}' for field {Field}."
        elif n > 1:
            err = f"Ambiguous value '{this}' for field {Field}."
        else:
            out = found.index(True) + 1
    else:
        err = f"Missing field '{Field}'."
    if nargout < 2:
        if err:
            raise ValueError(err)
        return out
    return out, err

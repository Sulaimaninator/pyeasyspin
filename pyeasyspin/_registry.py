"""Function registry. Ported code calls EasySpin functions through F (F.ham(...), F.sop(...))
so that a test harness can substitute any callee.

Each MATLAB function NAME lives in pyeasyspin/fn/NAME.py as a function NAME. Unported functions
are stubs made with @stub: they raise NotImplementedError and carry the MATLAB signature."""
import importlib


class NotPortedError(NotImplementedError):
    pass


def stub(nout, matlab):
    """Mark a function as an unported stub. nout = number of MATLAB outputs; matlab = the
    MATLAB function line."""
    def deco(f):
        def wrapper(*args, **kwargs):
            raise NotPortedError(f'{f.__name__} is not ported yet (MATLAB: {matlab})')
        wrapper.__name__ = f.__name__
        wrapper.__doc__ = f.__doc__
        wrapper._es_stub = True
        wrapper._es_nout = nout
        wrapper._es_matlab = matlab
        return wrapper
    return deco


def nout(n):
    """Declare the number of MATLAB outputs of a ported function (used by the mocks)."""
    def deco(f):
        f._es_nout = n
        return f
    return deco


class _Registry:
    def __init__(self):
        self._cache = {}
        self._override = {}   # name -> callable; set by test harnesses only

    def get(self, name):
        f = self._cache.get(name)
        if f is None:
            mod = importlib.import_module(f'pyeasyspin.fn.{name}')
            f = getattr(mod, name)
            self._cache[name] = f
        return f

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        o = self._override.get(name)
        if o is not None:
            return o
        return self.get(name)

    def is_stub(self, name):
        return getattr(self.get(name), '_es_stub', False)


F = _Registry()

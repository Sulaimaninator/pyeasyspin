"""Struct: a dict with attribute access, standing in for MATLAB structs (Sys, Exp, Opt, ...)."""


class Struct(dict):
    """dict subclass: s.g and s['g'] are the same. Field order is insertion order, as in MATLAB."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name) from None

    def copy(self):
        return Struct(self)

    def __repr__(self):
        return 'Struct(' + ', '.join(f'{k}={v!r}' for k, v in self.items()) + ')'


def isfield(s, name):
    """MATLAB isfield for a single field name (False for non-structs)."""
    return isinstance(s, dict) and name in s


def fieldnames(s):
    return list(s.keys()) if isinstance(s, dict) else []

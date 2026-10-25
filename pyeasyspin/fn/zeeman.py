"""Obsolete EasySpin function: raises an error naming its replacement."""


def zeeman(*varargin):
    raise ValueError("The function zeeman is obsolete.\nUse ham_ez (for electrons), ham_nz (for nuclei) or ham_oz (for orbital angular momenta) instead.")

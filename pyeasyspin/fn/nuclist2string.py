"""Convert a list of isotope names to a comma-separated string."""


def nuclist2string(Nucs=None):
    """'1H,14N' from ['1H','14N']."""
    if not isinstance(Nucs, (list, tuple)):
        raise ValueError('Nuclear isotope list must be a cell array!')
    return ','.join(Nucs)

"""Module-level settings replacing EasySpin's global variables."""
import logging

log = logging.getLogger('pyeasyspin')
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('%(message)s'))
    log.addHandler(_h)
    log.propagate = False


class _Config:
    """EasySpinLogLevel -> config.LogLevel (None = logging off, as an empty global in MATLAB)."""

    def __init__(self):
        self.LogLevel = None


config = _Config()

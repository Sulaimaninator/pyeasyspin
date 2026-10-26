"""Module-level settings replacing EasySpin's global variables."""
import logging

log = logging.getLogger('pyeasyspin')
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('%(message)s'))
    log.addHandler(_h)
    log.propagate = False


class _Config:
    """EasySpinLogLevel -> config.LogLevel (None = logging off, as an empty global in MATLAB).

    Multicore settings (see pyeasyspin/_parallel.py):
      Workers          worker processes for independent simulations; 'auto' = all cores, 1 = serial
      BLASThreads      BLAS threads inside a pyeasyspin call; None = library default
      ParallelMinTime  seconds a batch must take serially before it goes to the worker pool

    MatlabShapes: False (default) returns MATLAB row/column vectors from public functions as 1-D
    arrays, so they plot and index like ordinary NumPy vectors; True keeps MATLAB's 1 x N / N x 1."""

    def __init__(self):
        self.LogLevel = None
        self.MatlabShapes = False
        self.Workers = 'auto'
        self.BLASThreads = 1
        self.ParallelMinTime = 0.2


config = _Config()

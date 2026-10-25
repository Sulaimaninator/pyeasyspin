"""Apply an RC filter to a signal"""
import numpy as np
from scipy.signal import lfilter


def rcfilt(y=None, SampleTime=None, TimeConstant=None, UpDown=None):
    """rcfilt(y, SampleTime, TimeConstant, UpDown='up') -> RC-filtered y (columns; 'down'/'dn'
    filters in reverse)."""
    if UpDown is None:
        UpDown = 'up'
    err = ''
    if np.iscomplexobj(TimeConstant):
        err = 'Time constant must be real!'
    if np.iscomplexobj(SampleTime):
        err = 'Sampling time must be real!'
    if np.size(TimeConstant) != 1:
        err = 'Time constant must be a scalar!'
    if np.size(SampleTime) != 1:
        err = 'Sampling time must be a scalar!'
    if np.size(TimeConstant) == 1 and TimeConstant < 0:
        err = 'Time constant must be positive or zero!'
    if np.size(SampleTime) == 1 and SampleTime <= 0:
        err = 'Sampling time must be positive!'
    if UpDown not in ('up', 'down', 'dn'):
        err = "Specify either 'up' or 'down'."
    if err:
        raise ValueError(err)
    y = np.asarray(y)
    if y.ndim < 2:
        y = y.reshape(1, -1)
    RowVector = y.shape[0] == 1
    if RowVector:
        y = y.T
    Invert = UpDown in ('dn', 'down')
    if Invert:
        y = y[::-1, :]
    if TimeConstant == 0:
        yF = y.copy()
    else:
        e = np.exp(-SampleTime / TimeConstant)
        yF = lfilter([1 - e], [1, -e], y, axis=0)
    if Invert:
        yF = yF[::-1, :]
    if RowVector:
        yF = yF.T
    return yF

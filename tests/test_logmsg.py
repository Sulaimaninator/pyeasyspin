"""logmsg(-Inf, ...) is EasySpin's always-show warning; MATLAB's own level check made it crash.
The port logs it as a warning at any log level (docs/DEVIATIONS.md)."""
import logging

import pytest

from pyeasyspin.config import config, log
from pyeasyspin._registry import F


def test_minus_inf_always_logged(caplog):
    old = config.LogLevel
    try:
        config.LogLevel = 0
        with caplog.at_level(logging.INFO, logger=log.name):
            F.logmsg(float('-inf'), 'WARNING: Cannot compute derivative of a stick spectrum!')
        assert any(r.levelno == logging.WARNING and 'stick spectrum' in r.getMessage() for r in caplog.records)
    finally:
        config.LogLevel = old


def test_other_negative_levels_still_rejected():
    old = config.LogLevel
    try:
        config.LogLevel = 1
        with pytest.raises(ValueError):
            F.logmsg(-1, 'x')
    finally:
        config.LogLevel = old

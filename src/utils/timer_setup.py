"""Windows timer resolution setup.

Windows sleep() has ~15ms granularity by default. This sets it to 1ms,
which is critical for smooth mouse movement.
"""

import ctypes
import atexit

_timer_set = False


def enable_high_resolution_timer():
    """Set Windows timer resolution to 1ms. Call once at program startup."""
    global _timer_set
    if _timer_set:
        return
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
        _timer_set = True
        atexit.register(_cleanup)
    except Exception:
        pass


def _cleanup():
    """Restore default timer resolution on exit."""
    global _timer_set
    if _timer_set:
        try:
            ctypes.windll.winmm.timeEndPeriod(1)
        except Exception:
            pass
        _timer_set = False

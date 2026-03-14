"""Low-level mouse input via Win32 SendInput (DirectInput compatible)."""

import ctypes
import ctypes.wintypes as wt
import time

# ── Win32 constants ──────────────────────────────────────────────────────────
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040

INPUT_MOUSE = 0

user32 = ctypes.windll.user32


# ── Structures ───────────────────────────────────────────────────────────────
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class INPUT(ctypes.Structure):
    class _INPUT_UNION(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]

    _fields_ = [
        ("type", ctypes.c_ulong),
        ("union", _INPUT_UNION),
    ]


def _send_mouse_input(dx: int, dy: int, flags: int) -> None:
    """Send a single mouse input event via SendInput."""
    mi = MOUSEINPUT(
        dx=dx,
        dy=dy,
        mouseData=0,
        dwFlags=flags,
        time=0,
        dwExtraInfo=ctypes.pointer(ctypes.c_ulong(0)),
    )
    inp = INPUT(type=INPUT_MOUSE)
    inp.union.mi = mi
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def move_relative(dx: int, dy: int) -> None:
    """Move mouse by relative pixels. This is what CS2 reads for aim."""
    _send_mouse_input(dx, dy, MOUSEEVENTF_MOVE)


def click(button: str = "left", hold_ms: float = 0) -> None:
    """Click a mouse button with optional hold duration."""
    if button == "left":
        down_flag, up_flag = MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP
    elif button == "right":
        down_flag, up_flag = MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP
    else:
        down_flag, up_flag = MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP

    _send_mouse_input(0, 0, down_flag)
    if hold_ms > 0:
        time.sleep(hold_ms / 1000.0)
    _send_mouse_input(0, 0, up_flag)


def mouse_down(button: str = "left") -> None:
    """Press mouse button down."""
    flags = {
        "left": MOUSEEVENTF_LEFTDOWN,
        "right": MOUSEEVENTF_RIGHTDOWN,
        "middle": MOUSEEVENTF_MIDDLEDOWN,
    }
    _send_mouse_input(0, 0, flags.get(button, MOUSEEVENTF_LEFTDOWN))


def mouse_up(button: str = "left") -> None:
    """Release mouse button."""
    flags = {
        "left": MOUSEEVENTF_LEFTUP,
        "right": MOUSEEVENTF_RIGHTUP,
        "middle": MOUSEEVENTF_MIDDLEUP,
    }
    _send_mouse_input(0, 0, flags.get(button, MOUSEEVENTF_LEFTUP))

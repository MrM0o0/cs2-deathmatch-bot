"""Low-level mouse input via Win32.

Uses SendInput for mouse MOVEMENT (CS2 reads raw input for aim).
Uses mouse_event for CLICKS (CS2 blocks SendInput clicks but accepts mouse_event).
"""

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

# ── Button state tracking ────────────────────────────────────────────────────
_button_state = {"left": False, "right": False, "middle": False}

_DOWN_FLAGS = {
    "left": MOUSEEVENTF_LEFTDOWN,
    "right": MOUSEEVENTF_RIGHTDOWN,
    "middle": MOUSEEVENTF_MIDDLEDOWN,
}
_UP_FLAGS = {
    "left": MOUSEEVENTF_LEFTUP,
    "right": MOUSEEVENTF_RIGHTUP,
    "middle": MOUSEEVENTF_MIDDLEUP,
}


# ── Structures (for SendInput mouse movement) ────────────────────────────────
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


def move_relative(dx: int, dy: int) -> None:
    """Move mouse by relative pixels via SendInput. CS2 reads this for aim."""
    mi = MOUSEINPUT(
        dx=int(dx), dy=int(dy), mouseData=0, dwFlags=MOUSEEVENTF_MOVE,
        time=0, dwExtraInfo=ctypes.pointer(ctypes.c_ulong(0)),
    )
    inp = INPUT(type=INPUT_MOUSE)
    inp.union.mi = mi
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def click(button: str = "left", hold_ms: float = 0) -> None:
    """Click a mouse button with optional hold duration. Uses mouse_event."""
    down_flag = _DOWN_FLAGS.get(button, MOUSEEVENTF_LEFTDOWN)
    up_flag = _UP_FLAGS.get(button, MOUSEEVENTF_LEFTUP)

    _button_state[button] = True
    user32.mouse_event(down_flag, 0, 0, 0, 0)
    if hold_ms > 0:
        time.sleep(hold_ms / 1000.0)
    user32.mouse_event(up_flag, 0, 0, 0, 0)
    _button_state[button] = False


def mouse_down(button: str = "left") -> None:
    """Press mouse button down. Uses mouse_event (works in CS2)."""
    _button_state[button] = True
    flag = _DOWN_FLAGS.get(button, MOUSEEVENTF_LEFTDOWN)
    user32.mouse_event(flag, 0, 0, 0, 0)


def mouse_up(button: str = "left") -> None:
    """Release mouse button. Uses mouse_event (works in CS2)."""
    flag = _UP_FLAGS.get(button, MOUSEEVENTF_LEFTUP)
    user32.mouse_event(flag, 0, 0, 0, 0)
    _button_state[button] = False


def is_button_down(button: str = "left") -> bool:
    """Check if a button is currently held down."""
    return _button_state.get(button, False)


def ensure_released(button: str = "left") -> None:
    """Release button only if it's currently held. Safety net."""
    if _button_state.get(button, False):
        mouse_up(button)


def release_all_buttons() -> None:
    """Force release all mouse buttons. Emergency safety."""
    for button in ("left", "right", "middle"):
        if _button_state.get(button, False):
            mouse_up(button)

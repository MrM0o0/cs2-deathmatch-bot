"""Low-level mouse input via Win32.

Uses SendInput for mouse MOVEMENT (CS2 reads raw input for aim).
Uses mouse_event for CLICKS (CS2 blocks SendInput clicks but accepts mouse_event).
"""

import ctypes
import time

# ── Win32 constants ──────────────────────────────────────────────────────────
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_XDOWN = 0x0080
MOUSEEVENTF_XUP = 0x0100
XBUTTON1 = 0x0001  # "mouse4"
XBUTTON2 = 0x0002  # "mouse5"
WHEEL_DELTA = 120

INPUT_MOUSE = 0

user32 = ctypes.windll.user32

# ── Button state tracking ────────────────────────────────────────────────────
_button_state = {"left": False, "right": False, "middle": False}
_x_state = {XBUTTON1: False, XBUTTON2: False}

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
        dx=int(dx),
        dy=int(dy),
        mouseData=0,
        dwFlags=MOUSEEVENTF_MOVE,
        time=0,
        dwExtraInfo=ctypes.pointer(ctypes.c_ulong(0)),
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


def scroll(notches: int = -1) -> None:
    """Scroll the wheel. Negative = down (toward you). One notch = WHEEL_DELTA.
    `notches` is interpreted as a signed 16-bit value for mouse_event."""
    data = int(notches) * WHEEL_DELTA
    if data < 0:
        data += 0x100000000  # mouse_event wants unsigned for a negative delta
    user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, data, 0)


def jump() -> None:
    """Jump. The user binds +jump to mwheeldown, so one scroll-down = one jump."""
    scroll(-1)


def xbutton_down(which: int = XBUTTON2) -> None:
    _x_state[which] = True
    user32.mouse_event(MOUSEEVENTF_XDOWN, 0, 0, which, 0)


def xbutton_up(which: int = XBUTTON2) -> None:
    user32.mouse_event(MOUSEEVENTF_XUP, 0, 0, which, 0)
    _x_state[which] = False


def crouch_down() -> None:
    """Hold crouch. The user binds +duck to mouse5 (XBUTTON2)."""
    xbutton_down(XBUTTON2)


def crouch_up() -> None:
    xbutton_up(XBUTTON2)


def is_crouching() -> bool:
    return _x_state.get(XBUTTON2, False)


def release_all_buttons() -> None:
    """Force release all mouse buttons (incl. crouch/x-buttons). Emergency safety."""
    for button in ("left", "right", "middle"):
        if _button_state.get(button, False):
            mouse_up(button)
    for xb, down in list(_x_state.items()):
        if down:
            xbutton_up(xb)

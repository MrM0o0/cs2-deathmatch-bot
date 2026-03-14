"""Keyboard input via Win32 SendInput with DirectInput scancodes."""

import ctypes
import time
import random

# ── Win32 constants ──────────────────────────────────────────────────────────
INPUT_KEYBOARD = 1
KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_KEYUP = 0x0002

user32 = ctypes.windll.user32

# DirectInput scancodes (what CS2 actually reads)
SCANCODES = {
    "a": 0x1E, "b": 0x30, "c": 0x2E, "d": 0x20, "e": 0x12, "f": 0x21,
    "g": 0x22, "h": 0x23, "i": 0x17, "j": 0x24, "k": 0x25, "l": 0x26,
    "m": 0x32, "n": 0x31, "o": 0x18, "p": 0x19, "q": 0x10, "r": 0x13,
    "s": 0x1F, "t": 0x14, "u": 0x16, "v": 0x2F, "w": 0x11, "x": 0x2D,
    "y": 0x15, "z": 0x2C,
    "1": 0x02, "2": 0x03, "3": 0x04, "4": 0x05, "5": 0x06,
    "6": 0x07, "7": 0x08, "8": 0x09, "9": 0x0A, "0": 0x0B,
    "space": 0x39, "enter": 0x1C, "escape": 0x01, "tab": 0x0F,
    "shift": 0x2A, "ctrl": 0x1D, "alt": 0x38,
    "f1": 0x3B, "f2": 0x3C, "f3": 0x3D, "f4": 0x3E, "f5": 0x3F,
}


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class INPUT(ctypes.Structure):
    class _INPUT_UNION(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT)]

    _fields_ = [
        ("type", ctypes.c_ulong),
        ("union", _INPUT_UNION),
    ]


def _get_scancode(key: str) -> int:
    """Get DirectInput scancode for a key name."""
    return SCANCODES.get(key.lower(), 0)


def _send_key_input(scancode: int, key_up: bool = False) -> None:
    """Send a single key event via SendInput using scancodes."""
    flags = KEYEVENTF_SCANCODE
    if key_up:
        flags |= KEYEVENTF_KEYUP

    ki = KEYBDINPUT(
        wVk=0,
        wScan=scancode,
        dwFlags=flags,
        time=0,
        dwExtraInfo=ctypes.pointer(ctypes.c_ulong(0)),
    )
    inp = INPUT(type=INPUT_KEYBOARD)
    inp.union.ki = ki
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def key_down(key: str) -> None:
    """Press a key down."""
    sc = _get_scancode(key)
    if sc:
        _send_key_input(sc, key_up=False)


def key_up(key: str) -> None:
    """Release a key."""
    sc = _get_scancode(key)
    if sc:
        _send_key_input(sc, key_up=True)


def key_press(key: str, hold_ms: float = 0) -> None:
    """Press and release a key with optional hold time."""
    key_down(key)
    duration = hold_ms if hold_ms > 0 else random.uniform(30, 80)
    time.sleep(duration / 1000.0)
    key_up(key)


# Track which keys are currently held for cleanup
_held_keys: set[str] = set()


def hold_key(key: str) -> None:
    """Hold a key down (remembered for cleanup)."""
    key_down(key)
    _held_keys.add(key)


def release_key(key: str) -> None:
    """Release a held key."""
    key_up(key)
    _held_keys.discard(key)


def release_all() -> None:
    """Release all currently held keys. Call on shutdown."""
    for key in list(_held_keys):
        key_up(key)
    _held_keys.clear()

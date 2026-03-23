"""Fire test — tries both click methods to see which works in CS2.

Press ENTER to start each test. Switch to CS2 after pressing ENTER.

Usage:
    python tools/fire_test.py
"""

import ctypes
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.timer_setup import enable_high_resolution_timer
enable_high_resolution_timer()

user32 = ctypes.windll.user32

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_MOVE = 0x0001
INPUT_MOUSE = 0


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


def sendinput_down():
    mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0,
                    ctypes.pointer(ctypes.c_ulong(0)))
    inp = INPUT(type=INPUT_MOUSE)
    inp.union.mi = mi
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def sendinput_up():
    mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0,
                    ctypes.pointer(ctypes.c_ulong(0)))
    inp = INPUT(type=INPUT_MOUSE)
    inp.union.mi = mi
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def mouse_event_down():
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)


def mouse_event_up():
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def main():
    print("=" * 55)
    print("  CS2 Fire Test")
    print("=" * 55)
    print()

    # Test 1: SendInput burst
    input("TEST 1 (SendInput): Press ENTER, switch to CS2, burst fires in 3s...")
    time.sleep(3)
    print("FIRING with SendInput...")
    sendinput_down()
    time.sleep(0.8)
    sendinput_up()
    time.sleep(0.3)
    sendinput_down()
    time.sleep(0.8)
    sendinput_up()
    print("Done.")
    r1 = input("Did it fire? (y/n): ").strip().lower()

    # Test 2: mouse_event burst
    input("TEST 2 (mouse_event): Press ENTER, switch to CS2, burst fires in 3s...")
    time.sleep(3)
    print("FIRING with mouse_event...")
    mouse_event_down()
    time.sleep(0.8)
    mouse_event_up()
    time.sleep(0.3)
    mouse_event_down()
    time.sleep(0.8)
    mouse_event_up()
    print("Done.")
    r2 = input("Did it fire? (y/n): ").strip().lower()

    # Test 3: Rapid tap clicks with SendInput
    input("TEST 3 (rapid taps): Press ENTER, switch to CS2, taps in 3s...")
    time.sleep(3)
    print("TAP FIRING...")
    for _ in range(10):
        sendinput_down()
        time.sleep(0.05)
        sendinput_up()
        time.sleep(0.1)
    print("Done.")
    r3 = input("Did it fire? (y/n): ").strip().lower()

    print()
    print("=== Results ===")
    print(f"Test 1 (SendInput hold):   {'WORKS' if r1 == 'y' else 'BLOCKED'}")
    print(f"Test 2 (mouse_event hold): {'WORKS' if r2 == 'y' else 'BLOCKED'}")
    print(f"Test 3 (rapid taps):       {'WORKS' if r3 == 'y' else 'BLOCKED'}")


if __name__ == "__main__":
    main()

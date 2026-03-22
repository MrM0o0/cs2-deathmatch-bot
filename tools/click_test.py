"""Test which mouse click method works in CS2.

Tests each method one at a time with a pause between.
Press Enter in the terminal to advance to the next method.
"""

import ctypes
import time
import sys

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


def sendinput_click():
    mi_down = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0,
                         ctypes.pointer(ctypes.c_ulong(0)))
    inp_down = INPUT(type=INPUT_MOUSE)
    inp_down.union.mi = mi_down
    user32.SendInput(1, ctypes.byref(inp_down), ctypes.sizeof(INPUT))
    time.sleep(0.05)
    mi_up = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0,
                       ctypes.pointer(ctypes.c_ulong(0)))
    inp_up = INPUT(type=INPUT_MOUSE)
    inp_up.union.mi = mi_up
    user32.SendInput(1, ctypes.byref(inp_up), ctypes.sizeof(INPUT))


def mouse_event_click():
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.05)
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def sendinput_move_and_click():
    mi_down = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_MOVE | MOUSEEVENTF_LEFTDOWN, 0,
                         ctypes.pointer(ctypes.c_ulong(0)))
    inp_down = INPUT(type=INPUT_MOUSE)
    inp_down.union.mi = mi_down
    user32.SendInput(1, ctypes.byref(inp_down), ctypes.sizeof(INPUT))
    time.sleep(0.05)
    mi_up = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_MOVE | MOUSEEVENTF_LEFTUP, 0,
                       ctypes.pointer(ctypes.c_ulong(0)))
    inp_up = INPUT(type=INPUT_MOUSE)
    inp_up.union.mi = mi_up
    user32.SendInput(1, ctypes.byref(inp_up), ctypes.sizeof(INPUT))


def main():
    print("=== CS2 Mouse Click Test ===")
    print()

    # Method 1
    input("Press ENTER, then switch to CS2. Method 1 fires in 3 seconds...")
    time.sleep(3)
    print(">>> METHOD 1: SendInput (firing now!)")
    sendinput_click()
    time.sleep(0.3)
    sendinput_click()
    time.sleep(0.3)
    sendinput_click()
    print()
    result1 = input("Did your gun fire? (y/n): ").strip().lower()

    # Method 2
    input("Press ENTER, then switch to CS2. Method 2 fires in 3 seconds...")
    time.sleep(3)
    print(">>> METHOD 2: mouse_event (firing now!)")
    mouse_event_click()
    time.sleep(0.3)
    mouse_event_click()
    time.sleep(0.3)
    mouse_event_click()
    print()
    result2 = input("Did your gun fire? (y/n): ").strip().lower()

    # Method 3
    input("Press ENTER, then switch to CS2. Method 3 fires in 3 seconds...")
    time.sleep(3)
    print(">>> METHOD 3: SendInput+MOVE (firing now!)")
    sendinput_move_and_click()
    time.sleep(0.3)
    sendinput_move_and_click()
    time.sleep(0.3)
    sendinput_move_and_click()
    print()
    result3 = input("Did your gun fire? (y/n): ").strip().lower()

    print()
    print("=== Results ===")
    print(f"Method 1 (SendInput):      {'WORKS' if result1 == 'y' else 'BLOCKED'}")
    print(f"Method 2 (mouse_event):    {'WORKS' if result2 == 'y' else 'BLOCKED'}")
    print(f"Method 3 (SendInput+MOVE): {'WORKS' if result3 == 'y' else 'BLOCKED'}")


if __name__ == "__main__":
    main()

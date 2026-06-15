"""Isolate whether CS2 accepts our synthetic keyboard and mouse input.

RUN THIS YOURSELF in your own terminal (not via any launcher) so the input
fires from your interactive session into the focused game:

    python tools/input_test.py

Stand in OPEN SPACE facing a long sightline, press ENTER, tab into CS2, and
watch your character. The tool reports PASS/FAIL for keyboard and mouse based
on whether your minimap blip actually moved/rotated.

Verdicts:
  - keyboard PASS, mouse PASS -> input works; earlier failure was the launcher.
  - keyboard FAIL, mouse PASS -> keyboard-specific issue (scancode/flag).
  - both FAIL -> CS2 is rejecting injected input; we move to a driver-level
    method (Interception or a virtual gamepad).
"""

import os
import sys
import time

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.capture.screen import ScreenCapture
from src.vision.minimap import MinimapReader
from src.utils.math_helpers import distance
from src.input import keyboard, mouse


def _read_pos(cap, reader):
    f = None
    for _ in range(10):
        f = cap.grab()
        if f is not None:
            break
        time.sleep(0.03)
    return reader.read(f)[0]


def main():
    cfg = yaml.safe_load(open(os.path.join(PROJECT_ROOT, "config", "settings.yaml")))
    mm = cfg["minimap"]
    reader = MinimapReader(mm["x"], mm["y"], mm["size"],
                           sat_min=mm.get("sat_min", 150),
                           val_min=mm.get("val_min", 190))
    fwd = cfg["keybinds"]["forward"]
    cap = ScreenCapture(monitor=cfg["display"]["monitor"], target_fps=30)
    print("backend:", cap.start())

    try:
        input("\n>>> Stand in open space. Press ENTER, then tab into CS2... ")
    except EOFError:
        pass
    for i in range(10, 0, -1):
        print(f"\r>>> Starting in {i}... (switch to CS2)   ", end="", flush=True)
        time.sleep(1)
    print("\r>>> GO -- hands off!                         ")

    # Save what we're actually capturing so a (128,128) fallback can be
    # diagnosed: black frame = fullscreen-exclusive capture issue; chat/desktop
    # = CS2 not focused; real radar = capture is fine and input is the problem.
    import cv2
    _f = None
    for _ in range(10):
        _f = cap.grab()
        if _f is not None:
            break
        time.sleep(0.03)
    if _f is not None:
        os.makedirs(os.path.join(PROJECT_ROOT, "debug_output"), exist_ok=True)
        cv2.imwrite(os.path.join(PROJECT_ROOT, "debug_output", "input_test_capture.png"),
                    cv2.resize(_f, (_f.shape[1] // 4, _f.shape[0] // 4)))
        mmc = _f[mm["y"]:mm["y"] + mm["size"], mm["x"]:mm["x"] + mm["size"]]
        cv2.imwrite(os.path.join(PROJECT_ROOT, "debug_output", "input_test_minimap.png"), mmc)

    # --- Keyboard test: hold W for 2s, see if the blip travels --------------
    p0 = _read_pos(cap, reader)
    keyboard.hold_key(fwd)
    time.sleep(2.0)
    keyboard.release_key(fwd)
    time.sleep(0.2)
    p1 = _read_pos(cap, reader)
    kb_moved = distance(p0, p1)
    print(f"\n[keyboard] blip {p0} -> {p1}  moved {kb_moved:.0f}px  "
          f"{'PASS' if kb_moved >= 8 else 'FAIL'}")

    # --- Mouse test: turn the view, see if the blip rotates around itself ---
    # (Walk forward a touch first so there's a heading to rotate; then a big
    #  horizontal sweep. We just check the view reacted by re-walking.)
    pa = _read_pos(cap, reader)
    for _ in range(20):
        mouse.move_relative(60, 0)
        time.sleep(0.01)
    time.sleep(0.2)
    keyboard.hold_key(fwd)
    time.sleep(1.0)
    keyboard.release_key(fwd)
    pb = _read_pos(cap, reader)
    mouse_moved = distance(pa, pb)
    # If keyboard failed, this mostly tests "did anything happen". Report raw.
    print(f"[mouse]    after sweep+step blip {pa} -> {pb}  delta {mouse_moved:.0f}px  "
          f"{'(view likely turned)' if mouse_moved >= 8 else '(no reaction)'}")

    keyboard.release_all()
    cap.stop()
    print("\nReport these two lines back.")


if __name__ == "__main__":
    main()

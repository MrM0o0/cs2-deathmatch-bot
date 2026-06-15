"""Auto-calibrate navigation turn sign + gain, in-game.

The NavigationController steers by heading error, but how a horizontal mouse
move maps to minimap rotation (which direction, how many counts per degree)
can't be known until it runs on your setup. This tool figures it out for you:

    walk forward -> read heading -> issue a known turn -> read heading again

From the rotation it observed, it derives ``nav_invert_turn`` and a sensible
``nav_turn_gain`` and (with --write) updates config/settings.yaml.

Prereqs (same as the recorder):
  - cl_radar_rotate 0  (fixed radar; your blip must MOVE on the minimap)
  - Stand somewhere with open space ahead so walking forward doesn't wall you.

Usage:
    python tools/calibrate_nav.py            # report only
    python tools/calibrate_nav.py --write    # also save to settings.yaml
"""

import argparse
import os
import sys
import time

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.capture.screen import ScreenCapture
from src.vision.minimap import MinimapReader
from src.movement.navigator import compute_turn_calibration
from src.utils.math_helpers import angle_between, distance
from src.input import keyboard, mouse


def _measure_heading(capture, reader, keybind_forward: str,
                     walk_time: float = 2.2, move_thresh: float = 4.0):
    """Hold forward briefly and estimate heading from net blip displacement.

    The radar is zoomed to show the whole map, so the dot only travels a few
    px/sec -- walk long enough (>2s) to clear the noise floor before measuring.
    """
    positions = []
    keyboard.hold_key(keybind_forward)
    t0 = time.perf_counter()
    try:
        while time.perf_counter() - t0 < walk_time:
            frame = capture.grab()
            if frame is not None:
                pos, _ = reader.read(frame)
                positions.append((float(pos[0]), float(pos[1])))
            time.sleep(0.02)
    finally:
        keyboard.release_key(keybind_forward)

    if len(positions) < 2:
        return None
    start, end = positions[0], positions[-1]
    if distance(start, end) < move_thresh:
        return None
    return angle_between(start, end)


def calibrate(write: bool, test_counts: int, monitor: int, countdown: int = 6) -> None:
    with open(os.path.join(PROJECT_ROOT, "config", "settings.yaml")) as f:
        cfg = yaml.safe_load(f)
    mm = cfg["minimap"]
    reader = MinimapReader(mm["x"], mm["y"], mm["size"],
                           player_arrow_color=tuple(mm.get("player_arrow_color", (0, 255, 255))),
                           sat_min=mm.get("sat_min", 150),
                           val_min=mm.get("val_min", 190))
    fwd = cfg["keybinds"]["forward"]
    monitor = monitor if monitor is not None else cfg["display"]["monitor"]

    capture = ScreenCapture(monitor=monitor, target_fps=30)
    print(f"[Calibrate] Capture backend: {capture.start()}")
    print("[Calibrate] Stand in OPEN SPACE facing a long sightline. The tool "
          "will walk + turn your character itself -- don't touch the controls.")
    try:
        input(">>> Press ENTER to arm, then switch to CS2... ")
    except EOFError:
        pass
    for i in range(countdown, 0, -1):
        print(f"\r>>> Starting in {i}...  (switch to CS2, stand still)   ", end="", flush=True)
        time.sleep(1)
    print("\r>>> GO -- calibrating, hands off!                          ")

    try:
        h_before = _measure_heading(capture, reader, fwd)
        if h_before is None:
            print("[Calibrate] FAILED: blip didn't move while walking. "
                  "Check cl_radar_rotate 0 and minimap region in settings.yaml.")
            return
        print(f"[Calibrate] Heading before turn: {h_before:.1f} deg")

        # Issue the known turn in small steps so the game registers it smoothly.
        steps = 8
        per = int(test_counts / steps)
        for _ in range(steps):
            mouse.move_relative(per, 0)
            time.sleep(0.02)
        time.sleep(0.2)

        h_after = _measure_heading(capture, reader, fwd)
        if h_after is None:
            print("[Calibrate] FAILED: blip didn't move on the second walk.")
            return
        print(f"[Calibrate] Heading after turn:  {h_after:.1f} deg")

        result = compute_turn_calibration(h_before, h_after, test_counts)
        if result is None:
            print("[Calibrate] Turn too small to measure. Re-run with a larger "
                  "--counts value.")
            return
        invert, gain = result
        # Clamp gain to a sane band; raw measurement can be noisy. Upper bound is
        # generous because high in-game sensitivity needs many counts per degree.
        gain = max(0.3, min(80.0, gain))
        print("\n[Calibrate] RESULT:")
        print(f"  nav_invert_turn: {str(invert).lower()}")
        print(f"  nav_turn_gain:   {gain:.2f}")

        if write:
            cfg["navigation"]["nav_invert_turn"] = bool(invert)
            cfg["navigation"]["nav_turn_gain"] = round(float(gain), 2)
            with open(os.path.join(PROJECT_ROOT, "config", "settings.yaml"), "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            print("[Calibrate] Wrote values to config/settings.yaml")
        else:
            print("[Calibrate] (report only -- re-run with --write to save)")
    finally:
        keyboard.release_all()
        capture.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-calibrate nav turn sign + gain")
    parser.add_argument("--write", action="store_true", help="Save results to settings.yaml")
    parser.add_argument("--counts", type=int, default=1500,
                       help="Total mouse counts for the test turn")
    parser.add_argument("--monitor", type=int, default=None, help="Monitor index override")
    parser.add_argument("--countdown", type=int, default=6,
                       help="Seconds to count down after ENTER before driving")
    args = parser.parse_args()

    calibrate(args.write, args.counts, args.monitor, args.countdown)

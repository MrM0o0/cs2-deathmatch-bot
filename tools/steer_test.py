"""M2: steer-to-bearing -- turn the view to face a target compass bearing.

First piece that actually DRIVES the bot. Two phases:

  1. CALIBRATE: emit a known horizontal mouse move, read the cone heading
     before/after, and derive the turn sign + gain (mouse counts per degree) --
     the one thing that can't be known offline. Uses compute_turn_calibration.
  2. STEER: cycle through target bearings (default the four cardinals) and turn
     to face each, holding briefly, then the next. Closed-loop on the cone
     heading from minimap.py (instant + works while standing still, so it turns
     precisely in place -- no need to walk to keep a heading estimate alive).

This is the validation harness for the turn controller before it goes into the
roam loop. It only moves the mouse (and walks if --walk); it never fires.

    python tools/steer_test.py                 # spin to face N/E/S/W in a loop
    python tools/steer_test.py --bearings 0    # just face east and hold
    python tools/steer_test.py --walk          # walk forward while steering

Watch: does it turn the RIGHT way and SETTLE on each bearing without spinning
or oscillating? END quits. Logs heading/target/error per tick to logs/<ts>/.
"""

import argparse
import ctypes
import math
import os
import sys
import time

import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.capture.screen import ScreenCapture
from src.input import keyboard, mouse
from src.movement.navigator import compute_turn_calibration
from src.utils.math_helpers import clamp, normalize_angle
from src.utils.session_logger import SessionLogger
from src.utils.timer_setup import enable_high_resolution_timer
from src.vision.minimap import MinimapReader

END_VK = 0x23


def load_reader():
    with open(os.path.join(PROJECT_ROOT, "config", "settings.yaml")) as f:
        cfg = yaml.safe_load(f)
    mm = cfg["minimap"]
    reader = MinimapReader(
        mm["x"],
        mm["y"],
        mm["size"],
        sat_min=mm.get("sat_min", 150),
        val_min=mm.get("val_min", 190),
    )
    return reader, cfg


def read_stable_heading(reader, cap, n=8):
    """Circular-mean the cone heading over a few frames for a stable reading."""
    sins, coss = [], []
    tries = 0
    while len(sins) < n and tries < n * 6:
        tries += 1
        frame = cap.grab()
        if frame is None:
            time.sleep(0.005)
            continue
        _, deg = reader.read(frame)
        r = math.radians(deg)
        sins.append(math.sin(r))
        coss.append(math.cos(r))
        time.sleep(0.01)
    if not sins:
        return None
    return math.degrees(math.atan2(sum(sins) / len(sins), sum(coss) / len(coss)))


def calibrate(reader, cap, start_counts):
    """Learn (invert_turn, turn_gain) by emitting a known turn and reading the
    cone heading before/after. Escalates the turn size until the heading moves
    enough to measure (a small turn can be below the noise floor)."""
    counts = max(50, start_counts)
    for attempt in range(4):
        h0 = read_stable_heading(reader, cap)
        if h0 is None:
            return None
        # Move in small steps so the radar tracks the turn smoothly.
        step = 25
        moved = 0
        for _ in range(max(1, counts // step)):
            mouse.move_relative(step, 0)
            moved += step
            time.sleep(0.02)
        time.sleep(0.2)  # let the radar settle
        h1 = read_stable_heading(reader, cap)
        if h1 is None:
            return None
        delta = normalize_angle(h1 - h0)
        print(f"[steer] calib {attempt + 1}: {h0:.0f}->{h1:.0f} (d={delta:.0f}deg) +{moved}c")
        cal = compute_turn_calibration(h0, h1, moved)
        if cal is not None:
            return cal
        counts *= 2  # heading barely moved -> turn harder and retry
    return None


def main():
    ap = argparse.ArgumentParser(description="M2: steer to a target bearing")
    ap.add_argument(
        "--bearings",
        default="0,90,180,-90",
        help="Comma list of target bearings (deg, atan2: 0=E,90=S,-90=N) to cycle",
    )
    ap.add_argument("--walk", action="store_true", help="Hold W while steering")
    ap.add_argument("--tolerance", type=float, default=8.0, help="Deg within target = on-bearing")
    ap.add_argument("--hold", type=float, default=0.6, help="Seconds on-bearing before next target")
    ap.add_argument(
        "--max-turn", type=int, default=25, help="Max mouse counts per fresh frame (smoothness)"
    )
    ap.add_argument(
        "--gain", type=float, default=0.0, help="Override counts/deg (0 = auto-calibrate)"
    )
    ap.add_argument("--invert", action="store_true", help="Force-invert turn sign (skip auto)")
    ap.add_argument(
        "--cal-counts", type=int, default=600, help="Starting counts for the calibration turn"
    )
    ap.add_argument("--delay", type=int, default=6, help="Countdown to alt-tab into CS2")
    ap.add_argument("--max-seconds", type=float, default=120.0, help="Auto-stop after N seconds")
    args = ap.parse_args()

    targets = [float(b) for b in args.bearings.split(",")]
    enable_high_resolution_timer()
    reader, cfg = load_reader()
    mm = cfg["minimap"]
    mx, my, msz = mm["x"], mm["y"], mm["size"]
    cap = ScreenCapture(monitor=cfg["display"]["monitor"], target_fps=60)
    cap.start()
    logger = SessionLogger(os.path.join(PROJECT_ROOT, "logs"), enabled=True)
    user32 = ctypes.windll.user32

    for i in range(args.delay, 0, -1):
        print(f"\r[steer] starting in {i}...  (switch to CS2 now)   ", end="", flush=True)
        time.sleep(1)
    print("\r[steer] GO                                              ")

    # --- calibrate turn sign + gain ---
    if args.gain > 0:
        turn_gain, invert_turn = args.gain, args.invert
        print(f"[steer] using manual gain={turn_gain:.2f}, invert={invert_turn}")
    else:
        cal = calibrate(reader, cap, args.cal_counts)
        if cal is None:
            print("[steer] CALIBRATION FAILED even at a big turn. Did the view actually")
            print("        rotate? Is there a directional cone/arrow on the dot (not just a")
            print("        plain dot)? If unsure, re-run tools/validate_signals.py to inspect.")
            cap.stop()
            logger.close()
            return
        invert_turn, turn_gain = cal
        print(f"[steer] calibrated: gain={turn_gain:.2f} counts/deg, invert={invert_turn}")

    if args.walk:
        keyboard.key_down("w")

    # --- steer through the target bearings ---
    idx = 0
    on_since = None
    status_t = 0.0
    last_mm = None  # last minimap region, to detect duplicate (stale) captures
    start = time.perf_counter()
    try:
        while True:
            if user32.GetAsyncKeyState(END_VK) & 0x8000:
                break
            now = time.perf_counter()
            if args.max_seconds and now - start > args.max_seconds:
                print("\n[steer] max-seconds reached.")
                break

            frame = cap.grab()
            if frame is None:
                time.sleep(0.002)
                continue
            # Only steer on a FRESH frame. The loop runs faster than the capture
            # refreshes; acting on a duplicate frame turns blind between sensor
            # updates and over-rotates past the target (the jiggle). Skip dupes
            # so each turn is sensed before the next.
            region = frame[my : my + msz, mx : mx + msz]
            if last_mm is not None and np.array_equal(region, last_mm):
                time.sleep(0.002)
                continue
            last_mm = region.copy()

            _, heading = reader.read(frame)
            target = targets[idx]
            err = normalize_angle(target - heading)

            if abs(err) <= args.tolerance:
                if on_since is None:
                    on_since = now
                elif now - on_since >= args.hold:
                    idx = (idx + 1) % len(targets)  # advance to next bearing
                    on_since = None
                turn = 0
            else:
                on_since = None
                turn = int(clamp(err * turn_gain, -args.max_turn, args.max_turn))
                if invert_turn:
                    turn = -turn
                if turn:
                    mouse.move_relative(turn, 0)

            logger.log_tick(
                t=round(now - start, 3),
                heading=round(heading, 1),
                target=round(target, 1),
                err=round(err, 1),
                turn=turn,
            )

            if now - status_t >= 0.3:
                state = "ON " if on_since else "turn"
                print(
                    f"\r[steer] {state} | head {heading:6.0f} -> tgt {target:5.0f} "
                    f"| err {err:6.0f} | turn {turn:4d}   ",
                    end="",
                    flush=True,
                )
                status_t = now
    finally:
        if args.walk:
            keyboard.key_up("w")
        mouse.release_all_buttons()
        cap.stop()
        logger.close()
        print("\n[steer] stopped.")


if __name__ == "__main__":
    main()

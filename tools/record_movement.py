"""Record movement training data: what you see + what you press, while you play.

This is the data-collection step for the behavioural-cloning movement model.
Play deathmatch normally; it grabs a downscaled frame plus your held movement
keys ~15 times a second and writes them to disk. Your mouse turns are NOT
captured here -- they're recovered at training time from how the image shifts
between frames (phase correlation), so all this tool needs is the screen and
the keyboard, both of which are easy and reliable.

Output: data/movement/<timestamp>/
    frames/000001.jpg ...        downscaled BGR frames
    actions.jsonl                one line per frame: held keys (+ frame name)

Usage:
    python tools/record_movement.py --test      # verify key capture, no saving
    python tools/record_movement.py --secs 600  # record ~10 min then stop
    python tools/record_movement.py             # record until END pressed
"""

import argparse
import ctypes
import json
import os
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import cv2

from src.capture.screen import ScreenCapture

# Virtual-key codes for the movement keys we clone. Order is fixed -- it defines
# the label vector the model learns, so don't reorder without retraining.
KEYS = [
    ("w", 0x57),
    ("a", 0x41),
    ("s", 0x53),
    ("d", 0x44),
    ("space", 0x20),  # jump
    ("ctrl", 0x11),  # crouch
    ("shift", 0x10),  # walk
]
END_VK = 0x23  # END = stop recording

# Small enough to train cheaply, big enough to read the geometry ahead.
FRAME_W, FRAME_H = 160, 90
SAMPLE_HZ = 15


def _key_state(user32) -> list[int]:
    """Read which movement keys are currently held (1/0), in KEYS order."""
    return [1 if (user32.GetAsyncKeyState(vk) & 0x8000) else 0 for _, vk in KEYS]


def _countdown(user32, seconds: int) -> None:
    try:
        input("\n>>> Press ENTER, then tab into CS2 and start playing... ")
    except EOFError:
        pass
    for i in range(seconds, 0, -1):
        print(f"\r>>> Recording starts in {i}...  (get into CS2)   ", end="", flush=True)
        time.sleep(1)
    print("\r>>> GO -- play normally. END to stop.                    ")


def main():
    parser = argparse.ArgumentParser(description="Record movement training data")
    parser.add_argument(
        "--test", action="store_true", help="Print live key state + fps, save nothing"
    )
    parser.add_argument(
        "--secs", type=int, default=0, help="Auto-stop after N seconds (0 = until END)"
    )
    parser.add_argument("--countdown", type=int, default=6)
    parser.add_argument("--monitor", type=int, default=None)
    args = parser.parse_args()

    user32 = ctypes.windll.user32
    import yaml

    cfg = yaml.safe_load(open(os.path.join(PROJECT_ROOT, "config", "settings.yaml")))
    monitor = args.monitor if args.monitor is not None else cfg["display"]["monitor"]

    cap = ScreenCapture(monitor=monitor, target_fps=30)
    print("backend:", cap.start())

    session_dir = frames_dir = None
    actions_fh = None
    if not args.test:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(PROJECT_ROOT, "data", "movement", stamp)
        frames_dir = os.path.join(session_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        actions_fh = open(os.path.join(session_dir, "actions.jsonl"), "w", encoding="utf-8")
        print(f"[record] -> {session_dir}")

    _countdown(user32, args.countdown if not args.test else 3)
    if args.test:
        print("[record] TEST mode: press W/A/S/D/space/ctrl/shift, watch the vector. END to quit.")

    interval = 1.0 / SAMPLE_HZ
    idx = 0
    t_start = time.perf_counter()
    last_fps_print = t_start
    frames_since = 0
    try:
        while True:
            tick = time.perf_counter()
            if user32.GetAsyncKeyState(END_VK) & 0x8000:
                print("\n[record] END pressed -- stopping.")
                break
            if args.secs and (tick - t_start) >= args.secs:
                print(f"\n[record] reached {args.secs}s -- stopping.")
                break

            frame = cap.grab()
            if frame is None:
                time.sleep(0.005)
                continue
            keys = _key_state(user32)

            if args.test:
                frames_since += 1
                if tick - last_fps_print >= 0.5:
                    fps = frames_since / (tick - last_fps_print)
                    pressed = [name for (name, _), k in zip(KEYS, keys) if k]
                    print(f"\r[record] {fps:4.1f} fps  held={pressed!s:30}", end="")
                    last_fps_print = tick
                    frames_since = 0
            else:
                small = cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
                name = f"{idx:06d}.jpg"
                cv2.imwrite(os.path.join(frames_dir, name), small, [cv2.IMWRITE_JPEG_QUALITY, 80])
                actions_fh.write(json.dumps({"f": name, "keys": keys}) + "\n")
                idx += 1
                if idx % 150 == 0:
                    print(f"\r[record] {idx} frames ({(tick - t_start):.0f}s)   ", end="")

            sleep = interval - (time.perf_counter() - tick)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\n[record] interrupted.")
    finally:
        cap.stop()
        if actions_fh:
            actions_fh.close()
            print(f"\n[record] saved {idx} frames to {session_dir}")


if __name__ == "__main__":
    main()

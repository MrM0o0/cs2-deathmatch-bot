"""Record a movement MACRO: your real inputs + position + facing, per frame.

Unlike record_waypoints (which only saves a position trail), this captures what
you actually DID -- WASD + crouch states and your real mouse motion (Raw Input)
-- so it can be replayed faithfully by macro_play.py. It also stores the dot
position and cone facing each frame, which macro_play uses to detect drift and
re-sync (so the replay doesn't fall apart).

Note: jump is bound to the scroll wheel, which Raw Input here doesn't capture
(needs a wheel hook) -- so jumps aren't recorded yet. Crouch (mouse5) is.

    python tools/record_macro.py -m dust2          # 30 Hz, walk your route, END to save
    python tools/record_macro.py -m dust2 --rate 60

Saves config/maps/<map>_macro.json. END (or Ctrl+C) stops + saves.
"""

import argparse
import ctypes
import json
import os
import sys
import time

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.input.raw_mouse import RawMouseListener
from src.utils.timer_setup import enable_high_resolution_timer
from src.vision.minimap import MinimapReader

# Movement keys to record (held state). Jump=scroll (not captured); crouch=mouse5.
KEYS = [("w", 0x57), ("a", 0x41), ("s", 0x53), ("d", 0x44), ("crouch", 0x06)]
END_VK = 0x23


def held(vk):
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


def main():
    ap = argparse.ArgumentParser(description="Record a movement macro (inputs + pos + facing)")
    ap.add_argument(
        "--map", "-m", required=True, help="Output name -> config/maps/<map>_macro.json"
    )
    ap.add_argument("--rate", type=int, default=30, help="Record/replay frame rate (Hz)")
    ap.add_argument("--countdown", type=int, default=5, help="Seconds before recording starts")
    ap.add_argument("--max-seconds", type=float, default=300.0, help="Auto-stop")
    args = ap.parse_args()

    enable_high_resolution_timer()
    with open(os.path.join(PROJECT_ROOT, "config", "settings.yaml")) as f:
        cfg = yaml.safe_load(f)
    mm = cfg["minimap"]
    reader = MinimapReader(
        mm["x"], mm["y"], mm["size"], sat_min=mm["sat_min"], val_min=mm["val_min"]
    )

    from src.capture.screen import ScreenCapture

    cap = ScreenCapture(monitor=cfg["display"]["monitor"], target_fps=max(60, args.rate))
    cap.start()
    raw = RawMouseListener()
    raw.start()

    for i in range(args.countdown, 0, -1):
        print(
            f"\r[macro] recording in {i}...  (switch to CS2, then PLAY normally)   ",
            end="",
            flush=True,
        )
        time.sleep(1)
    print("\r[macro] REC -- walk your route. END to stop.                      ")

    dt = 1.0 / args.rate
    frames = []
    raw.read_delta()  # clear any startup accumulation
    start = time.perf_counter()
    status_t = 0.0
    try:
        while True:
            if held(END_VK):
                break
            now = time.perf_counter()
            if args.max_seconds and now - start > args.max_seconds:
                break
            frame = cap.grab()
            if frame is None:
                time.sleep(0.002)
                continue
            (x, y), heading = reader.read(frame)
            keys = [1 if held(vk) else 0 for _, vk in KEYS]
            mdx, mdy = raw.read_delta()
            frames.append({"k": keys, "m": [mdx, mdy], "p": [x, y], "h": round(heading, 1)})

            if now - status_t >= 0.5:
                print(
                    f"\r[macro] frames {len(frames)} | pos ({x:3d},{y:3d}) | keys {keys}   ",
                    end="",
                    flush=True,
                )
                status_t = now

            # pace to the fixed frame rate
            sleep_for = dt - (time.perf_counter() - now)
            if sleep_for > 0:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        pass
    finally:
        raw.stop()
        cap.stop()

    out = os.path.join(PROJECT_ROOT, "config", "maps", f"{args.map}_macro.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"rate": args.rate, "frames": frames}, f)
    secs = len(frames) / args.rate
    print(f"\n[macro] saved {len(frames)} frames ({secs:.0f}s) -> {out}")


if __name__ == "__main__":
    main()

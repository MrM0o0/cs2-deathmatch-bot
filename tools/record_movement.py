"""Record movement training data: what you see + what you press, while you play.

This is the data-collection step for the behavioural-cloning movement model.
Play normally; ~20 times a second it grabs a downscaled frame plus your held
movement keys (GetAsyncKeyState) and your real mouse-x/y motion (Raw Input,
which works despite CS2 clamping the cursor). The mouse gives clean turn-label
ground truth -- far better than guessing turns from inter-frame image shift.

Jump is captured too: it's commonly bound to the scroll wheel, which Raw Input
reports, so RawMouseListener.read_wheel() gives us the jump signal (negative =
scroll-down). That makes box-boosts / gap-jumps learnable instead of a known gap.

Output: data/movement/<timestamp>/
    frames/000001.jpg ...   downscaled BGR frames (320x180)
    actions.jsonl           per frame: {"f", "t", "keys": [...], "m": [dx, dy],
                                         "wheel", "fire"}
    meta.json               schema, map, resolution, sample rate, key order

Usage:
    python tools/record_movement.py --test          # verify key + mouse + WHEEL
    python tools/record_movement.py --map dust2 --secs 600   # ~10 min then stop
    python tools/record_movement.py --map dust2     # record until END pressed
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
from src.input.raw_mouse import RawMouseListener

# Virtual-key codes for the movement signals we clone. Order is fixed -- it
# defines the label vector the model learns, so don't reorder without retraining.
# Jump is captured separately via the scroll wheel (RawMouseListener.read_wheel),
# since the common bind is mwheel and there's no key-state to poll. Crouch is
# often a mouse side button -- both side buttons are polled so whichever you use
# is captured.
KEYS = [
    ("w", 0x57),
    ("a", 0x41),
    ("s", 0x53),
    ("d", 0x44),
    ("shift", 0x10),  # walk (keyboard)
    ("ctrl", 0x11),  # crouch (keyboard bind, if any)
    ("mouse4", 0x05),  # XBUTTON1 -- common crouch bind
    ("mouse5", 0x06),  # XBUTTON2 -- common crouch bind
]
END_VK = 0x23  # END = stop recording
FIRE_VK = 0x01  # left mouse button -- recorded (not a model output) to flag and
#                 drop combat frames in prep, so the movement model stays pure
#                 navigation and never learns to spray/counter-strafe at nothing.

# Record at a generous resolution; training can downscale, but it can never
# upscale, so we capture more detail than the model strictly needs.
FRAME_W, FRAME_H = 320, 180
SAMPLE_HZ = 20
SCHEMA = 2  # actions.jsonl row format version (2 = adds wheel/jump + timestamp)

# GetAsyncKeyState bits: 0x8000 = down right now; 0x0001 = pressed since the
# previous call. OR-ing them means a quick tap that happened *between* samples
# (e.g. a counter-strafe) still registers, instead of being missed.
_KEY_ACTIVE_MASK = 0x8001


def _key_state(user32) -> list[int]:
    """Which movement keys were active this interval (held OR tapped), in KEYS order."""
    return [1 if (user32.GetAsyncKeyState(vk) & _KEY_ACTIVE_MASK) else 0 for _, vk in KEYS]


def _banked_seconds(base: str) -> float:
    """Total seconds of movement data already recorded across all sessions.

    Lets the live counter show cumulative progress toward the training goal, so
    each short session visibly adds to the pile instead of starting from zero.
    """
    if not os.path.isdir(base):
        return 0.0
    frames = 0
    for d in os.listdir(base):
        aj = os.path.join(base, d, "actions.jsonl")
        if os.path.exists(aj):
            with open(aj, encoding="utf-8") as f:
                frames += sum(1 for _ in f)
    return frames / SAMPLE_HZ


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
    parser.add_argument("--map", default="unknown", help="Map name, stored in session meta")
    parser.add_argument("--goal", type=int, default=60, help="Target total minutes (progress bar)")
    args = parser.parse_args()

    user32 = ctypes.windll.user32
    import yaml

    cfg = yaml.safe_load(open(os.path.join(PROJECT_ROOT, "config", "settings.yaml")))
    monitor = args.monitor if args.monitor is not None else cfg["display"]["monitor"]

    cap = ScreenCapture(monitor=monitor, target_fps=30)
    print("backend:", cap.start())
    rawmouse = RawMouseListener()
    rawmouse.start()

    session_dir = frames_dir = None
    actions_fh = None
    if not args.test:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(PROJECT_ROOT, "data", "movement", stamp)
        frames_dir = os.path.join(session_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        actions_fh = open(os.path.join(session_dir, "actions.jsonl"), "w", encoding="utf-8")
        meta = {
            "schema": SCHEMA,
            "map": args.map,
            "frame_w": FRAME_W,
            "frame_h": FRAME_H,
            "sample_hz": SAMPLE_HZ,
            "keys": [name for name, _ in KEYS],
            "row": "f, t(s), keys[], m[dx,dy], wheel, fire",
            "wheel_note": "negative wheel = scroll-down (jump for mwheeldown binds)",
        }
        with open(os.path.join(session_dir, "meta.json"), "w", encoding="utf-8") as mf:
            json.dump(meta, mf, indent=2)
        print(f"[record] -> {session_dir}  (map={args.map}, {FRAME_W}x{FRAME_H}, {SAMPLE_HZ}Hz)")

    _countdown(user32, args.countdown if not args.test else 3)
    if args.test:
        print("[record] TEST mode: tap/hold W/A/S/D/shift/m5, watch the vector. END to quit.")

    prior_secs = (
        0.0 if args.test else _banked_seconds(os.path.join(PROJECT_ROOT, "data", "movement"))
    )
    goal_secs = max(1, args.goal * 60)
    if not args.test:
        print(f"[record] already banked {prior_secs / 60:.1f} min toward {args.goal} min goal")

    interval = 1.0 / SAMPLE_HZ
    idx = 0
    t_start = time.perf_counter()
    last_fps_print = t_start
    frames_since = 0
    seen = [0] * len(KEYS)  # keys touched since the last test print (catches taps)
    mdx_accum = 0  # mouse-x accumulated since the last test print
    wheel_accum = 0  # scroll-wheel accumulated since the last test print
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
            mdx, mdy = rawmouse.read_delta()
            wheel = rawmouse.read_wheel()
            fire = 1 if (user32.GetAsyncKeyState(FIRE_VK) & 0x8001) else 0

            if args.test:
                frames_since += 1
                seen = [s | k for s, k in zip(seen, keys)]
                mdx_accum += mdx
                wheel_accum += wheel
                if tick - last_fps_print >= 0.5:
                    fps = frames_since / (tick - last_fps_print)
                    touched = [name for (name, _), k in zip(KEYS, seen) if k]
                    jump = "JUMP" if wheel_accum < 0 else ("scroll+" if wheel_accum > 0 else "    ")
                    print(
                        f"\r[record] {fps:4.1f} fps  mouseX={mdx_accum:+5d}  "
                        f"wheel={wheel_accum:+4d} {jump}  touched={touched!s:30}",
                        end="",
                    )
                    last_fps_print = tick
                    frames_since = 0
                    seen = [0] * len(KEYS)
                    mdx_accum = 0
                    wheel_accum = 0
            else:
                small = cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
                name = f"{idx:06d}.jpg"
                cv2.imwrite(os.path.join(frames_dir, name), small, [cv2.IMWRITE_JPEG_QUALITY, 80])
                actions_fh.write(
                    json.dumps(
                        {
                            "f": name,
                            "t": round(tick - t_start, 4),
                            "keys": keys,
                            "m": [mdx, mdy],
                            "wheel": wheel,
                            "fire": fire,
                        }
                    )
                    + "\n"
                )
                idx += 1
                if idx % 40 == 0:
                    banked = prior_secs + idx / SAMPLE_HZ
                    pct = min(100.0, 100.0 * banked / goal_secs)
                    filled = int(pct / 5)
                    bar = "#" * filled + "-" * (20 - filled)
                    print(
                        f"\r[record] banked {banked / 60:5.1f}/{args.goal} min  "
                        f"[{bar}] {pct:3.0f}%   (this session {idx / SAMPLE_HZ:.0f}s)   ",
                        end="",
                    )

            sleep = interval - (time.perf_counter() - tick)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\n[record] interrupted.")
    finally:
        rawmouse.stop()
        cap.stop()
        if actions_fh:
            actions_fh.close()
            print(f"\n[record] saved {idx} frames to {session_dir}")


if __name__ == "__main__":
    main()

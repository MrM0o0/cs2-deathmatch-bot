"""Simple, robust roam -- walk the map without getting stuck. NO heading.

Deliberately uses only the signal that's proven rock-solid: the minimap dot
POSITION. It never reads the fragile cone heading.

  - Walk forward.
  - Dot advancing? keep going.
  - Dot stalled (stuck on a wall/box)? jump + turn a chunk OPEN-LOOP (just move
    the mouse, no sensor feedback) and keep walking until it moves again.

That's enough to wander a map convincingly. Facing-dependent niceties (peeks)
are a later problem; this is the reliable backbone.

SAFETY -- dead-man switch: the bot only acts WHILE YOU HOLD the run key (Insert).
Let go and it stops instantly (keys + mouse released). END quits entirely;
--max-seconds auto-stops. You cannot lose control: release the key and it's done.

    python tools/roam.py            # HOLD Insert to roam, release to stop, END to quit
"""

import argparse
import ctypes
import os
import random
import sys
import time
from collections import deque

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.capture.screen import ScreenCapture
from src.input import keyboard, mouse
from src.utils.math_helpers import distance
from src.utils.session_logger import SessionLogger
from src.utils.timer_setup import enable_high_resolution_timer
from src.vision.minimap import MinimapReader

VK = {"insert": 0x2D, "end": 0x23, "home": 0x24, "rshift": 0xA1, "rctrl": 0xA3}
END_VK = 0x23


def held(vk):
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


def load_reader():
    with open(os.path.join(PROJECT_ROOT, "config", "settings.yaml")) as f:
        cfg = yaml.safe_load(f)
    mm = cfg["minimap"]
    reader = MinimapReader(
        mm["x"], mm["y"], mm["size"], sat_min=mm["sat_min"], val_min=mm["val_min"]
    )
    return reader, cfg


def main():
    ap = argparse.ArgumentParser(description="Simple position-based roam (no heading)")
    ap.add_argument("--run-key", default="insert", choices=sorted(VK), help="HOLD this to roam")
    ap.add_argument("--sample", type=float, default=0.25, help="Seconds between position samples")
    ap.add_argument("--stuck-dist", type=float, default=5.0, help="Min px moved/window or stuck")
    ap.add_argument("--stuck-time", type=float, default=1.2, help="Walk this long before judging")
    ap.add_argument("--recover-time", type=float, default=0.7, help="Seconds to turn when stuck")
    ap.add_argument("--turn-step", type=int, default=18, help="Mouse counts/tick while recovering")
    ap.add_argument("--no-jump", action="store_true", help="Don't jump when stuck")
    ap.add_argument("--max-seconds", type=float, default=180.0, help="Hard auto-stop")
    args = ap.parse_args()

    enable_high_resolution_timer()
    reader, cfg = load_reader()
    run_vk = VK[args.run_key]
    cap = ScreenCapture(monitor=cfg["display"]["monitor"], target_fps=60)
    cap.start()
    logger = SessionLogger(os.path.join(PROJECT_ROOT, "logs"), enabled=True)
    print(f"[roam] ready. HOLD {args.run_key.upper()} to roam, release to STOP, END to quit.")
    print(f"[roam] logging to {logger.session_dir}")

    def stop_all():
        for k in ("w", "a", "d"):
            keyboard.key_up(k)
        mouse.release_all_buttons()

    walking = False
    state = "walk"
    recover_until = 0.0
    recover_dir = 1
    walk_started = 0.0
    samples = deque(maxlen=8)  # (t, pos)
    last_sample = 0.0
    period = 1.0 / 30.0
    status_t = 0.0
    start = time.perf_counter()
    try:
        while True:
            if held(END_VK):
                break
            now = time.perf_counter()
            if args.max_seconds and now - start > args.max_seconds:
                print("\n[roam] max-seconds reached.")
                break

            # --- dead-man switch: only act while the run key is held ---
            if not held(run_vk):
                if walking:
                    stop_all()
                    walking = False
                    state = "walk"
                    samples.clear()
                time.sleep(0.02)
                if now - status_t >= 0.5:
                    print("\r[roam] IDLE (hold key to roam)        ", end="", flush=True)
                    status_t = now
                continue

            if not walking:  # just (re)engaged -> start walking
                keyboard.key_down("w")
                walking = True
                walk_started = now
                samples.clear()

            frame = cap.grab()
            if frame is None:
                time.sleep(0.002)
                continue
            pos, _ = reader.read(frame)

            # sample position sparsely (capture has surely refreshed by then)
            if now - last_sample >= args.sample:
                samples.append((now, pos))
                last_sample = now

            moved = 0.0
            if len(samples) >= 2:
                moved = distance(samples[0][1], samples[-1][1])

            if state == "walk":
                # stuck = walked long enough but the dot barely moved over the window
                window = samples[-1][0] - samples[0][0] if len(samples) >= 2 else 0.0
                if (
                    now - walk_started > args.stuck_time
                    and window >= args.stuck_time
                    and moved < args.stuck_dist
                ):
                    state = "recover"
                    recover_until = now + args.recover_time
                    recover_dir = random.choice((-1, 1))
                    if not args.no_jump:
                        keyboard.key_press("space", hold_ms=40)
            else:  # recover: turn open-loop (no sensing) to face somewhere new
                mouse.move_relative(recover_dir * args.turn_step, 0)
                if now >= recover_until:
                    state = "walk"
                    walk_started = now
                    samples.clear()
                    last_sample = now

            logger.log_tick(
                t=round(now - start, 3),
                x=pos[0],
                y=pos[1],
                moved=round(moved, 1),
                state=state,
            )
            if now - status_t >= 0.4:
                print(
                    f"\r[roam] {state:7s} | pos ({pos[0]:3d},{pos[1]:3d}) | mv {moved:4.0f}px  ",
                    end="",
                    flush=True,
                )
                status_t = now

            elapsed = time.perf_counter() - now
            if elapsed < period:
                time.sleep(period - elapsed)
    finally:
        stop_all()
        cap.stop()
        logger.close()
        print("\n[roam] stopped.")


if __name__ == "__main__":
    main()

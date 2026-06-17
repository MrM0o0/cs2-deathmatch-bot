"""Simple, robust roam -- walk the map and never stay stuck. NO heading.

Uses only the rock-solid signal (minimap dot POSITION), never the fragile cone
heading. Turns are OPEN-LOOP, sized by ANGLE (counts = deg / (sens*m_yaw),
~53 counts/deg here) so they're real swings, not baby steps.

Behaviour:
  - WALK forward; occasionally a small wander turn so it isn't dead straight.
  - When the dot stops advancing (wall/box) -> ESCAPE: jump, then turn steadily
    while pushing forward UNTIL it's moving again. If a full sweep finds nothing,
    BACK UP briefly and sweep the other way. So it never grinds in place.

SAFETY -- dead-man switch: acts only WHILE YOU HOLD the run key (default L).
Release = instant stop (keys + mouse released). END quits; --max-seconds backstops.

    python tools/roam.py                 # HOLD L to roam, release to stop, END to quit
    python tools/roam.py --cpd 40        # eye-calibrate turn size if needed
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

SPECIAL = {"insert": 0x2D, "end": 0x23, "home": 0x24, "rshift": 0xA1, "rctrl": 0xA3}
END_VK = 0x23


def key_vk(name: str) -> int:
    name = name.lower()
    if name in SPECIAL:
        return SPECIAL[name]
    if len(name) == 1 and name.isalpha():
        return ord(name.upper())
    raise ValueError(f"unknown run-key: {name!r}")


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
    ap = argparse.ArgumentParser(description="Position-based roam that never stays stuck")
    ap.add_argument("--run-key", default="l", help="HOLD this to roam (a letter, or insert/rctrl)")
    ap.add_argument("--stuck-secs", type=float, default=0.8, help="No-progress time -> escape")
    ap.add_argument(
        "--stuck-dist", type=float, default=6.0, help="Px/window below this = no progress"
    )
    ap.add_argument(
        "--clear-dist", type=float, default=9.0, help="Recent px moved that means 'freed'"
    )
    ap.add_argument(
        "--escape-slice", type=int, default=200, help="Mouse counts/tick while escaping"
    )
    ap.add_argument(
        "--escape-timeout", type=float, default=3.5, help="Sweep this long then back up"
    )
    ap.add_argument(
        "--backup-secs", type=float, default=0.5, help="How long to walk back when boxed in"
    )
    ap.add_argument(
        "--wander-secs", type=float, default=6.0, help="Seconds between gentle wander turns"
    )
    ap.add_argument(
        "--cpd", type=float, default=0.0, help="Counts per degree (0 = from sensitivity)"
    )
    ap.add_argument("--max-seconds", type=float, default=180.0, help="Hard auto-stop")
    args = ap.parse_args()

    enable_high_resolution_timer()
    reader, cfg = load_reader()
    g = cfg["game"]
    cpd = args.cpd if args.cpd > 0 else 1.0 / (g["sensitivity"] * g["m_yaw"])
    run_vk = key_vk(args.run_key)
    cap = ScreenCapture(monitor=cfg["display"]["monitor"], target_fps=60)
    cap.start()
    logger = SessionLogger(os.path.join(PROJECT_ROOT, "logs"), enabled=True)
    print(
        f"[roam] {cpd:.1f} counts/deg. HOLD {args.run_key.upper()} to roam, release=STOP, END=quit."
    )

    keys_down = set()

    def press(k):
        if k not in keys_down:
            keyboard.key_down(k)
            keys_down.add(k)

    def release(k):
        if k in keys_down:
            keyboard.key_up(k)
            keys_down.discard(k)

    def stop_all():
        for k in list(keys_down):
            release(k)
        mouse.release_all_buttons()

    def moved_since(t_ago, now, samples):
        """Px the dot moved over roughly the last t_ago seconds."""
        if len(samples) < 2:
            return 999.0
        cutoff = now - t_ago
        old = next((p for (t, p) in samples if t >= cutoff), samples[0][1])
        return distance(old, samples[-1][1])

    active = False
    state = "walk"
    samples = deque(maxlen=12)  # (t, pos) ~3s at 0.25s
    last_sample = 0.0
    wander_left = 0  # signed counts remaining for a wander turn
    phase_start = 0.0  # when current straight-walk phase began
    escape_start = 0.0
    escape_dir = 1
    backup_until = 0.0
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

            # dead-man: only act while the run key is held
            if not held(run_vk):
                if active:
                    stop_all()
                    active = False
                    samples.clear()
                time.sleep(0.02)
                if now - status_t >= 0.5:
                    print("\r[roam] IDLE (hold key to roam)        ", end="", flush=True)
                    status_t = now
                continue

            if not active:
                active = True
                state = "walk"
                phase_start = now
                wander_left = 0
                samples.clear()
                press("w")

            frame = cap.grab()
            if frame is None:
                time.sleep(0.002)
                continue
            pos, _ = reader.read(frame)
            if now - last_sample >= 0.25:
                samples.append((now, pos))
                last_sample = now

            if state == "walk":
                release("s")
                press("w")
                if wander_left != 0:  # finishing a gentle wander turn
                    step = max(-args.escape_slice, min(args.escape_slice, wander_left))
                    mouse.move_relative(step, 0)
                    wander_left -= step
                else:
                    win = samples[-1][0] - samples[0][0] if len(samples) >= 2 else 0.0
                    progress = moved_since(args.stuck_secs, now, samples)
                    if (
                        now - phase_start > args.stuck_secs
                        and win >= args.stuck_secs
                        and progress < args.stuck_dist
                    ):
                        state = "escape"  # wall -> start sweeping
                        escape_start = now
                        escape_dir = random.choice((-1, 1))
                        keyboard.key_press("space", hold_ms=40)
                    elif now - phase_start > args.wander_secs:
                        deg = random.uniform(15, 35) * random.choice((-1, 1))
                        wander_left = int(deg * cpd)
                        phase_start = now

            elif state == "escape":
                press("w")
                mouse.move_relative(escape_dir * args.escape_slice, 0)  # sweep
                if moved_since(0.6, now, samples) > args.clear_dist:
                    state = "walk"  # moving again -> resume
                    phase_start = now
                elif now - escape_start > args.escape_timeout:
                    state = "backup"  # boxed in -> reverse out
                    backup_until = now + args.backup_secs
                    release("w")
                    press("s")

            elif state == "backup":
                if now >= backup_until:
                    release("s")
                    press("w")
                    state = "escape"
                    escape_start = now
                    escape_dir = -escape_dir  # try the other way

            logger.log_tick(t=round(now - start, 3), x=pos[0], y=pos[1], state=state)
            if now - status_t >= 0.4:
                print(
                    f"\r[roam] {state:6s} | pos {pos[0]:3d},{pos[1]:3d}        ", end="", flush=True
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

"""Simple, robust roam -- walk the map without getting stuck. NO heading.

Uses only the rock-solid signal (minimap dot POSITION), never the fragile cone
heading. Turns are OPEN-LOOP and sized by ANGLE: counts = degrees / (sens*m_yaw)
(~53 counts/deg here), so a "turn 120 deg" is a real, visible swing -- not the
~7 deg baby-step the old fixed counts produced.

  - Walk forward.
  - Every few seconds, wander: small random turn so it doesn't just go straight.
  - Dot stalled (stuck on a wall/box)? jump + big random turn until it moves.

SAFETY -- dead-man switch: the bot only acts WHILE YOU HOLD the run key (default
L). Let go and it stops instantly (keys + mouse released). END quits entirely;
--max-seconds auto-stops. You cannot lose control: release the key and it's done.

    python tools/roam.py                 # HOLD L to roam, release to stop, END to quit
    python tools/roam.py --run-key j     # different hold key
    python tools/roam.py --cpd 40        # override counts-per-degree (eye-calibrate turns)
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
    """Virtual-key code for a run-key: a single letter (e.g. 'l') or a name."""
    name = name.lower()
    if name in SPECIAL:
        return SPECIAL[name]
    if len(name) == 1 and name.isalpha():
        return ord(name.upper())
    raise ValueError(f"unknown run-key: {name!r} (use a single letter or {sorted(SPECIAL)})")


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
    ap.add_argument("--run-key", default="l", help="HOLD this to roam (a letter, or insert/rctrl)")
    ap.add_argument("--sample", type=float, default=0.25, help="Seconds between position samples")
    ap.add_argument("--stuck-dist", type=float, default=5.0, help="Min px moved/window or stuck")
    ap.add_argument("--stuck-time", type=float, default=1.2, help="Walk this long before judging")
    ap.add_argument("--wander-secs", type=float, default=4.0, help="Seconds between wander turns")
    ap.add_argument(
        "--turn-slice", type=int, default=300, help="Max mouse counts per tick in a turn"
    )
    ap.add_argument(
        "--cpd", type=float, default=0.0, help="Counts per degree (0 = from sensitivity)"
    )
    ap.add_argument("--no-jump", action="store_true", help="Don't jump when stuck")
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
    print(f"[roam] {cpd:.1f} counts/deg (360 = {cpd * 360:.0f} counts).")
    print(f"[roam] ready. HOLD {args.run_key.upper()} to roam, release to STOP, END to quit.")

    def stop_all():
        for k in ("w", "a", "d"):
            keyboard.key_up(k)
        mouse.release_all_buttons()

    walking = False
    turn_remaining = 0  # signed mouse counts still to emit for the current turn
    phase_start = 0.0  # when the current straight-walk phase began
    samples = deque(maxlen=8)  # (t, pos)
    last_sample = 0.0
    period = 1.0 / 30.0
    status_t = 0.0
    state = "walk"
    start = time.perf_counter()

    def queue_turn(deg_lo, deg_hi):
        nonlocal turn_remaining
        deg = random.uniform(deg_lo, deg_hi) * random.choice((-1, 1))
        turn_remaining = int(deg * cpd)

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
                    turn_remaining = 0
                    samples.clear()
                time.sleep(0.02)
                if now - status_t >= 0.5:
                    print("\r[roam] IDLE (hold key to roam)            ", end="", flush=True)
                    status_t = now
                continue

            if not walking:  # just (re)engaged
                keyboard.key_down("w")
                walking = True
                phase_start = now
                samples.clear()
                turn_remaining = 0

            frame = cap.grab()
            if frame is None:
                time.sleep(0.002)
                continue
            pos, _ = reader.read(frame)
            if now - last_sample >= args.sample:
                samples.append((now, pos))
                last_sample = now

            if turn_remaining != 0:
                # Emit the next slice of the in-progress turn (open-loop).
                step = max(-args.turn_slice, min(args.turn_slice, turn_remaining))
                mouse.move_relative(step, 0)
                turn_remaining -= step
                state = "turn"
                if turn_remaining == 0:  # turn finished -> fresh straight-walk phase
                    phase_start = now
                    samples.clear()
                    last_sample = now
            else:
                moved = distance(samples[0][1], samples[-1][1]) if len(samples) >= 2 else 999.0
                window = samples[-1][0] - samples[0][0] if len(samples) >= 2 else 0.0
                if (
                    now - phase_start > args.stuck_time
                    and window >= args.stuck_time
                    and moved < args.stuck_dist
                ):
                    state = "stuck"
                    if not args.no_jump:
                        keyboard.key_press("space", hold_ms=40)
                    queue_turn(110, 160)  # big turn to escape
                elif now - phase_start > args.wander_secs:
                    state = "wander"
                    queue_turn(20, 50)  # gentle course change
                    phase_start = now
                else:
                    state = "walk"

            logger.log_tick(
                t=round(now - start, 3), x=pos[0], y=pos[1], state=state, tr=turn_remaining
            )
            if now - status_t >= 0.4:
                print(
                    f"\r[roam] {state:6s} | pos {pos[0]:3d},{pos[1]:3d} | t {turn_remaining:5d}  ",
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

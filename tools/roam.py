"""Radar-aware roam -- steer toward OPEN radar space, away from black. NO cone.

Finally uses the radar as a live map (the original idea). Each tick it casts rays
out from the dot on the radar to see which directions are walkable (lit) vs
blacked-out (walls / out of bounds), and steers toward the most-open direction.
It knows which way it's facing from the dot's recent MOTION (reliable while
walking) -- not the fragile cone. Turns are sized by angle (counts/deg from
sensitivity), so they're real swings.

It saves an annotated radar image every ~0.7s (rays green=open / red=blocked,
blue=chosen direction) to logs/<ts>/ so we can SEE what it reads.

SAFETY -- dead-man switch: acts only WHILE YOU HOLD the run key (default L).
Release = instant stop. END quits; --max-seconds backstops.

    python tools/roam.py                 # HOLD L to roam, release to stop, END to quit
    python tools/roam.py --black 60      # tweak the black/walkable brightness cutoff
"""

import argparse
import ctypes
import math
import os
import sys
import time
from collections import deque

import yaml

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.capture.screen import ScreenCapture
from src.input import keyboard, mouse
from src.movement.occupancy import best_direction, radar_clearances
from src.utils.math_helpers import distance, normalize_angle
from src.utils.session_logger import SessionLogger
from src.utils.timer_setup import enable_high_resolution_timer
from src.vision.minimap import MinimapReader

SPECIAL = {"insert": 0x2D, "end": 0x23, "home": 0x24, "rshift": 0xA1, "rctrl": 0xA3}
END_VK = 0x23
N_RAYS = 24
MAX_R = 46
START_R = 5


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
    return reader, cfg, mm


def annotate(region, bx, by, clrs, black_thresh, chosen_deg, motion_deg):
    img = region.copy()
    for k, dist in enumerate(clrs):
        a = 2 * math.pi * k / len(clrs)
        ex = int(bx + dist * math.cos(a))
        ey = int(by + dist * math.sin(a))
        color = (0, 200, 0) if dist >= MAX_R * 0.6 else (0, 0, 255)
        cv2.line(img, (bx, by), (ex, ey), color, 1)
    for deg, color, ln in ((motion_deg, (0, 255, 255), 22), (chosen_deg, (255, 80, 0), 28)):
        if deg is not None:
            r = math.radians(deg)
            cv2.line(
                img, (bx, by), (int(bx + ln * math.cos(r)), int(by + ln * math.sin(r))), color, 2
            )
    cv2.circle(img, (bx, by), 3, (255, 255, 255), 1)
    return cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser(description="Radar-aware roam")
    ap.add_argument("--run-key", default="l", help="HOLD this to roam (a letter, or insert/rctrl)")
    ap.add_argument("--black", type=int, default=0, help="Black cutoff (0 = auto from floor)")
    ap.add_argument(
        "--deadzone", type=float, default=14.0, help="Deg error before it bothers turning"
    )
    ap.add_argument(
        "--slice", type=int, default=200, help="Max mouse counts per tick (turn smoothness)"
    )
    ap.add_argument(
        "--cpd", type=float, default=0.0, help="Counts per degree (0 = from sensitivity)"
    )
    ap.add_argument(
        "--stuck-secs", type=float, default=1.0, help="No motion this long -> blind sweep"
    )
    ap.add_argument("--max-seconds", type=float, default=180.0, help="Hard auto-stop")
    args = ap.parse_args()

    if cv2 is None:
        print("[roam] cv2/numpy missing -- aborting.")
        return

    enable_high_resolution_timer()
    reader, cfg, mm = load_reader()
    mx, my, msz = mm["x"], mm["y"], mm["size"]
    g = cfg["game"]
    cpd = args.cpd if args.cpd > 0 else 1.0 / (g["sensitivity"] * g["m_yaw"])
    run_vk = key_vk(args.run_key)
    cap = ScreenCapture(monitor=cfg["display"]["monitor"], target_fps=60)
    cap.start()
    logger = SessionLogger(os.path.join(PROJECT_ROOT, "logs"), enabled=True)
    print(f"[roam] {cpd:.1f} counts/deg. HOLD {args.run_key.upper()} = roam, release = STOP.")

    keys_down = set()

    def press(k):
        if k not in keys_down:
            keyboard.key_down(k)
            keys_down.add(k)

    def release_all():
        for k in list(keys_down):
            keyboard.key_up(k)
            keys_down.discard(k)
        mouse.release_all_buttons()

    active = False
    hist = deque(maxlen=20)  # (t, pos) ~0.66s for motion heading
    motion_deg = None
    last_move_t = 0.0
    sweep_dir = 1
    last_dump = 0.0
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

            if not held(run_vk):  # dead-man
                if active:
                    release_all()
                    active = False
                    hist.clear()
                    motion_deg = None
                time.sleep(0.02)
                if now - status_t >= 0.5:
                    print("\r[roam] IDLE (hold key to roam)        ", end="", flush=True)
                    status_t = now
                continue

            if not active:
                active = True
                press("w")
                hist.clear()
                motion_deg = None
                last_move_t = now

            frame = cap.grab()
            if frame is None:
                time.sleep(0.002)
                continue
            (bx, by), _ = reader.read(frame)
            region = frame[my : my + msz, mx : mx + msz]
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

            # walkable/black cutoff: auto from the floor brightness right by the dot
            if args.black > 0:
                black = args.black
            else:
                y0, y1 = max(0, by - 7), min(msz, by + 8)
                x0, x1 = max(0, bx - 7), min(msz, bx + 8)
                ref = float(np.median(gray[y0:y1, x0:x1]))
                black = max(35.0, ref * 0.5)

            clrs = radar_clearances(gray, bx, by, N_RAYS, MAX_R, START_R, black)

            # motion heading from how the dot has moved recently
            hist.append((now, (bx, by)))
            if len(hist) >= 2:
                old = next((p for (t, p) in hist if t >= now - 0.5), hist[0][1])
                if distance(old, (bx, by)) >= 4.0:
                    motion_deg = math.degrees(math.atan2(by - old[1], bx - old[0]))
                    last_move_t = now

            press("w")
            chosen = None
            if motion_deg is None or now - last_move_t > args.stuck_secs:
                # no reliable facing (just started / wedged) -> blind sweep till moving
                mouse.move_relative(sweep_dir * args.slice, 0)
                state = "sweep"
            else:
                chosen = best_direction(clrs, motion_deg)
                err = normalize_angle(chosen - motion_deg)
                if abs(err) > args.deadzone:
                    tc = int(max(-args.slice, min(args.slice, err * cpd)))
                    mouse.move_relative(tc, 0)
                    state = "steer"
                else:
                    state = "walk"
                sweep_dir = 1 if err >= 0 else -1  # remember which way open was

            if now - last_dump >= 0.7:
                stamp = f"{now - start:06.1f}".replace(".", "_")
                cv2.imwrite(
                    os.path.join(logger.session_dir, f"radar_{stamp}_{state}.jpg"),
                    annotate(region, bx, by, clrs, black, chosen, motion_deg),
                    [cv2.IMWRITE_JPEG_QUALITY, 85],
                )
                last_dump = now

            logger.log_tick(
                t=round(now - start, 3),
                x=bx,
                y=by,
                state=state,
                motion=None if motion_deg is None else round(motion_deg),
                chosen=None if chosen is None else round(chosen),
                fwd_clear=clrs[int(round((motion_deg % 360) / 360 * N_RAYS)) % N_RAYS]
                if motion_deg is not None
                else 0,
            )
            if now - status_t >= 0.4:
                m = "--" if motion_deg is None else f"{motion_deg:4.0f}"
                c = "--" if chosen is None else f"{chosen:4.0f}"
                print(
                    f"\r[roam] {state:6s} | motion {m} -> open {c} | blk {black:.0f}   ",
                    end="",
                    flush=True,
                )
                status_t = now

            elapsed = time.perf_counter() - now
            if elapsed < 1.0 / 30.0:
                time.sleep(1.0 / 30.0 - elapsed)
    finally:
        release_all()
        cap.stop()
        logger.close()
        print("\n[roam] stopped.")


if __name__ == "__main__":
    main()

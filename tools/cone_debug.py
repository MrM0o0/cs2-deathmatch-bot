"""Diagnostic: why does the cone heading read go bistable at some angles?

Read-only -- it does NOT move the mouse. YOU slowly sweep your view left/right
through the angle where steer_test jiggles. For every frame it records what the
cone reader actually sees:

  - the chosen heading (same math minimap.py uses),
  - EVERY separate white blob near the dot (area + its angle from the dot) via
    connected components -- so if the read is flipping between two competing
    blobs, it shows up as two components,
  - and it saves the upscaled marker crop, especially around any sudden jump,
    so I can look at the real pixels at both states of the flip.

    python tools/cone_debug.py              # 20s, sweep your view slowly

Then send me logs/<newest>/. END quits early.
"""

import argparse
import ctypes
import math
import os
import sys
import time

import yaml

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.capture.screen import ScreenCapture
from src.utils.session_logger import SessionLogger
from src.utils.timer_setup import enable_high_resolution_timer
from src.vision.minimap import MinimapReader

END_VK = 0x23


def load_reader():
    with open(os.path.join(PROJECT_ROOT, "config", "settings.yaml")) as f:
        cfg = yaml.safe_load(f)
    mm = cfg["minimap"]
    reader = MinimapReader(
        mm["x"], mm["y"], mm["size"], sat_min=mm["sat_min"], val_min=mm["val_min"]
    )
    return reader, cfg, mm


def white_components(hsv_win, reader, dot_local):
    """Every white blob in the window: (area, angle-from-dot-deg, dist)."""
    white = cv2.inRange(
        hsv_win,
        np.array([0, 0, reader.cone_val_min], dtype=np.uint8),
        np.array([179, reader.cone_sat_max, 255], dtype=np.uint8),
    )
    n, _, stats, centroids = cv2.connectedComponentsWithStats(white, connectivity=8)
    dx0, dy0 = dot_local
    comps = []
    for i in range(1, n):  # skip background label 0
        area = int(stats[i, cv2.CC_STAT_AREA])
        cxw, cyw = centroids[i]
        rx, ry = cxw - dx0, cyw - dy0
        comps.append(
            {
                "area": area,
                "ang": round(math.degrees(math.atan2(ry, rx)), 1),
                "dist": round(math.hypot(rx, ry), 1),
            }
        )
    comps.sort(key=lambda c: -c["area"])
    return int(white.sum() // 255), comps


def main():
    ap = argparse.ArgumentParser(description="Diagnose cone-heading bistability")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--delay", type=int, default=6)
    ap.add_argument("--jump", type=float, default=15.0, help="Deg jump that triggers a crop dump")
    args = ap.parse_args()

    if cv2 is None:
        print("[cone] cv2/numpy missing -- aborting.")
        return

    enable_high_resolution_timer()
    reader, cfg, mm = load_reader()
    mx, my, msz = mm["x"], mm["y"], mm["size"]
    r = reader.cone_radius
    cap = ScreenCapture(monitor=cfg["display"]["monitor"], target_fps=60)
    cap.start()
    logger = SessionLogger(os.path.join(PROJECT_ROOT, "logs"), enabled=True)
    print(f"[cone] logging to {logger.session_dir}")

    for i in range(args.delay, 0, -1):
        print(f"\r[cone] starting in {i}...  (switch to CS2)   ", end="", flush=True)
        time.sleep(1)
    print("\r[cone] GO -- sweep your view SLOWLY left/right through the bad angle   ")

    def save_crop(minimap, bx, by, tag):
        x1, y1 = max(0, bx - r), max(0, by - r)
        x2, y2 = min(msz, bx + r), min(msz, by + r)
        crop = minimap[y1:y2, x1:x2]
        if crop.size:
            big = cv2.resize(crop, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(
                os.path.join(logger.session_dir, f"cone_{tag}.jpg"),
                big,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )

    last_heading = None
    last_save = 0.0
    ticks = 0
    start = time.perf_counter()
    while True:
        if ctypes.windll.user32.GetAsyncKeyState(END_VK) & 0x8000:
            break
        now = time.perf_counter()
        if now - start > args.seconds:
            break
        frame = cap.grab()
        if frame is None:
            time.sleep(0.003)
            continue
        (bx, by), heading = reader.read(frame)
        ticks += 1
        minimap = frame[my : my + msz, mx : mx + msz]
        x1, y1 = max(0, bx - r), max(0, by - r)
        x2, y2 = min(msz, bx + r), min(msz, by + r)
        hsv_win = cv2.cvtColor(minimap[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        white_n, comps = white_components(hsv_win, reader, (bx - x1, by - y1))

        jumped = last_heading is not None and abs(heading - last_heading) >= args.jump
        logger.log_tick(
            t=round(now - start, 3),
            heading=round(heading, 1),
            jump=jumped,
            white_px=white_n,
            ncomp=len(comps),
            comps=comps[:4],
        )
        stamp = f"{now - start:06.2f}".replace(".", "_")
        if jumped:  # save the frames around a flip
            save_crop(minimap, bx, by, f"JUMP_{stamp}_h{heading:.0f}")
        elif now - last_save >= 0.5:
            save_crop(minimap, bx, by, f"{stamp}_h{heading:.0f}")
            last_save = now
        last_heading = heading

        if now - last_save < 0.05 or ticks % 15 == 0:
            print(
                f"\r[cone] head {heading:6.0f} | white {white_n:3d}px | comps {len(comps)}   ",
                end="",
                flush=True,
            )

    cap.stop()
    logger.close()
    print(f"\n[cone] done, {ticks} frames. Send me {logger.session_dir}")


if __name__ == "__main__":
    main()

"""M1: validate the navigation signals (position + heading) from one walk.

This does NOT drive the bot. You walk a route in CS2 yourself; it watches the
minimap and records, every tick:

  - the detected blip position,
  - the ellipse-fit "angle" from minimap.py (the suspect one),
  - a motion-derived heading (net blip displacement over ~1s, like navigator),

to logs/<ts>/session.jsonl, plus every second it saves two images:

  - minimap_*.jpg  -- the full 256px minimap, annotated with the detected blip
                      and both heading guesses drawn as rays, so I can eyeball
                      whether the reads match reality;
  - marker_*.jpg   -- a small crop around the blip, upscaled 8x, so I can see
                      the raw marker pixels and judge whether the ARROW TIP is
                      legible (a clean tip = instant, lag-free heading).

At the end it prints a moves/frozen verdict for position and how often the
motion-heading was actually available.

    python tools/validate_signals.py                 # 90s walk, 6s countdown
    python tools/validate_signals.py --seconds 120 --delay 8

Then send me the newest logs/<ts>/ folder. END stops early.
"""

import argparse
import math
import os
import sys
import time
from collections import deque

import yaml

try:
    import cv2
except ImportError:
    cv2 = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import ctypes

from src.capture.screen import ScreenCapture
from src.utils.math_helpers import angle_between, distance
from src.utils.session_logger import SessionLogger
from src.utils.timer_setup import enable_high_resolution_timer
from src.vision.minimap import MinimapReader

END_VK = 0x23


def load_minimap_reader():
    with open(os.path.join(PROJECT_ROOT, "config", "settings.yaml")) as f:
        cfg = yaml.safe_load(f)
    mm = cfg["minimap"]
    reader = MinimapReader(
        mm["x"],
        mm["y"],
        mm["size"],
        player_arrow_color=tuple(mm.get("player_arrow_color", (0, 255, 255))),
        sat_min=mm.get("sat_min", 150),
        val_min=mm.get("val_min", 190),
    )
    return reader, cfg, mm


def annotate_minimap(minimap, blip, ellipse_deg, motion_deg):
    """Draw the detected blip + both heading rays onto a copy of the minimap."""
    img = minimap.copy()
    bx, by = blip
    cv2.circle(img, (bx, by), 4, (0, 255, 0), 1)
    # ellipse angle (cyan) and motion heading (magenta), each a 30px ray
    for deg, color in ((ellipse_deg, (255, 255, 0)), (motion_deg, (255, 0, 255))):
        if deg is None:
            continue
        rad = math.radians(deg)
        ex = int(bx + 30 * math.cos(rad))
        ey = int(by + 30 * math.sin(rad))
        cv2.line(img, (bx, by), (ex, ey), color, 1)
    cv2.putText(
        img,
        f"e={ellipse_deg:.0f} m={'--' if motion_deg is None else f'{motion_deg:.0f}'}",
        (4, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )
    return img


def marker_crop(minimap, blip, half=18, scale=8):
    """Crop a small box around the blip and upscale it (nearest) so the raw
    marker pixels -- and any arrow tip -- are clearly visible."""
    bx, by = blip
    h, w = minimap.shape[:2]
    x1, y1 = max(0, bx - half), max(0, by - half)
    x2, y2 = min(w, bx + half), min(h, by + half)
    crop = minimap[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser(description="M1: validate nav signals from a walk")
    ap.add_argument("--seconds", type=float, default=90.0, help="How long to record")
    ap.add_argument("--delay", type=int, default=6, help="Countdown to alt-tab into CS2")
    ap.add_argument("--save-every", type=float, default=1.0, help="Seconds between image dumps")
    ap.add_argument("--history", type=int, default=30, help="Frames for motion-heading window")
    ap.add_argument(
        "--move-threshold", type=float, default=3.0, help="Min px move to trust heading"
    )
    args = ap.parse_args()

    if cv2 is None:
        print("[validate] cv2 not available -- aborting.")
        return

    enable_high_resolution_timer()
    reader, cfg, mm = load_minimap_reader()
    mx, my, msize = mm["x"], mm["y"], mm["size"]

    cap = ScreenCapture(monitor=cfg["display"]["monitor"], target_fps=60)
    cap.start()
    logger = SessionLogger(os.path.join(PROJECT_ROOT, "logs"), enabled=True)
    print(f"[validate] logging to {logger.session_dir}")

    for i in range(args.delay, 0, -1):
        print(f"\r[validate] starting in {i}...  (switch to CS2 now)   ", end="", flush=True)
        time.sleep(1)
    print("\r[validate] GO -- walk your route!                          ")

    positions = deque(maxlen=max(2, args.history))
    trail = []
    heading_available = 0
    ticks = 0
    last_save = 0.0
    status_t = 0.0
    start = time.perf_counter()
    while True:
        if ctypes.windll.user32.GetAsyncKeyState(END_VK) & 0x8000:
            break
        now = time.perf_counter()
        if now - start > args.seconds:
            break

        frame = cap.grab()
        if frame is None:
            time.sleep(0.002)
            continue

        (bx, by), ellipse_deg = reader.read(frame)
        ticks += 1
        positions.append((float(bx), float(by)))
        trail.append((bx, by))

        # motion heading: net displacement over the window, if it moved enough
        motion_deg = None
        if len(positions) >= 2:
            ox, oy = positions[0]
            nx, ny = positions[-1]
            if distance((ox, oy), (nx, ny)) >= args.move_threshold:
                motion_deg = angle_between((ox, oy), (nx, ny))
                heading_available += 1

        logger.log_tick(
            t=round(now - start, 3),
            x=bx,
            y=by,
            ellipse_deg=round(float(ellipse_deg), 1),
            motion_deg=None if motion_deg is None else round(motion_deg, 1),
        )

        if now - last_save >= args.save_every:
            minimap = frame[my : my + msize, mx : mx + msize]
            if minimap.size:
                stamp = f"{now - start:06.1f}".replace(".", "_")
                cv2.imwrite(
                    os.path.join(logger.session_dir, f"minimap_{stamp}.jpg"),
                    annotate_minimap(minimap, (bx, by), ellipse_deg, motion_deg),
                    [cv2.IMWRITE_JPEG_QUALITY, 85],
                )
                mc = marker_crop(minimap, (bx, by))
                if mc is not None:
                    cv2.imwrite(
                        os.path.join(logger.session_dir, f"marker_{stamp}.jpg"),
                        mc,
                        [cv2.IMWRITE_JPEG_QUALITY, 90],
                    )
            last_save = now

        if now - status_t >= 0.5:
            md = "--" if motion_deg is None else f"{motion_deg:5.0f}"
            print(
                f"\r[validate] blip=({bx:4d},{by:4d}) ellipse={ellipse_deg:5.0f} "
                f"motion={md} | ticks {ticks}   ",
                end="",
                flush=True,
            )
            status_t = now

    cap.stop()
    logger.close()

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if not trail:
        print("[validate] VERDICT: no samples -- minimap never detected. Check")
        print("           the minimap x/y/size in settings.yaml.")
        return
    xs = [p[0] for p in trail]
    ys = [p[1] for p in trail]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    path_len = sum(distance(trail[i - 1], trail[i]) for i in range(1, len(trail)))
    print(f"[validate] ticks: {ticks} | blip span: {span}px | path length: {path_len:.0f}px")
    if span < 10:
        print("[validate] POSITION: FROZEN -- blip barely moved. Tracking is broken")
        print("           (rotating radar? wrong region? not in a game?).")
    else:
        print("[validate] POSITION: MOVES -- tracking looks alive. ")
    pct = 100.0 * heading_available / max(1, ticks)
    print(
        f"[validate] MOTION-HEADING available on {pct:.0f}% of ticks "
        f"({'good' if pct > 50 else 'sparse -- lots of stationary/jitter ticks'})."
    )
    print(
        f"[validate] Saved annotated minimaps + marker crops in:\n           {logger.session_dir}"
    )
    print("[validate] Send me that folder.")


if __name__ == "__main__":
    main()

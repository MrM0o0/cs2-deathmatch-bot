"""Rung 2 of the assistant ladder: aim assist (soft aim).

Builds on Rung 1 (threat highlight). Same detector, but now it nudges the mouse.
While you HOLD the activation key, it gently pulls the crosshair toward the
nearest enemy's chest -- you still do most of the aiming, it just helps you
track. Release the key and it does nothing. It never fires (that is Rung 3).

The pull is fractional and capped per tick, with a deadzone near the target, so
it reads as "sticky aim help," not an instant snap. Strength is tunable below.

    python tools/aim_assist.py                  # hold RIGHT MOUSE to assist
    python tools/aim_assist.py --key mouse5     # hold side button instead
    python tools/aim_assist.py --strength 0.30  # stronger pull
    python tools/aim_assist.py --no-window      # headless, less overhead

Safety: END quits instantly, the run auto-stops after --max-seconds, and the
mouse is only ever touched while you hold the key.

NOTE: synthetic aim input is VAC-bannable in real matchmaking. Offline /
private / with-friends only -- this is a prank-and-learning tool.
"""

import argparse
import ctypes
import os
import sys
import time

import cv2
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.capture.screen import ScreenCapture
from src.input import mouse
from src.utils.math_helpers import bbox_to_aim_point, distance, screen_delta_to_mouse
from src.vision.confirmation_filter import ConfirmationFilter
from src.vision.detector import YOLODetector

# Virtual-key codes for the activation key (held to enable the pull).
KEYS = {
    "rmb": 0x02,  # right mouse button (default)
    "mouse4": 0x05,  # XBUTTON1 (side, back)
    "mouse5": 0x06,  # XBUTTON2 (side, forward)
    "shift": 0x10,
    "alt": 0x12,
    "ctrl": 0x11,
    "c": 0x43,
}
END_VK = 0x23


def main():
    ap = argparse.ArgumentParser(description="Rung 2: aim assist (soft aim)")
    ap.add_argument(
        "--key",
        default="rmb",
        choices=sorted(KEYS),
        help="Hold this to enable the pull (default: rmb)",
    )
    ap.add_argument(
        "--strength",
        type=float,
        default=0.20,
        help="Fraction of the remaining gap closed per tick (0.1 soft .. 0.4 sticky)",
    )
    ap.add_argument(
        "--max-step", type=int, default=60, help="Max mouse counts moved per tick (anti-snap cap)"
    )
    ap.add_argument(
        "--deadzone", type=float, default=6.0, help="Screen px from target where the pull stops"
    )
    ap.add_argument(
        "--track-radius",
        type=float,
        default=130.0,
        help="Within this px of target, pull harder (sticky tracking) to hold the lock",
    )
    ap.add_argument(
        "--track-strength",
        type=float,
        default=0.50,
        help="Pull fraction used once the crosshair is inside track-radius",
    )
    ap.add_argument(
        "--lead",
        type=float,
        default=0.05,
        help="Seconds of motion to lead a moving target (aim where it's going, 0 = off)",
    )
    ap.add_argument("--head", action="store_true", help="Aim head instead of chest")
    ap.add_argument("--scale", type=float, default=0.5, help="Debug window scale")
    ap.add_argument("--no-window", action="store_true", help="Run headless (no debug window)")
    ap.add_argument("--monitor", type=int, default=None)
    ap.add_argument("--max-seconds", type=float, default=120.0, help="Auto-stop after N seconds")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(PROJECT_ROOT, "config", "settings.yaml")))
    det_cfg = cfg["detection"]
    game = cfg["game"]
    monitor = args.monitor if args.monitor is not None else cfg["display"]["monitor"]
    activate_vk = KEYS[args.key]

    cap = ScreenCapture(monitor=monitor, target_fps=cfg["display"]["capture_fps"])
    print("[assist] capture:", cap.start())

    # Auto-detect resolution + crosshair centre from a real frame.
    cx, cy, screen_w, screen_h = 1720, 720, 3440, 1440
    for _ in range(40):
        f = cap.grab()
        if f is not None:
            screen_h, screen_w = f.shape[:2]
            cx, cy = screen_w // 2, screen_h // 2
            break
        time.sleep(0.02)
    print(f"[assist] resolution {screen_w}x{screen_h} crosshair ({cx},{cy})")

    det = YOLODetector(
        model_path=os.path.join(PROJECT_ROOT, det_cfg["model_path"]),
        input_size=det_cfg["input_size"],
        confidence_threshold=det_cfg["confidence_threshold"],
        nms_threshold=det_cfg["nms_threshold"],
        classes=det_cfg["classes"],
    )
    det.load()
    conf = ConfirmationFilter()
    user32 = ctypes.windll.user32

    win = "Aim Assist (END to quit)"
    if not args.no_window:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print(
        f"[assist] running -- HOLD {args.key.upper()} to assist, END to quit "
        f"(strength={args.strength}, deadzone={args.deadzone}px)"
    )

    fps_t, frames, fps = time.perf_counter(), 0, 0.0
    start = time.perf_counter()
    # Target-velocity state for lead prediction (aim where it's heading).
    prev_aim = None
    prev_aim_t = 0.0
    vel_x = vel_y = 0.0
    while True:
        if user32.GetAsyncKeyState(END_VK) & 0x8000:
            break
        if args.max_seconds and time.perf_counter() - start > args.max_seconds:
            print("[assist] max-seconds reached, stopping.")
            break

        frame = cap.grab()
        if frame is None:
            time.sleep(0.003)
            continue

        dets = conf.update(det.detect(frame))
        # Enemies are ct_player bodies; fall back to any non-head box.
        enemies = [d for d in dets if d.class_name == "ct_player"]
        if not enemies:
            enemies = [d for d in dets if not getattr(d, "is_head", False)]

        # Nearest-to-crosshair target.
        target = None
        if enemies:
            target = min(enemies, key=lambda d: distance(d.center, (cx, cy)))

        held = bool(user32.GetAsyncKeyState(activate_vk) & 0x8000)
        pulling = False
        screen_dist = 0.0

        # Track on-screen target velocity every frame (independent of `held`), so
        # the lead estimate is warm the instant you press the key. Reset velocity
        # when the aim point jumps far -- that's a target switch, not motion.
        now_t = time.perf_counter()
        if target is not None:
            aim_x, aim_y = bbox_to_aim_point(
                target.x1, target.y1, target.x2, target.y2, head_aim=args.head
            )
            if prev_aim is not None:
                dt = now_t - prev_aim_t
                jump = distance((aim_x, aim_y), prev_aim)
                if dt > 0 and jump < 250:
                    # Low-pass the velocity so it doesn't whip around on jitter.
                    vx = (aim_x - prev_aim[0]) / dt
                    vy = (aim_y - prev_aim[1]) / dt
                    vel_x = 0.5 * vel_x + 0.5 * vx
                    vel_y = 0.5 * vel_y + 0.5 * vy
                else:
                    vel_x = vel_y = 0.0  # new target / detection gap
            prev_aim = (aim_x, aim_y)
            prev_aim_t = now_t
        else:
            prev_aim = None
            vel_x = vel_y = 0.0

        if target is not None and held:
            # Lead the target: aim where it's heading, not where it was last frame.
            lead_x = aim_x + vel_x * args.lead
            lead_y = aim_y + vel_y * args.lead
            dx_pixels = lead_x - cx
            dy_pixels = lead_y - cy
            screen_dist = distance((cx, cy), (aim_x, aim_y))

            if screen_dist > args.deadzone:
                mouse_dx, mouse_dy = screen_delta_to_mouse(
                    dx_pixels,
                    dy_pixels,
                    game["sensitivity"],
                    game["m_yaw"],
                    game["m_pitch"],
                    screen_w,
                    screen_h,
                    game["fov_horizontal"],
                    scale=0.85,
                )
                # Sticky once locked: pull harder inside track-radius to hold the
                # target through your own movement; gentler when acquiring far.
                strength = args.track_strength if screen_dist < args.track_radius else args.strength
                step_dx = mouse_dx * strength
                step_dy = mouse_dy * strength
                step_dx = max(-args.max_step, min(args.max_step, step_dx))
                step_dy = max(-args.max_step, min(args.max_step, step_dy))
                ix, iy = int(round(step_dx)), int(round(step_dy))
                if ix or iy:
                    mouse.move_relative(ix, iy)
                    pulling = True

        frames += 1
        now = time.perf_counter()
        if now - fps_t >= 0.5:
            fps = frames / (now - fps_t)
            fps_t, frames = now, 0

        if not args.no_window:
            vis = frame
            for d in dets:
                is_t = d is target
                head = getattr(d, "is_head", False)
                colour = (0, 255, 0) if is_t else ((0, 200, 255) if head else (0, 0, 255))
                x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
                cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2 + int(is_t))
            cv2.drawMarker(vis, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 22, 2)
            state = "PULLING" if pulling else ("ARMED" if held else "idle")
            scolour = (0, 255, 0) if pulling else (0, 255, 255)
            hud = f"{state} | {args.key.upper()} | {len(enemies)} enemy | dist {screen_dist:.0f}"
            cv2.putText(
                vis,
                hud,
                (24, 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                scolour,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                f"{fps:4.1f} fps | infer {det.inference_ms:.0f}ms | END quits",
                (24, vis.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            small = cv2.resize(
                vis, (int(vis.shape[1] * args.scale), int(vis.shape[0] * args.scale))
            )
            cv2.imshow(win, small)
            cv2.waitKey(1)

    cap.stop()
    if not args.no_window:
        cv2.destroyAllWindows()
    print("[assist] stopped.")


if __name__ == "__main__":
    main()

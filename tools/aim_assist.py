"""Rung 2 of the assistant ladder: aim assist (soft aim).

Builds on Rung 1 (threat highlight). Same detector, but now it nudges the mouse.
While the assist is active it gently eases the crosshair toward the nearest
enemy's chest -- you still do most of the aiming, it just helps you track. It
never fires (that is Rung 3).

ARCHITECTURE: detection runs in a background thread (~slow, capped by inference)
while the mouse runs on its own fast control loop (~333 Hz). The control loop
eases toward the *latest* target with sub-pixel residual carried between ticks,
so the motion is smooth instead of one chunky jump per detection frame. Between
detections the target is extrapolated from its measured on-screen velocity, so
tracking keeps up through your own movement.

    python tools/aim_assist.py                 # hold RIGHT MOUSE to assist
    python tools/aim_assist.py --mode always   # always on, no key
    python tools/aim_assist.py --mode toggle   # tap RMB to flip on/off
    python tools/aim_assist.py --key mouse5     # use a side button
    python tools/aim_assist.py --strength 0.35  # snappier ease

Safety: END quits instantly, the run auto-stops after --max-seconds, and the
mouse is only ever touched while the assist is active.

NOTE: synthetic aim input is VAC-bannable in real matchmaking. Offline /
private / with-friends only -- this is a prank-and-learning tool.
"""

import argparse
import ctypes
import math
import os
import sys
import threading
import time

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.capture.screen import ScreenCapture
from src.input import mouse
from src.utils.math_helpers import bbox_to_aim_point, distance, screen_delta_to_mouse
from src.utils.timer_setup import enable_high_resolution_timer
from src.vision.confirmation_filter import ConfirmationFilter
from src.vision.detector import YOLODetector

# Virtual-key codes for the activation key.
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
REF_HZ = 60.0  # strength is calibrated to feel the same as one tick at 60 Hz


class TargetTracker(threading.Thread):
    """Background capture + detection. Publishes the latest target as a snapshot.

    The snapshot is a single immutable tuple swapped atomically, so the control
    loop never sees a half-updated target.
    """

    def __init__(self, cfg, fps, head, confirm):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.fps = fps
        self.head = head
        self.confirm = confirm
        self.running = True
        self.ready = threading.Event()
        self.cx = self.cy = 0
        self.screen_w = self.screen_h = 0
        # (aim_x, aim_y, vel_x, vel_y, half_w, half_h, stamp) or None
        self.snapshot = None
        self.frames = 0
        self.n_enemies = 0
        self.infer_ms = 0.0

    def run(self):
        det_cfg = self.cfg["detection"]
        cap = ScreenCapture(monitor=self.cfg["display"]["monitor"], target_fps=self.fps)
        cap.start()
        det = YOLODetector(
            model_path=os.path.join(PROJECT_ROOT, det_cfg["model_path"]),
            input_size=det_cfg["input_size"],
            confidence_threshold=det_cfg["confidence_threshold"],
            nms_threshold=det_cfg["nms_threshold"],
            classes=det_cfg["classes"],
        )
        det.load()
        # Acquire after `confirm` frames (lower = faster lock, default 2 vs the
        # 3 used for autonomous fire); survive a couple missed frames to hold it.
        conf = ConfirmationFilter(min_confirm_frames=self.confirm, max_missing_frames=3)

        prev_aim = None
        prev_t = 0.0
        vel_x = vel_y = 0.0
        while self.running:
            frame = cap.grab()
            if frame is None:
                time.sleep(0.002)
                continue
            if not self.ready.is_set():
                self.screen_h, self.screen_w = frame.shape[:2]
                self.cx, self.cy = self.screen_w // 2, self.screen_h // 2
                self.ready.set()
            self.frames += 1

            dets = conf.update(det.detect(frame))
            self.infer_ms = det.inference_ms
            enemies = [d for d in dets if d.class_name == "ct_player"]
            if not enemies:
                enemies = [d for d in dets if not getattr(d, "is_head", False)]
            self.n_enemies = len(enemies)

            if not enemies:
                self.snapshot = None
                prev_aim = None
                vel_x = vel_y = 0.0
                continue

            target = min(enemies, key=lambda d: distance(d.center, (self.cx, self.cy)))
            aim_x, aim_y = bbox_to_aim_point(
                target.x1, target.y1, target.x2, target.y2, head_aim=self.head
            )
            now = time.perf_counter()
            if prev_aim is not None:
                dt = now - prev_t
                jump = distance((aim_x, aim_y), prev_aim)
                if dt > 0 and jump < 250:  # same target -> measure its velocity
                    vx = (aim_x - prev_aim[0]) / dt
                    vy = (aim_y - prev_aim[1]) / dt
                    vel_x = 0.5 * vel_x + 0.5 * vx
                    vel_y = 0.5 * vel_y + 0.5 * vy
                else:
                    vel_x = vel_y = 0.0  # target switch / detection gap
            prev_aim = (aim_x, aim_y)
            prev_t = now

            half_w = (target.x2 - target.x1) * 0.5
            half_h = (target.y2 - target.y1) * 0.5
            self.snapshot = (aim_x, aim_y, vel_x, vel_y, half_w, half_h, now)
        cap.stop()


def main():
    ap = argparse.ArgumentParser(description="Rung 2: aim assist (soft aim)")
    ap.add_argument("--key", default="rmb", choices=sorted(KEYS), help="Activation key")
    ap.add_argument(
        "--mode",
        default="hold",
        choices=["hold", "toggle", "always"],
        help="hold = while key held; toggle = tap to flip; always = always on",
    )
    ap.add_argument(
        "--strength",
        type=float,
        default=0.20,
        help="Ease factor when acquiring a far target (0.1 soft .. 0.4 snappy)",
    )
    ap.add_argument(
        "--track-strength",
        type=float,
        default=0.45,
        help="Ease factor once locked (inside --track-radius), for a firm hold",
    )
    ap.add_argument(
        "--track-radius", type=float, default=140.0, help="Screen px to switch to track-strength"
    )
    ap.add_argument("--deadzone", type=float, default=4.0, help="Screen px where the pull stops")
    ap.add_argument(
        "--lead", type=float, default=0.0, help="Seconds to lead a moving target (0 = off)"
    )
    ap.add_argument("--max-step", type=int, default=18, help="Max mouse counts per control tick")
    ap.add_argument("--fps", type=int, default=144, help="Detection capture rate")
    ap.add_argument(
        "--confirm", type=int, default=2, help="Frames before acquiring a target (lower = faster)"
    )
    ap.add_argument(
        "--control-hz", type=float, default=333.0, help="Mouse control loop rate (smoothness)"
    )
    ap.add_argument("--head", action="store_true", help="Aim head instead of chest")
    ap.add_argument("--max-seconds", type=float, default=120.0, help="Auto-stop after N seconds")
    args = ap.parse_args()

    enable_high_resolution_timer()
    cfg = yaml.safe_load(open(os.path.join(PROJECT_ROOT, "config", "settings.yaml")))
    game = cfg["game"]
    activate_vk = KEYS[args.key]
    user32 = ctypes.windll.user32

    tracker = TargetTracker(cfg, args.fps, args.head, args.confirm)
    tracker.start()
    print("[assist] loading detector + capture...")
    if not tracker.ready.wait(timeout=30):
        print("[assist] capture never produced a frame -- aborting.")
        return
    cx, cy, sw, sh = tracker.cx, tracker.cy, tracker.screen_w, tracker.screen_h
    print(f"[assist] {sw}x{sh} crosshair ({cx},{cy})")
    how = {"hold": f"HOLD {args.key.upper()}", "toggle": f"TAP {args.key.upper()}", "always": "ON"}[
        args.mode
    ]
    print(f"[assist] running -- {how} to assist, END to quit (mode={args.mode})")

    period = 1.0 / args.control_hz
    residual_x = residual_y = 0.0
    # Closed-loop accounting: how much we've already moved toward the CURRENT
    # detection snapshot. Subtracting this from the gap stops the fast control
    # loop over-driving past a stale target between detections (the orbit bug).
    last_stamp = None
    applied_x = applied_y = 0.0
    toggled_on = False
    key_was_down = False
    last = time.perf_counter()
    status_t = last
    start = last
    while True:
        if user32.GetAsyncKeyState(END_VK) & 0x8000:
            break
        if args.max_seconds and time.perf_counter() - start > args.max_seconds:
            print("[assist] max-seconds reached, stopping.")
            break

        now = time.perf_counter()
        dt = now - last
        last = now

        # Resolve whether the assist is active this tick from the chosen mode.
        key_down = bool(user32.GetAsyncKeyState(activate_vk) & 0x8000)
        if args.mode == "always":
            active = True
        elif args.mode == "toggle":
            if key_down and not key_was_down:
                toggled_on = not toggled_on
            active = toggled_on
        else:
            active = key_down
        key_was_down = key_down

        snap = tracker.snapshot
        pulling = False
        if active and snap is not None:
            aim_x, aim_y, vel_x, vel_y, half_w, half_h, stamp = snap
            if stamp != last_stamp:  # fresh detection -> reset what we've applied
                last_stamp = stamp
                applied_x = applied_y = 0.0
            # Small lead only (clamped to the body); no stale-time extrapolation,
            # which fed the orbit during your own turns.
            off_x = max(-half_w, min(half_w, vel_x * args.lead))
            off_y = max(-half_h, min(half_h, vel_y * args.lead))
            px = aim_x + off_x - cx
            py = aim_y + off_y - cy
            screen_dist = math.hypot(px, py)

            if screen_dist > args.deadzone:
                gap_dx, gap_dy = screen_delta_to_mouse(
                    px,
                    py,
                    game["sensitivity"],
                    game["m_yaw"],
                    game["m_pitch"],
                    sw,
                    sh,
                    game["fov_horizontal"],
                    scale=0.85,
                )
                # Remaining gap = full gap minus what we've already emitted toward
                # this snapshot. Converges to the target and STOPS, no overshoot.
                rem_x = gap_dx - applied_x
                rem_y = gap_dy - applied_y
                s = args.track_strength if screen_dist < args.track_radius else args.strength
                # Framerate-independent ease: same feel as `s` per tick at 60 Hz,
                # but spread over the fast control rate so motion stays smooth.
                frac = 1.0 - (1.0 - s) ** (dt * REF_HZ)
                desired_x = max(-args.max_step, min(args.max_step, rem_x * frac))
                desired_y = max(-args.max_step, min(args.max_step, rem_y * frac))
                applied_x += desired_x
                applied_y += desired_y
                out_x = desired_x + residual_x
                out_y = desired_y + residual_y
                ix, iy = int(out_x), int(out_y)
                residual_x = out_x - ix  # carry the sub-pixel remainder
                residual_y = out_y - iy
                if ix or iy:
                    mouse.move_relative(ix, iy)
                    pulling = True
        else:
            residual_x = residual_y = 0.0
            applied_x = applied_y = 0.0
            last_stamp = None

        if now - status_t >= 0.5:
            state = "PULL" if pulling else ("ARMED" if active else "idle")
            print(
                f"\r[assist] {state:5s} | enemies {tracker.n_enemies} | "
                f"cap {tracker.frames} | infer {tracker.infer_ms:.0f}ms   ",
                end="",
                flush=True,
            )
            status_t = now

        # Pace the control loop without busy-spinning the CPU.
        sleep_for = period - (time.perf_counter() - now)
        if sleep_for > 0:
            time.sleep(sleep_for)

    tracker.running = False
    print("\n[assist] stopped.")


if __name__ == "__main__":
    main()

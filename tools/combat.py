"""Rung 4 of the assistant ladder: autonomous combat (aim + fire combined).

Merges Rung 2 (aim assist) and Rung 3 (triggerbot) into ONE process driven by a
single detector -- so it runs the screen capture and YOLO once and feeds both
the aim and the trigger, instead of two scripts fighting over the GPU.

Per tick it eases the crosshair toward the nearest enemy (R2) and, once the
crosshair is on the body, fires a recoil-controlled burst (R3). It is fully
autonomous by default but every "tell" is a knob you can soften:

    --reaction-min/max   wait like a human before engaging a fresh target
    --aim-strength       how fast it pulls on (lower = lazier, more human)
    --aim-max-step       cap mouse counts/tick (lower = no inhuman teleport)
    --inset              how centred it must be before firing
    --recoil-pull        AK climb compensation (his tuned value carries over)

Run any subset -- the rungs are independent flags:

    python tools/combat.py                  # full auto: aim + fire
    python tools/combat.py --no-fire        # aim assist only (R2)
    python tools/combat.py --no-aim         # triggerbot only (R3)
    python tools/combat.py --mode hold --key rmb   # only while holding RMB

Safety: END quits instantly, the run auto-stops after --max-seconds, the mouse
is only ever driven while active, and all buttons are released on exit.

NOTE: synthetic aim/fire is VAC-bannable in real matchmaking. Offline / private
/ with-friends only -- this is a prank-and-learning tool.
"""

import argparse
import ctypes
import math
import os
import random
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

KEYS = {
    "rmb": 0x02,
    "mouse4": 0x05,
    "mouse5": 0x06,
    "shift": 0x10,
    "alt": 0x12,
    "ctrl": 0x11,
    "c": 0x43,
}
END_VK = 0x23
REF_HZ = 60.0  # aim strength is calibrated to feel like one tick at 60 Hz


class TargetTracker(threading.Thread):
    """Background capture + detection. Publishes the nearest enemy as an atomic
    snapshot: aim point (chest/head), its on-screen velocity, and the body box."""

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
        # (aim_x, aim_y, vel_x, vel_y, x1, y1, x2, y2, stamp) or None
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
            # Single-class model: every detection is an enemy player (skip any
            # stray head box for safety; the model emits none).
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
                if dt > 0 and jump < 250:  # same target -> measure velocity
                    vx = (aim_x - prev_aim[0]) / dt
                    vy = (aim_y - prev_aim[1]) / dt
                    vel_x = 0.5 * vel_x + 0.5 * vx
                    vel_y = 0.5 * vel_y + 0.5 * vy
                else:
                    vel_x = vel_y = 0.0
            prev_aim = (aim_x, aim_y)
            prev_t = now

            self.snapshot = (
                aim_x,
                aim_y,
                vel_x,
                vel_y,
                target.x1,
                target.y1,
                target.x2,
                target.y2,
                now,
            )
        cap.stop()


def crosshair_in_box(cx, cy, box, inset):
    """True if (cx, cy) is inside `box` after shrinking it by `inset` each side."""
    x1, y1, x2, y2 = box
    mx = (x2 - x1) * inset
    my = (y2 - y1) * inset
    return (x1 + mx) <= cx <= (x2 - mx) and (y1 + my) <= cy <= (y2 - my)


def spray_burst(count, interval, pull, on_target, kill_grace):
    """Hold fire while pulling the view DOWN each shot to fight the AK climb,
    then release so the pattern resets. Eases the pull in over the first two
    rounds. Cuts off shortly after the target is gone (kill_grace) so a kill
    eats 1-2 rounds, not the mag. END polled throughout. Returns False on END."""
    user32 = ctypes.windll.user32
    step = 0.012
    total = count * interval
    next_shot = interval
    shot_i = 1
    lost_for = None
    elapsed = 0.0
    mouse.mouse_down("left")
    try:
        while elapsed < total:
            if user32.GetAsyncKeyState(END_VK) & 0x8000:
                return False
            time.sleep(step)
            elapsed += step
            if elapsed >= next_shot and shot_i < count:
                ramp = min(1.0, (shot_i + 1) / 2.0)
                dy = int(round(pull * ramp))
                if dy:
                    mouse.move_relative(0, dy)
                shot_i += 1
                next_shot += interval
            if on_target():
                lost_for = None
            else:
                lost_for = step if lost_for is None else lost_for + step
                if lost_for >= kill_grace:
                    break
    finally:
        mouse.mouse_up("left")
    return True


def main():
    ap = argparse.ArgumentParser(description="Rung 4: autonomous combat (aim + fire)")
    ap.add_argument("--key", default="rmb", choices=sorted(KEYS), help="Activation key")
    ap.add_argument(
        "--mode",
        default="always",
        choices=["hold", "toggle", "always"],
        help="always = autonomous (default); hold = while key held; toggle = tap to flip",
    )
    ap.add_argument(
        "--aim", action=argparse.BooleanOptionalAction, default=True, help="Enable aim assist (R2)"
    )
    ap.add_argument(
        "--fire", action=argparse.BooleanOptionalAction, default=True, help="Enable firing (R3)"
    )
    # --- aim (R2) ---
    ap.add_argument(
        "--aim-strength", type=float, default=0.30, help="Ease when acquiring (0.1..0.5)"
    )
    ap.add_argument(
        "--track-strength", type=float, default=0.50, help="Ease once locked (inside track-radius)"
    )
    ap.add_argument("--track-radius", type=float, default=140.0, help="Px to switch to track ease")
    ap.add_argument("--aim-deadzone", type=float, default=3.0, help="Px where the aim pull stops")
    ap.add_argument(
        "--aim-max-step", type=int, default=35, help="Max mouse counts/tick (lower=smoother/human)"
    )
    ap.add_argument(
        "--smooth",
        type=float,
        default=0.4,
        help="Target low-pass per detection frame (lower = smoother/glidier, higher = snappier)",
    )
    ap.add_argument("--head", action="store_true", help="Aim head instead of chest")
    # --- fire (R3) ---
    ap.add_argument("--inset", type=float, default=0.15, help="Shrink hit box each side (stricter)")
    ap.add_argument("--reaction-min", type=float, default=0.05, help="Min engage delay (s)")
    ap.add_argument("--reaction-max", type=float, default=0.12, help="Max engage delay (s)")
    ap.add_argument("--burst", type=int, default=9, help="Bullets per burst")
    ap.add_argument("--rpm", type=float, default=600.0, help="Fire rate (AK = 600)")
    ap.add_argument("--recoil-pull", type=float, default=152.0, help="Counts pulled DOWN per shot")
    ap.add_argument("--cooldown", type=float, default=0.4, help="Min seconds between bursts")
    ap.add_argument(
        "--kill-grace", type=float, default=0.10, help="Keep firing N s after target gone"
    )
    ap.add_argument("--repeat", action="store_true", help="Re-spray while still aligned")
    # --- shared ---
    ap.add_argument("--fps", type=int, default=144, help="Detection capture rate")
    ap.add_argument("--confirm", type=int, default=3, help="Frames before a target is usable")
    ap.add_argument("--control-hz", type=float, default=333.0, help="Aim control loop rate")
    ap.add_argument("--max-seconds", type=float, default=120.0, help="Auto-stop after N seconds")
    args = ap.parse_args()

    enable_high_resolution_timer()
    cfg = yaml.safe_load(open(os.path.join(PROJECT_ROOT, "config", "settings.yaml")))
    game = cfg["game"]
    activate_vk = KEYS[args.key]
    user32 = ctypes.windll.user32

    tracker = TargetTracker(cfg, args.fps, args.head, args.confirm)
    tracker.start()
    print("[combat] loading detector + capture...")
    if not tracker.ready.wait(timeout=30):
        print("[combat] capture never produced a frame -- aborting.")
        return
    cx, cy, sw, sh = tracker.cx, tracker.cy, tracker.screen_w, tracker.screen_h
    rungs = "+".join([r for r, on in (("aim", args.aim), ("fire", args.fire)) if on]) or "none"
    print(f"[combat] {sw}x{sh} crosshair ({cx},{cy}) | rungs: {rungs}")
    how = {"hold": f"HOLD {args.key.upper()}", "toggle": f"TAP {args.key.upper()}", "always": "ON"}[
        args.mode
    ]
    print(f"[combat] running -- {how}, END to quit (mode={args.mode})")

    period = 1.0 / args.control_hz
    residual_x = residual_y = 0.0
    last_stamp = None
    applied_x = applied_y = 0.0
    smooth_x = smooth_y = None  # low-passed target the aim glides toward
    toggled_on = False
    key_was_down = False
    aligned_since = None
    react_delay = 0.0
    fired_this_onset = False
    last_fire = 0.0
    last = time.perf_counter()
    status_t = last
    start = last
    try:
        while True:
            if user32.GetAsyncKeyState(END_VK) & 0x8000:
                break
            if args.max_seconds and time.perf_counter() - start > args.max_seconds:
                print("\n[combat] max-seconds reached, stopping.")
                break

            now = time.perf_counter()
            dt = now - last
            last = now

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
            state = "idle"
            if active and snap is not None:
                aim_x, aim_y, vel_x, vel_y, x1, y1, x2, y2, stamp = snap

                # --- aim (R2): ease the crosshair toward the body ---
                if args.aim:
                    if stamp != last_stamp:
                        last_stamp = stamp
                        applied_x = applied_y = 0.0
                        # Low-pass the target across detection frames so box
                        # jitter doesn't make the mouse twitch. Snap on a big
                        # jump (target switch) instead of gliding across screen.
                        if smooth_x is None or math.hypot(aim_x - smooth_x, aim_y - smooth_y) > 200:
                            smooth_x, smooth_y = aim_x, aim_y
                        else:
                            smooth_x += args.smooth * (aim_x - smooth_x)
                            smooth_y += args.smooth * (aim_y - smooth_y)
                    px = smooth_x - cx
                    py = smooth_y - cy
                    screen_dist = math.hypot(px, py)
                    if screen_dist > args.aim_deadzone:
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
                        rem_x = gap_dx - applied_x
                        rem_y = gap_dy - applied_y
                        s = (
                            args.track_strength
                            if screen_dist < args.track_radius
                            else args.aim_strength
                        )
                        frac = 1.0 - (1.0 - s) ** (dt * REF_HZ)
                        dx = max(-args.aim_max_step, min(args.aim_max_step, rem_x * frac))
                        dy = max(-args.aim_max_step, min(args.aim_max_step, rem_y * frac))
                        applied_x += dx
                        applied_y += dy
                        out_x = dx + residual_x
                        out_y = dy + residual_y
                        ix, iy = int(out_x), int(out_y)
                        residual_x = out_x - ix
                        residual_y = out_y - iy
                        if ix or iy:
                            mouse.move_relative(ix, iy)
                    state = "aim"

                # --- fire (R3): burst when the crosshair is on the body ---
                if args.fire:
                    aligned = crosshair_in_box(cx, cy, (x1, y1, x2, y2), args.inset)
                    if aligned:
                        state = "ontgt"
                        if aligned_since is None:
                            aligned_since = now
                            react_delay = random.uniform(args.reaction_min, args.reaction_max)
                            fired_this_onset = False
                        ready = now - aligned_since >= react_delay
                        cooled = now - last_fire >= args.cooldown
                        allowed = args.repeat or not fired_this_onset
                        if ready and cooled and allowed:
                            interval = 60.0 / args.rpm

                            def on_target():
                                s = tracker.snapshot
                                if s is None:
                                    return False
                                return crosshair_in_box(
                                    cx, cy, (s[4], s[5], s[6], s[7]), args.inset
                                )

                            alive = spray_burst(
                                args.burst, interval, args.recoil_pull, on_target, args.kill_grace
                            )
                            last_fire = time.perf_counter()
                            fired_this_onset = True
                            state = "fire"
                            # The burst moved the view; re-acquire aim cleanly.
                            last_stamp = None
                            applied_x = applied_y = 0.0
                            residual_x = residual_y = 0.0
                            smooth_x = smooth_y = None
                            if not alive:
                                break
                    else:
                        aligned_since = None
            else:
                residual_x = residual_y = 0.0
                applied_x = applied_y = 0.0
                last_stamp = None
                smooth_x = smooth_y = None
                aligned_since = None

            if now - status_t >= 0.5:
                print(
                    f"\r[combat] {state:5s} | enemies {tracker.n_enemies} | "
                    f"cap {tracker.frames} | infer {tracker.infer_ms:.0f}ms   ",
                    end="",
                    flush=True,
                )
                status_t = now

            sleep_for = period - (time.perf_counter() - now)
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        tracker.running = False
        mouse.release_all_buttons()
        print("\n[combat] stopped.")


if __name__ == "__main__":
    main()

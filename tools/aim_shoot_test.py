"""Aim + shoot test v6 — smooth Bezier paths, proper recoil, death detection.

Non-blocking Bezier curves for smooth aim.
Recoil pull-down during spray.
Stops shooting shortly after target dies.

Usage:
    python tools/aim_shoot_test.py

Press Q in debug window or Ctrl+C to stop.
"""

import sys
import os
import time
import random
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.timer_setup import enable_high_resolution_timer
enable_high_resolution_timer()

import cv2
import numpy as np
from src.capture.screen import ScreenCapture
from src.vision.detector import YOLODetector, Detection
from src.vision.confirmation_filter import ConfirmationFilter
from src.aim.aim_path import AimPath
from src.input import mouse
from src.utils.math_helpers import bbox_to_aim_point, distance, screen_delta_to_mouse

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "cs2_yolov8n.onnx")
LOG_PATH = os.path.join(os.path.dirname(__file__), "diagnostic_log.txt")


class DiagLogger:
    def __init__(self, path):
        self.f = open(path, "w")
        self.start = time.perf_counter()
        self.log("=== Diagnostic Log Started ===")

    def log(self, msg):
        t = time.perf_counter() - self.start
        self.f.write(f"[{t:8.3f}s] {msg}\n")
        self.f.flush()

    def close(self):
        self.log("=== Log Ended ===")
        self.f.close()


FIRE_IDLE = "idle"
FIRE_AIMING = "aiming"
FIRE_SHOOTING = "shooting"
FIRE_COOLDOWN = "cooldown"


def main():
    print("=" * 55)
    print("  CS2 Aim + Shoot Test v6")
    print("  Smooth Bezier | Recoil | Death detect")
    print("  Press Q in debug window or Ctrl+C to stop")
    print("=" * 55)

    log = DiagLogger(LOG_PATH)

    # Init systems
    detector = YOLODetector(model_path=MODEL_PATH, confidence_threshold=0.50, nms_threshold=0.5)
    detector.load()

    conf_filter = ConfirmationFilter(
        min_confirm_frames=2, max_missing_frames=3,
        match_distance=200.0, match_size_ratio=0.3,
    )

    aim_path = AimPath()

    capture = ScreenCapture(target_fps=120)
    backend = capture.start()

    # Auto-detect resolution
    time.sleep(0.5)
    cx, cy = 960, 540
    for _ in range(30):
        f = capture.grab()
        if f is not None:
            h, w = f.shape[:2]
            cx, cy = w // 2, h // 2
            break
        time.sleep(0.05)

    screen_w, screen_h = cx * 2, cy * 2
    print(f"Resolution: {screen_w}x{screen_h} | Crosshair: ({cx}, {cy})")
    log.log(f"Resolution: {screen_w}x{screen_h}")

    # Game settings
    sensitivity = 0.85
    m_yaw = 0.022
    m_pitch = 0.022
    fov_h = 122.0

    # Fire settings
    bullets_per_burst = 8
    bullet_interval = 0.1       # 100ms per bullet
    burst_cooldown_time = 0.15
    recoil_per_bullet = 4       # Pull down per bullet

    log.log(f"Settings: sens={sensitivity} fov={fov_h} burst={bullets_per_burst}")

    print(f"\nRunning! Switch to CS2. Press Q to stop.\n")

    cv2.namedWindow("Bot Debug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Bot Debug", 960, 540)

    # State
    fire_state = FIRE_IDLE
    fire_start_time = 0
    bullets_fired = 0
    cooldown_until = 0
    last_target_pos = None
    last_target_time = 0
    target_lost_grace = 0.3       # Only keep firing 300ms after target gone
    frame_count = 0
    reaction_until = 0
    last_aim_dist = 999

    # FPS counter
    fps_counter = 0
    fps_timer = time.perf_counter()
    display_fps = 0

    # Detection results
    raw = []
    confirmed = []

    # Recoil tracking
    total_recoil_applied = 0

    try:
        while True:
            frame = capture.grab()
            if frame is None:
                time.sleep(0.001)
                continue

            now = time.perf_counter()
            frame_count += 1

            # FPS counter
            fps_counter += 1
            if now - fps_timer >= 1.0:
                display_fps = fps_counter
                fps_counter = 0
                fps_timer = now

            # Safety: force release after 2s
            if fire_state == FIRE_SHOOTING and (now - fire_start_time) > 2.0:
                mouse.mouse_up("left")
                fire_state = FIRE_IDLE
                total_recoil_applied = 0
                log.log("SAFETY: force release after 2s")

            # ── DETECT (every 2nd frame) ──
            if frame_count % 2 == 0:
                raw = detector.detect(frame)
                confirmed = conf_filter.update(raw)

            # Pick target
            players = [d for d in confirmed if d.class_name == "ct_player"]
            target = None
            if players:
                if last_target_pos is not None:
                    near = [p for p in players if distance(p.center, last_target_pos) < 300]
                    if near:
                        near.sort(key=lambda d: distance(d.center, last_target_pos))
                        target = near[0]
                if target is None:
                    players.sort(key=lambda d: distance(d.center, (cx, cy)))
                    target = players[0]

            # Track target
            if target:
                tcx, tcy = target.center
                is_new = (last_target_pos is None or
                         distance(last_target_pos, (tcx, tcy)) > 400)
                last_target_pos = (tcx, tcy)
                last_target_time = now

                if is_new and fire_state != FIRE_SHOOTING:
                    reaction_ms = random.uniform(60, 150)
                    reaction_until = now + reaction_ms / 1000.0
                    aim_path.cancel()
                    log.log(f"NEW TARGET ({tcx:.0f},{tcy:.0f}) reaction={reaction_ms:.0f}ms")

            has_target = target is not None
            target_recently_seen = (now - last_target_time) < target_lost_grace
            status = "IDLE"

            # ── STEP BEZIER PATH (every frame, sub-frame smoothing) ──
            if aim_path.is_active:
                aim_path.apply_frame(mouse.move_relative)

            # ── AIM + FIRE LOGIC ──
            if has_target and now >= reaction_until:
                aim_x, aim_y = bbox_to_aim_point(
                    target.x1, target.y1, target.x2, target.y2,
                    head_aim=False,
                )
                dx_pixels = aim_x - cx
                dy_pixels = aim_y - cy
                screen_dist = distance((cx, cy), (aim_x, aim_y))
                last_aim_dist = screen_dist

                mouse_dx, mouse_dy = screen_delta_to_mouse(
                    dx_pixels, dy_pixels, sensitivity, m_yaw, m_pitch,
                    screen_w, screen_h, fov_h
                )

                if fire_state == FIRE_SHOOTING:
                    # While firing: gentle tracking + recoil
                    spray_time = now - fire_start_time
                    bullets_fired = int(spray_time / bullet_interval)

                    # Recoil pull-down (cumulative)
                    target_recoil = bullets_fired * recoil_per_bullet
                    recoil_to_apply = target_recoil - total_recoil_applied
                    if recoil_to_apply > 0:
                        mouse.move_relative(0, recoil_to_apply)
                        total_recoil_applied = target_recoil

                    # Track target while firing (small corrections)
                    track_dx = int(mouse_dx * 0.2)
                    track_dy = int(mouse_dy * 0.2)
                    if abs(track_dx) > 1 or abs(track_dy) > 1:
                        mouse.move_relative(track_dx, track_dy)

                    # End burst
                    if bullets_fired >= bullets_per_burst:
                        mouse.mouse_up("left")
                        fire_state = FIRE_COOLDOWN
                        cooldown_until = now + burst_cooldown_time
                        log.log(f"BURST END: {bullets_fired} bullets, recoil={total_recoil_applied}")
                        total_recoil_applied = 0
                        status = "BURST END"
                    else:
                        status = f"FIRING ({bullets_fired}/{bullets_per_burst})"

                elif fire_state == FIRE_COOLDOWN:
                    if now >= cooldown_until:
                        fire_state = FIRE_AIMING
                        status = "RE-AIMING"
                    else:
                        status = "COOLDOWN"

                elif screen_dist > 50:
                    # Need to aim — but don't restart path if one is already running
                    # Only start new path if: no path active, or path is done and still far
                    if not aim_path.is_active:
                        aim_path.start(mouse_dx, mouse_dy)
                        log.log(f"AIM PATH: dist={screen_dist:.0f}px mouse=({mouse_dx},{mouse_dy}) "
                               f"steps={aim_path.remaining_steps}")
                    fire_state = FIRE_AIMING
                    status = f"AIMING ({screen_dist:.0f}px)"

                elif screen_dist <= 50 and not aim_path.is_active:
                    # On target — start firing
                    if fire_state != FIRE_SHOOTING:
                        # Small correction before first shot
                        if abs(mouse_dx) > 2 or abs(mouse_dy) > 2:
                            mouse.move_relative(int(mouse_dx * 0.5), int(mouse_dy * 0.5))

                        mouse.mouse_down("left")
                        fire_state = FIRE_SHOOTING
                        fire_start_time = now
                        bullets_fired = 0
                        total_recoil_applied = 0
                        log.log(f"BURST START: dist={screen_dist:.0f}px")
                        status = "FIRING (start)"

                elif aim_path.is_active:
                    status = f"CURVING ({aim_path.remaining_steps} steps)"

            elif has_target and now < reaction_until:
                status = "REACTING"

            elif not has_target:
                if fire_state == FIRE_SHOOTING:
                    if target_recently_seen:
                        # Keep firing briefly — target might be between frames
                        spray_time = now - fire_start_time
                        bullets_fired = int(spray_time / bullet_interval)

                        # Still apply recoil
                        target_recoil = bullets_fired * recoil_per_bullet
                        recoil_to_apply = target_recoil - total_recoil_applied
                        if recoil_to_apply > 0:
                            mouse.move_relative(0, recoil_to_apply)
                            total_recoil_applied = target_recoil

                        if bullets_fired >= bullets_per_burst:
                            mouse.mouse_up("left")
                            fire_state = FIRE_IDLE
                            total_recoil_applied = 0
                            log.log(f"BURST END (persist): {bullets_fired} bullets")
                        status = f"PERSIST ({bullets_fired}/{bullets_per_burst})"
                    else:
                        # Target gone — stop firing
                        mouse.mouse_up("left")
                        fire_state = FIRE_IDLE
                        total_recoil_applied = 0
                        log.log("FIRE STOP: target lost")
                        status = "TARGET LOST"
                else:
                    fire_state = FIRE_IDLE
                    mouse.ensure_released("left")
                    aim_path.cancel()
                    if not target_recently_seen:
                        last_target_pos = None
                    status = "NO TARGET"

            # ── DEBUG OVERLAY ──
            if frame_count % 3 == 0:
                vis = frame.copy()

                for det in raw:
                    x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (128, 128, 128), 1)

                for det in confirmed:
                    x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
                    is_target = (det == target)
                    color = (0, 0, 255) if is_target else (0, 255, 0)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2 + is_target)

                cv2.drawMarker(vis, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

                if fire_state == FIRE_SHOOTING:
                    cv2.putText(vis, f"BURST {bullets_fired}/{bullets_per_burst}",
                               (cx - 80, cy + 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                info = [
                    f"{status}",
                    f"Raw:{len(raw)} Conf:{len(confirmed)} | {fire_state}",
                    f"Inf:{detector.inference_ms:.0f}ms FPS:{display_fps}",
                ]
                for i, line in enumerate(info):
                    cv2.putText(vis, line, (10, 28 + i * 26),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                h, w = vis.shape[:2]
                if w > 1920:
                    s = 1920 / w
                    vis = cv2.resize(vis, (1920, int(h * s)))
                cv2.imshow("Bot Debug", vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Log every second
            if frame_count % 50 == 0:
                log.log(f"TICK: f={frame_count} {status} fire={fire_state} "
                       f"raw={len(raw)} conf={len(confirmed)} "
                       f"bezier={'active' if aim_path.is_active else 'idle'} "
                       f"inf={detector.inference_ms:.0f}ms fps={display_fps}")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        mouse.release_all_buttons()
        capture.stop()
        cv2.destroyAllWindows()
        log.close()
        print(f"Done. Log: {LOG_PATH}")


if __name__ == "__main__":
    main()

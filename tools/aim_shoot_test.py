"""Aim + shoot test v5 — non-blocking aim, simple fire control.

The main loop never blocks. Each frame:
1. Detect enemies
2. Calculate how much to move mouse
3. Apply a fraction of that movement (smooth, non-blocking)
4. If on target: fire burst (8 bullets), pull down, stop, re-aim

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


# ── Fire states ──
FIRE_IDLE = "idle"           # Not firing, looking for target
FIRE_AIMING = "aiming"       # Moving crosshair toward target
FIRE_SHOOTING = "shooting"   # Mouse held down, spraying
FIRE_COOLDOWN = "cooldown"   # Brief pause between bursts


def main():
    print("=" * 55)
    print("  CS2 Aim + Shoot Test v5 (non-blocking)")
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

    capture = ScreenCapture(target_fps=60)
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

    # Aim settings
    aim_smoothing = 0.35          # Apply 35% of needed movement per frame (smooth tracking)
    aim_threshold = 50            # Start firing when within 50px
    aim_micro_threshold = 15      # Close enough, just fire

    # Fire settings
    bullets_per_burst = 8         # Bullets per burst
    bullet_interval = 0.1         # ~100ms per bullet (600 RPM)
    burst_cooldown_time = 0.15    # Pause between bursts
    recoil_per_bullet = 3         # Pull down per bullet (pixels)

    log.log(f"Settings: sens={sensitivity} fov={fov_h} smooth={aim_smoothing} "
           f"burst={bullets_per_burst} bullets")

    print(f"\nRunning! Switch to CS2. Press Q to stop.\n")

    cv2.namedWindow("Bot Debug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Bot Debug", 960, 540)

    # State
    fire_state = FIRE_IDLE
    fire_start_time = 0
    bullets_fired = 0
    cooldown_until = 0
    last_target_pos = None        # Last known target position
    last_target_time = 0          # When we last saw a target
    target_lost_grace = 0.5       # Keep state for 500ms after losing target
    frame_count = 0
    reaction_until = 0

    try:
        while True:
            frame = capture.grab()
            if frame is None:
                time.sleep(0.001)
                continue

            now = time.perf_counter()
            frame_count += 1

            # ── DETECT ──
            raw = detector.detect(frame)
            confirmed = conf_filter.update(raw)

            # Pick closest player to crosshair
            players = [d for d in confirmed if d.class_name == "ct_player"]
            target = None
            if players:
                # If we have a last known position, prefer targets near it (target lock)
                if last_target_pos is not None:
                    near = [p for p in players if distance(p.center, last_target_pos) < 300]
                    if near:
                        near.sort(key=lambda d: distance(d.center, last_target_pos))
                        target = near[0]

                if target is None:
                    players.sort(key=lambda d: distance(d.center, (cx, cy)))
                    target = players[0]

            # Update target tracking
            if target:
                tcx, tcy = target.center
                is_new = (last_target_pos is None or
                         distance(last_target_pos, (tcx, tcy)) > 400)
                last_target_pos = (tcx, tcy)
                last_target_time = now

                if is_new and fire_state != FIRE_SHOOTING:
                    reaction_ms = random.uniform(100, 220)
                    reaction_until = now + reaction_ms / 1000.0
                    log.log(f"NEW TARGET ({tcx:.0f},{tcy:.0f}) reaction={reaction_ms:.0f}ms")

            has_target = target is not None
            target_recently_seen = (now - last_target_time) < target_lost_grace

            # ── AIM (non-blocking) ──
            status = "IDLE"

            if has_target and now >= reaction_until:
                aim_x, aim_y = bbox_to_aim_point(
                    target.x1, target.y1, target.x2, target.y2,
                    head_aim=False,
                )
                dx_pixels = aim_x - cx
                dy_pixels = aim_y - cy
                screen_dist = distance((cx, cy), (aim_x, aim_y))

                # Convert to mouse counts
                mouse_dx, mouse_dy = screen_delta_to_mouse(
                    dx_pixels, dy_pixels, sensitivity, m_yaw, m_pitch,
                    screen_w, screen_h, fov_h
                )

                if fire_state == FIRE_SHOOTING:
                    # While firing: gentle tracking (25% correction)
                    track_dx = int(mouse_dx * 0.25)
                    track_dy = int(mouse_dy * 0.25)
                    if abs(track_dx) > 1 or abs(track_dy) > 1:
                        mouse.move_relative(track_dx, track_dy)
                    status = f"FIRING+TRACK"
                else:
                    # Not firing: smooth aim toward target
                    # Apply a fraction of the needed movement (non-blocking!)
                    if screen_dist > aim_micro_threshold:
                        smooth = aim_smoothing
                        # Move faster when far away
                        if screen_dist > 300:
                            smooth = 0.6
                        elif screen_dist > 150:
                            smooth = 0.45

                        move_dx = int(mouse_dx * smooth)
                        move_dy = int(mouse_dy * smooth)

                        # Ensure we always move at least 1 pixel if there's a delta
                        if move_dx == 0 and abs(mouse_dx) > 0:
                            move_dx = 1 if mouse_dx > 0 else -1
                        if move_dy == 0 and abs(mouse_dy) > 0:
                            move_dy = 1 if mouse_dy > 0 else -1

                        mouse.move_relative(move_dx, move_dy)
                        status = f"AIMING ({screen_dist:.0f}px)"
                    else:
                        status = "ON TARGET"

                # ── FIRE STATE MACHINE ──
                if fire_state == FIRE_IDLE or fire_state == FIRE_AIMING:
                    if screen_dist <= aim_threshold:
                        fire_state = FIRE_SHOOTING
                        mouse.mouse_down("left")
                        fire_start_time = now
                        bullets_fired = 0
                        log.log(f"BURST START: dist={screen_dist:.0f}px")

                elif fire_state == FIRE_SHOOTING:
                    spray_time = now - fire_start_time
                    bullets_fired = int(spray_time / bullet_interval)

                    # Pull down for recoil each frame
                    if bullets_fired > 0:
                        mouse.move_relative(0, recoil_per_bullet)

                    # End burst after enough bullets
                    if bullets_fired >= bullets_per_burst:
                        mouse.mouse_up("left")
                        fire_state = FIRE_COOLDOWN
                        cooldown_until = now + burst_cooldown_time
                        log.log(f"BURST END: {bullets_fired} bullets in {spray_time:.2f}s")
                        bullets_fired = 0

                    status = f"FIRING ({bullets_fired}/{bullets_per_burst})"

                elif fire_state == FIRE_COOLDOWN:
                    if now >= cooldown_until:
                        fire_state = FIRE_AIMING  # Ready to fire again
                        status = "RE-AIMING"
                    else:
                        status = "COOLDOWN"

            elif has_target and now < reaction_until:
                status = "REACTING"

            elif not has_target:
                if fire_state == FIRE_SHOOTING and target_recently_seen:
                    # Keep firing briefly — target might reappear
                    spray_time = now - fire_start_time
                    bullets_fired = int(spray_time / bullet_interval)
                    if bullets_fired > 0:
                        mouse.move_relative(0, recoil_per_bullet)
                    if bullets_fired >= bullets_per_burst:
                        mouse.mouse_up("left")
                        fire_state = FIRE_IDLE
                        log.log(f"BURST END (persist): {bullets_fired} bullets")
                    status = f"FIRING (persist {bullets_fired}/{bullets_per_burst})"
                elif fire_state == FIRE_SHOOTING:
                    # Target gone too long, stop
                    mouse.mouse_up("left")
                    fire_state = FIRE_IDLE
                    log.log("FIRE STOP: target lost")
                    status = "TARGET LOST"
                else:
                    if fire_state != FIRE_IDLE:
                        fire_state = FIRE_IDLE
                        mouse.ensure_released("left")
                    if not target_recently_seen:
                        last_target_pos = None
                    status = "NO TARGET"

            # Safety: never hold fire for more than 3s
            if fire_state == FIRE_SHOOTING and (now - fire_start_time) > 3.0:
                mouse.mouse_up("left")
                fire_state = FIRE_IDLE
                log.log("SAFETY: force release after 3s")

            # ── DEBUG OVERLAY ──
            if frame_count % 2 == 0:
                vis = frame.copy()

                for det in raw:
                    x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (128, 128, 128), 1)

                for det in confirmed:
                    x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
                    is_target = (det == target)
                    color = (0, 0, 255) if is_target else (0, 255, 0)
                    thickness = 3 if is_target else 2
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

                cv2.drawMarker(vis, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

                # Fire state indicator
                if fire_state == FIRE_SHOOTING:
                    cv2.putText(vis, f"BURST {bullets_fired}/{bullets_per_burst}",
                               (cx - 80, cy + 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                info = [
                    f"{status}",
                    f"Raw:{len(raw)} Conf:{len(confirmed)} | {fire_state}",
                    f"Inf:{detector.inference_ms:.0f}ms FPS:{capture.fps:.0f}",
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
            if frame_count % 30 == 0:
                log.log(f"TICK: f={frame_count} {status} fire={fire_state} "
                       f"raw={len(raw)} conf={len(confirmed)} "
                       f"inf={detector.inference_ms:.0f}ms fps={capture.fps:.0f}")

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

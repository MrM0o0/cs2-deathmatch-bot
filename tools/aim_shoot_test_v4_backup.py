"""Aim + shoot test v4 — fire persistence, better spray control.

BACKUP — this was the last working iteration before v5 rewrite.

Usage:
    python tools/aim_shoot_test_v4_backup.py

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
from src.aim.mouse_mover import MouseMover
from src.humanizer.mistakes import MistakeMaker
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


def select_target(detections, cx, cy, locked_target_center, lock_radius=250):
    players = [d for d in detections if d.class_name == "ct_player"]
    if not players:
        return None
    if locked_target_center is not None:
        for p in players:
            if distance(p.center, locked_target_center) < lock_radius:
                return p
    players.sort(key=lambda d: distance(d.center, (cx, cy)))
    return players[0]


def main():
    print("=" * 55)
    print("  CS2 Aim + Shoot Test v4 (BACKUP)")
    print("  Fire persistence | Better spray | Target lock")
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

    mover = MouseMover(base_speed=8.0, noise_amplitude=1.5)
    mistakes = MistakeMaker(overshoot_chance=0.25, overshoot_magnitude=1.2, tracking_error=5.0)

    capture = ScreenCapture(target_fps=30)
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
    log.log(f"Resolution: {screen_w}x{screen_h} | Crosshair: ({cx}, {cy})")

    sensitivity = 0.85
    m_yaw = 0.022
    m_pitch = 0.022
    fov_h = 122.0

    log.log(f"Settings: sens={sensitivity} fov={fov_h}")

    print("\nRunning! Switch to CS2. Press Q to stop.\n")

    cv2.namedWindow("Aim+Shoot Debug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Aim+Shoot Debug", 960, 540)

    # State
    locked_target_center = None
    lock_time = 0
    max_lock_time = 2.5
    reaction_until = 0
    is_firing = False
    fire_start_time = 0
    last_target_seen_time = 0
    fire_persist_time = 0.4
    burst_count = 0
    max_burst = random.randint(4, 8)
    burst_cooldown_until = 0
    last_safety_check = 0
    frame_count = 0

    try:
        while True:
            frame = capture.grab()
            if frame is None:
                time.sleep(0.001)
                continue

            now = time.perf_counter()
            frame_count += 1

            if is_firing and (now - fire_start_time) > 2.0:
                mouse.mouse_up("left")
                is_firing = False
                burst_count = 0
                log.log("SAFETY: Force released mouse after 2s")

            if now - last_safety_check > 2.0:
                if not is_firing:
                    mouse.ensure_released("left")
                last_safety_check = now

            raw = detector.detect(frame)
            confirmed = conf_filter.update(raw)

            target = select_target(confirmed, cx, cy, locked_target_center)

            status = "IDLE"

            if target:
                last_target_seen_time = now
                tcx, tcy = target.center

                if locked_target_center is None or distance(locked_target_center, (tcx, tcy)) > 250:
                    locked_target_center = (tcx, tcy)
                    lock_time = now
                    reaction_ms = random.uniform(120, 250)
                    reaction_until = now + reaction_ms / 1000.0
                    if is_firing:
                        mouse.mouse_up("left")
                        is_firing = False
                        burst_count = 0
                    log.log(f"LOCKED TARGET at ({tcx:.0f},{tcy:.0f}) reaction={reaction_ms:.0f}ms")
                else:
                    locked_target_center = (tcx, tcy)

                if now - lock_time > max_lock_time:
                    locked_target_center = None
                    lock_time = now
                    status = "LOCK TIMEOUT"
                elif now < reaction_until:
                    status = "REACTING"
                elif now < burst_cooldown_until:
                    status = "BURST COOLDOWN"
                else:
                    aim_x, aim_y = bbox_to_aim_point(
                        target.x1, target.y1, target.x2, target.y2,
                        head_aim=False,
                    )
                    dx_pixels = aim_x - cx
                    dy_pixels = aim_y - cy
                    screen_dist = distance((cx, cy), (aim_x, aim_y))

                    if screen_dist > 50 and not is_firing:
                        mouse_dx, mouse_dy = screen_delta_to_mouse(
                            dx_pixels, dy_pixels, sensitivity, m_yaw, m_pitch,
                            screen_w, screen_h, fov_h
                        )
                        err_dx, err_dy = mistakes.apply_aim_error(mouse_dx, mouse_dy)

                        if abs(err_dx) < 80 and abs(err_dy) < 80:
                            mover.micro_correct(err_dx, err_dy, delay_ms=25)
                        else:
                            mover.move_to_delta(err_dx, err_dy)

                        log.log(f"AIM: dist={screen_dist:.0f}px mouse=({mouse_dx},{mouse_dy})")
                        status = f"AIMING ({screen_dist:.0f}px)"
                    else:
                        if not is_firing:
                            mouse_dx, mouse_dy = screen_delta_to_mouse(
                                dx_pixels, dy_pixels, sensitivity, m_yaw, m_pitch,
                                screen_w, screen_h, fov_h
                            )
                            if abs(mouse_dx) > 2 or abs(mouse_dy) > 2:
                                mover.micro_correct(mouse_dx, mouse_dy, delay_ms=15)

                            mouse.mouse_down("left")
                            is_firing = True
                            fire_start_time = now
                            burst_count = 0
                            max_burst = random.randint(4, 8)
                            log.log(f"FIRE START: dist={screen_dist:.0f}px burst_max={max_burst}")

                        spray_time = now - fire_start_time

                        recoil_pull = int(spray_time * 5)
                        if recoil_pull > 0:
                            mouse.move_relative(0, recoil_pull)

                        mouse_dx, mouse_dy = screen_delta_to_mouse(
                            dx_pixels, dy_pixels, sensitivity, m_yaw, m_pitch,
                            screen_w, screen_h, fov_h
                        )
                        if abs(mouse_dx) > 3 or abs(mouse_dy) > 3:
                            mouse.move_relative(int(mouse_dx * 0.25), int(mouse_dy * 0.25))

                        burst_count = int(spray_time * 10)

                        if burst_count >= max_burst:
                            mouse.mouse_up("left")
                            is_firing = False
                            cooldown = random.uniform(0.05, 0.12)
                            burst_cooldown_until = now + cooldown
                            log.log(f"BURST END: {burst_count} bullets, cd={cooldown:.2f}s")
                            burst_count = 0
                            status = "BURST END"
                        else:
                            status = f"FIRING ({burst_count}/{max_burst})"
            else:
                time_since_target = now - last_target_seen_time

                if is_firing and time_since_target < fire_persist_time:
                    spray_time = now - fire_start_time
                    recoil_pull = int(spray_time * 5)
                    if recoil_pull > 0:
                        mouse.move_relative(0, recoil_pull)
                    burst_count = int(spray_time * 10)
                    if burst_count >= max_burst:
                        mouse.mouse_up("left")
                        is_firing = False
                        burst_count = 0
                        log.log(f"BURST END (persist): {burst_count} bullets")
                    status = f"FIRING (persist {burst_count}/{max_burst})"
                elif is_firing:
                    mouse.mouse_up("left")
                    is_firing = False
                    burst_count = 0
                    log.log("Released fire: target gone")
                    status = "TARGET LOST"
                else:
                    if not time_since_target < fire_persist_time:
                        locked_target_center = None
                    status = "NO TARGET"

            if frame_count % 2 == 0:
                vis = frame.copy()
                for det in raw:
                    x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (128, 128, 128), 1)
                for det in confirmed:
                    x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
                    is_locked = (det == target)
                    color = (0, 0, 255) if is_locked else (0, 255, 0)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    if is_locked:
                        cv2.putText(vis, "LOCKED", (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.drawMarker(vis, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

                if is_firing:
                    cv2.putText(vis, f"FIRING {burst_count}/{max_burst}",
                               (cx - 80, cy + 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                info = [
                    f"Status: {status}",
                    f"Raw: {len(raw)} | Confirmed: {len(confirmed)}",
                    f"Inf: {detector.inference_ms:.0f}ms | FPS: {capture.fps:.0f}",
                    f"Fire: {'DOWN' if is_firing else 'up'} | Lock: {'YES' if locked_target_center else 'no'}",
                ]
                for i, line in enumerate(info):
                    cv2.putText(vis, line, (10, 30 + i * 28),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

                h, w = vis.shape[:2]
                if w > 1920:
                    s = 1920 / w
                    vis = cv2.resize(vis, (1920, int(h * s)))
                cv2.imshow("Aim+Shoot Debug", vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if frame_count % 30 == 0:
                log.log(f"TICK: frame={frame_count} status={status} "
                       f"raw={len(raw)} confirmed={len(confirmed)} "
                       f"firing={is_firing} burst={burst_count}/{max_burst} "
                       f"inf={detector.inference_ms:.1f}ms fps={capture.fps:.0f}")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if is_firing:
            mouse.mouse_up("left")
        mouse.release_all_buttons()
        capture.stop()
        cv2.destroyAllWindows()
        log.close()
        print(f"Done. Log: {LOG_PATH}")


if __name__ == "__main__":
    main()

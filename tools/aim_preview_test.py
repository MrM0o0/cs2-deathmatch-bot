"""Aim preview test — shows what the bot WOULD do without moving your mouse.

Visualizes:
- Confirmed detections (green boxes)
- Crosshair position (white cross)
- Aim target point on the selected enemy (yellow circle)
- Bezier curve path the mouse WOULD follow (cyan curve)
- Overshoot point if applicable (orange circle)
- Aim state info (reaction time, fire mode, etc.)

Usage:
    python tools/aim_preview_test.py

Press Q to quit the preview window.
"""

import sys
import os
import time
import random
import math

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from src.capture.screen import ScreenCapture
from src.vision.detector import YOLODetector, Detection
from src.vision.confirmation_filter import ConfirmationFilter
from src.aim.targeting import TargetingSystem
from src.humanizer.mistakes import MistakeMaker
from src.utils.math_helpers import cubic_bezier, bbox_to_aim_point, distance

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "cs2_yolov8n.onnx")

# Screen center (crosshair) — auto-detected below in main()
CX, CY = 960, 540

# Colors (BGR)
COLOR_CONFIRMED = (0, 255, 0)       # Green - confirmed detection
COLOR_RAW = (128, 128, 128)         # Gray - unconfirmed
COLOR_CROSSHAIR = (255, 255, 255)   # White - crosshair
COLOR_AIM_TARGET = (0, 255, 255)    # Yellow - where we'd aim
COLOR_BEZIER = (255, 255, 0)        # Cyan - bezier path
COLOR_OVERSHOOT = (0, 128, 255)     # Orange - overshoot point
COLOR_HEAD = (0, 0, 255)            # Red - head aim point
COLOR_INFO_BG = (0, 0, 0)          # Black background for text


def generate_bezier_preview(start_x, start_y, end_x, end_y, num_points=30):
    """Generate a Bezier curve path for visualization."""
    dx = end_x - start_x
    dy = end_y - start_y
    dist = math.sqrt(dx * dx + dy * dy)

    p0 = (float(start_x), float(start_y))
    p3 = (float(end_x), float(end_y))

    ctrl_spread = dist * 0.3
    p1 = (
        start_x + dx * random.uniform(0.2, 0.4) + random.gauss(0, ctrl_spread * 0.3),
        start_y + dy * random.uniform(0.2, 0.4) + random.gauss(0, ctrl_spread * 0.3),
    )
    p2 = (
        start_x + dx * random.uniform(0.6, 0.8) + random.gauss(0, ctrl_spread * 0.2),
        start_y + dy * random.uniform(0.6, 0.8) + random.gauss(0, ctrl_spread * 0.2),
    )

    points = []
    for i in range(num_points + 1):
        t = i / num_points
        bx, by = cubic_bezier(t, p0, p1, p2, p3)
        points.append((int(bx), int(by)))

    return points


def draw_aim_preview(frame, raw_detections, confirmed_detections, target,
                     aim_point, bezier_points, overshoot_point,
                     inference_ms, cap_fps, aim_info):
    """Draw the full aim preview visualization."""
    vis = frame.copy()

    # Draw raw detections (thin gray)
    for det in raw_detections:
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_RAW, 1)

    # Draw confirmed detections (thick green)
    for det in confirmed_detections:
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_CONFIRMED, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw, y1), COLOR_CONFIRMED, -1)
        cv2.putText(vis, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Draw crosshair
    cv2.drawMarker(vis, (CX, CY), COLOR_CROSSHAIR, cv2.MARKER_CROSS, 20, 2)

    if target and aim_point:
        aim_x, aim_y = int(aim_point[0]), int(aim_point[1])

        # Highlight selected target box
        x1, y1, x2, y2 = int(target.x1), int(target.y1), int(target.x2), int(target.y2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_AIM_TARGET, 3)

        # Draw aim target point
        cv2.circle(vis, (aim_x, aim_y), 8, COLOR_AIM_TARGET, 2)
        cv2.circle(vis, (aim_x, aim_y), 3, COLOR_AIM_TARGET, -1)

        # Draw Bezier curve path
        if bezier_points and len(bezier_points) > 1:
            for i in range(len(bezier_points) - 1):
                thickness = max(1, 3 - i // 10)
                cv2.line(vis, bezier_points[i], bezier_points[i + 1], COLOR_BEZIER, thickness)

        # Draw overshoot point if applicable
        if overshoot_point:
            ox, oy = int(overshoot_point[0]), int(overshoot_point[1])
            cv2.circle(vis, (ox, oy), 10, COLOR_OVERSHOOT, 2)
            cv2.putText(vis, "OVERSHOOT", (ox + 12, oy - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_OVERSHOOT, 1)


    # Info panel (top-left)
    lines = [
        f"Inference: {inference_ms:.1f}ms | FPS: {cap_fps:.1f}",
        f"Raw: {len(raw_detections)} | Confirmed: {len(confirmed_detections)}",
    ]
    lines.extend(aim_info)

    for i, line in enumerate(lines):
        y = 25 + i * 28
        # Background
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis, (8, y - th - 4), (14 + tw, y + 6), COLOR_INFO_BG, -1)
        cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    # Legend (bottom-left)
    legend = [
        ("Gray box = raw detection", COLOR_RAW),
        ("Green box = confirmed target", COLOR_CONFIRMED),
        ("Yellow box = selected target", COLOR_AIM_TARGET),
        ("Cyan line = aim path (Bezier)", COLOR_BEZIER),
        ("Orange circle = overshoot", COLOR_OVERSHOOT),
        ("NO MOUSE MOVEMENT - preview only", (0, 0, 255)),
    ]
    h = vis.shape[0]
    for i, (text, color) in enumerate(legend):
        y = h - 20 - (len(legend) - 1 - i) * 22
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return vis


def main():
    global CX, CY

    print("=" * 55)
    print("  CS2 Aim Preview Test (NO mouse movement)")
    print("  This only SHOWS what the bot would do")
    print("  Your mouse will NOT be moved")
    print("  Press Q in the preview window to quit")
    print("=" * 55)

    # Init detector
    print("\nLoading YOLO model...")
    detector = YOLODetector(
        model_path=MODEL_PATH,
        confidence_threshold=0.50,
        nms_threshold=0.5,
    )
    detector.load()

    # Init confirmation filter
    conf_filter = ConfirmationFilter(
        min_confirm_frames=3,
        max_missing_frames=2,
        match_distance=150.0,
        match_size_ratio=0.4,
    )

    # Init targeting (for aim calculations only, no mouse movement)
    targeting = TargetingSystem(
        screen_center_x=CX, screen_center_y=CY,
        sensitivity=2.0, m_yaw=0.022, m_pitch=0.022,
        head_aim_chance=0.3,
    )

    # Init mistake maker (for overshoot preview)
    mistake_maker = MistakeMaker(
        overshoot_chance=0.35,
        overshoot_magnitude=1.3,
        tracking_error=8.0,
    )

    # Init screen capture
    print("Starting screen capture...")
    capture = ScreenCapture(target_fps=30)
    backend = capture.start()
    print(f"Capture backend: {backend}")

    # Auto-detect resolution for crosshair center
    test_frame = None
    for _ in range(30):
        test_frame = capture.grab()
        if test_frame is not None:
            break
        time.sleep(0.05)
    if test_frame is not None:
        h, w = test_frame.shape[:2]
        CX, CY = w // 2, h // 2
        print(f"Resolution: {w}x{h} | Crosshair: ({CX}, {CY})")

    time.sleep(0.5)

    print("\nRunning! Switch to CS2 and check the preview window.")
    print("Your mouse will NOT be moved — this is preview only.")
    print("Press Q to quit.\n")

    cv2.namedWindow("CS2 Aim Preview (NO MOUSE)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CS2 Aim Preview (NO MOUSE)", 960, 540)

    # Track last bezier generation to avoid regenerating every frame
    last_target_center = None
    bezier_points = []
    overshoot_point = None
    will_overshoot = False

    try:
        while True:
            frame = capture.grab()
            if frame is None:
                time.sleep(0.01)
                continue

            # Detection pipeline
            raw_detections = detector.detect(frame)
            confirmed_detections = conf_filter.update(raw_detections)

            # Target selection
            target = targeting.select_target(confirmed_detections)

            aim_point = None
            aim_info = []

            if target:
                # Calculate aim point
                head_aim = random.random() < 0.3 or target.is_head
                aim_x, aim_y = bbox_to_aim_point(
                    target.x1, target.y1, target.x2, target.y2,
                    head_aim=head_aim,
                )
                aim_point = (aim_x, aim_y)

                # Screen distance
                screen_dist = distance((CX, CY), (aim_x, aim_y))

                # Check if target changed significantly (regenerate bezier)
                tcx, tcy = target.center
                target_moved = (last_target_center is None or
                               distance(last_target_center, (tcx, tcy)) > 30)

                if target_moved:
                    last_target_center = (tcx, tcy)

                    # Generate bezier preview
                    bezier_points = generate_bezier_preview(CX, CY, aim_x, aim_y)

                    # Check overshoot
                    will_overshoot = mistake_maker.should_overshoot()
                    if will_overshoot:
                        ox, oy = mistake_maker.overshoot_target(CX, CY, aim_x, aim_y)
                        overshoot_point = (ox, oy)
                        bezier_points = generate_bezier_preview(CX, CY, ox, oy)
                    else:
                        overshoot_point = None

                # Aim info
                on_target = targeting.is_on_target(target)
                aim_type = "HEAD" if head_aim else "BODY"
                aim_info = [
                    f"Target: {target.class_name} ({target.confidence:.2f})",
                    f"Distance: {screen_dist:.0f}px | Aim: {aim_type}",
                    f"On target: {'YES' if on_target else 'NO'}",
                    f"Overshoot: {'YES' if will_overshoot else 'NO'}",
                ]
                if on_target:
                    aim_info.append("Action: FIRE")
                else:
                    aim_info.append("Action: FLICK -> FIRE")
            else:
                aim_info = ["No target — ROAMING"]
                bezier_points = []
                overshoot_point = None
                last_target_center = None

            # Draw everything
            vis = draw_aim_preview(
                frame, raw_detections, confirmed_detections,
                target, aim_point, bezier_points,
                overshoot_point if will_overshoot else None,
                detector.inference_ms, capture.fps, aim_info,
            )

            # Scale down
            h, w = vis.shape[:2]
            if w > 1920:
                scale = 1920 / w
                vis = cv2.resize(vis, (1920, int(h * scale)))

            cv2.imshow("CS2 Aim Preview (NO MOUSE)", vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        capture.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()

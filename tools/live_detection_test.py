"""Live detection preview — captures screen + runs YOLO + draws bounding boxes.

Shows both raw detections (thin gray boxes) and confirmed detections (thick
colored boxes) so you can see the confirmation filter in action.

Usage:
    python tools/live_detection_test.py

Press Q to quit the preview window.
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from src.capture.screen import ScreenCapture
from src.vision.detector import YOLODetector
from src.vision.confirmation_filter import ConfirmationFilter

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "cs2_yolov8n.onnx")

# Colors (BGR)
COLORS_CONFIRMED = {
    "ct_player": (0, 255, 0),   # Green
    "head": (0, 0, 255),         # Red
}
COLOR_RAW = (128, 128, 128)      # Gray for unconfirmed
DEFAULT_COLOR = (255, 255, 0)


def draw_detections(frame: np.ndarray, raw_detections, confirmed_detections,
                    inference_ms: float, cap_fps: float,
                    filter_info: str) -> np.ndarray:
    """Draw bounding boxes and labels on a frame."""
    vis = frame.copy()

    # Draw raw detections as thin gray boxes
    for det in raw_detections:
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_RAW, 1)

    # Draw confirmed detections as thick colored boxes
    for det in confirmed_detections:
        color = COLORS_CONFIRMED.get(det.class_name, DEFAULT_COLOR)
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)

        # Draw box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{det.class_name} {det.confidence:.2f} [OK]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(vis, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Stats overlay
    line1 = f"Inference: {inference_ms:.1f}ms | FPS: {cap_fps:.1f}"
    line2 = f"Raw: {len(raw_detections)} | Confirmed: {len(confirmed_detections)} | {filter_info}"
    cv2.putText(vis, line1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(vis, line2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return vis


def main():
    print("=" * 50)
    print("  CS2 Live Detection Test (with confirmation filter)")
    print("  Gray boxes = raw detections (unconfirmed)")
    print("  Colored boxes = confirmed targets (3+ frames)")
    print("  Press Q in the preview window to quit")
    print("=" * 50)

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

    # Init screen capture
    print("Starting screen capture...")
    capture = ScreenCapture(target_fps=30)
    backend = capture.start()
    print(f"Capture backend: {backend}")

    # Give capture a moment to initialize
    time.sleep(0.5)

    print("\nRunning! Switch to CS2 and check the preview window.")
    print("Press Q to quit.\n")

    cv2.namedWindow("CS2 Detection Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CS2 Detection Preview", 960, 540)

    try:
        while True:
            frame = capture.grab()
            if frame is None:
                time.sleep(0.01)
                continue

            # Run detection
            raw_detections = detector.detect(frame)

            # Run confirmation filter
            confirmed_detections = conf_filter.update(raw_detections)

            # Filter info
            filter_info = f"Tracking: {conf_filter.tracker_count}"

            # Draw results
            vis = draw_detections(
                frame, raw_detections, confirmed_detections,
                detector.inference_ms, capture.fps, filter_info,
            )

            # Scale down for preview (full res can be huge)
            h, w = vis.shape[:2]
            if w > 1920:
                scale = 1920 / w
                vis = cv2.resize(vis, (1920, int(h * scale)))

            cv2.imshow("CS2 Detection Preview", vis)

            # Q to quit
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

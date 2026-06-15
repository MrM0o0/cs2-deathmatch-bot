"""Rung 1 of the assistant ladder: threat highlighter.

Pure assist -- no input control. Captures the screen, runs the YOLO detector,
and shows a window with a red box on every detected enemy plus a flashing red
border when any threat is on screen. This is the foundation the later rungs
(aim assist, triggerbot) build on, and it reuses the detector you already
trained -- no new model.

Run it on a second monitor, or with CS2 in windowed/borderless mode so you can
see both. Press END to quit.

    python tools/threat_highlight.py
    python tools/threat_highlight.py --scale 0.6   # window size

NOTE: this only *shows* you things. It does not touch your mouse or keyboard.
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
from src.vision.confirmation_filter import ConfirmationFilter
from src.vision.detector import YOLODetector

END_VK = 0x23


def main():
    ap = argparse.ArgumentParser(description="Rung 1: threat highlighter (view-only)")
    ap.add_argument("--scale", type=float, default=0.5, help="Display window scale")
    ap.add_argument("--monitor", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(PROJECT_ROOT, "config", "settings.yaml")))
    det_cfg = cfg["detection"]
    monitor = args.monitor if args.monitor is not None else cfg["display"]["monitor"]

    cap = ScreenCapture(monitor=monitor, target_fps=cfg["display"]["capture_fps"])
    print("[highlight] capture:", cap.start())
    det = YOLODetector(
        model_path=os.path.join(PROJECT_ROOT, det_cfg["model_path"]),
        input_size=det_cfg["input_size"],
        confidence_threshold=det_cfg["confidence_threshold"],
        nms_threshold=det_cfg["nms_threshold"],
        classes=det_cfg["classes"],
    )
    det.load()
    conf = ConfirmationFilter()  # smooth out single-frame flickers
    user32 = ctypes.windll.user32

    win = "Threat Highlight (END to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("[highlight] running -- END to quit")

    fps_t, frames, fps = time.perf_counter(), 0, 0.0
    flash = 0
    while True:
        if user32.GetAsyncKeyState(END_VK) & 0x8000:
            break
        frame = cap.grab()
        if frame is None:
            time.sleep(0.003)
            continue

        dets = conf.update(det.detect(frame))
        vis = frame

        for d in dets:
            head = getattr(d, "is_head", False)
            colour = (0, 200, 255) if head else (0, 0, 255)  # head=orange, body=red
            x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
            cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 3)
            cv2.putText(
                vis,
                f"{d.class_name} {d.confidence:.2f}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                colour,
                2,
                cv2.LINE_AA,
            )

        # Flash a red border whenever a threat is present.
        if dets:
            flash = 6
        if flash > 0:
            flash -= 1
            h, w = vis.shape[:2]
            cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (0, 0, 255), 14)
            cv2.putText(
                vis,
                f"THREAT x{len(dets)}",
                (24, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

        frames += 1
        now = time.perf_counter()
        if now - fps_t >= 0.5:
            fps = frames / (now - fps_t)
            fps_t, frames = now, 0
        cv2.putText(
            vis,
            f"{fps:4.1f} fps | infer {det.inference_ms:.0f}ms | {len(dets)} threats",
            (24, vis.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        small = cv2.resize(vis, (int(vis.shape[1] * args.scale), int(vis.shape[0] * args.scale)))
        cv2.imshow(win, small)
        cv2.waitKey(1)

    cap.stop()
    cv2.destroyAllWindows()
    print("[highlight] stopped.")


if __name__ == "__main__":
    main()

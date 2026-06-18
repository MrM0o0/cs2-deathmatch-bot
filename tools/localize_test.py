"""Phase 2: validate landmark localization (VIEW-ONLY, no input control).

Runs your trained LANDMARK model (not the player model), feeds detections to the
Localizer, and prints the area it thinks you're in -- live, as you walk the map.
This proves the keystone (YOLO localization) before any movement is built on it.

    python tools/localize_test.py --model models/landmarks_dust2.onnx \
        --classes mid_doors goose b_car b_doors xbox catwalk t_spawn ct_spawn

Walk dust2 and watch the printed area. If it tracks where you actually are,
localization works and we move to danger pre-aim. If it lags or names the wrong
area, that's a cheap, early signal to fix before building movement on top.

It only READS the screen. It does not touch your mouse or keyboard. Press END to
quit. Optional --window shows the detected landmarks with boxes.
"""

import argparse
import ctypes
import os
import sys
import time

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.capture.screen import ScreenCapture
from src.movement.localizer import Localizer
from src.vision.detector import YOLODetector

END_VK = 0x23


def main():
    ap = argparse.ArgumentParser(description="Phase 2: localization validator (view-only)")
    ap.add_argument("--model", default="models/landmarks_dust2.onnx", help="Landmark ONNX model")
    ap.add_argument("--classes", nargs="+", required=True, help="Landmark class names, in order")
    ap.add_argument("--map", default="dust2", help="Map name (loads config/maps/<map>_areas.json)")
    ap.add_argument("--conf", type=float, default=0.30, help="Detection confidence threshold")
    ap.add_argument("--confirm", type=int, default=3, help="Frames to confirm an area switch")
    ap.add_argument("--monitor", type=int, default=None)
    ap.add_argument("--window", action="store_true", help="Show detection overlay window")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(PROJECT_ROOT, "config", "settings.yaml")))
    monitor = args.monitor if args.monitor is not None else cfg["display"]["monitor"]

    model_path = os.path.join(PROJECT_ROOT, args.model)
    if not os.path.exists(model_path):
        print(f"[localize] model not found: {model_path}")
        print("[localize] train the landmark model first (Phase 1), then re-run.")
        return

    areas_file = os.path.join(PROJECT_ROOT, "config", "maps", f"{args.map}_areas.json")
    loc = Localizer.from_file(areas_file, confirm_frames=args.confirm)

    cap = ScreenCapture(monitor=monitor, target_fps=cfg["display"]["capture_fps"])
    print("[localize] capture:", cap.start())
    det = YOLODetector(
        model_path=model_path,
        input_size=cfg["detection"]["input_size"],
        confidence_threshold=args.conf,
        nms_threshold=cfg["detection"]["nms_threshold"],
        classes=args.classes,
    )
    det.load()

    cv2 = None
    if args.window:
        import cv2 as _cv2

        cv2 = _cv2
        cv2.namedWindow("Localize (END to quit)", cv2.WINDOW_NORMAL)

    user32 = ctypes.windll.user32
    print("[localize] running -- walk the map, watch the area. END to quit.")
    last_line = ""
    while True:
        if user32.GetAsyncKeyState(END_VK) & 0x8000:
            break
        frame = cap.grab()
        if frame is None:
            time.sleep(0.003)
            continue

        dets = det.detect(frame)
        frame_area = float(frame.shape[0] * frame.shape[1])
        area = loc.update(dets, frame_area=frame_area)

        seen = ",".join(sorted({d.class_name for d in dets})) or "-"
        line = (
            f"area={area or '???':<14} conf={loc.confidence:0.2f} "
            f"stale={loc.stale:<3} seeing=[{seen}]"
        )
        if line != last_line:
            print("[localize] " + line)
            last_line = line

        if cv2 is not None:
            vis = frame
            for d in dets:
                x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(
                    vis,
                    f"{d.class_name} {d.confidence:.2f}",
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 220, 0),
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                vis,
                f"AREA: {area or '???'}  ({loc.confidence:.2f})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 220, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Localize (END to quit)", vis)
            cv2.waitKey(1)

    cap.stop()
    if cv2 is not None:
        cv2.destroyAllWindows()
    print("[localize] stopped.")


if __name__ == "__main__":
    main()

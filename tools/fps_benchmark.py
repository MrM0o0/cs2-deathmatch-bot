"""Quick FPS benchmark — tests capture + detection speed without overlay."""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.capture.screen import ScreenCapture
from src.vision.detector import YOLODetector

detector = YOLODetector(model_path="models/cs2_yolov8n.onnx", confidence_threshold=0.50, nms_threshold=0.5)
detector.load()

capture = ScreenCapture(target_fps=120)
capture.start()
time.sleep(1)

count = 0
t0 = time.perf_counter()
while time.perf_counter() - t0 < 5.0:
    f = capture.grab()
    if f is not None:
        detector.detect(f)
        count += 1

elapsed = time.perf_counter() - t0
print(f"Detection only: {count/elapsed:.0f} FPS (inference avg: {detector.inference_ms:.1f}ms)")

capture.stop()

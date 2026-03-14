"""Debug visualization overlay for development and tuning."""

import time
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from src.vision.detector import Detection
from src.brain.state_machine import BotState


class DebugOverlay:
    """Draws debug information on frames for visualization."""

    COLORS = {
        "ct": (255, 150, 50),     # Blue-ish (BGR)
        "t": (50, 50, 255),       # Red
        "head_ct": (255, 200, 100),
        "head_t": (100, 100, 255),
        "crosshair": (0, 255, 0), # Green
        "text": (255, 255, 255),  # White
        "state": (0, 255, 255),   # Yellow
    }

    def __init__(self, window_name: str = "CS2 Bot Debug", scale: float = 0.5):
        self.window_name = window_name
        self.scale = scale
        self._fps_counter = 0
        self._fps_time = time.perf_counter()
        self._fps = 0.0

    def draw(self, frame: np.ndarray, detections: list[Detection],
             state: BotState, hud_info: str = "",
             inference_ms: float = 0, extra_lines: list[str] | None = None) -> np.ndarray:
        """Draw debug info on a frame copy.

        Returns the annotated frame (does not modify original).
        """
        if not _HAS_CV2:
            return frame

        vis = frame.copy()

        # Draw detections
        for det in detections:
            color = self.COLORS.get(det.class_name, (200, 200, 200))
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(vis, label, (x1, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw center dot
            cx, cy = int(det.center[0]), int(det.center[1])
            cv2.circle(vis, (cx, cy), 3, color, -1)

        # Draw crosshair
        h, w = vis.shape[:2]
        cx, cy = w // 2, h // 2
        cv2.drawMarker(vis, (cx, cy), self.COLORS["crosshair"],
                      cv2.MARKER_CROSS, 20, 1)

        # FPS counter
        self._fps_counter += 1
        now = time.perf_counter()
        if now - self._fps_time >= 1.0:
            self._fps = self._fps_counter / (now - self._fps_time)
            self._fps_counter = 0
            self._fps_time = now

        # Info panel
        y_offset = 25
        lines = [
            f"FPS: {self._fps:.1f}",
            f"State: {state.name}",
            f"Detections: {len(detections)}",
            f"Inference: {inference_ms:.1f}ms",
        ]
        if hud_info:
            lines.append(hud_info)
        if extra_lines:
            lines.extend(extra_lines)

        for line in lines:
            cv2.putText(vis, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       self.COLORS["text"], 1)
            y_offset += 22

        return vis

    def show(self, frame: np.ndarray) -> bool:
        """Display frame in a window. Returns False if window closed."""
        if not _HAS_CV2:
            return True

        if self.scale != 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * self.scale)
            new_h = int(h * self.scale)
            frame = cv2.resize(frame, (new_w, new_h))

        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return key != 27  # ESC to close

    def cleanup(self) -> None:
        """Destroy debug window."""
        if _HAS_CV2:
            cv2.destroyAllWindows()

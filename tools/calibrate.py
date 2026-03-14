"""Interactive HUD region calibration tool.

Captures a screenshot and lets you click to define HUD regions,
then saves them to settings.yaml.
"""

import os
import sys
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    import cv2
except ImportError:
    print("OpenCV required: pip install opencv-python")
    sys.exit(1)

from src.capture.screen import ScreenCapture


class RegionCalibrator:
    """Interactive region selection for HUD calibration."""

    def __init__(self):
        self.regions: dict[str, list[int]] = {}
        self._click_start = None
        self._current_name = ""
        self._frame = None
        self._display = None

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._click_start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self._click_start:
            x1, y1 = self._click_start
            w = abs(x - x1)
            h = abs(y - y1)
            rx = min(x, x1)
            ry = min(y, y1)

            # Scale back to original resolution
            scale = param if param else 1.0
            rx = int(rx / scale)
            ry = int(ry / scale)
            w = int(w / scale)
            h = int(h / scale)

            if w > 5 and h > 5:
                self.regions[self._current_name] = [rx, ry, w, h]
                print(f"  {self._current_name}: [{rx}, {ry}, {w}, {h}]")

            self._click_start = None

    def calibrate(self):
        """Run the calibration process."""
        print("[Calibrate] Capturing screenshot...")
        capture = ScreenCapture(target_fps=5)
        capture.start()

        import time
        time.sleep(0.5)
        frame = capture.grab()
        capture.stop()

        if frame is None:
            print("Failed to capture screenshot!")
            return

        self._frame = frame
        h, w = frame.shape[:2]
        scale = min(1.0, 1280.0 / w)

        print(f"[Calibrate] Resolution: {w}x{h}")
        print("[Calibrate] For each region, click and drag to select.")
        print("[Calibrate] Press SPACE to confirm and move to next region.")
        print("[Calibrate] Press ESC to skip a region.")
        print()

        region_names = [
            "health", "armor", "ammo_clip", "ammo_reserve",
            "killfeed", "alive_ct", "alive_t",
        ]

        window = "Calibration"
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, self._mouse_callback, scale)

        for name in region_names:
            self._current_name = name
            print(f"Select region: {name}")

            while True:
                display = frame.copy()
                if scale != 1.0:
                    display = cv2.resize(display,
                                        (int(w * scale), int(h * scale)))

                # Draw existing regions
                for rname, (rx, ry, rw, rh) in self.regions.items():
                    srx, sry = int(rx * scale), int(ry * scale)
                    srw, srh = int(rw * scale), int(rh * scale)
                    cv2.rectangle(display, (srx, sry),
                                (srx + srw, sry + srh), (0, 255, 0), 2)
                    cv2.putText(display, rname, (srx, sry - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.putText(display, f"Select: {name} (SPACE=next, ESC=skip)",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                           (0, 255, 255), 2)

                cv2.imshow(window, display)
                key = cv2.waitKey(30) & 0xFF

                if key == 32:  # SPACE
                    break
                elif key == 27:  # ESC
                    print(f"  Skipped {name}")
                    break

        cv2.destroyAllWindows()

        # Save to settings
        if self.regions:
            self._save_regions()

    def _save_regions(self):
        """Save calibrated regions to settings.yaml."""
        settings_path = os.path.join(PROJECT_ROOT, "config", "settings.yaml")

        with open(settings_path, "r") as f:
            settings = yaml.safe_load(f)

        settings["regions"] = {k: v for k, v in self.regions.items()}

        with open(settings_path, "w") as f:
            yaml.dump(settings, f, default_flow_style=False, sort_keys=False)

        print(f"\nSaved {len(self.regions)} regions to {settings_path}")


if __name__ == "__main__":
    cal = RegionCalibrator()
    cal.calibrate()

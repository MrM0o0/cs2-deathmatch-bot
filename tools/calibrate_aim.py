"""Auto-calibration — measures actual pixel shift vs expected to find scale.

Stand still in CS2 looking at a wall. This captures a frame, moves the mouse,
captures another frame, and uses image correlation to measure the actual
pixel shift. Then calculates the correct scale factor.

Usage:
    python tools/calibrate_aim.py
"""

import sys
import os
import time
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.timer_setup import enable_high_resolution_timer
enable_high_resolution_timer()

import cv2
import numpy as np
from src.capture.screen import ScreenCapture
from src.input.mouse import move_relative


def measure_pixel_shift(before, after):
    """Use phase correlation to measure horizontal pixel shift between frames."""
    # Convert to grayscale
    g1 = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Use center strip (avoid HUD elements at edges)
    h, w = g1.shape
    strip_y1 = h // 4
    strip_y2 = 3 * h // 4
    strip_x1 = w // 6
    strip_x2 = 5 * w // 6

    g1_strip = g1[strip_y1:strip_y2, strip_x1:strip_x2]
    g2_strip = g2[strip_y1:strip_y2, strip_x1:strip_x2]

    # Phase correlation
    shift, response = cv2.phaseCorrelate(g1_strip, g2_strip)
    return shift[0], shift[1], response  # dx, dy, confidence


def main():
    print("=" * 55)
    print("  CS2 Auto-Calibration")
    print("=" * 55)
    print()
    print("Stand still in CS2 looking at a wall.")
    print("DON'T move your mouse at all during the test!")
    print()
    input("Ready? Press ENTER, then switch to CS2...")
    print("Starting in 5 seconds... DON'T TOUCH THE MOUSE!")
    time.sleep(5)

    # Init capture
    capture = ScreenCapture(target_fps=30)
    capture.start()
    time.sleep(0.5)

    # Detect resolution
    frame = None
    for _ in range(30):
        frame = capture.grab()
        if frame is not None:
            break
        time.sleep(0.05)

    if frame is None:
        print("ERROR: Couldn't capture screen")
        return

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    print(f"Resolution: {w}x{h}")

    # Game settings
    sensitivity = 0.85
    m_yaw = 0.022
    fov_h = 90.0

    # Calculate focal length
    half_fov = math.radians(fov_h / 2.0)
    focal_length = (w / 2.0) / math.tan(half_fov)

    results = []

    # Test multiple movement amounts
    test_counts = [100, 200, 400]

    for counts in test_counts:
        # Capture before
        time.sleep(0.3)
        before = capture.grab()
        if before is None:
            continue
        time.sleep(0.1)

        # Move mouse right
        move_relative(counts, 0)
        time.sleep(0.3)

        # Capture after
        after = capture.grab()
        if after is None:
            continue

        # Measure actual pixel shift
        dx_pixels, dy_pixels, confidence = measure_pixel_shift(before, after)

        # Expected angle from counts
        angle_deg = counts * sensitivity * m_yaw

        # Expected pixel shift from that angle
        expected_pixels = focal_length * math.tan(math.radians(angle_deg))

        ratio = abs(dx_pixels) / expected_pixels if expected_pixels > 0 else 0

        print(f"  Sent {counts} counts → moved {dx_pixels:.1f}px "
              f"(expected {expected_pixels:.1f}px) "
              f"ratio={ratio:.3f} confidence={confidence:.3f}")

        results.append((counts, dx_pixels, expected_pixels, ratio, confidence))

        # Move back
        move_relative(-counts, 0)
        time.sleep(0.3)

    capture.stop()

    # Calculate scale factor
    # Filter out low-confidence results
    good_results = [(c, dx, ex, r, conf) for c, dx, ex, r, conf in results if conf > 0.1]

    if not good_results:
        print("\nERROR: Couldn't measure pixel shift reliably.")
        print("Make sure you're looking at a textured wall (not sky or flat color)")
        return

    avg_ratio = sum(r for _, _, _, r, _ in good_results) / len(good_results)

    # The ratio tells us: actual_pixels / expected_pixels
    # If ratio > 1: mouse moved more than expected → scale down
    # If ratio < 1: mouse moved less than expected → scale up
    scale = 1.0 / avg_ratio if avg_ratio > 0 else 1.0

    print()
    print(f"{'=' * 55}")
    print(f"  Actual/Expected pixel ratio: {avg_ratio:.3f}")
    print(f"  Recommended scale factor:    {scale:.3f}")
    print(f"{'=' * 55}")
    print()
    print(f"Tell me this scale: {scale:.3f}")


if __name__ == "__main__":
    main()

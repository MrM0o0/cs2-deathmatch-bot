"""Collect CS2 screenshots for training data.

Run this while playing CS2 to automatically capture frames at intervals.
Press F5 to save a screenshot, or use auto mode to capture every N seconds.
"""

import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.capture.screen import ScreenCapture

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def collect_screenshots(
    output_dir: str,
    interval: float = 2.0,
    max_count: int = 1000,
    monitor: int = 0,
    delay: float = 5.0,
    prefix: str = "cs2",
):
    """Capture screenshots at regular intervals.

    Args:
        output_dir: Directory to save screenshots.
        interval: Seconds between captures.
        max_count: Maximum number of screenshots.
        monitor: Monitor index.
        delay: Countdown before capturing starts, so you can alt-tab into CS2.
    """
    os.makedirs(output_dir, exist_ok=True)
    existing = len([f for f in os.listdir(output_dir) if f.endswith(".png")])
    print(f"[Collector] Output: {output_dir} ({existing} existing)")
    print(f"[Collector] Interval: {interval}s, Max: {max_count}")

    if delay > 0:
        print(f"[Collector] Alt-tab into CS2 now -- starting in {delay:.0f}s...")
        for remaining in range(int(delay), 0, -1):
            print(f"\r[Collector] {remaining}... ", end="", flush=True)
            time.sleep(1)
        print("\r[Collector] Go!        ")
    print("[Collector] Press Ctrl+C to stop")

    capture = ScreenCapture(monitor=monitor, target_fps=5)
    backend = capture.start()
    print(f"[Collector] Capture backend: {backend}")

    count = existing
    try:
        while count < max_count:
            frame = capture.grab()
            if frame is None:
                time.sleep(0.1)
                continue

            filename = f"{prefix}_{count:05d}.png"
            filepath = os.path.join(output_dir, filename)

            if _HAS_CV2:
                cv2.imwrite(filepath, frame)
            else:
                # Fallback using PIL
                from PIL import Image

                img = Image.fromarray(frame[:, :, ::-1])  # BGR -> RGB
                img.save(filepath)

            count += 1
            print(f"\r[Collector] Saved {count}/{max_count}: {filename}", end="")

            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n[Collector] Stopped. Total: {count} screenshots")
    finally:
        capture.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS2 Screenshot Collector")
    parser.add_argument(
        "--output", "-o", default="models/training/dataset/images", help="Output directory"
    )
    parser.add_argument(
        "--interval", "-i", type=float, default=2.0, help="Capture interval in seconds"
    )
    parser.add_argument("--max", "-m", type=int, default=1000, help="Maximum screenshots")
    parser.add_argument("--monitor", type=int, default=0, help="Monitor index")
    parser.add_argument(
        "--delay", "-d", type=float, default=5.0, help="Countdown before starting (alt-tab time)"
    )
    parser.add_argument(
        "--prefix", "-p", default="cs2", help="Filename prefix (use a unique one per batch)"
    )
    args = parser.parse_args()

    collect_screenshots(args.output, args.interval, args.max, args.monitor, args.delay, args.prefix)

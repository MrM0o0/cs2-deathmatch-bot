"""Screen capture using DXcam with mss fallback."""

import time
import numpy as np

_dxcam_available = False
_mss_available = False

try:
    import dxcam
    _dxcam_available = True
except ImportError:
    pass

try:
    import mss
    import mss.tools
    _mss_available = True
except ImportError:
    pass


class ScreenCapture:
    """High-performance screen capture for CS2."""

    def __init__(self, monitor: int = 0, target_fps: int = 30,
                 region: tuple[int, int, int, int] | None = None):
        """
        Args:
            monitor: Monitor index.
            target_fps: Target capture framerate.
            region: (left, top, right, bottom) capture region, or None for full screen.
        """
        self.monitor = monitor
        self.target_fps = target_fps
        self.region = region
        self._camera = None
        self._mss = None
        self._backend = None
        self._frame_count = 0
        self._start_time = 0.0

    def start(self) -> str:
        """Initialize capture backend. Returns backend name."""
        if _dxcam_available:
            try:
                self._camera = dxcam.create(
                    device_idx=self.monitor,
                    output_color="BGR",
                )
                self._camera.start(
                    target_fps=self.target_fps,
                    region=self.region,
                )
                self._backend = "dxcam"
                self._start_time = time.perf_counter()
                return "dxcam"
            except Exception as e:
                print(f"[ScreenCapture] DXcam failed: {e}, falling back to mss")
                self._camera = None

        if _mss_available:
            self._mss = mss.mss()
            self._backend = "mss"
            self._start_time = time.perf_counter()
            return "mss"

        raise RuntimeError("No screen capture backend available. Install dxcam or mss.")

    def grab(self) -> np.ndarray | None:
        """Grab a single frame as BGR numpy array (H, W, 3)."""
        self._frame_count += 1

        if self._backend == "dxcam":
            frame = self._camera.get_latest_frame()
            return frame  # Already BGR numpy array

        if self._backend == "mss":
            mon = self._mss.monitors[self.monitor + 1]  # mss is 1-indexed
            if self.region:
                mon = {
                    "left": self.region[0],
                    "top": self.region[1],
                    "width": self.region[2] - self.region[0],
                    "height": self.region[3] - self.region[1],
                }
            shot = self._mss.grab(mon)
            # mss returns BGRA, convert to BGR
            frame = np.array(shot)[:, :, :3].copy()
            return frame

        return None

    @property
    def fps(self) -> float:
        """Current average FPS."""
        elapsed = time.perf_counter() - self._start_time
        if elapsed <= 0:
            return 0.0
        return self._frame_count / elapsed

    def stop(self) -> None:
        """Clean up capture resources."""
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception:
                pass
            self._camera = None

        if self._mss is not None:
            self._mss.close()
            self._mss = None

        self._backend = None

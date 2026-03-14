"""Detect when the bot is stuck and initiate recovery."""

import time
import numpy as np


class StuckDetector:
    """Detects when the bot isn't moving despite movement inputs."""

    def __init__(self, timeout: float = 2.0, similarity_threshold: float = 0.98):
        """
        Args:
            timeout: Seconds of no movement before declaring stuck.
            similarity_threshold: Frame similarity to consider "not moving".
        """
        self.timeout = timeout
        self.sim_threshold = similarity_threshold
        self._last_frame: np.ndarray | None = None
        self._stuck_start: float | None = None
        self._is_stuck = False
        self._recovery_count = 0

    def update(self, frame: np.ndarray, is_moving: bool) -> bool:
        """Update stuck detection with current frame.

        Args:
            frame: Current game frame (or a center crop of it).
            is_moving: Whether movement keys are being pressed.

        Returns:
            True if bot appears to be stuck.
        """
        if not is_moving:
            # Not pressing movement keys, can't be "stuck"
            self._stuck_start = None
            self._is_stuck = False
            return False

        if self._last_frame is not None:
            similarity = self._frame_similarity(frame, self._last_frame)

            if similarity > self.sim_threshold:
                # Frames are very similar despite movement input
                if self._stuck_start is None:
                    self._stuck_start = time.perf_counter()
                elif time.perf_counter() - self._stuck_start > self.timeout:
                    self._is_stuck = True
                    self._recovery_count += 1
                    return True
            else:
                # Moving successfully
                self._stuck_start = None
                self._is_stuck = False

        # Store downsampled frame for comparison
        self._last_frame = self._downsample(frame)
        return False

    def _frame_similarity(self, frame_a: np.ndarray,
                          frame_b: np.ndarray) -> float:
        """Compare two frames for similarity (0 = different, 1 = identical)."""
        a = self._downsample(frame_a)
        b = frame_b  # Already downsampled

        if a.shape != b.shape:
            return 0.0

        # Normalized cross-correlation
        a_f = a.astype(np.float32).flatten()
        b_f = b.astype(np.float32).flatten()

        a_norm = np.linalg.norm(a_f)
        b_norm = np.linalg.norm(b_f)

        if a_norm == 0 or b_norm == 0:
            return 0.0

        return float(np.dot(a_f, b_f) / (a_norm * b_norm))

    def _downsample(self, frame: np.ndarray) -> np.ndarray:
        """Downsample frame for fast comparison."""
        # Take every 16th pixel in each dimension
        return frame[::16, ::16].copy()

    def reset(self) -> None:
        """Reset stuck detection state."""
        self._last_frame = None
        self._stuck_start = None
        self._is_stuck = False

    @property
    def is_stuck(self) -> bool:
        return self._is_stuck

    @property
    def recovery_count(self) -> int:
        return self._recovery_count

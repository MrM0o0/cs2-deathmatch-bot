"""Reactive wall-following exploration when no waypoints available."""

import random
import time
import math

import numpy as np


class WallFollower:
    """Reactive navigation by detecting and avoiding walls.

    Uses the center of the screen to detect if we're facing a wall,
    and turns away. This is the MVP navigation before waypoints.
    """

    def __init__(self, wall_threshold: int = 200, turn_speed: float = 5.0):
        """
        Args:
            wall_threshold: Brightness/edge threshold for wall detection.
            turn_speed: How fast to turn when avoiding walls.
        """
        self.wall_threshold = wall_threshold
        self.turn_speed = turn_speed
        self._current_dir = random.choice(["forward", "forward_left", "forward_right"])
        self._dir_change_time = time.perf_counter()
        self._dir_hold_time = random.uniform(1.0, 4.0)

    def get_movement(self, frame: np.ndarray) -> dict:
        """Analyze frame and decide movement direction.

        Args:
            frame: Current BGR frame.

        Returns:
            Dict with keys: forward, left, right, turn_x (mouse)
        """
        h, w = frame.shape[:2]

        # Check three vertical strips for wall proximity
        strip_w = w // 6
        center_y = h // 2
        strip_h = h // 4

        left_strip = frame[center_y - strip_h:center_y + strip_h,
                          w // 6:w // 6 + strip_w]
        center_strip = frame[center_y - strip_h:center_y + strip_h,
                            w // 2 - strip_w // 2:w // 2 + strip_w // 2]
        right_strip = frame[center_y - strip_h:center_y + strip_h,
                           5 * w // 6 - strip_w:5 * w // 6]

        # Detect walls by edge density (walls have more edges/detail up close)
        left_wall = self._wall_score(left_strip)
        center_wall = self._wall_score(center_strip)
        right_wall = self._wall_score(right_strip)

        result = {"forward": True, "left": False, "right": False, "turn_x": 0}

        # Wall in center - need to turn
        if center_wall > self.wall_threshold:
            if left_wall < right_wall:
                result["turn_x"] = -int(self.turn_speed)
                result["left"] = True
            else:
                result["turn_x"] = int(self.turn_speed)
                result["right"] = True
            return result

        # Periodically change direction for variety
        now = time.perf_counter()
        if now - self._dir_change_time > self._dir_hold_time:
            self._dir_change_time = now
            self._dir_hold_time = random.uniform(1.0, 4.0)
            self._current_dir = random.choice([
                "forward", "forward", "forward",  # Bias toward forward
                "forward_left", "forward_right",
            ])

        if self._current_dir == "forward_left":
            result["turn_x"] = -2
        elif self._current_dir == "forward_right":
            result["turn_x"] = 2

        return result

    def _wall_score(self, strip: np.ndarray) -> float:
        """Score how likely a strip contains a nearby wall.

        Uses variance of pixel values - nearby walls have high detail/edges.
        """
        if strip.size == 0:
            return 0

        gray = np.mean(strip, axis=2)
        # High variance = lots of detail = close to wall
        return float(np.std(gray))

"""Minimap reading for player position and orientation."""

import math
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class MinimapReader:
    """Reads player position from the CS2 minimap."""

    def __init__(self, minimap_x: int, minimap_y: int, minimap_size: int,
                 player_arrow_color: tuple[int, int, int] = (0, 255, 0)):
        self.x = minimap_x
        self.y = minimap_y
        self.size = minimap_size
        # BGR format for the player arrow color
        self.arrow_color = np.array(player_arrow_color[::-1], dtype=np.uint8)
        self._last_position = (minimap_size // 2, minimap_size // 2)
        self._last_angle = 0.0

    def read(self, frame: np.ndarray) -> tuple[tuple[int, int], float]:
        """Read player position and facing angle from minimap.

        Returns:
            ((x, y) on minimap, angle in degrees)
        """
        minimap = frame[self.y:self.y + self.size,
                        self.x:self.x + self.size]

        if minimap.size == 0 or cv2 is None:
            return self._last_position, self._last_angle

        # Find the player arrow by color matching
        # The player icon is a bright colored triangle/arrow
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)

        # Green arrow: H=40-80, S>100, V>100
        lower = np.array([40, 100, 100])
        upper = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        # Find centroid of the mask
        moments = cv2.moments(mask)
        if moments["m00"] > 10:  # Minimum area threshold
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            self._last_position = (cx, cy)

            # Estimate facing direction from arrow shape
            self._last_angle = self._estimate_angle(mask, cx, cy)

        return self._last_position, self._last_angle

    def _estimate_angle(self, mask: np.ndarray, cx: int, cy: int) -> float:
        """Estimate facing angle from arrow mask shape."""
        if cv2 is None:
            return self._last_angle

        # Find contours of the arrow
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._last_angle

        # Use the largest contour
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 5:
            return self._last_angle

        # Fit an ellipse to get orientation
        try:
            ellipse = cv2.fitEllipse(contour)
            angle = ellipse[2]  # Rotation angle
            return angle
        except cv2.error:
            return self._last_angle

    @property
    def position(self) -> tuple[int, int]:
        return self._last_position

    @property
    def angle(self) -> float:
        return self._last_angle

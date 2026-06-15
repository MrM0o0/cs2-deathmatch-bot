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
                 player_arrow_color: tuple[int, int, int] = (0, 255, 0),
                 sat_min: int = 150, val_min: int = 190,
                 min_blob_area: int = 6, lock_radius: float = 60.0,
                 **_legacy):
        """
        Args:
            sat_min, val_min: the player dot is detected hue-agnostically as the
                brightest, most-saturated blob (S>=sat_min, V>=val_min). CS2
                assigns your radar blip a different colour per match (cyan one
                game, yellow the next), so matching a fixed hue is unreliable --
                but it's always a bright, saturated dot, and the map terrain is
                not.
            min_blob_area: ignore blobs smaller than this (px).
            lock_radius: once locked, prefer the candidate blob within this many
                px of the last position (temporal stability vs. other markers).
            **_legacy: absorbs old hsv_lower/hsv_upper kwargs so existing callers
                don't break.
        """
        self.x = minimap_x
        self.y = minimap_y
        self.size = minimap_size
        # BGR format for the player arrow color (kept for reference/overlay).
        self.arrow_color = np.array(player_arrow_color[::-1], dtype=np.uint8)
        self.sat_min = sat_min
        self.val_min = val_min
        self.min_blob_area = min_blob_area
        self.lock_radius = lock_radius
        self._last_position = (minimap_size // 2, minimap_size // 2)
        self._last_angle = 0.0
        self._locked = False

    def read(self, frame: np.ndarray) -> tuple[tuple[int, int], float]:
        """Read player position and facing angle from minimap.

        Returns:
            ((x, y) on minimap, angle in degrees)
        """
        minimap = frame[self.y:self.y + self.size,
                        self.x:self.x + self.size]

        if minimap.size == 0 or cv2 is None:
            return self._last_position, self._last_angle

        # The player dot is the brightest, most-saturated blob -- hue agnostic,
        # so it survives CS2 reassigning the blip colour between matches.
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([0, self.sat_min, self.val_min], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_blob_area:
                continue
            m = cv2.moments(c)
            if m["m00"] <= 0:
                continue
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            candidates.append((area, (cx, cy), c))

        if not candidates:
            return self._last_position, self._last_angle

        # Once locked on, prefer the blob nearest where the player just was
        # (rejects bomb/teammate markers); otherwise take the largest.
        chosen = None
        if self._locked:
            lx, ly = self._last_position
            near = [t for t in candidates
                    if ((t[1][0] - lx) ** 2 + (t[1][1] - ly) ** 2) ** 0.5 <= self.lock_radius]
            if near:
                chosen = max(near, key=lambda t: t[0])
        if chosen is None:
            chosen = max(candidates, key=lambda t: t[0])

        _, (cx, cy), contour = chosen
        self._last_position = (cx, cy)
        self._locked = True
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

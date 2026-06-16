"""Minimap reading for player position and orientation."""

import math

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class MinimapReader:
    """Reads player position from the CS2 minimap."""

    def __init__(
        self,
        minimap_x: int,
        minimap_y: int,
        minimap_size: int,
        player_arrow_color: tuple[int, int, int] = (0, 255, 0),
        sat_min: int = 150,
        val_min: int = 190,
        min_blob_area: int = 6,
        lock_radius: float = 60.0,
        cone_sat_max: int = 60,
        cone_val_min: int = 180,
        cone_radius: int = 16,
        cone_inner: float = 4.0,
        min_cone_area: int = 8,
        **_legacy,
    ):
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
            cone_sat_max, cone_val_min: the facing arrow/cone next to the dot is
                near-WHITE (low saturation, high value), which separates it
                cleanly from the saturated dot. Heading is the direction from the
                dot to that white cone -- instant and unambiguous (full 360),
                unlike fitting an ellipse to the round dot (180-ambiguous + noisy).
            cone_radius: search box half-size (px) around the dot for the cone.
            cone_inner: ignore white pixels within this radius (the dot's halo).
            min_cone_area: need at least this many white px to trust a heading.
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
        self.cone_sat_max = cone_sat_max
        self.cone_val_min = cone_val_min
        self.cone_radius = cone_radius
        self.cone_inner = cone_inner
        self.min_cone_area = min_cone_area
        self._last_position = (minimap_size // 2, minimap_size // 2)
        self._last_angle = 0.0
        self._locked = False

    def read(self, frame: np.ndarray) -> tuple[tuple[int, int], float]:
        """Read player position and facing angle from minimap.

        Returns:
            ((x, y) on minimap, angle in degrees)
        """
        minimap = frame[self.y : self.y + self.size, self.x : self.x + self.size]

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

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
            near = [
                t
                for t in candidates
                if ((t[1][0] - lx) ** 2 + (t[1][1] - ly) ** 2) ** 0.5 <= self.lock_radius
            ]
            if near:
                chosen = max(near, key=lambda t: t[0])
        if chosen is None:
            chosen = max(candidates, key=lambda t: t[0])

        _, (cx, cy), _contour = chosen
        self._last_position = (cx, cy)
        self._locked = True
        heading = self._cone_heading(hsv, cx, cy)
        if heading is not None:
            self._last_angle = heading
        return self._last_position, self._last_angle

    def _cone_heading(self, hsv: np.ndarray, cx: int, cy: int) -> float | None:
        """Heading (deg) from the white facing-cone beside the dot.

        The cone is near-white (low saturation, high value) so it isolates
        cleanly from the saturated dot. We take the direction from the dot
        centre to the cone's pixels, weighted toward the tip (pixels farther
        out), giving an instant, unambiguous full-360 heading. Returns None if
        no clear cone is visible (caller keeps the last heading).

        Angle convention matches the rest of the nav stack: degrees via
        atan2(dy, dx) in minimap pixels (x right, y down) -- 0 = east, +90 = south.
        """
        if cv2 is None:
            return None

        h, w = hsv.shape[:2]
        r = self.cone_radius
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r), min(h, cy + r)
        win = hsv[y1:y2, x1:x2]
        if win.size == 0:
            return None

        white = cv2.inRange(
            win,
            np.array([0, 0, self.cone_val_min], dtype=np.uint8),
            np.array([179, self.cone_sat_max, 255], dtype=np.uint8),
        )
        ys, xs = np.where(white > 0)
        if len(xs) < self.min_cone_area:
            return None

        # Positions relative to the dot centre; drop the dot's own halo.
        rx = xs.astype(np.float64) + x1 - cx
        ry = ys.astype(np.float64) + y1 - cy
        rad = np.hypot(rx, ry)
        keep = rad >= self.cone_inner
        if keep.sum() < self.min_cone_area:
            return None
        rx, ry, rad = rx[keep], ry[keep], rad[keep]

        # Weight toward the tip so the direction tracks where the cone points.
        wsum = rad.sum()
        mx = float((rx * rad).sum() / wsum)
        my = float((ry * rad).sum() / wsum)
        return math.degrees(math.atan2(my, mx))

    @property
    def position(self) -> tuple[int, int]:
        return self._last_position

    @property
    def angle(self) -> float:
        return self._last_angle

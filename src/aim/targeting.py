"""Convert detection bounding boxes to aim deltas."""

import random

from src.vision.detector import Detection
from src.utils.math_helpers import (
    bbox_to_aim_point, screen_delta_to_mouse, distance
)


class TargetingSystem:
    """Converts detections into aim commands with humanization."""

    def __init__(self, screen_center_x: int, screen_center_y: int,
                 sensitivity: float = 2.0, m_yaw: float = 0.022,
                 m_pitch: float = 0.022, head_aim_chance: float = 0.3):
        self.cx = screen_center_x
        self.cy = screen_center_y
        self.sensitivity = sensitivity
        self.m_yaw = m_yaw
        self.m_pitch = m_pitch
        self.head_aim_chance = head_aim_chance

    def get_aim_delta(self, detection: Detection) -> tuple[int, int, float]:
        """Calculate mouse movement needed to aim at a detection.

        Returns:
            (mouse_dx, mouse_dy, screen_distance_pixels)
        """
        # Decide aim point (head or body)
        head_aim = random.random() < self.head_aim_chance or detection.is_head

        aim_x, aim_y = bbox_to_aim_point(
            detection.x1, detection.y1,
            detection.x2, detection.y2,
            head_aim=head_aim,
        )

        # Screen pixel delta from crosshair to target
        dx_pixels = aim_x - self.cx
        dy_pixels = aim_y - self.cy

        # Distance for priority calculation
        screen_dist = distance((self.cx, self.cy), (aim_x, aim_y))

        # Convert to mouse counts
        mouse_dx, mouse_dy = screen_delta_to_mouse(
            dx_pixels, dy_pixels,
            self.sensitivity, self.m_yaw, self.m_pitch
        )

        return mouse_dx, mouse_dy, screen_dist

    def select_target(self, detections: list[Detection],
                      our_team: str = "ct") -> Detection | None:
        """Select the best target from a list of detections.

        Prioritizes:
        1. Enemies only (skip our team)
        2. Closest to crosshair
        3. Higher confidence
        """
        enemies = [d for d in detections if our_team not in d.class_name]
        if not enemies:
            return None

        # Score each target (lower is better)
        def target_score(det: Detection) -> float:
            cx, cy = det.center
            dist = distance((self.cx, self.cy), (cx, cy))
            # Bonus for head detections
            head_bonus = -100 if det.is_head else 0
            # Confidence factor
            conf_bonus = (1.0 - det.confidence) * 50
            return dist + head_bonus + conf_bonus

        enemies.sort(key=target_score)
        return enemies[0]

    def is_on_target(self, detection: Detection, threshold: float = 30.0) -> bool:
        """Check if crosshair is already on/near the target."""
        cx, cy = detection.center
        dist = distance((self.cx, self.cy), (cx, cy))
        return dist < threshold

"""Target selection and threat assessment."""

from src.vision.detector import Detection
from src.utils.math_helpers import distance


class ThreatAssessor:
    """Evaluate and prioritize detected enemies."""

    def __init__(self, screen_center: tuple[int, int]):
        self.cx, self.cy = screen_center

    def assess_threat(self, detection: Detection) -> float:
        """Score a detection's threat level (higher = more threatening).

        Factors:
        - Distance to crosshair (closer = more threat)
        - Size of bounding box (closer enemies = bigger bbox)
        - Confidence (higher confidence = more certain it's real)
        """
        cx, cy = detection.center
        crosshair_dist = distance((self.cx, self.cy), (cx, cy))

        # Closer to crosshair = higher threat
        proximity_score = max(0, 1000 - crosshair_dist)

        # Larger bbox = closer enemy = more threat
        size_score = detection.area / 100.0

        # Confidence boost
        conf_score = detection.confidence * 200

        return proximity_score + size_score + conf_score

    def prioritize_targets(self, detections: list[Detection],
                           our_team: str = "ct") -> list[Detection]:
        """Sort detections by threat level, filtering out teammates.

        Returns enemies sorted by threat (highest first).
        """
        enemies = [d for d in detections if our_team not in d.class_name]

        if not enemies:
            return []

        # Sort by threat score (descending)
        enemies.sort(key=lambda d: self.assess_threat(d), reverse=True)
        return enemies

    def should_switch_target(self, current: Detection | None,
                             new_targets: list[Detection],
                             switch_threshold: float = 200) -> bool:
        """Decide if we should switch to a new target.

        Avoids constant target switching (looks robotic).
        """
        if current is None:
            return True

        if not new_targets:
            return False

        best = new_targets[0]
        current_threat = self.assess_threat(current)
        best_threat = self.assess_threat(best)

        # Only switch if new target is significantly more threatening
        return best_threat > current_threat + switch_threshold

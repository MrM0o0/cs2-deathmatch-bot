"""Deliberate error injection for human-like imperfection."""

import random
import math


class MistakeMaker:
    """Injects human-like mistakes into aim and actions."""

    def __init__(self, overshoot_chance: float = 0.35,
                 overshoot_magnitude: float = 1.3,
                 tracking_error: float = 8.0):
        self.overshoot_chance = overshoot_chance
        self.overshoot_magnitude = overshoot_magnitude
        self.tracking_error = tracking_error

    def apply_aim_error(self, target_x: float, target_y: float) -> tuple[float, float]:
        """Add random error to an aim target point.

        Returns adjusted (x, y) with human-like inaccuracy.
        """
        # Add tracking error (normal distribution around target)
        err_x = random.gauss(0, self.tracking_error)
        err_y = random.gauss(0, self.tracking_error)
        return target_x + err_x, target_y + err_y

    def should_overshoot(self) -> bool:
        """Decide if this aim movement should overshoot."""
        return random.random() < self.overshoot_chance

    def overshoot_target(self, current_x: float, current_y: float,
                         target_x: float, target_y: float) -> tuple[float, float]:
        """Calculate an overshoot point past the target.

        Returns a point that goes past the target, requiring a correction.
        """
        dx = target_x - current_x
        dy = target_y - current_y

        # Overshoot by the magnitude factor, with some randomness
        factor = self.overshoot_magnitude * random.uniform(0.8, 1.2)
        overshoot_x = current_x + dx * factor
        overshoot_y = current_y + dy * factor

        # Add slight angular deviation
        angle_err = random.gauss(0, 3)  # degrees
        rad = math.radians(angle_err)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        rel_x = overshoot_x - current_x
        rel_y = overshoot_y - current_y
        overshoot_x = current_x + rel_x * cos_a - rel_y * sin_a
        overshoot_y = current_y + rel_x * sin_a + rel_y * cos_a

        return overshoot_x, overshoot_y

    def micro_correction_count(self, corrections_range: list[int]) -> int:
        """How many micro-corrections to make after the initial flick."""
        lo, hi = corrections_range
        return random.randint(lo, hi)

    def should_whiff(self, base_chance: float = 0.05) -> bool:
        """Occasionally completely miss (panic flick / brain fart)."""
        return random.random() < base_chance

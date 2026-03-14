"""Basic spray/recoil compensation."""

import random

from src.input.mouse import move_relative


# Simplified recoil patterns (vertical pull-down values per shot)
# These are rough approximations - real patterns are weapon-specific
RECOIL_PATTERNS = {
    "ak47": [
        (0, -3), (0, -5), (0, -7), (0, -8), (0, -9),
        (-1, -10), (-2, -10), (-3, -9), (-3, -8), (-2, -7),
        (1, -6), (2, -6), (3, -5), (3, -5), (2, -4),
        (1, -4), (0, -3), (-1, -3), (-2, -3), (-2, -2),
        (-1, -2), (0, -2), (1, -2), (1, -1), (0, -1),
        (0, -1), (0, -1), (0, -1), (0, 0), (0, 0),
    ],
    "m4a4": [
        (0, -2), (0, -4), (0, -6), (0, -7), (0, -7),
        (-1, -8), (-1, -7), (-2, -7), (-2, -6), (-1, -5),
        (0, -5), (1, -4), (2, -4), (2, -3), (1, -3),
        (0, -3), (-1, -3), (-1, -2), (0, -2), (0, -2),
        (0, -2), (0, -1), (0, -1), (0, -1), (0, -1),
        (0, -1), (0, 0), (0, 0), (0, 0), (0, 0),
    ],
    "default": [
        (0, -2), (0, -4), (0, -5), (0, -6), (0, -6),
        (0, -5), (0, -5), (0, -4), (0, -4), (0, -3),
        (0, -3), (0, -2), (0, -2), (0, -2), (0, -1),
    ],
}


class RecoilCompensator:
    """Compensates for weapon recoil during spraying."""

    def __init__(self, compensation_factor: float = 0.5, weapon: str = "default"):
        """
        Args:
            compensation_factor: 0.0 = no compensation, 1.0 = perfect.
                Human players are typically 0.4-0.7.
            weapon: Weapon name for recoil pattern lookup.
        """
        self.compensation_factor = compensation_factor
        self.weapon = weapon
        self._shot_count = 0

    def get_pattern(self) -> list[tuple[int, int]]:
        """Get recoil pattern for current weapon."""
        return RECOIL_PATTERNS.get(self.weapon, RECOIL_PATTERNS["default"])

    def compensate(self) -> tuple[int, int]:
        """Get recoil compensation for the current shot.

        Returns:
            (dx, dy) mouse adjustment to counteract recoil.
        """
        pattern = self.get_pattern()

        if self._shot_count >= len(pattern):
            # Beyond pattern length, minimal compensation
            base_dx, base_dy = 0, -1
        else:
            base_dx, base_dy = pattern[self._shot_count]

        self._shot_count += 1

        # Apply compensation factor (imperfect, like a human)
        comp_dx = int(-base_dx * self.compensation_factor * random.uniform(0.7, 1.3))
        comp_dy = int(-base_dy * self.compensation_factor * random.uniform(0.7, 1.3))

        return comp_dx, comp_dy

    def apply(self) -> None:
        """Apply recoil compensation as mouse movement."""
        dx, dy = self.compensate()
        if dx != 0 or dy != 0:
            move_relative(dx, dy)

    def reset(self) -> None:
        """Reset shot counter (new spray)."""
        self._shot_count = 0

    def set_weapon(self, weapon: str) -> None:
        """Switch weapon recoil pattern."""
        self.weapon = weapon
        self.reset()

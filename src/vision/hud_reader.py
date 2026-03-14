"""Read HUD elements (health, ammo, alive status) from screen regions."""

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class HUDState:
    """Current game HUD state."""

    def __init__(self):
        self.health: int = 100
        self.armor: int = 0
        self.ammo_clip: int = 30
        self.ammo_reserve: int = 90
        self.is_alive: bool = True

    def __repr__(self) -> str:
        status = "ALIVE" if self.is_alive else "DEAD"
        return f"HUD({status} HP:{self.health} Armor:{self.armor} Ammo:{self.ammo_clip}/{self.ammo_reserve})"


class HUDReader:
    """Reads game state from HUD screen regions using OCR-free methods.

    Uses color-based digit recognition since CS2 HUD digits have
    consistent styling. Falls back to brightness-based alive detection.
    """

    def __init__(self, regions: dict):
        """
        Args:
            regions: Dict of region names -> (x, y, w, h) tuples from settings.
        """
        self.regions = regions
        self._prev_state = HUDState()

    def read(self, frame: np.ndarray) -> HUDState:
        """Read HUD state from a full frame."""
        state = HUDState()

        # Detect alive/dead by checking if HUD elements are visible
        state.is_alive = self._detect_alive(frame)

        if state.is_alive:
            state.health = self._read_number(frame, "health", default=100)
            state.armor = self._read_number(frame, "armor", default=0)
            state.ammo_clip = self._read_number(frame, "ammo_clip", default=30)
            state.ammo_reserve = self._read_number(frame, "ammo_reserve", default=90)

        self._prev_state = state
        return state

    def _detect_alive(self, frame: np.ndarray) -> bool:
        """Detect if player is alive by checking health region brightness.

        When dead, the HUD fades out / grays out significantly.
        """
        if "health" not in self.regions:
            return True

        region = self.regions["health"]
        x, y, w, h = region
        roi = frame[y:y + h, x:x + w]

        if roi.size == 0:
            return self._prev_state.is_alive

        # Health numbers are bright white/colored when alive
        # When dead, region is very dark or absent
        mean_brightness = np.mean(roi)
        return mean_brightness > 30

    def _read_number(self, frame: np.ndarray, region_name: str,
                     default: int = 0) -> int:
        """Read a number from a HUD region using brightness analysis.

        Simple approach: estimate value by analyzing white pixel density.
        More accurate OCR can be added later with digit templates.
        """
        if region_name not in self.regions:
            return default

        region = self.regions[region_name]
        x, y, w, h = region
        roi = frame[y:y + h, x:x + w]

        if roi.size == 0:
            return default

        # Convert to grayscale and threshold to isolate HUD text
        if cv2 is not None:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = np.mean(roi, axis=2).astype(np.uint8)

        # HUD numbers are typically bright (>200) on dark background
        bright_pixels = np.sum(gray > 180)
        total_pixels = gray.size

        if total_pixels == 0:
            return default

        # Rough estimation based on digit coverage
        # This is a placeholder - proper digit template matching
        # should replace this for accuracy
        density = bright_pixels / total_pixels

        if region_name == "health":
            if density < 0.02:
                return 0
            # Rough mapping: more bright pixels = higher number
            return max(1, min(100, int(density * 500)))

        if region_name in ("ammo_clip", "ammo_reserve"):
            return max(0, min(999, int(density * 800)))

        return default

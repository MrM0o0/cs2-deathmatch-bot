"""Perlin/simplex noise for natural-looking jitter and variation."""

import time
import math

try:
    from opensimplex import OpenSimplex
    _HAS_OPENSIMPLEX = True
except ImportError:
    _HAS_OPENSIMPLEX = False


class NoiseGenerator:
    """Generates smooth noise for humanizing mouse movement and timing."""

    def __init__(self, seed: int | None = None):
        self._seed = seed or int(time.time() * 1000) % (2**31)
        self._offset = 0.0

        if _HAS_OPENSIMPLEX:
            self._noise = OpenSimplex(seed=self._seed)
        else:
            self._noise = None

    def sample_1d(self, x: float, scale: float = 1.0) -> float:
        """Sample 1D noise at position x.

        Returns value in [-1, 1].
        """
        if self._noise is not None:
            return self._noise.noise2(x * scale, 0.0)
        # Fallback: cheap pseudo-noise using sin
        return math.sin(x * scale * 7.31 + self._seed) * math.cos(x * scale * 3.17)

    def sample_2d(self, x: float, y: float, scale: float = 1.0) -> float:
        """Sample 2D noise at position (x, y).

        Returns value in [-1, 1].
        """
        if self._noise is not None:
            return self._noise.noise2(x * scale, y * scale)
        return (math.sin(x * scale * 5.13 + y * scale * 3.77 + self._seed) *
                math.cos(x * scale * 2.91 - y * scale * 4.23))

    def time_noise(self, scale: float = 1.0, amplitude: float = 1.0) -> float:
        """Get noise value based on current time. Useful for continuous jitter."""
        t = time.perf_counter()
        return self.sample_1d(t, scale) * amplitude

    def mouse_jitter(self, amplitude: float = 2.0) -> tuple[float, float]:
        """Generate a 2D jitter offset for mouse movement.

        Returns (dx, dy) jitter in pixels.
        """
        t = time.perf_counter()
        jx = self.sample_2d(t, 0.0, scale=3.0) * amplitude
        jy = self.sample_2d(0.0, t, scale=3.0) * amplitude
        return jx, jy

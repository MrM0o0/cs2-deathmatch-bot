"""Bezier curve mouse movement for human-like aim paths."""

import time
import random
import math

from src.utils.math_helpers import cubic_bezier, distance
from src.humanizer.noise import NoiseGenerator
from src.input.mouse import move_relative


def _smoothstep(t: float) -> float:
    """Hermite smoothstep easing — accelerates then decelerates like a real hand."""
    return t * t * (3.0 - 2.0 * t)


def _spin_wait(seconds: float) -> None:
    """Precise sub-millisecond wait using busy spin instead of sleep()."""
    if seconds <= 0:
        return
    target = time.perf_counter() + seconds
    while time.perf_counter() < target:
        pass


class MouseMover:
    """Moves mouse along Bezier curves with noise for human-like aiming."""

    def __init__(self, base_speed: float = 6.0, noise_amplitude: float = 2.0):
        """
        Args:
            base_speed: Base mouse speed in pixels per millisecond.
            noise_amplitude: Amplitude of Perlin noise jitter.
        """
        self.base_speed = base_speed
        self.noise_amplitude = noise_amplitude
        self._noise = NoiseGenerator()
        self._residual_x = 0.0
        self._residual_y = 0.0

    def move_to_delta(self, dx: float, dy: float, duration_ms: float | None = None,
                      steps: int | None = None) -> None:
        """Move mouse by (dx, dy) along a humanized Bezier curve.

        Args:
            dx: Total horizontal mouse movement (raw counts).
            dy: Total vertical mouse movement.
            duration_ms: Total movement time. Auto-calculated if None.
            steps: Number of intermediate points. Auto-calculated if None.
        """
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1:
            return

        # Auto-calculate duration using Fitts's Law inspired curve
        # Fast flicks with slight variance
        if duration_ms is None:
            duration_ms = 30 + 40 * math.log2(1 + dist / 50)
            duration_ms *= random.uniform(0.85, 1.15)
            duration_ms = max(16, min(300, duration_ms))

        if steps is None:
            # ~2ms per step for smooth movement (was 8ms = chunky)
            steps = max(8, int(duration_ms / 2))

        # Generate Bezier control points
        p0 = (0.0, 0.0)
        p3 = (float(dx), float(dy))

        # Randomized control points for natural curve
        ctrl_spread = dist * 0.3
        p1 = (
            dx * random.uniform(0.2, 0.4) + random.gauss(0, ctrl_spread * 0.3),
            dy * random.uniform(0.2, 0.4) + random.gauss(0, ctrl_spread * 0.3),
        )
        p2 = (
            dx * random.uniform(0.6, 0.8) + random.gauss(0, ctrl_spread * 0.2),
            dy * random.uniform(0.6, 0.8) + random.gauss(0, ctrl_spread * 0.2),
        )

        step_delay = duration_ms / steps / 1000.0
        prev_x, prev_y = 0.0, 0.0

        for i in range(1, steps + 1):
            # Linear parameter
            t_linear = i / steps

            # Apply easing (accelerate then decelerate)
            t = _smoothstep(t_linear)

            # Evaluate Bezier
            bx, by = cubic_bezier(t, p0, p1, p2, p3)

            # Add Perlin noise jitter (decreasing toward end)
            fade = 1.0 - (t_linear ** 2)
            jx, jy = self._noise.mouse_jitter(self.noise_amplitude * fade)
            bx += jx
            by += jy

            # Calculate delta from previous position
            step_dx = bx - prev_x + self._residual_x
            step_dy = by - prev_y + self._residual_y

            # SendInput needs integers, accumulate residual
            int_dx = int(round(step_dx))
            int_dy = int(round(step_dy))
            self._residual_x = step_dx - int_dx
            self._residual_y = step_dy - int_dy

            if int_dx != 0 or int_dy != 0:
                move_relative(int_dx, int_dy)

            prev_x, prev_y = bx - jx, by - jy

            # Use spin-wait for precise timing instead of sleep()
            if step_delay > 0:
                _spin_wait(step_delay)

    def move_instant(self, dx: int, dy: int) -> None:
        """Instant mouse movement (for small corrections)."""
        move_relative(dx, dy)

    def micro_correct(self, dx: float, dy: float, delay_ms: float = 40) -> None:
        """Small correction movement after initial flick."""
        steps = random.randint(3, 6)
        self.move_to_delta(dx, dy, duration_ms=delay_ms, steps=steps)

"""Non-blocking Bezier aim path — precomputes a smooth curve with high step density.

Generates many fine-grained steps along a Bezier curve. Each frame, multiple
micro-steps are applied with tiny delays between them, creating fluid mouse
movement instead of visible jumps.
"""

import math
import random
import time
from src.utils.math_helpers import cubic_bezier


def _human_ease(t: float) -> float:
    """Human flick: brief ramp-up → fast committed leap → slow micro-adjust.

    First 10%: slight acceleration (hand reacting)
    Next 50%: fast committed motion (covers ~85% of distance)
    Last 40%: slow settle with micro-corrections
    """
    if t < 0.1:
        # Brief ramp-up
        return (t / 0.1) ** 1.5 * 0.05
    elif t < 0.6:
        # Fast committed leap
        progress = (t - 0.1) / 0.5
        return 0.05 + progress * 0.8
    else:
        # Slow settle / micro-corrections
        progress = (t - 0.6) / 0.4
        return 0.85 + (1.0 - (1.0 - progress) ** 2) * 0.15


def _spin_wait(seconds: float) -> None:
    """Precise sub-millisecond wait."""
    if seconds <= 0:
        return
    target = time.perf_counter() + seconds
    while time.perf_counter() < target:
        pass


class AimPath:
    """Manages a non-blocking Bezier aim path with sub-frame smoothing.

    Usage:
        path = AimPath()

        # When new target acquired:
        path.start(mouse_dx, mouse_dy)

        # Each frame (call once per frame, handles timing internally):
        path.apply_frame(mouse_move_relative_fn)
    """

    def __init__(self):
        self._steps: list[tuple[int, int]] = []
        self._index = 0
        self._active = False
        self._steps_per_frame = 1
        self._step_delay = 0.0
        self._total_steps = 0

    def start(self, total_dx: float, total_dy: float,
              duration_ms: float | None = None, noise: float = 1.0) -> None:
        """Generate a new Bezier aim path with high step density.

        Args:
            total_dx: Total horizontal mouse movement needed.
            total_dy: Total vertical mouse movement needed.
            duration_ms: How long the move should take. Auto-calculated if None.
            noise: Jitter amplitude for humanization.
        """
        dist = math.sqrt(total_dx * total_dx + total_dy * total_dy)
        if dist < 2:
            self._active = False
            return

        # Auto duration — fast flick + settle time
        if duration_ms is None:
            duration_ms = 40 + 35 * math.log2(1 + dist / 40)
            duration_ms *= random.uniform(0.9, 1.1)
            duration_ms = max(30, min(250, duration_ms))

        # Step density: ~1 step per 3ms (balance smoothness vs CPU)
        total_steps = max(5, int(duration_ms / 3))

        # How many steps to apply per frame (~20ms per frame at 50fps)
        frame_ms = 20.0
        self._steps_per_frame = max(1, int(frame_ms / 2))  # ~10 steps per frame
        self._step_delay = 0.001  # 1ms between sub-steps (less CPU than 2ms)

        # Bezier control points
        p0 = (0.0, 0.0)
        p3 = (float(total_dx), float(total_dy))

        ctrl_spread = dist * 0.2
        p1 = (
            total_dx * random.uniform(0.25, 0.4) + random.gauss(0, ctrl_spread * 0.15),
            total_dy * random.uniform(0.25, 0.4) + random.gauss(0, ctrl_spread * 0.15),
        )
        p2 = (
            total_dx * random.uniform(0.6, 0.8) + random.gauss(0, ctrl_spread * 0.1),
            total_dy * random.uniform(0.6, 0.8) + random.gauss(0, ctrl_spread * 0.1),
        )

        # Generate all steps
        self._steps = []
        prev_x, prev_y = 0.0, 0.0
        residual_x, residual_y = 0.0, 0.0

        for i in range(1, total_steps + 1):
            t_linear = i / total_steps
            t = _human_ease(t_linear)

            bx, by = cubic_bezier(t, p0, p1, p2, p3)

            # Subtle noise that fades
            fade = 1.0 - t_linear ** 2
            bx += random.gauss(0, noise * fade)
            by += random.gauss(0, noise * fade)

            step_dx = bx - prev_x + residual_x
            step_dy = by - prev_y + residual_y

            int_dx = int(round(step_dx))
            int_dy = int(round(step_dy))
            residual_x = step_dx - int_dx
            residual_y = step_dy - int_dy

            self._steps.append((int_dx, int_dy))
            prev_x, prev_y = bx, by

        self._index = 0
        self._total_steps = total_steps
        self._active = True

    def apply_frame(self, move_fn) -> None:
        """Apply this frame's portion of the path. Call once per main loop iteration.

        Args:
            move_fn: Function that takes (dx, dy) to move the mouse. Usually mouse.move_relative.
        """
        if not self._active:
            return

        # Apply multiple micro-steps this frame with small delays between
        steps_this_frame = min(self._steps_per_frame, len(self._steps) - self._index)

        for _ in range(steps_this_frame):
            if self._index >= len(self._steps):
                self._active = False
                break

            dx, dy = self._steps[self._index]
            self._index += 1

            if dx != 0 or dy != 0:
                move_fn(dx, dy)

            # Tiny delay between sub-steps for smoothness
            if self._step_delay > 0 and self._index < len(self._steps):
                _spin_wait(self._step_delay)

        if self._index >= len(self._steps):
            self._active = False

    @property
    def is_active(self) -> bool:
        return self._active and self._index < len(self._steps)

    @property
    def progress(self) -> float:
        if not self._total_steps:
            return 1.0
        return self._index / self._total_steps

    def cancel(self) -> None:
        self._active = False
        self._steps = []
        self._index = 0

    @property
    def remaining_steps(self) -> int:
        if not self._active:
            return 0
        return max(0, len(self._steps) - self._index)

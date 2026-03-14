"""Vector math, angle calculations, and geometry helpers."""

import math
import random


def distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Euclidean distance between two 2D points."""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def angle_between(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Angle in degrees from p1 to p2 (0 = right, 90 = down)."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-180, 180] range."""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation."""
    return a + (b - a) * t


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range."""
    return max(min_val, min(max_val, value))


def random_in_range(low: float, high: float) -> float:
    """Random float in range."""
    return random.uniform(low, high)


def screen_delta_to_mouse(dx_pixels: float, dy_pixels: float,
                          sensitivity: float, m_yaw: float,
                          m_pitch: float) -> tuple[int, int]:
    """Convert screen pixel delta to mouse movement counts.

    CS2 mouse input: pixels_moved = counts * sensitivity * m_yaw
    So: counts = pixels / (sensitivity * m_yaw)

    Args:
        dx_pixels: Horizontal screen pixel difference.
        dy_pixels: Vertical screen pixel difference.
        sensitivity: CS2 in-game sensitivity.
        m_yaw: CS2 m_yaw value (default 0.022).
        m_pitch: CS2 m_pitch value (default 0.022).

    Returns:
        (mouse_dx, mouse_dy) in raw mouse counts.
    """
    # The actual conversion depends on resolution and FOV
    # This is a simplified model - needs calibration per setup
    mouse_dx = int(dx_pixels / (sensitivity * m_yaw))
    mouse_dy = int(dy_pixels / (sensitivity * m_pitch))
    return mouse_dx, mouse_dy


def bbox_to_aim_point(x1: float, y1: float, x2: float, y2: float,
                      head_aim: bool = False) -> tuple[float, float]:
    """Convert a detection bounding box to an aim point.

    Args:
        x1, y1, x2, y2: Bounding box coordinates.
        head_aim: If True, aim for upper portion (head area).

    Returns:
        (x, y) aim point in screen coordinates.
    """
    cx = (x1 + x2) / 2

    if head_aim:
        # Aim at top ~20% of bbox (head area)
        cy = y1 + (y2 - y1) * 0.15
    else:
        # Aim at upper-center (chest area)
        cy = y1 + (y2 - y1) * 0.35

    return cx, cy


def cubic_bezier(t: float, p0: tuple[float, float], p1: tuple[float, float],
                 p2: tuple[float, float], p3: tuple[float, float]) -> tuple[float, float]:
    """Evaluate cubic Bezier curve at parameter t in [0, 1]."""
    u = 1 - t
    x = (u**3 * p0[0] + 3 * u**2 * t * p1[0] +
         3 * u * t**2 * p2[0] + t**3 * p3[0])
    y = (u**3 * p0[1] + 3 * u**2 * t * p1[1] +
         3 * u * t**2 * p2[1] + t**3 * p3[1])
    return x, y

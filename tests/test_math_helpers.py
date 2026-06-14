"""Tests for math helper functions."""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.math_helpers import (
    distance, angle_between, normalize_angle, lerp, clamp,
    bbox_to_aim_point, cubic_bezier, screen_delta_to_mouse,
)


def test_distance():
    assert distance((0, 0), (3, 4)) == 5.0
    assert distance((1, 1), (1, 1)) == 0.0
    assert abs(distance((0, 0), (1, 1)) - math.sqrt(2)) < 1e-9


def test_angle_between():
    assert angle_between((0, 0), (1, 0)) == 0.0
    assert angle_between((0, 0), (0, 1)) == 90.0
    assert abs(angle_between((0, 0), (-1, 0)) - 180.0) < 1e-9


def test_normalize_angle():
    assert normalize_angle(0) == 0
    assert normalize_angle(360) == 0
    assert normalize_angle(-360) == 0
    assert normalize_angle(270) == -90
    assert normalize_angle(-270) == 90


def test_lerp():
    assert lerp(0, 10, 0.5) == 5.0
    assert lerp(0, 10, 0.0) == 0.0
    assert lerp(0, 10, 1.0) == 10.0


def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-5, 0, 10) == 0
    assert clamp(15, 0, 10) == 10


def test_bbox_to_aim_point():
    # Body aim
    x, y = bbox_to_aim_point(100, 100, 200, 300, head_aim=False)
    assert x == 150  # Center X
    assert 100 < y < 200  # Upper portion (chest)

    # Head aim
    x, y = bbox_to_aim_point(100, 100, 200, 300, head_aim=True)
    assert x == 150
    assert y < 150  # Near top


def test_cubic_bezier():
    p0, p3 = (0.0, 0.0), (10.0, 10.0)
    p1, p2 = (3.0, 0.0), (7.0, 10.0)

    # Start and end points
    bx, by = cubic_bezier(0.0, p0, p1, p2, p3)
    assert abs(bx) < 1e-9 and abs(by) < 1e-9

    bx, by = cubic_bezier(1.0, p0, p1, p2, p3)
    assert abs(bx - 10.0) < 1e-9 and abs(by - 10.0) < 1e-9


def test_screen_delta_to_mouse():
    # FOV-based conversion: pixel offset -> angle (via focal length) -> mouse counts.
    # focal = (3440/2) / tan(45deg) = 1720
    # angle_x = atan2(100, 1720) = 3.328deg -> 3.328 / (2.0*0.022) = 75
    dx, dy = screen_delta_to_mouse(100, 50, 2.0, 0.022, 0.022)

    import math
    focal = (3440 / 2.0) / math.tan(math.radians(45.0))
    expected_dx = int(math.degrees(math.atan2(100, focal)) / (2.0 * 0.022))
    expected_dy = int(math.degrees(math.atan2(50, focal)) / (2.0 * 0.022))
    assert dx == expected_dx
    assert dy == expected_dy

    # Smaller pixel deltas must produce smaller mouse moves (monotonic), and
    # the near-crosshair regime stays roughly linear.
    assert dx == 75 and dy == 37
    assert abs(dx) < abs(screen_delta_to_mouse(300, 50, 2.0, 0.022, 0.022)[0])

"""Tests for radar occupancy ray-casting (steer toward open, away from black)."""

import numpy as np

from src.movement.occupancy import best_direction, clearance_at, radar_clearances


def _open_east():
    """80x80 brightness map: a lit walkable corridor running east from centre,
    black everywhere else."""
    g = np.zeros((80, 80), dtype=np.uint8)
    g[36:45, 40:79] = 160  # horizontal lit band to the right of (40, 40)
    return g


def test_clearance_open_in_corridor_blocked_otherwise():
    g = _open_east()
    clrs = radar_clearances(g, 40, 40, n_rays=24, max_r=35, start_r=4, black_thresh=45)
    assert clearance_at(clrs, 0) >= 25  # east is open far
    assert clearance_at(clrs, 180) <= 8  # west is black quickly
    assert clearance_at(clrs, 90) <= 10  # south is black quickly


def test_best_direction_picks_the_open_way():
    g = _open_east()
    clrs = radar_clearances(g, 40, 40, n_rays=24, max_r=35, start_r=4, black_thresh=45)
    # heading roughly east already -> stays east
    chosen = best_direction(clrs, motion_deg=0.0)
    assert abs(((chosen - 0 + 180) % 360) - 180) <= 30


def test_best_direction_turns_away_from_a_wall_ahead():
    g = _open_east()
    # travelling WEST (into the black) -> must choose ~east (the open corridor)
    chosen = best_direction(
        radar_clearances(g, 40, 40, n_rays=24, max_r=35, start_r=4, black_thresh=45),
        motion_deg=180.0,
        turn_penalty=0.05,
    )
    assert abs(((chosen - 0 + 180) % 360) - 180) <= 45

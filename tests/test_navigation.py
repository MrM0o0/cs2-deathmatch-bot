"""Tests for A* pathfinding and the face-aware NavigationController."""

import math

from src.movement.navigator import (
    Waypoint, WaypointGraph, NavigationController, compute_turn_calibration,
)


def _grid_graph(n: int = 5) -> WaypointGraph:
    """Build an n x n grid graph, 100px spacing, 4-connected."""
    g = WaypointGraph()

    def wid(r, c):
        return r * n + c

    for r in range(n):
        for c in range(n):
            g.add_waypoint(Waypoint(wid(r, c), c * 100.0, r * 100.0))
    for r in range(n):
        for c in range(n):
            neigh = []
            if r > 0:
                neigh.append(wid(r - 1, c))
            if r < n - 1:
                neigh.append(wid(r + 1, c))
            if c > 0:
                neigh.append(wid(r, c - 1))
            if c < n - 1:
                neigh.append(wid(r, c + 1))
            g.waypoints[wid(r, c)].neighbors = neigh
    return g


# --- A* ---------------------------------------------------------------------

def test_astar_straight_line():
    g = _grid_graph(5)
    path = g.shortest_path(0, 4)  # top row, left to right
    assert path == [0, 1, 2, 3, 4]


def test_astar_corner_is_shortest_length():
    g = _grid_graph(5)
    # From (0,0)=id0 to (4,4)=id24: Manhattan distance 8 hops -> 9 nodes.
    path = g.shortest_path(0, 24)
    assert path[0] == 0 and path[-1] == 24
    assert len(path) == 9


def test_astar_same_node():
    g = _grid_graph(3)
    assert g.shortest_path(4, 4) == [4]


def test_astar_unknown_or_disconnected():
    g = _grid_graph(3)
    assert g.shortest_path(0, 999) == []
    # Add an island with no neighbors -> unreachable.
    g.add_waypoint(Waypoint(500, 9999.0, 9999.0))
    assert g.shortest_path(0, 500) == []


# --- Heading estimation -----------------------------------------------------

def test_heading_from_motion_east():
    g = _grid_graph(5)
    nav = NavigationController(g, move_threshold=2.0)
    # Walk east (x increasing) -> heading ~0 deg.
    for x in range(0, 60, 10):
        nav.update((float(x), 0.0))
    assert nav.heading is not None
    assert abs(nav.heading) < 1e-6


def test_heading_from_motion_south():
    g = _grid_graph(5)
    nav = NavigationController(g, move_threshold=2.0)
    # y increasing == south == +90 deg in this (y-down) frame.
    for y in range(0, 60, 10):
        nav.update((0.0, float(y)))
    assert nav.heading is not None
    assert abs(nav.heading - 90.0) < 1e-6


def test_heading_not_set_when_still():
    g = _grid_graph(5)
    nav = NavigationController(g, move_threshold=5.0)
    for _ in range(6):
        nav.update((10.0, 10.0))  # not moving
    assert nav.heading is None


# --- Turn command sign ------------------------------------------------------

def test_turn_toward_target_on_the_right():
    g = _grid_graph(5)
    nav = NavigationController(g, move_threshold=2.0, turn_gain=1.0, max_turn=999)
    # Establish heading = east (0 deg).
    for x in range(0, 60, 10):
        nav.update((float(x), 0.0))
    # Goal forced due south of current pos: target heading +90, error +90.
    nav._route = [g.nearest(50, 0).id, g.nearest(50, 400).id]
    nav._route_idx = 1
    cmd = nav.update((50.0, 0.0))
    assert cmd["yaw_error_deg"] > 0     # target is clockwise of heading
    assert cmd["turn_x"] > 0            # turning right (positive mouse-x)
    assert cmd["forward"] is True       # still advancing while rounding


def test_turn_inverts_with_flag():
    g = _grid_graph(5)
    base = NavigationController(g, move_threshold=2.0, turn_gain=1.0, max_turn=999)
    inv = NavigationController(g, move_threshold=2.0, turn_gain=1.0,
                              max_turn=999, invert_turn=True)
    for nav in (base, inv):
        for x in range(0, 60, 10):
            nav.update((float(x), 0.0))
        nav._route = [g.nearest(50, 0).id, g.nearest(50, 400).id]
        nav._route_idx = 1
    c1 = base.update((50.0, 0.0))
    c2 = inv.update((50.0, 0.0))
    assert c1["turn_x"] == -c2["turn_x"]


def test_target_behind_keeps_walking_while_turning():
    g = _grid_graph(5)
    nav = NavigationController(g, move_threshold=2.0)
    for x in range(0, 60, 10):
        nav.update((float(x), 0.0))  # heading east
    # Goal due west -> error ~180. Must KEEP walking (so heading stays live)
    # while turning hard, not freeze in place -- that caused the spin spiral.
    nav._route = [g.nearest(50, 0).id, g.nearest(0, 0).id]
    nav._route_idx = 1
    cmd = nav.update((50.0, 0.0))
    assert abs(cmd["yaw_error_deg"]) > 150
    assert cmd["forward"] is True
    assert cmd["turn_x"] != 0  # turning hard toward the target


def test_no_waypoints_is_idle():
    nav = NavigationController(WaypointGraph())
    cmd = nav.update((0.0, 0.0))
    assert cmd["has_route"] is False
    assert cmd["forward"] is False and cmd["turn_x"] == 0


# --- Auto-calibration -------------------------------------------------------

def test_calibration_same_sign_no_invert():
    # +400 counts produced +40deg rotation -> assumption holds, gain 10 c/deg.
    inv, gain = compute_turn_calibration(0.0, 40.0, 400)
    assert inv is False
    assert abs(gain - 10.0) < 1e-6


def test_calibration_opposite_sign_inverts():
    # +400 counts produced -40deg rotation -> minimap rotates opposite -> invert.
    inv, gain = compute_turn_calibration(0.0, -40.0, 400)
    assert inv is True
    assert abs(gain - 10.0) < 1e-6


def test_calibration_wraps_across_180():
    # 170 -> -170 is a +20deg step (clockwise) through the wrap, not -340.
    inv, gain = compute_turn_calibration(170.0, -170.0, 200)
    assert inv is False
    assert abs(gain - 10.0) < 1e-6


def test_calibration_too_small_returns_none():
    assert compute_turn_calibration(0.0, 2.0, 400) is None
    assert compute_turn_calibration(0.0, 40.0, 0) is None

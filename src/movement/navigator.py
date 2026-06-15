"""Waypoint-based navigation system."""

import heapq
import json
import math
import os
import random
from collections import deque

from src.utils.math_helpers import (
    angle_between,
    clamp,
    distance,
    normalize_angle,
)


def compute_turn_calibration(
    heading_before: float, heading_after: float, mouse_counts: int
) -> tuple[bool, float] | None:
    """Derive nav turn-sign and gain from a known turn experiment.

    The bot walks forward, we read its heading (from blip motion), issue a
    known horizontal mouse move, then read the heading again. From the observed
    rotation we learn the two things that can't be known offline:

      - ``invert_turn``: the NavigationController assumes a positive mouse-x
        increases heading (clockwise in the y-down minimap frame). If the
        observed rotation went the other way, the sign must be flipped.
      - ``turn_gain``: mouse counts to emit per degree of heading error.

    Args:
        heading_before: Heading (deg) measured before the test turn.
        heading_after: Heading (deg) measured after the test turn.
        mouse_counts: The horizontal mouse counts issued during the test
            (sign included).

    Returns:
        ``(invert_turn, turn_gain)``, or ``None`` if the turn was too small to
        measure reliably (blip barely rotated -- redo with a bigger move).
    """
    delta = normalize_angle(heading_after - heading_before)
    if abs(delta) < 5.0 or mouse_counts == 0:
        return None
    invert_turn = (delta * mouse_counts) < 0
    turn_gain = abs(mouse_counts) / abs(delta)
    return invert_turn, turn_gain


class Waypoint:
    """A navigation waypoint on a map."""

    __slots__ = ("id", "x", "y", "neighbors", "tags")

    def __init__(
        self,
        wp_id: int,
        x: float,
        y: float,
        neighbors: list[int] | None = None,
        tags: list[str] | None = None,
    ):
        self.id = wp_id
        self.x = x
        self.y = y
        self.neighbors = neighbors or []
        self.tags = tags or []

    def pos(self) -> tuple[float, float]:
        return (self.x, self.y)


class WaypointGraph:
    """Graph of waypoints for map navigation."""

    def __init__(self):
        self.waypoints: dict[int, Waypoint] = {}

    def add_waypoint(self, wp: Waypoint) -> None:
        self.waypoints[wp.id] = wp

    def nearest(self, x: float, y: float) -> Waypoint | None:
        """Find the nearest waypoint to a position."""
        if not self.waypoints:
            return None
        return min(self.waypoints.values(), key=lambda wp: distance((x, y), wp.pos()))

    def random_neighbor(self, wp_id: int) -> Waypoint | None:
        """Get a random neighbor of a waypoint."""
        wp = self.waypoints.get(wp_id)
        if not wp or not wp.neighbors:
            return None
        neighbor_id = random.choice(wp.neighbors)
        return self.waypoints.get(neighbor_id)

    def shortest_path(self, start_id: int, goal_id: int) -> list[int]:
        """A* shortest path between two waypoints.

        Uses straight-line distance as both edge cost and heuristic, so it
        always returns an optimal route when one exists.

        Returns:
            List of waypoint ids from start to goal (inclusive), or an empty
            list if either endpoint is unknown or no route exists.
        """
        if start_id not in self.waypoints or goal_id not in self.waypoints:
            return []
        if start_id == goal_id:
            return [start_id]

        goal = self.waypoints[goal_id]

        def h(wp_id: int) -> float:
            return distance(self.waypoints[wp_id].pos(), goal.pos())

        # (f_score, counter, wp_id); counter breaks ties deterministically.
        counter = 0
        open_heap: list[tuple[float, int, int]] = [(h(start_id), counter, start_id)]
        came_from: dict[int, int] = {}
        g_score: dict[int, float] = {start_id: 0.0}
        closed: set[int] = set()

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current == goal_id:
                # Reconstruct path.
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            if current in closed:
                continue
            closed.add(current)

            cur_wp = self.waypoints[current]
            for nb_id in cur_wp.neighbors:
                nb = self.waypoints.get(nb_id)
                if nb is None or nb_id in closed:
                    continue
                tentative = g_score[current] + distance(cur_wp.pos(), nb.pos())
                if tentative < g_score.get(nb_id, math.inf):
                    came_from[nb_id] = current
                    g_score[nb_id] = tentative
                    counter += 1
                    heapq.heappush(open_heap, (tentative + h(nb_id), counter, nb_id))

        return []

    def save(self, path: str) -> None:
        """Save waypoint graph to JSON."""
        data = []
        for wp in self.waypoints.values():
            data.append(
                {
                    "id": wp.id,
                    "x": wp.x,
                    "y": wp.y,
                    "neighbors": wp.neighbors,
                    "tags": wp.tags,
                }
            )
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load waypoint graph from JSON."""
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        for item in data:
            wp = Waypoint(
                wp_id=item["id"],
                x=item["x"],
                y=item["y"],
                neighbors=item.get("neighbors", []),
                tags=item.get("tags", []),
            )
            self.add_waypoint(wp)


class NavigationController:
    """Face-aware waypoint navigation.

    The old approach turned the view using only the x-component of the
    direction-to-waypoint vector, which ignored where the bot was actually
    facing -- so it could never reliably walk a path. This controller fixes
    that by:

      1. Estimating the bot's current heading from how its minimap blip has
         *moved* over the last few frames (more reliable than reading the
         arrow's rotation, which is noisy and direction-ambiguous).
      2. Routing to a goal waypoint with A* over the graph.
      3. Emitting an explicit turn + key command each tick to steer toward the
         next node on that route.

    Coordinate convention (must match the recorder): minimap pixels, x to the
    right, y downward. Headings are degrees via atan2(dy, dx): 0 = east,
    +90 = south. ``invert_turn`` / ``turn_gain`` absorb the one unknown that
    can't be derived offline -- how a positive mouse-x maps to minimap rotation
    -- and are pinned down during in-game calibration.
    """

    def __init__(
        self,
        graph: WaypointGraph,
        reach_dist: float = 40.0,
        turn_gain: float = 1.6,
        max_turn: int = 22,
        move_threshold: float = 3.0,
        history: int = 30,
        invert_turn: bool = False,
        turn_deadzone: float = 18.0,
    ):
        """
        Args:
            graph: Waypoint graph to navigate.
            reach_dist: Distance (minimap px) at which a node counts as reached.
            turn_gain: Mouse counts emitted per degree of heading error.
            max_turn: Clamp on per-tick turn (mouse counts) for smooth motion.
            move_threshold: Min displacement (px) over the history window before
                a fresh heading estimate is trusted.
            history: How many recent positions to keep for heading estimation.
                The radar shows the whole map so the dot moves only a few px/sec
                -- the window must span ~1s (≈30 ticks) to see real movement.
            invert_turn: Flip turn sign if the minimap rotates opposite to the
                assumed convention (set during calibration).
        """
        self.graph = graph
        self.reach_dist = reach_dist
        self.turn_gain = turn_gain
        self.max_turn = max_turn
        self.move_threshold = move_threshold
        self.invert_turn = invert_turn
        self.turn_deadzone = turn_deadzone

        self._positions: deque[tuple[float, float]] = deque(maxlen=max(2, history))
        self._heading: float | None = None
        self._route: list[int] = []
        self._route_idx = 0

    def has_waypoints(self) -> bool:
        return len(self.graph.waypoints) > 0

    @property
    def heading(self) -> float | None:
        """Last trusted heading in degrees, or None if not yet moving."""
        return self._heading

    def reset_route(self) -> None:
        self._route = []
        self._route_idx = 0

    def update(self, pos: tuple[float, float]) -> dict:
        """Advance navigation by one tick.

        Args:
            pos: Current (x, y) minimap-pixel position of the bot.

        Returns:
            Command dict: ``turn_x`` (mouse counts), ``forward``/``back``/
            ``left``/``right`` (bool), plus ``heading_deg``, ``yaw_error_deg``,
            and ``has_route`` for debugging/overlay.
        """
        idle = {
            "turn_x": 0,
            "forward": False,
            "back": False,
            "left": False,
            "right": False,
            "heading_deg": self._heading,
            "yaw_error_deg": 0.0,
            "has_route": False,
        }

        if not self.has_waypoints():
            return idle

        self._positions.append((float(pos[0]), float(pos[1])))
        self._update_heading()

        target = self._current_route_target(pos)
        if target is None:
            # No reachable goal yet -- nudge forward to start moving so we can
            # bootstrap a heading estimate.
            return {**idle, "forward": True}

        target_heading = angle_between(pos, target.pos())

        # Without a heading estimate yet, walk forward to get one.
        if self._heading is None:
            return {**idle, "forward": True, "has_route": True}

        yaw_error = normalize_angle(target_heading - self._heading)

        # Deadzone: when roughly aligned, stop turning and walk straight. The
        # motion-derived heading lags, so turning right up to 0 error overshoots
        # and the bot circles; walking straight lets the estimate settle first.
        if abs(yaw_error) <= self.turn_deadzone:
            turn = 0
        else:
            turn = int(clamp(yaw_error * self.turn_gain, -self.max_turn, self.max_turn))
            if self.invert_turn:
                turn = -turn

        cmd = {
            "turn_x": turn,
            "forward": False,
            "back": False,
            "left": False,
            "right": False,
            "heading_deg": self._heading,
            "yaw_error_deg": yaw_error,
            "has_route": True,
        }

        # ALWAYS keep walking while turning. Heading is derived from how the
        # dot *moves*, so if the bot ever stops to "turn in place" the estimate
        # freezes and it spins forever (target stays behind). Walking + turning
        # arcs toward the target and keeps the heading estimate alive.
        cmd["forward"] = True
        abs_err = abs(yaw_error)
        if abs_err > 35:
            # Strafe toward the turn to tighten the arc / round corners.
            if yaw_error > 0:
                cmd["right"] = True
            else:
                cmd["left"] = True
        return cmd

    def _update_heading(self) -> None:
        """Estimate heading from net displacement over the position window."""
        if len(self._positions) < 2:
            return
        ox, oy = self._positions[0]
        nx, ny = self._positions[-1]
        if distance((ox, oy), (nx, ny)) >= self.move_threshold:
            self._heading = angle_between((ox, oy), (nx, ny))

    def _current_route_target(self, pos: tuple[float, float]) -> Waypoint | None:
        """Return the next waypoint to walk toward, (re)planning as needed."""
        # Advance past nodes we've already reached.
        while self._route_idx < len(self._route):
            wp = self.graph.waypoints.get(self._route[self._route_idx])
            if wp is None:
                self._route_idx += 1
                continue
            if distance(pos, wp.pos()) < self.reach_dist:
                self._route_idx += 1
                continue
            return wp

        # Route finished or empty -> plan a new one to a random far node.
        self._plan_new_route(pos)
        if self._route_idx < len(self._route):
            return self.graph.waypoints.get(self._route[self._route_idx])
        return None

    def _plan_new_route(self, pos: tuple[float, float]) -> None:
        start = self.graph.nearest(pos[0], pos[1])
        if start is None:
            self._route, self._route_idx = [], 0
            return
        # Prefer a goal that's a real distance away so the bot actually travels.
        candidates = [
            wp
            for wp in self.graph.waypoints.values()
            if distance(start.pos(), wp.pos()) > self.reach_dist * 4
        ]
        if not candidates:
            candidates = [wp for wp in self.graph.waypoints.values() if wp.id != start.id]
        if not candidates:
            self._route, self._route_idx = [], 0
            return
        goal = random.choice(candidates)
        self._route = self.graph.shortest_path(start.id, goal.id)
        self._route_idx = 1 if len(self._route) > 1 else 0

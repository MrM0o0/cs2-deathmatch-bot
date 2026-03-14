"""Waypoint-based navigation system."""

import json
import os
import random
import math
import time

from src.utils.math_helpers import distance


class Waypoint:
    """A navigation waypoint on a map."""

    __slots__ = ("id", "x", "y", "neighbors", "tags")

    def __init__(self, wp_id: int, x: float, y: float,
                 neighbors: list[int] | None = None,
                 tags: list[str] | None = None):
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
        return min(self.waypoints.values(),
                   key=lambda wp: distance((x, y), wp.pos()))

    def random_neighbor(self, wp_id: int) -> Waypoint | None:
        """Get a random neighbor of a waypoint."""
        wp = self.waypoints.get(wp_id)
        if not wp or not wp.neighbors:
            return None
        neighbor_id = random.choice(wp.neighbors)
        return self.waypoints.get(neighbor_id)

    def save(self, path: str) -> None:
        """Save waypoint graph to JSON."""
        data = []
        for wp in self.waypoints.values():
            data.append({
                "id": wp.id,
                "x": wp.x,
                "y": wp.y,
                "neighbors": wp.neighbors,
                "tags": wp.tags,
            })
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load waypoint graph from JSON."""
        if not os.path.exists(path):
            return
        with open(path, "r") as f:
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


class Navigator:
    """Waypoint-based map navigation."""

    def __init__(self, graph: WaypointGraph | None = None,
                 reach_distance: float = 50):
        self.graph = graph or WaypointGraph()
        self.reach_dist = reach_distance
        self._current_wp: Waypoint | None = None
        self._target_wp: Waypoint | None = None
        self._path: list[Waypoint] = []

    def set_position(self, x: float, y: float) -> None:
        """Update current position and snap to nearest waypoint."""
        if not self.graph.waypoints:
            return
        nearest = self.graph.nearest(x, y)
        if nearest and distance((x, y), nearest.pos()) < self.reach_dist:
            self._current_wp = nearest

    def get_movement_direction(self, current_x: float, current_y: float) -> tuple[float, float] | None:
        """Get direction to move toward next waypoint.

        Returns:
            (dx, dy) normalized direction, or None if no destination.
        """
        if not self._target_wp:
            self._pick_next_target()

        if not self._target_wp:
            return None

        dx = self._target_wp.x - current_x
        dy = self._target_wp.y - current_y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < self.reach_dist:
            # Reached target, pick next
            self._current_wp = self._target_wp
            self._pick_next_target()
            if not self._target_wp:
                return None
            dx = self._target_wp.x - current_x
            dy = self._target_wp.y - current_y
            dist = math.sqrt(dx * dx + dy * dy)

        if dist < 1:
            return None

        return (dx / dist, dy / dist)

    def _pick_next_target(self) -> None:
        """Pick next waypoint to navigate to."""
        if self._current_wp:
            self._target_wp = self.graph.random_neighbor(self._current_wp.id)
        elif self.graph.waypoints:
            # No current waypoint, pick random one
            self._target_wp = random.choice(list(self.graph.waypoints.values()))

    def has_waypoints(self) -> bool:
        return len(self.graph.waypoints) > 0

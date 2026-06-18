"""Landmark-based localization: which map area am I in?

Consumes landmark detections (same YOLO `Detection` shape as the player
detector, but trained on map landmarks like ``goose``/``mid_doors``) and reports
the current semantic area from the hand-authored area graph
(``config/maps/<map>_areas.json``).

This is deliberately independent of the radar dot: in deathmatch the minimap is
full of other players' dots and the brightest-blob reader grabs the wrong one
(see CLAUDE.md / project notes). A landmark in the first-person view is an
unambiguous "I am here" signal that the dot soup can't corrupt.

Design:
- Each landmark votes for every area whose graph entry lists it, weighted by
  detection confidence and (optionally) on-screen size as a proximity proxy.
- A graph-adjacency prior penalises areas that aren't the current area or one of
  its neighbours -- you can't teleport across the map, so an ambiguous landmark
  is resolved toward where you could actually be.
- A temporal confirm filter (like the detection ConfirmationFilter) requires a
  new area to win N frames in a row before we commit, so a single noisy frame
  never flips the reported area.
"""

import json
import math


class Localizer:
    """Tracks the current map area from landmark detections."""

    def __init__(
        self,
        area_landmarks: dict[str, list[str]],
        neighbors: dict[str, list[str]],
        confirm_frames: int = 3,
        jump_penalty: float = 0.3,
    ):
        """
        Args:
            area_landmarks: area name -> landmark class names visible there.
            neighbors: area name -> directly connected area names.
            confirm_frames: a new area must win this many consecutive frames
                before it becomes the committed current area.
            jump_penalty: score multiplier for areas that are neither the
                current area nor a neighbour (0..1; lower = stickier).
        """
        self.neighbors = neighbors
        self.confirm_frames = confirm_frames
        self.jump_penalty = jump_penalty

        # reverse index: landmark class -> areas that contain it
        self.landmark_to_areas: dict[str, list[str]] = {}
        for area, landmarks in area_landmarks.items():
            for lm in landmarks:
                self.landmark_to_areas.setdefault(lm, []).append(area)

        self.current_area: str | None = None
        self.confidence: float = 0.0  # share of vote mass on the winning area
        self.stale: int = 0  # frames since we last saw any known landmark
        self._candidate: str | None = None
        self._candidate_count: int = 0

    @classmethod
    def from_file(cls, path: str, **kwargs) -> "Localizer":
        """Build from a ``<map>_areas.json`` file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        areas = data["areas"]
        area_landmarks = {name: rec.get("landmarks", []) for name, rec in areas.items()}
        neighbors = {name: rec.get("neighbors", []) for name, rec in areas.items()}
        loc = cls(area_landmarks, neighbors, **kwargs)
        loc._areas = areas  # keep full records for downstream (danger_from, etc.)
        return loc

    def area_record(self, area: str | None = None) -> dict:
        """Full graph record (neighbors, danger_from, ...) for an area."""
        area = area or self.current_area
        if area is None or not hasattr(self, "_areas"):
            return {}
        return self._areas.get(area, {})

    def _allowed(self) -> set[str] | None:
        """Areas reachable from the current one (current + neighbours)."""
        if self.current_area is None:
            return None  # cold start: anywhere is allowed
        return {self.current_area} | set(self.neighbors.get(self.current_area, []))

    def update(self, detections, frame_area: float | None = None) -> str | None:
        """Feed one frame of landmark detections; returns the current area.

        Args:
            detections: iterable of objects with ``.class_name``, ``.confidence``
                and (optionally) ``.area`` -- the YOLO ``Detection`` type fits.
            frame_area: total frame pixel area, used to turn a landmark's
                on-screen size into a 0..1 proximity weight. If None, only
                confidence is used.
        """
        scores: dict[str, float] = {}
        for det in detections:
            areas = self.landmark_to_areas.get(det.class_name)
            if not areas:
                continue
            weight = det.confidence
            if frame_area and getattr(det, "area", None):
                prox = min(1.0, det.area / frame_area)
                # never zero-out a small/distant landmark, just trust it less
                weight *= 0.4 + 0.6 * math.sqrt(prox)
            for area in areas:
                scores[area] = scores.get(area, 0.0) + weight

        if not scores:
            self.stale += 1
            self._candidate = None
            self._candidate_count = 0
            return self.current_area

        self.stale = 0

        # graph-adjacency prior: discourage impossible jumps
        allowed = self._allowed()
        if allowed is not None:
            for area in list(scores):
                if area not in allowed:
                    scores[area] *= self.jump_penalty

        best = max(scores, key=scores.get)
        total = sum(scores.values())
        self.confidence = scores[best] / total if total else 0.0

        if best == self.current_area:
            # reaffirmed; clear any pending switch
            self._candidate = None
            self._candidate_count = 0
            return self.current_area

        # candidate must win consecutive frames before we commit the switch
        if best == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = best
            self._candidate_count = 1

        if self._candidate_count >= self.confirm_frames:
            self.current_area = best
            self._candidate = None
            self._candidate_count = 0

        return self.current_area

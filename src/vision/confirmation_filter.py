"""Temporal confirmation filter — requires detections to persist across multiple
frames before they are treated as real targets.

Brief false positives (fire hydrants, dark patches, etc.) get filtered out
because they only appear for 1-2 frames. Real players persist for many frames
in roughly the same screen location.
"""

import math
from src.vision.detector import Detection


class TrackedObject:
    """A detection being tracked across frames."""

    __slots__ = (
        "class_name", "cx", "cy", "width", "height", "confidence",
        "frames_seen", "frames_missing", "confirmed", "_last_detection",
    )

    def __init__(self, detection: Detection):
        self.class_name = detection.class_name
        self.cx, self.cy = detection.center
        self.width = detection.width
        self.height = detection.height
        self.confidence = detection.confidence
        self.frames_seen = 1
        self.frames_missing = 0
        self.confirmed = False
        self._last_detection = detection

    def update(self, detection: Detection) -> None:
        """Update tracker with a new matching detection."""
        self.cx, self.cy = detection.center
        self.width = detection.width
        self.height = detection.height
        self.confidence = detection.confidence
        self.frames_seen += 1
        self.frames_missing = 0
        self._last_detection = detection

    def mark_missing(self) -> None:
        """Called when no matching detection was found this frame."""
        self.frames_missing += 1

    @property
    def last_detection(self) -> Detection:
        return self._last_detection

    def distance_to(self, detection: Detection) -> float:
        """Euclidean distance between this tracker's center and a detection's center."""
        dcx, dcy = detection.center
        return math.hypot(self.cx - dcx, self.cy - dcy)

    def size_ratio(self, detection: Detection) -> float:
        """How similar the bounding box size is (1.0 = identical)."""
        area_self = self.width * self.height
        area_det = detection.width * detection.height
        if area_self == 0 or area_det == 0:
            return 0.0
        ratio = min(area_self, area_det) / max(area_self, area_det)
        return ratio


class ConfirmationFilter:
    """Filters detections by requiring temporal consistency.

    A detection must appear in `min_confirm_frames` out of the last few frames
    in roughly the same screen location to be considered confirmed.

    Args:
        min_confirm_frames: Consecutive frames needed to confirm a target.
        max_missing_frames: How many frames a tracker survives without a match.
        match_distance: Max pixel distance between frames to count as same target.
        match_size_ratio: Min size similarity (0-1) to count as same target.
    """

    def __init__(
        self,
        min_confirm_frames: int = 3,
        max_missing_frames: int = 2,
        match_distance: float = 150.0,
        match_size_ratio: float = 0.4,
    ):
        self.min_confirm_frames = min_confirm_frames
        self.max_missing_frames = max_missing_frames
        self.match_distance = match_distance
        self.match_size_ratio = match_size_ratio
        self._trackers: list[TrackedObject] = []

    def update(self, detections: list[Detection]) -> list[Detection]:
        """Process a new frame's detections and return only confirmed ones.

        Call this every frame with the raw detector output.
        Returns detections that have persisted long enough to be trusted.
        """
        # Match incoming detections to existing trackers
        used_detections: set[int] = set()
        matched_trackers: set[int] = set()

        # Greedy matching: closest pairs first
        pairs = []
        for ti, tracker in enumerate(self._trackers):
            for di, det in enumerate(detections):
                if det.class_name != tracker.class_name:
                    continue
                dist = tracker.distance_to(det)
                if dist > self.match_distance:
                    continue
                if tracker.size_ratio(det) < self.match_size_ratio:
                    continue
                pairs.append((dist, ti, di))

        pairs.sort(key=lambda x: x[0])

        for _, ti, di in pairs:
            if ti in matched_trackers or di in used_detections:
                continue
            self._trackers[ti].update(detections[di])
            matched_trackers.add(ti)
            used_detections.add(di)

        # Mark unmatched trackers as missing
        for ti, tracker in enumerate(self._trackers):
            if ti not in matched_trackers:
                tracker.mark_missing()

        # Create new trackers for unmatched detections
        for di, det in enumerate(detections):
            if di not in used_detections:
                self._trackers.append(TrackedObject(det))

        # Remove stale trackers
        self._trackers = [
            t for t in self._trackers
            if t.frames_missing <= self.max_missing_frames
        ]

        # Update confirmed status
        for tracker in self._trackers:
            if tracker.frames_seen >= self.min_confirm_frames:
                tracker.confirmed = True

        # Return confirmed detections
        confirmed = [
            t.last_detection for t in self._trackers
            if t.confirmed and t.frames_missing == 0
        ]

        return confirmed

    def reset(self) -> None:
        """Clear all tracked objects."""
        self._trackers.clear()

    @property
    def tracker_count(self) -> int:
        """Number of active trackers (confirmed + unconfirmed)."""
        return len(self._trackers)

    @property
    def confirmed_count(self) -> int:
        """Number of currently confirmed targets."""
        return sum(1 for t in self._trackers if t.confirmed)

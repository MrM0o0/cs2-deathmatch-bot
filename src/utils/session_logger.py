"""Per-run session logging so bot behaviour can be diagnosed after the fact.

Each run creates a timestamped folder under ``logs/`` containing:
  - ``session.jsonl`` -- one compact JSON record per logged tick (state,
    decision, position, heading, nav error, detection counts, fps).
  - ``frames/`` -- annotated frames saved at a throttled interval.

The point: "looks like a player" can only be judged in-game, but the *why*
behind a weird moment (spun in a circle, walked into a wall, never engaged)
is recoverable from this folder without anyone watching live. Zip the folder,
send it over, and the behaviour is reconstructable from data.
"""

import json
import os
import time
from datetime import datetime

try:
    import cv2
except ImportError:
    cv2 = None


class SessionLogger:
    """Lightweight append-only logger for a single bot run."""

    def __init__(self, root: str, frame_interval: float = 2.0, enabled: bool = True):
        """
        Args:
            root: Project root; logs go under ``<root>/logs/<timestamp>/``.
            frame_interval: Min seconds between saved frames (throttle disk use).
            enabled: If False, every method is a cheap no-op.
        """
        self.enabled = enabled
        self.frame_interval = frame_interval
        self._last_frame_time = 0.0
        self._frame_count = 0
        self._tick_count = 0
        self._fh = None
        self.session_dir = ""

        if not self.enabled:
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(root, "logs", stamp)
        self.frames_dir = os.path.join(self.session_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        self._fh = open(os.path.join(self.session_dir, "session.jsonl"), "w", encoding="utf-8")
        print(f"[SessionLogger] Logging to {self.session_dir}")

    def log_tick(self, **fields) -> None:
        """Append one JSON record for this tick. Cheap; safe to call every tick."""
        if not self.enabled or self._fh is None:
            return
        self._tick_count += 1
        record = {"t": round(time.time(), 3), "tick": self._tick_count}
        record.update(fields)
        self._fh.write(json.dumps(record, separators=(",", ":"), default=_jsonable))
        self._fh.write("\n")

    def maybe_save_frame(self, frame, label: str = "") -> None:
        """Save an annotated frame, throttled to ``frame_interval`` seconds."""
        if not self.enabled or frame is None or cv2 is None:
            return
        now = time.time()
        if now - self._last_frame_time < self.frame_interval:
            return
        self._last_frame_time = now
        img = frame
        if label:
            img = frame.copy()
            cv2.putText(
                img, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
            )
        path = os.path.join(self.frames_dir, f"f_{self._frame_count:05d}.jpg")
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        self._frame_count += 1

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            print(
                f"[SessionLogger] Closed. {self._tick_count} ticks, "
                f"{self._frame_count} frames in {self.session_dir}"
            )


def _jsonable(obj):
    """Best-effort fallback so logging never crashes the bot."""
    return str(obj)

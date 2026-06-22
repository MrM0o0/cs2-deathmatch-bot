"""Detect when the bot is wedged and script an escape (the stuck reflex).

Behavioural cloning has a known blind spot: the human never demonstrated being
jammed in a corner, so the model has no idea how to get out of one. This module
is the cheap safety net for that. Two pieces:

- StuckDetector: vision-only. If the bot is *commanding* movement but the view
  hasn't changed over the last ~timeout seconds, it's wedged. Comparing against a
  frame from a full `timeout` ago (not just the previous frame) means genuinely
  slow movement still registers as movement -- only a truly frozen view trips it.
- EscapeManeuver: a short scripted break-out (reverse out, sweep the view a big
  angle, a hop in case it's a ledge), emitted as the same command dict the BC
  policy returns so it can simply override the policy while active. The turn
  direction alternates each time so repeated stucks don't loop into the same wall.
"""

import time
from collections import deque

import numpy as np


class StuckDetector:
    """Flags when the bot presses move keys but the view stops changing."""

    def __init__(self, timeout: float = 2.0, similarity_threshold: float = 0.98):
        """
        Args:
            timeout: Seconds of a frozen view (while moving) before declaring stuck.
            similarity_threshold: How alike two frames must be to count as "no
                movement". Internally this is a max allowed pixel change of
                (1 - similarity_threshold); 0.98 -> the view must change by >2%.
        """
        self.timeout = timeout
        self.min_change = max(1e-4, 1.0 - similarity_threshold)
        self._buf: deque[tuple[float, np.ndarray]] = deque()
        self._is_stuck = False
        self._recovery_count = 0

    def update(self, frame: np.ndarray, is_moving: bool, now: float | None = None) -> bool:
        """Feed the current frame + whether move keys are held. Returns True once
        the view has been frozen for `timeout` seconds while moving."""
        now = time.perf_counter() if now is None else now

        if not is_moving:
            # Standing still isn't stuck -- reset the window.
            self._buf.clear()
            self._is_stuck = False
            return False

        self._buf.append((now, self._downsample(frame)))
        # Keep a little more than one window of history.
        while self._buf and now - self._buf[0][0] > self.timeout * 1.5:
            self._buf.popleft()

        oldest_t, oldest_f = self._buf[0]
        if now - oldest_t < self.timeout:
            return False  # not enough history yet to judge

        change = self._frame_change(self._buf[-1][1], oldest_f)
        if change < self.min_change:
            self._is_stuck = True
            self._recovery_count += 1
            self._buf.clear()  # require a fresh window before flagging again
            return True

        self._is_stuck = False
        return False

    @staticmethod
    def _frame_change(a: np.ndarray, b: np.ndarray) -> float:
        """Mean absolute pixel difference, normalised to [0, 1]."""
        if a.shape != b.shape:
            return 1.0
        return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)

    @staticmethod
    def _downsample(frame: np.ndarray) -> np.ndarray:
        """Coarse, fast view of the frame for cheap comparison."""
        return frame[::16, ::16].copy()

    def reset(self) -> None:
        self._buf.clear()
        self._is_stuck = False

    @property
    def is_stuck(self) -> bool:
        return self._is_stuck

    @property
    def recovery_count(self) -> int:
        return self._recovery_count


class EscapeManeuver:
    """A short scripted break-out, emitted as a BC-style command dict.

    Turn magnitude is angle-sized via counts-per-degree (1 / (sens * m_yaw)) --
    the hard-won lesson that bot turns must be sized by angle, not magic counts.
    """

    def __init__(
        self,
        cpd: float = 53.5,  # mouse counts per degree at sens 0.85, m_yaw 0.022
        turn_deg: float = 130.0,  # big turn so it faces a genuinely new direction
        back_time: float = 0.4,  # reverse out of the wall first
        duration: float = 1.2,  # total maneuver length
        jump: bool = True,  # a hop in case it's a ledge/box
    ):
        self.back_time = back_time
        self.duration = duration
        self.jump = jump
        self.turn_rate = (turn_deg * cpd) / duration  # counts/sec, spread over the move
        self._dir = 1
        self._elapsed: float | None = None

    @property
    def active(self) -> bool:
        return self._elapsed is not None

    def start(self) -> None:
        """Begin an escape; alternate turn direction so repeats don't loop."""
        self._dir = -self._dir
        self._elapsed = 0.0

    def update(self, dt: float) -> dict | None:
        """Advance by dt seconds; return the override command, or None when done."""
        if self._elapsed is None:
            return None
        self._elapsed += dt
        if self._elapsed >= self.duration:
            self._elapsed = None
            return None
        e = self._elapsed
        return {
            "forward": False,
            "back": e < self.back_time,
            "left": False,
            "right": False,
            "crouch": False,
            "turn_x": int(self._dir * self.turn_rate * dt),
            "jump": self.jump and (0.15 < e < 0.45),
        }

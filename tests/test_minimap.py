"""Tests for hue-agnostic player-dot detection on the minimap.

CS2 reassigns the radar blip colour between matches, so detection must key on
"brightest, most-saturated blob", not a fixed hue.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.vision.minimap import MinimapReader


def _frame_with_dot(bgr, cx, cy, size=256, radius=6, origin=(16, 16)):
    """Full frame with the minimap at `origin` and one bright dot in it."""
    ox, oy = origin
    frame = np.zeros((oy + size + 20, ox + size + 20, 3), dtype=np.uint8)
    # dull grey terrain so only the dot is bright+saturated
    frame[oy : oy + size, ox : ox + size] = (60, 60, 62)
    cv2.circle(frame, (ox + cx, oy + cy), radius, bgr, -1)
    return frame


def test_detects_yellow_dot():
    r = MinimapReader(16, 16, 256)
    pos, _ = r.read(_frame_with_dot((0, 255, 255), 112, 184))  # yellow (BGR)
    assert abs(pos[0] - 112) <= 2 and abs(pos[1] - 184) <= 2


def test_detects_cyan_dot_same_reader():
    """Same reader, different match colour -> still found (no reconfig)."""
    r = MinimapReader(16, 16, 256)
    pos, _ = r.read(_frame_with_dot((255, 255, 0), 70, 90))  # cyan (BGR)
    assert abs(pos[0] - 70) <= 2 and abs(pos[1] - 90) <= 2


def test_ignores_dull_terrain():
    """A frame with no bright dot keeps the last position (no false lock)."""
    r = MinimapReader(16, 16, 256)
    blank = np.full((300, 300, 3), (60, 60, 62), dtype=np.uint8)
    pos, _ = r.read(blank)
    assert pos == (128, 128)  # fallback / last position unchanged


def test_lock_prefers_nearby_blob_over_larger_far_one():
    """Once locked, a small dot near the last pos beats a big far marker."""
    r = MinimapReader(16, 16, 256, lock_radius=40)
    # lock onto the dot near (60,60)
    r.read(_frame_with_dot((0, 255, 255), 60, 60, radius=6))
    assert r._locked
    # now a frame with a BIGGER bright blob far away + the player dot nearby
    frame = _frame_with_dot((0, 255, 255), 62, 61, radius=5)
    cv2.circle(frame, (16 + 200, 16 + 200), 12, (0, 255, 255), -1)  # big far marker
    pos, _ = r.read(frame)
    assert abs(pos[0] - 62) <= 3 and abs(pos[1] - 61) <= 3  # stuck to the player

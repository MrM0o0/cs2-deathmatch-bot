"""Tests for the stuck reflex: detector + escape maneuver."""

import numpy as np

from src.movement.stuck_detector import EscapeManeuver, StuckDetector


def _frame(value: int) -> np.ndarray:
    """A uniform 32x32 BGR frame of the given brightness."""
    return np.full((32, 32, 3), value, dtype=np.uint8)


# --- StuckDetector ----------------------------------------------------------


def test_not_moving_is_never_stuck():
    det = StuckDetector(timeout=2.0)
    for t in range(10):
        assert det.update(_frame(0), is_moving=False, now=float(t)) is False
    assert det.is_stuck is False


def test_frozen_view_while_moving_trips_after_timeout():
    det = StuckDetector(timeout=2.0)
    # Same frame every tick while "moving" -> view never changes -> stuck.
    flags = [det.update(_frame(100), is_moving=True, now=t) for t in (0.0, 0.5, 1.0, 1.5, 2.0)]
    assert flags[:-1] == [False, False, False, False]  # not enough history yet
    assert flags[-1] is True  # at t=2.0 the oldest frame is a full timeout old
    assert det.recovery_count == 1


def test_changing_view_is_not_stuck():
    det = StuckDetector(timeout=2.0)
    # View changes meaningfully each tick (real movement) -> never stuck.
    stuck = any(
        det.update(_frame(v), is_moving=True, now=t)
        for v, t in zip((0, 40, 80, 120, 160, 200), (0.0, 0.5, 1.0, 1.5, 2.0, 2.5))
    )
    assert stuck is False


def test_reset_clears_state():
    det = StuckDetector(timeout=2.0)
    for t in (0.0, 1.0, 2.0):
        det.update(_frame(0), is_moving=True, now=t)
    det.reset()
    assert det.is_stuck is False
    # after reset it needs a fresh full window before it can flag again
    assert det.update(_frame(0), is_moving=True, now=2.1) is False


# --- EscapeManeuver ---------------------------------------------------------


def test_inactive_until_started():
    em = EscapeManeuver()
    assert em.active is False
    assert em.update(0.1) is None


def test_reverses_then_finishes():
    em = EscapeManeuver(back_time=0.4, duration=1.2)
    em.start()
    assert em.active is True
    first = em.update(0.1)  # e=0.1 -> still backing up
    assert first["back"] is True
    assert first["turn_x"] != 0
    # run it out
    for _ in range(20):
        if em.update(0.1) is None:
            break
    assert em.active is False
    assert em.update(0.1) is None


def test_turn_is_angle_sized_and_alternates():
    em = EscapeManeuver(cpd=53.5, turn_deg=130.0, duration=1.2)
    target = 130.0 * 53.5  # ~6955 counts total

    def run_total():
        em.start()
        total = 0
        while True:
            cmd = em.update(0.1)
            if cmd is None:
                break
            total += cmd["turn_x"]
        return total

    first = run_total()
    second = run_total()
    # roughly the right magnitude (discretisation loses ~the last tick)
    assert 0.8 * target < abs(first) < 1.05 * target
    # alternating direction so repeated stucks don't loop into the same wall
    assert np.sign(first) == -np.sign(second)

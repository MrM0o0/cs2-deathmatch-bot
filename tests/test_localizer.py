"""Tests for landmark-based area localization."""

import os

from src.movement.localizer import Localizer

AREAS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "maps",
    "dust2_areas.json",
)


class FakeDet:
    """Minimal stand-in for the YOLO Detection type."""

    def __init__(self, class_name, confidence=0.9, area=1000.0):
        self.class_name = class_name
        self.confidence = confidence
        self.area = area


def _feed(loc, dets, times):
    """Feed the same detection list `times` frames in a row."""
    out = None
    for _ in range(times):
        out = loc.update(dets)
    return out


# --- small hand-built graph for deterministic logic tests ---------------------


def _toy():
    # mid <-> catwalk <-> a_site ;  b_site is NOT adjacent to mid
    area_landmarks = {
        "mid": ["mid_doors"],
        "catwalk": ["catwalk"],
        "a_site": ["goose"],
        "b_site": ["b_car"],
    }
    neighbors = {
        "mid": ["catwalk"],
        "catwalk": ["mid", "a_site"],
        "a_site": ["catwalk"],
        "b_site": [],
    }
    return Localizer(area_landmarks, neighbors, confirm_frames=3, jump_penalty=0.3)


def test_cold_start_needs_confirmation():
    loc = _toy()
    # two frames of mid_doors is not enough (confirm_frames=3)
    loc.update([FakeDet("mid_doors")])
    loc.update([FakeDet("mid_doors")])
    assert loc.current_area is None
    loc.update([FakeDet("mid_doors")])
    assert loc.current_area == "mid"


def test_reaffirm_does_not_require_reconfirm():
    loc = _toy()
    _feed(loc, [FakeDet("mid_doors")], 3)
    assert loc.current_area == "mid"
    # staying put keeps reporting mid immediately
    assert loc.update([FakeDet("mid_doors")]) == "mid"


def test_no_landmarks_holds_area_and_marks_stale():
    loc = _toy()
    _feed(loc, [FakeDet("mid_doors")], 3)
    assert loc.update([]) == "mid"
    assert loc.update([]) == "mid"
    assert loc.stale == 2


def test_moves_to_adjacent_area_after_confirm():
    loc = _toy()
    _feed(loc, [FakeDet("mid_doors")], 3)  # at mid
    out = _feed(loc, [FakeDet("catwalk")], 3)  # catwalk is a neighbour of mid
    assert out == "catwalk"


def test_single_frame_does_not_flip_area():
    loc = _toy()
    _feed(loc, [FakeDet("mid_doors")], 3)  # at mid
    assert loc.update([FakeDet("catwalk")]) == "mid"  # one frame -> no switch yet


def test_non_adjacent_jump_is_resisted():
    """Standing at mid, a single strong b_car sighting must not teleport us to
    b_site (not a neighbour). The adjacency penalty + confirm filter hold mid."""
    loc = _toy()
    _feed(loc, [FakeDet("mid_doors")], 3)
    # even a few frames of a far landmark shouldn't instantly win once; but it
    # WILL eventually commit after confirm_frames since it's the only evidence.
    assert loc.update([FakeDet("b_car")]) == "mid"
    assert loc.update([FakeDet("b_car")]) == "mid"


def test_proximity_weight_prefers_the_closer_landmark():
    """When two landmarks vote for different areas, the on-screen-larger
    (closer) one should win."""
    area_landmarks = {"near_area": ["goose"], "far_area": ["b_car"]}
    neighbors = {"near_area": ["far_area"], "far_area": ["near_area"]}
    loc = Localizer(area_landmarks, neighbors, confirm_frames=1)
    big = FakeDet("goose", confidence=0.8, area=8000.0)
    small = FakeDet("b_car", confidence=0.8, area=500.0)
    out = loc.update([big, small], frame_area=20000.0)
    assert out == "near_area"


# --- real dust2 graph loads and behaves -------------------------------------


def test_loads_real_dust2_graph():
    loc = Localizer.from_file(AREAS_FILE)
    assert "goose" in loc.landmark_to_areas
    # goose is a landmark of multiple A-side areas
    assert "a_long" in loc.landmark_to_areas["goose"]


def test_real_graph_localizes_and_exposes_danger():
    loc = Localizer.from_file(AREAS_FILE, confirm_frames=2)
    _feed(loc, [FakeDet("mid_doors")], 2)
    assert loc.current_area in ("mid", "mid_doors")
    rec = loc.area_record()
    assert "danger_from" in rec
    assert "neighbors" in rec

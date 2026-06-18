"""Consistency checks for every map area graph (config/maps/*_areas.json).

These run against all area graphs so a typo in a hand-authored map (dangling
neighbour, one-way connection, danger pointing at a non-existent area) is caught
before it confuses routing in-game.
"""

import glob
import json
import os

import pytest

MAPS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "maps"
)
GRAPHS = sorted(glob.glob(os.path.join(MAPS_DIR, "*_areas.json")))


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["areas"]


@pytest.mark.parametrize("path", GRAPHS, ids=[os.path.basename(p) for p in GRAPHS])
def test_graph_is_consistent(path):
    areas = _load(path)
    names = set(areas)
    assert names, f"{path} has no areas"

    for area, rec in areas.items():
        for n in rec.get("neighbors", []):
            assert n in names, f"{area} -> neighbour '{n}' does not exist"
            assert area in areas[n].get("neighbors", []), (
                f"one-way connection: {area} -> {n} but not {n} -> {area}"
            )
        for d in rec.get("danger_from", []):
            assert d in names, f"{area} danger_from '{d}' does not exist"


@pytest.mark.parametrize("path", GRAPHS, ids=[os.path.basename(p) for p in GRAPHS])
def test_no_orphan_areas(path):
    """Every area should be reachable from at least one neighbour link."""
    areas = _load(path)
    linked = set()
    for area, rec in areas.items():
        neighbours = rec.get("neighbors", [])
        if neighbours:
            linked.add(area)
        linked.update(neighbours)
    orphans = set(areas) - linked
    assert not orphans, f"{path} has unreachable areas: {orphans}"

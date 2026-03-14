"""Tests for humanizer components."""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.humanizer.timing import ReactionTimer, ActionCooldown
from src.humanizer.mistakes import MistakeMaker
from src.humanizer.personality import Personality, load_personality
from src.humanizer.noise import NoiseGenerator


def test_reaction_timer_range():
    timer = ReactionTimer(mean_ms=250, std_ms=50, min_ms=150, max_ms=500)
    delays = []
    for _ in range(100):
        d = timer.start_reaction()
        delays.append(d)
    assert all(0.15 <= d <= 0.5 for d in delays)


def test_reaction_timer_ready():
    timer = ReactionTimer(mean_ms=10, std_ms=1, min_ms=10, max_ms=20)
    timer.start_reaction()
    assert not timer.is_ready()  # Should not be ready immediately
    time.sleep(0.03)
    assert timer.is_ready()


def test_action_cooldown():
    cd = ActionCooldown(cooldown_ms=50)
    assert cd.try_action()  # First action should succeed
    assert not cd.try_action()  # Too soon
    time.sleep(0.06)
    assert cd.try_action()  # Cooldown elapsed


def test_mistake_maker_aim_error():
    mm = MistakeMaker(tracking_error=10.0)
    # Run many times, should produce different results
    results = set()
    for _ in range(10):
        x, y = mm.apply_aim_error(100, 100)
        results.add((round(x), round(y)))
    assert len(results) > 1  # Should have variation


def test_mistake_maker_overshoot():
    mm = MistakeMaker(overshoot_magnitude=1.5)
    ox, oy = mm.overshoot_target(0, 0, 100, 100)
    # Overshoot should go past the target
    assert ox > 100 or oy > 100


def test_personality_load():
    config_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "personalities"
    )
    p = load_personality("average", config_dir)
    assert p.name == "average"
    assert p.reaction_mean_ms == 250
    assert p.aim_speed == 6.0


def test_noise_generator():
    ng = NoiseGenerator(seed=42)
    v1 = ng.sample_1d(1.0)
    v2 = ng.sample_1d(2.0)
    assert -1.0 <= v1 <= 1.0
    assert v1 != v2  # Different inputs should give different outputs

    jx, jy = ng.mouse_jitter(amplitude=5.0)
    assert -10 <= jx <= 10
    assert -10 <= jy <= 10

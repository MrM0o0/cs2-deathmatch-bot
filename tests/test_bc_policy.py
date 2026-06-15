"""Smoke test for the behavioural-cloning movement policy.

Skips cleanly where onnxruntime or the trained model aren't present (e.g. CI
without the heavy runtime deps).
"""

import os

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("cv2")

from src.movement.bc_policy import BCMovementPolicy

MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "movement.onnx")


@pytest.mark.skipif(not os.path.exists(MODEL), reason="movement.onnx not trained yet")
def test_policy_act_returns_valid_command():
    p = BCMovementPolicy(MODEL)
    p.load()
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    cmd = p.act(frame)
    for k in ("forward", "back", "left", "right", "crouch"):
        assert isinstance(cmd[k], bool)
    assert cmd["turn_class"] in range(5)
    assert isinstance(cmd["turn_x"], int)
    assert len(cmd["key_probs"]) == 5


@pytest.mark.skipif(not os.path.exists(MODEL), reason="movement.onnx not trained yet")
def test_invert_flips_turn_sign():
    a = BCMovementPolicy(MODEL, invert_turn=False)
    b = BCMovementPolicy(MODEL, invert_turn=True)
    a.load()
    b.load()
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    assert a.act(frame)["turn_x"] == -b.act(frame)["turn_x"]

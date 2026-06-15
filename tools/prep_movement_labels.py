"""Turn the raw movement recording into a training-ready label set.

For each frame we already have the held keys. This adds the turn label, which
is recovered from how the game world shifts horizontally between consecutive
frames (phase correlation on a HUD-free crop -- the static crosshair/HUD must
be excluded or it dominates the correlation). The continuous shift is bucketed
into 5 coarse turn classes; the deadzone around 0 absorbs strafe/jitter noise.

Output (per session): labels.npy, an (N, 6) int array per frame:
    [w, a, s, d, crouch, turn_class]   turn_class in 0..4
    0 = hard left, 1 = left, 2 = straight, 3 = right, 4 = hard right
(The left/right sign is a convention; it's flippable at integration time, like
nav_invert_turn, if the bot ends up turning the wrong way.)

Usage:
    python tools/prep_movement_labels.py                 # newest session
    python tools/prep_movement_labels.py --session <dir> # a specific one
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# HUD-free world crop (rows, cols) on the 160x90 frame: below the radar band,
# above the gun/HUD, away from the left radar and right ammo.
CROP = (slice(20, 60), slice(30, 130))
# Turn-class thresholds in px of horizontal shift.
DEAD, HARD = 2.0, 7.0
# Reject phase-correlation results below this confidence (scene cuts, deaths).
MIN_RESPONSE = 0.10
# Clip wild outliers before bucketing.
CLIP = 25.0


def _bucket(dx: float) -> int:
    if dx <= -HARD:
        return 0
    if dx <= -DEAD:
        return 1
    if dx < DEAD:
        return 2
    if dx < HARD:
        return 3
    return 4


def _turn_from_mouse(acts: list) -> list[int]:
    """Bucket the recorded raw mouse-x per frame into turn classes.

    Thresholds adapt to the player's sensitivity/DPI: the deadzone and hard-turn
    cutoffs come from percentiles of the actual |mdx| distribution.
    """
    mdx = np.array([a["m"][0] for a in acts], dtype=np.float32)
    amag = np.abs(mdx)
    dead = max(8.0, float(np.percentile(amag, 55)))
    hard = (
        max(dead * 3, float(np.percentile(amag[amag > dead], 70)))
        if (amag > dead).any()
        else dead * 3
    )
    print(
        f"[prep] mouse turn: |mdx| mean={amag.mean():.0f} p90={np.percentile(amag, 90):.0f} "
        f"-> dead={dead:.0f} hard={hard:.0f}"
    )

    def bucket(dx):
        if dx <= -hard:
            return 0
        if dx <= -dead:
            return 1
        if dx < dead:
            return 2
        if dx < hard:
            return 3
        return 4

    return [bucket(float(v)) for v in mdx]


def prep(session_dir: str) -> None:
    acts = [json.loads(line) for line in open(os.path.join(session_dir, "actions.jsonl"))]
    frames_dir = os.path.join(session_dir, "frames")

    if "m" in acts[0]:
        # Real mouse recorded -> clean turn labels (no optical-flow guessing).
        print("[prep] using recorded mouse for turn labels")
        turn = _turn_from_mouse(acts)
    else:
        # Legacy recording without mouse -> recover turn from inter-frame flow.
        print("[prep] no mouse data; recovering turn from optical flow")

        def world_gray(name):
            g = cv2.imread(os.path.join(frames_dir, name), cv2.IMREAD_GRAYSCALE)
            return g[CROP].astype(np.float32)

        win = None
        turn = []
        prev = world_gray(acts[0]["f"])
        last_dx = 0.0
        for a in acts:
            cur = world_gray(a["f"])
            if win is None:
                win = np.outer(np.hanning(cur.shape[0]), np.hanning(cur.shape[1])).astype(
                    np.float32
                )
            (dx, _), resp = cv2.phaseCorrelate(prev, cur, win)
            prev = cur
            if resp >= MIN_RESPONSE:
                last_dx = float(np.clip(dx, -CLIP, CLIP))
            turn.append(_bucket(last_dx))

    labels = []
    for a, t in zip(acts, turn):
        keys = a["keys"]  # [w,a,s,d,shift,ctrl,m4,m5]
        crouch = 1 if (keys[5] or keys[6] or keys[7]) else 0
        labels.append([keys[0], keys[1], keys[2], keys[3], crouch, t])

    arr = np.array(labels, dtype=np.int16)
    out = os.path.join(session_dir, "labels.npy")
    np.save(out, arr)

    names = ["hardL", "left", "straight", "right", "hardR"]
    print(f"[prep] {len(arr)} labels -> {out}")
    print("[prep] turn-class distribution:")
    for c in range(5):
        print(f"    {names[c]:9} {100 * np.mean(arr[:, 5] == c):5.1f}%")
    print(
        f"[prep] WASD active rate: {100 * np.mean(arr[:, :4].any(axis=1)):.1f}%  "
        f"crouch: {100 * np.mean(arr[:, 4]):.1f}%"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build turn labels for a movement recording")
    parser.add_argument("--session", default=None, help="Session dir (default: newest)")
    args = parser.parse_args()

    base = os.path.join(PROJECT_ROOT, "data", "movement")
    session = args.session
    if session is None:
        sessions = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
        session = os.path.join(base, sessions[-1])
    prep(session)

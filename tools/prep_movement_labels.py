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

# Filtering: a frame is "moving" if WASD held or the mouse turned at least this
# much (counts) -- otherwise it's idle/camping and dropped. Combat frames (fire
# or a visible enemy) are dropped with a margin so the model never learns to
# spray/counter-strafe while roaming.
IDLE_MOUSE = 30.0
COMBAT_MARGIN = 8  # frames (~0.4s) dropped around each combat frame


def _dilate(mask: np.ndarray, margin: int) -> np.ndarray:
    """Grow True regions by `margin` frames on each side (1D)."""
    out = mask.copy()
    for s in range(1, margin + 1):
        out[s:] |= mask[:-s]
        out[:-s] |= mask[s:]
    return out


def _yolo_enemy_mask(names: list, frames_dir: str) -> np.ndarray:
    """True where the YOLO detector sees an enemy (combat). Slow; opt-in."""
    import yaml

    from src.vision.detector import YOLODetector

    cfg = yaml.safe_load(open(os.path.join(PROJECT_ROOT, "config", "settings.yaml")))["detection"]
    det = YOLODetector(
        model_path=os.path.join(PROJECT_ROOT, cfg["model_path"]),
        input_size=cfg["input_size"],
        confidence_threshold=0.35,
        nms_threshold=cfg["nms_threshold"],
        classes=cfg["classes"],
    )
    det.load()
    mask = np.zeros(len(names), dtype=bool)
    for i, n in enumerate(names):
        img = cv2.imread(os.path.join(frames_dir, n))
        mask[i] = bool(det.detect(img))
        if i % 500 == 0:
            print(f"\r[prep] yolo scan {i}/{len(names)}", end="")
    print()
    return mask


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


def prep(session_dir: str, yolo: bool = False) -> None:
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
    frame_names = [a["f"] for a in acts]
    n = len(arr)

    # --- Build the keep mask (drop idle/camping and combat frames) ----------
    wasd = arr[:, :4].any(axis=1)
    if "m" in acts[0]:
        turning = np.abs(np.array([a["m"][0] for a in acts])) >= IDLE_MOUSE
    else:
        turning = arr[:, 5] != 2  # non-straight from optical flow
    moving = wasd | turning

    combat = np.zeros(n, dtype=bool)
    if "fire" in acts[0]:
        combat |= _dilate(np.array([a["fire"] for a in acts], dtype=bool), COMBAT_MARGIN)
    if yolo:
        combat |= _dilate(_yolo_enemy_mask(frame_names, frames_dir), COMBAT_MARGIN)

    keep = moving & ~combat
    if keep.sum() < 50:  # safety: never emit an empty/tiny set
        print("[prep] WARNING: filter kept <50 frames; keeping all instead")
        keep = np.ones(n, dtype=bool)

    arr = arr[keep]
    kept_names = [fn for fn, k in zip(frame_names, keep) if k]
    np.save(os.path.join(session_dir, "labels.npy"), arr)
    with open(os.path.join(session_dir, "frames.txt"), "w") as f:
        f.write("\n".join(kept_names))

    names = ["hardL", "left", "straight", "right", "hardR"]
    print(
        f"[prep] kept {keep.sum()}/{n} frames "
        f"(dropped idle {100 * np.mean(~moving):.0f}%, combat {100 * np.mean(combat):.0f}%)"
    )
    print("[prep] turn-class distribution (kept):")
    for c in range(5):
        print(f"    {names[c]:9} {100 * np.mean(arr[:, 5] == c):5.1f}%")
    print(f"[prep] WASD active: {100 * np.mean(arr[:, :4].any(axis=1)):.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build turn labels for a movement recording")
    parser.add_argument("--session", default=None, help="Session dir (default: newest)")
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Also drop frames where YOLO sees an enemy (slow; for sessions "
        "recorded before the fire-flag)",
    )
    args = parser.parse_args()

    base = os.path.join(PROJECT_ROOT, "data", "movement")
    session = args.session
    if session is None:
        sessions = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
        session = os.path.join(base, sessions[-1])
    prep(session, yolo=args.yolo)

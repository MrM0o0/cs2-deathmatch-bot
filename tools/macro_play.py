"""Replay a recorded movement MACRO -- faithfully copy your inputs, with re-sync.

Plays back record_macro.json: feeds your recorded WASD + crouch + mouse motion
tick-for-tick, so the bot moves exactly like you did. Because replaying relative
inputs open-loop drifts, it self-corrects:

  PLAY      -- apply the recorded inputs for the current frame, advance (ping-pong
               at the ends). Each tick, compare the dot to the frame's recorded
               position.
  desync    -- drift too big: re-anchor to the recorded frame nearest where the
               bot ACTUALLY is, then:
  RECOVER   -- if far off the route, walk back to that frame's position, then
  REORIENT  -- turn the view to that frame's recorded facing (so the relative
               mouse replay lines up again), then resume PLAY.

SAFETY -- dead-man: acts only WHILE YOU HOLD the run key (default L). Release =
instant stop. END quits; --max-seconds backstops.

    python tools/macro_play.py            # HOLD L to replay the dust2 macro
    python tools/macro_play.py --map mirage
"""

import argparse
import ctypes
import json
import os
import sys
import time

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.capture.screen import ScreenCapture
from src.input import keyboard, mouse
from src.utils.math_helpers import angle_between, distance, normalize_angle
from src.utils.timer_setup import enable_high_resolution_timer
from src.vision.minimap import MinimapReader

KEY_NAMES = ["w", "a", "s", "d"]  # frame keys 0..3; index 4 = crouch (mouse5)
RUN_VK = 0x4C  # L
END_VK = 0x23


def held(vk):
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


def main():
    ap = argparse.ArgumentParser(description="Replay a movement macro with re-sync")
    ap.add_argument("--map", "-m", default="dust2", help="config/maps/<map>_macro.json")
    ap.add_argument("--resync", type=float, default=20.0, help="Px drift before re-syncing")
    ap.add_argument("--offroute", type=float, default=45.0, help="Px off-route -> walk back first")
    ap.add_argument("--reach", type=float, default=10.0, help="Px to a recover target = arrived")
    ap.add_argument("--head-tol", type=float, default=14.0, help="Deg facing error = oriented")
    ap.add_argument("--slice", type=int, default=200, help="Max mouse counts/tick during recovery")
    ap.add_argument("--cpd", type=float, default=0.0, help="Counts/deg (0 = from sensitivity)")
    ap.add_argument("--max-seconds", type=float, default=180.0, help="Hard auto-stop")
    args = ap.parse_args()

    path = os.path.join(PROJECT_ROOT, "config", "maps", f"{args.map}_macro.json")
    if not os.path.exists(path):
        print(f"[play] no macro for '{args.map}'. Record one: tools/record_macro.py -m {args.map}")
        return
    with open(path) as f:
        macro = json.load(f)
    frames = macro["frames"]
    dt = 1.0 / macro.get("rate", 30)
    if len(frames) < 2:
        print("[play] macro too short.")
        return

    enable_high_resolution_timer()
    with open(os.path.join(PROJECT_ROOT, "config", "settings.yaml")) as f:
        cfg = yaml.safe_load(f)
    mm, g = cfg["minimap"], cfg["game"]
    cpd = args.cpd if args.cpd > 0 else 1.0 / (g["sensitivity"] * g["m_yaw"])
    reader = MinimapReader(
        mm["x"], mm["y"], mm["size"], sat_min=mm["sat_min"], val_min=mm["val_min"]
    )
    cap = ScreenCapture(monitor=cfg["display"]["monitor"], target_fps=max(60, int(1 / dt)))
    cap.start()
    print(f"[play] {args.map}: {len(frames)} frames @ {macro.get('rate', 30)}Hz. HOLD L to replay.")

    keys_down = set()

    def set_keys(state):
        # state: [w,a,s,d, crouch]
        for j, name in enumerate(KEY_NAMES):
            if state[j] and name not in keys_down:
                keyboard.key_down(name)
                keys_down.add(name)
            elif not state[j] and name in keys_down:
                keyboard.key_up(name)
                keys_down.discard(name)
        if state[4] and not mouse.is_crouching():
            mouse.crouch_down()
        elif not state[4] and mouse.is_crouching():
            mouse.crouch_up()

    def release_all():
        for k in list(keys_down):
            keyboard.key_up(k)
            keys_down.discard(k)
        mouse.release_all_buttons()

    def nearest_frame(p):
        return min(range(len(frames)), key=lambda j: distance(p, frames[j]["p"]))

    def turn_to(target_deg, cur_deg):
        err = normalize_angle(target_deg - cur_deg)
        if abs(err) > args.head_tol:
            mouse.move_relative(int(max(-args.slice, min(args.slice, err * cpd))), 0)
            return False
        return True

    active = False
    i = 0
    direction = 1
    state = "play"
    status_t = 0.0
    start = time.perf_counter()
    try:
        while True:
            if held(END_VK):
                break
            now = time.perf_counter()
            if args.max_seconds and now - start > args.max_seconds:
                print("\n[play] max-seconds reached.")
                break

            if not held(RUN_VK):  # dead-man
                if active:
                    release_all()
                    active = False
                time.sleep(0.02)
                if now - status_t >= 0.5:
                    print("\r[play] IDLE (hold L to replay)        ", end="", flush=True)
                    status_t = now
                continue

            frame = cap.grab()
            if frame is None:
                time.sleep(0.002)
                continue
            (x, y), heading = reader.read(frame)
            pos = (x, y)

            if not active:
                active = True
                i = nearest_frame(pos)  # sync to wherever we spawned
                state = "play"

            f = frames[i]
            drift = distance(pos, f["p"])

            if state == "play":
                if drift > args.resync:  # lost sync -> re-anchor + recover
                    i = nearest_frame(pos)
                    state = (
                        "recover" if distance(pos, frames[i]["p"]) > args.offroute else "reorient"
                    )
                    release_all()  # stop replaying inputs while we recover
                else:
                    set_keys(f["k"])
                    if f["m"][0] or f["m"][1]:
                        mouse.move_relative(int(f["m"][0]), int(f["m"][1]))
                    i += direction  # advance along the macro, ping-pong at ends
                    if i >= len(frames):
                        i, direction = len(frames) - 2, -1
                    elif i < 0:
                        i, direction = 1, 1

            elif state == "recover":  # far off route -> walk back to the frame's spot
                keyboard.key_down("w")
                keys_down.add("w")
                if distance(pos, f["p"]) <= args.reach:
                    keyboard.key_up("w")
                    keys_down.discard("w")
                    state = "reorient"
                else:
                    turn_to(angle_between(pos, f["p"]), heading)

            elif state == "reorient":  # face the recorded direction, then resume
                if turn_to(f["h"], heading):
                    state = "play"

            if now - status_t >= 0.4:
                print(
                    f"\r[play] {state:8s} | frame {i}/{len(frames)} | drift {drift:4.0f}px   ",
                    end="",
                    flush=True,
                )
                status_t = now

            sleep_for = dt - (time.perf_counter() - now)
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        release_all()
        cap.stop()
        print("\n[play] stopped.")


if __name__ == "__main__":
    main()

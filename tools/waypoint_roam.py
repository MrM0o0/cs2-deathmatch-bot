"""Trail roam -- REPLAY the exact route you recorded, in order, on a loop.

Not routing -- replay. It follows the recorded breadcrumbs IN ORDER (the order
you walked them), so it only ever goes where you actually went. No A*, no random
goals, no auto shortcut-links (those made it cut through walls / go places you
never walked). Works from ANY spawn (DM is random): it snaps to the nearest
point on your trail, then traces the path from there, looping at the end.

Facing comes from the dot's MOTION (slow-radar tuned). If it drifts off the trail
it rejoins the nearest point; if it bumps something it sweeps past. Saves an
annotated map each ~0.7s to logs/<ts>/.

SAFETY -- dead-man: acts only WHILE YOU HOLD the run key (default L). Release =
instant stop. END quits; --max-seconds backstops.

    python tools/waypoint_roam.py                # HOLD L, replay the dust2 trail
    python tools/waypoint_roam.py --map mirage   # a different recorded trail
"""

import argparse
import ctypes
import math
import os
import sys
import time
from collections import deque

import yaml

try:
    import cv2
except ImportError:
    cv2 = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.capture.screen import ScreenCapture
from src.input import keyboard, mouse
from src.movement.navigator import WaypointGraph
from src.utils.math_helpers import angle_between, distance, normalize_angle
from src.utils.session_logger import SessionLogger
from src.utils.timer_setup import enable_high_resolution_timer
from src.vision.minimap import MinimapReader

SPECIAL = {"insert": 0x2D, "end": 0x23, "home": 0x24, "rshift": 0xA1, "rctrl": 0xA3}
END_VK = 0x23
N_RAYS = 24
MAX_R = 46


def key_vk(name):
    name = name.lower()
    if name in SPECIAL:
        return SPECIAL[name]
    if len(name) == 1 and name.isalpha():
        return ord(name.upper())
    raise ValueError(f"unknown run-key: {name!r}")


def held(vk):
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


def head_to_node(node_bearing, motion_deg, deadzone, slice_, cpd):
    """Steer straight toward the next node. With a DENSE path the next node is
    always close and on the safe route the user walked, so a straight segment
    never cuts into a wall -- no radar wall-judging needed. Returns sweep_dir."""
    err = normalize_angle(node_bearing - motion_deg)
    sweep_dir = 1 if err >= 0 else -1
    if abs(err) > deadzone:
        mouse.move_relative(int(max(-slice_, min(slice_, err * cpd))), 0)
    return sweep_dir


def load_cfg():
    with open(os.path.join(PROJECT_ROOT, "config", "settings.yaml")) as f:
        return yaml.safe_load(f)


def annotate(region, graph, route, idx, pos, target, motion_deg):
    img = region.copy()
    for w in graph.waypoints.values():  # edges (faint) + nodes
        for nb in w.neighbors:
            o = graph.waypoints.get(nb)
            if o:
                cv2.line(img, (int(w.x), int(w.y)), (int(o.x), int(o.y)), (70, 70, 70), 1)
    for i in range(1, len(route)):  # current route in cyan
        a = graph.waypoints[route[i - 1]]
        b = graph.waypoints[route[i]]
        cv2.line(img, (int(a.x), int(a.y)), (int(b.x), int(b.y)), (200, 200, 0), 1)
    if target is not None:
        cv2.circle(img, (int(target.x), int(target.y)), 4, (255, 80, 0), -1)
    bx, by = pos
    cv2.circle(img, (bx, by), 3, (255, 255, 255), 1)
    if motion_deg is not None:
        r = math.radians(motion_deg)
        cv2.line(
            img,
            (bx, by),
            (int(bx + 18 * math.cos(r)), int(by + 18 * math.sin(r))),
            (0, 255, 255),
            2,
        )
    return cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser(description="Waypoint roam (follow a map's routes)")
    ap.add_argument("--map", default="dust2", help="Map name (config/maps/<map>.json)")
    ap.add_argument("--run-key", default="l", help="HOLD this to roam")
    ap.add_argument("--reach", type=float, default=9.0, help="Px to a node to count as reached")
    ap.add_argument(
        "--lookahead", type=int, default=2, help="Aim this many points ahead (anticipate corners)"
    )
    ap.add_argument(
        "--resnap", type=float, default=35.0, help="Px off the trail before rejoining nearest point"
    )
    ap.add_argument("--deadzone", type=float, default=10.0, help="Deg error before turning")
    ap.add_argument("--slice", type=int, default=200, help="Max mouse counts per tick (smoothness)")
    ap.add_argument(
        "--cpd", type=float, default=0.0, help="Counts per degree (0 = from sensitivity)"
    )
    ap.add_argument(
        "--stuck-secs", type=float, default=1.0, help="No motion this long -> sweep past"
    )
    ap.add_argument(
        "--skip-after", type=float, default=2.5, help="Seconds truly stuck before giving up a node"
    )
    ap.add_argument("--max-seconds", type=float, default=180.0, help="Hard auto-stop")
    args = ap.parse_args()

    if cv2 is None:
        print("[wp] cv2/numpy missing -- aborting.")
        return

    graph = WaypointGraph()
    graph.load(os.path.join(PROJECT_ROOT, "config", "maps", f"{args.map}.json"))
    if not graph.waypoints:
        print(f"[wp] no waypoints for '{args.map}'. Record with tools/record_waypoints.py first.")
        return

    enable_high_resolution_timer()
    cfg = load_cfg()
    mm, g = cfg["minimap"], cfg["game"]
    mx, my, msz = mm["x"], mm["y"], mm["size"]
    cpd = args.cpd if args.cpd > 0 else 1.0 / (g["sensitivity"] * g["m_yaw"])
    reader = MinimapReader(mx, my, msz, sat_min=mm["sat_min"], val_min=mm["val_min"])
    run_vk = key_vk(args.run_key)
    cap = ScreenCapture(monitor=cfg["display"]["monitor"], target_fps=60)
    cap.start()
    logger = SessionLogger(os.path.join(PROJECT_ROOT, "logs"), enabled=True)
    print(f"[wp] {args.map}: {len(graph.waypoints)} nodes. HOLD {args.run_key.upper()} = roam.")

    keys_down = set()

    def press(k):
        if k not in keys_down:
            keyboard.key_down(k)
            keys_down.add(k)

    def release_all():
        for k in list(keys_down):
            keyboard.key_up(k)
            keys_down.discard(k)
        mouse.release_all_buttons()

    order = sorted(graph.waypoints)  # node ids in recorded (walk) order = the trail

    def nearest_idx(p):
        return min(range(len(order)), key=lambda i: distance(p, graph.waypoints[order[i]].pos()))

    active = False
    idx = 0
    need_snap = True  # snap to nearest trail point on (re)start / after drifting off
    hist = deque(maxlen=40)  # ~1.3s of positions for slow-dot motion estimate
    motion_deg = None
    last_move_t = 0.0
    stall_start = 0.0
    sweep_dir = 1
    last_dump = 0.0
    status_t = 0.0
    start = time.perf_counter()

    try:
        while True:
            if held(END_VK):
                break
            now = time.perf_counter()
            if args.max_seconds and now - start > args.max_seconds:
                print("\n[wp] max-seconds reached.")
                break

            if not held(run_vk):  # dead-man
                if active:
                    release_all()
                    active = False
                    hist.clear()
                    motion_deg = None
                    need_snap = True
                time.sleep(0.02)
                if now - status_t >= 0.5:
                    print("\r[wp] IDLE (hold key to roam)        ", end="", flush=True)
                    status_t = now
                continue

            if not active:
                active = True
                press("w")
                hist.clear()
                motion_deg = None
                last_move_t = now
                need_snap = True

            frame = cap.grab()
            if frame is None:
                time.sleep(0.002)
                continue
            (bx, by), _ = reader.read(frame)
            pos = (bx, by)
            region = frame[my : my + msz, mx : mx + msz]

            # motion heading from recent dot displacement
            # The radar dot moves slowly (~5 px/sec at whole-map zoom), so judge
            # motion over a ~1s window with a small threshold -- otherwise normal
            # walking reads as "stalled" and the bot sweep-turns forever.
            hist.append((now, pos))
            old = next((p for (t, p) in hist if t >= now - 1.0), hist[0][1])
            if distance(old, pos) >= 3.0:
                motion_deg = angle_between(old, pos)
                last_move_t = now

            # Follow the recorded trail IN ORDER (no A*, no random goals, no
            # shortcut links) -- so it only goes exactly where you walked, looping.
            if need_snap:
                idx = nearest_idx(pos)  # rejoin the trail at the closest point
                need_snap = False
            cur = graph.waypoints[order[idx]]
            d_cur = distance(pos, cur.pos())
            if d_cur < args.reach:
                idx = (idx + 1) % len(order)  # reached this point -> next on the trail
            elif d_cur > args.resnap:
                idx = nearest_idx(pos)  # drifted off -> rejoin nearest trail point
            target = graph.waypoints[order[idx]]
            # Aim a couple points further ALONG the trail so corners start early.
            steer_target = graph.waypoints[order[(idx + args.lookahead) % len(order)]]
            press("w")
            state = "walk"
            if target is not None:
                node_bearing = angle_between(pos, steer_target.pos())
                stalled = motion_deg is not None and now - last_move_t > args.stuck_secs
                if not stalled:
                    stall_start = now

                if motion_deg is None:
                    state = "spinup"  # walk to establish facing
                elif stalled:
                    # Bumped something off the path -> sweep-turn past it. With a
                    # dense path this is rare; if it can't clear, drop the node.
                    mouse.move_relative(sweep_dir * args.slice, 0)
                    state = "around"
                    if now - stall_start > args.skip_after:
                        idx = (idx + 1) % len(order)  # can't pass -> skip ahead on trail
                        stall_start = now
                        sweep_dir = -sweep_dir
                else:
                    sweep_dir = head_to_node(
                        node_bearing, motion_deg, args.deadzone, args.slice, cpd
                    )
                    state = "steer"

            if now - last_dump >= 0.7:
                stamp = f"{now - start:06.1f}".replace(".", "_")
                cv2.imwrite(
                    os.path.join(logger.session_dir, f"nav_{stamp}_{state}.jpg"),
                    annotate(region, graph, order, idx, pos, steer_target, motion_deg),
                    [cv2.IMWRITE_JPEG_QUALITY, 85],
                )
                last_dump = now

            logger.log_tick(
                t=round(now - start, 3),
                x=bx,
                y=by,
                state=state,
                node=order[idx],
                motion=None if motion_deg is None else round(motion_deg),
            )
            if now - status_t >= 0.4:
                tgt = f"#{order[idx]}"
                print(
                    f"\r[wp] {state:6s} | pos {bx:3d},{by:3d} -> node {tgt}   ", end="", flush=True
                )
                status_t = now

            elapsed = time.perf_counter() - now
            if elapsed < 1.0 / 30.0:
                time.sleep(1.0 / 30.0 - elapsed)
    finally:
        release_all()
        cap.stop()
        logger.close()
        print("\n[wp] stopped.")


if __name__ == "__main__":
    main()

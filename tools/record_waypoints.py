"""Record a waypoint graph by walking a map yourself.

This is the one step that needs you in the game. Load a map (a bot/casual
match or an empty server with `bot_kick`), run this, and just walk the normal
routes for ~2 minutes. The tool watches your blip on the minimap and drops a
waypoint every time you've travelled far enough, linking consecutive nodes
into a path graph. Because the nodes are literally your footsteps, the bot's
navigation inherits human-shaped routes for free.

It also auto-links nodes that end up physically close (within `merge_dist`),
so when you re-walk a corridor the graph forms real junctions instead of one
long spaghetti line -- that's what lets A* find shortcuts later.

Usage:
    python tools/record_waypoints.py --map dust2
    # walk around... Ctrl+C when done. Saves config/maps/dust2.json

Verify localisation first: run with --preview to see the detected blip
position printed live, and confirm it tracks you before recording for real.
"""

import argparse
import os
import sys
import time

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.capture.screen import ScreenCapture
from src.vision.minimap import MinimapReader
from src.movement.navigator import Waypoint, WaypointGraph
from src.utils.math_helpers import distance


def load_minimap_reader() -> tuple[MinimapReader, dict]:
    with open(os.path.join(PROJECT_ROOT, "config", "settings.yaml")) as f:
        cfg = yaml.safe_load(f)
    mm = cfg["minimap"]
    reader = MinimapReader(
        mm["x"], mm["y"], mm["size"],
        player_arrow_color=tuple(mm.get("player_arrow_color", (0, 255, 0))),
    )
    return reader, cfg


def record(map_name: str, drop_dist: float, merge_dist: float,
           preview: bool, monitor: int) -> None:
    reader, cfg = load_minimap_reader()
    monitor = monitor if monitor is not None else cfg["display"]["monitor"]

    out_dir = os.path.join(PROJECT_ROOT, "config", "maps")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{map_name}.json")

    graph = WaypointGraph()
    if os.path.exists(out_path):
        graph.load(out_path)
        print(f"[Recorder] Extending existing graph: {len(graph.waypoints)} nodes")

    next_id = (max(graph.waypoints) + 1) if graph.waypoints else 0
    last_node: Waypoint | None = None

    capture = ScreenCapture(monitor=monitor, target_fps=15)
    backend = capture.start()
    print(f"[Recorder] Capture backend: {backend}")
    if preview:
        print("[Recorder] PREVIEW mode -- printing blip position, not saving nodes.")
        print("[Recorder] Confirm the (x, y) tracks your movement, then re-run "
              "without --preview.")
    else:
        print(f"[Recorder] Recording '{map_name}'. Walk the routes. Ctrl+C to "
              f"save -> {out_path}")
        print(f"[Recorder] Dropping a node every {drop_dist}px, "
              f"merging within {merge_dist}px.")

    def nearest_existing(x: float, y: float) -> Waypoint | None:
        best, best_d = None, merge_dist
        for wp in graph.waypoints.values():
            d = distance((x, y), wp.pos())
            if d < best_d:
                best, best_d = wp, d
        return best

    def link(a: Waypoint, b: Waypoint) -> None:
        if a.id != b.id:
            if b.id not in a.neighbors:
                a.neighbors.append(b.id)
            if a.id not in b.neighbors:
                b.neighbors.append(a.id)

    try:
        while True:
            frame = capture.grab()
            if frame is None:
                time.sleep(0.02)
                continue
            (x, y), angle = reader.read(frame)

            if preview:
                print(f"\r[Recorder] blip=({x:4d},{y:4d}) angle={angle:6.1f}", end="")
                time.sleep(0.1)
                continue

            # Only consider dropping a node once we've travelled far enough.
            if last_node is not None and distance((x, y), last_node.pos()) < drop_dist:
                time.sleep(0.03)
                continue

            existing = nearest_existing(x, y)
            if existing is not None:
                # Re-walking known ground -> just connect to it, forming a junction.
                if last_node is not None:
                    link(last_node, existing)
                last_node = existing
            else:
                node = Waypoint(next_id, float(x), float(y))
                next_id += 1
                graph.add_waypoint(node)
                if last_node is not None:
                    link(last_node, node)
                last_node = node
                print(f"\r[Recorder] nodes={len(graph.waypoints)} "
                      f"last=({x},{y})   ", end="")
            time.sleep(0.03)

    except KeyboardInterrupt:
        if not preview:
            graph.save(out_path)
            edges = sum(len(w.neighbors) for w in graph.waypoints.values()) // 2
            print(f"\n[Recorder] Saved {len(graph.waypoints)} nodes / "
                  f"{edges} edges -> {out_path}")
        else:
            print("\n[Recorder] Preview ended.")
    finally:
        capture.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record a CS2 waypoint graph by walking")
    parser.add_argument("--map", "-m", required=True, help="Map name (output filename)")
    parser.add_argument("--drop-dist", type=float, default=22.0,
                       help="Minimap px travelled between dropped nodes")
    parser.add_argument("--merge-dist", type=float, default=16.0,
                       help="Snap to an existing node within this many px")
    parser.add_argument("--preview", action="store_true",
                       help="Print blip position only; verify tracking before recording")
    parser.add_argument("--monitor", type=int, default=None, help="Monitor index override")
    args = parser.parse_args()

    record(args.map, args.drop_dist, args.merge_dist, args.preview, args.monitor)

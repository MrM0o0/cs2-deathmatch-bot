"""Depth-vision roam -- steer toward open space using a monocular depth model.

No radar, no waypoints, no recorded data. Each frame it runs a pretrained depth
model (Depth-Anything-V2-small, ONNX/DirectML) on the game view, reads how far it
can see across an eye-level band, and steers toward the MOST OPEN direction while
walking. Because depth sees real 3D geometry, it knows about walls, boxes, ledges
and drops -- the obstacles the radar was blind to. Steering is purely from the
first-person image, so there's no facing signal to fight.

Saves a depth heatmap each ~0.5s (with the chosen direction marked) to logs/<ts>/
so we can confirm it's reading near/far correctly and tune.

SAFETY -- dead-man: acts only WHILE YOU HOLD L. Release = instant stop. END quits;
--max-seconds backstops.

    python tools/depth_nav.py
    python tools/depth_nav.py --invert     # if near/far is backwards in the heatmap
"""

import argparse
import ctypes
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import yaml

from src.capture.screen import ScreenCapture
from src.input import keyboard, mouse
from src.utils.timer_setup import enable_high_resolution_timer

RUN_VK = 0x4C  # L
END_VK = 0x23
SIZE = 518  # depth model input (multiple of 14)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def held(vk):
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


def load_depth():
    p = hf_hub_download("onnx-community/depth-anything-v2-small", "onnx/model_fp16.onnx")
    sess = ort.InferenceSession(p, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name


def preprocess(frame_bgr):
    """Center-square crop (the forward view) -> 518x518 normalized CHW."""
    h, w = frame_bgr.shape[:2]
    s = min(h, w)
    crop = frame_bgr[(h - s) // 2 : (h - s) // 2 + s, (w - s) // 2 : (w - s) // 2 + s]
    img = cv2.resize(crop, (SIZE, SIZE))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - _MEAN) / _STD
    return rgb.transpose(2, 0, 1)[None].astype(np.float32), crop


def main():
    ap = argparse.ArgumentParser(description="Depth-vision roam (steer toward open space)")
    ap.add_argument("--cols", type=int, default=9, help="Horizontal bands to compare")
    ap.add_argument("--turn-gain", type=int, default=350, help="Max mouse counts/tick toward open")
    ap.add_argument("--invert", action="store_true", help="Flip if near/far is backwards")
    ap.add_argument(
        "--band", type=float, default=0.5, help="Eye-level band centre (0=top,1=bottom)"
    )
    ap.add_argument("--no-jump", action="store_true", help="Disable jump-over-obstacle")
    ap.add_argument("--max-seconds", type=float, default=180.0, help="Hard auto-stop")
    args = ap.parse_args()

    enable_high_resolution_timer()
    with open(os.path.join(PROJECT_ROOT, "config", "settings.yaml")) as f:
        cfg = yaml.safe_load(f)
    print("[depth] loading model...")
    sess, in_name = load_depth()
    cap = ScreenCapture(monitor=cfg["display"]["monitor"], target_fps=60)
    cap.start()
    from src.utils.session_logger import SessionLogger

    logger = SessionLogger(os.path.join(PROJECT_ROOT, "logs"), enabled=True)
    print(f"[depth] ready ({sess.get_providers()[0]}). HOLD L to roam, END to quit.")
    print(f"[depth] logging to {logger.session_dir}")

    walking = False
    last_dump = 0.0
    last_jump = 0.0
    status_t = 0.0
    start = time.perf_counter()
    try:
        while True:
            if held(END_VK):
                break
            now = time.perf_counter()
            if args.max_seconds and now - start > args.max_seconds:
                break
            if not held(RUN_VK):  # dead-man
                if walking:
                    keyboard.key_up("w")
                    mouse.release_all_buttons()
                    walking = False
                time.sleep(0.02)
                continue
            if not walking:
                keyboard.key_down("w")
                walking = True

            frame = cap.grab()
            if frame is None:
                time.sleep(0.002)
                continue
            blob, crop = preprocess(frame)
            depth = sess.run(None, {in_name: blob})[0][0]  # (518,518), higher = closer
            h, w = depth.shape

            # Eye-level band: median depth per column -> openness (far = open).
            y0 = int(h * (args.band - 0.08))
            y1 = int(h * (args.band + 0.08))
            band = depth[max(0, y0) : min(h, y1), :]
            colw = w // args.cols
            near = np.array(
                [np.median(band[:, c * colw : (c + 1) * colw]) for c in range(args.cols)]
            )
            openness = near if args.invert else -near  # want the farthest column
            best = int(np.argmax(openness))
            center = (args.cols - 1) / 2.0
            offset = (best - center) / center  # -1 (left) .. +1 (right)
            turn = int(offset * args.turn_gain)
            if turn:
                mouse.move_relative(turn, 0)

            # Jump a low obstacle: near at the bottom-centre but open just above it.
            jumped = False
            if not args.no_jump and now - last_jump > 0.6:
                bc = depth[int(h * 0.66) : int(h * 0.82), int(w * 0.4) : int(w * 0.6)]
                mc = depth[int(h * 0.42) : int(h * 0.55), int(w * 0.4) : int(w * 0.6)]
                near_bot = float(np.median(bc))
                far_mid = float(np.median(mc))
                close = near_bot if not args.invert else -near_bot
                openmid = -far_mid if not args.invert else far_mid
                if close > np.median(near) and openmid > -np.median(near):
                    mouse.jump()
                    last_jump = now
                    jumped = True

            if now - last_dump >= 0.5:
                dn = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                heat = cv2.applyColorMap(dn, cv2.COLORMAP_INFERNO)
                heat = cv2.resize(heat, (crop.shape[1] // 3, crop.shape[0] // 3))
                hh, hw = heat.shape[:2]
                cx = int((best + 0.5) * hw / args.cols)
                cv2.line(heat, (cx, 0), (cx, hh), (0, 255, 0), 2)  # chosen direction
                cv2.line(heat, (hw // 2, 0), (hw // 2, hh), (255, 255, 255), 1)  # center
                stamp = f"{now - start:06.1f}".replace(".", "_")
                cv2.imwrite(
                    os.path.join(logger.session_dir, f"depth_{stamp}.jpg"),
                    heat,
                    [cv2.IMWRITE_JPEG_QUALITY, 80],
                )
                last_dump = now

            logger.log_tick(
                t=round(now - start, 3), best=best, offset=round(offset, 2), turn=turn, jump=jumped
            )
            if now - status_t >= 0.4:
                bar = "".join("#" if c == best else "." for c in range(args.cols))
                print(
                    f"\r[depth] open[{bar}] turn {turn:4d} {'JUMP' if jumped else '    '}  ",
                    end="",
                    flush=True,
                )
                status_t = now
    finally:
        keyboard.key_up("w")
        mouse.release_all_buttons()
        cap.stop()
        logger.close()
        print("\n[depth] stopped.")


if __name__ == "__main__":
    main()

"""Rung 3 of the assistant ladder: triggerbot.

Builds on Rung 2 (aim assist). Same detector, but instead of nudging the mouse
it watches whether your crosshair is already ON an enemy and, if it is, fires a
short burst -- holding fire while pulling the view down to counter the AK's
upward climb, then releasing so the spray pattern resets. YOU still aim -- this
only pulls the trigger the moment you are lined up, with a short human reaction
delay so it is not an inhuman 0 ms snap.

It does NOT move the mouse (that is Rung 2 -- run them together if you want both).

ARCHITECTURE mirrors aim_assist: detection runs in a background thread and
publishes the current enemy boxes as an atomic snapshot; the main loop checks at
a high rate whether the crosshair (screen centre) sits inside one of those boxes
and decides when to click.

    python tools/triggerbot.py                  # hold MOUSE5, burst when on target
    python tools/triggerbot.py --mode always    # always armed, no key
    python tools/triggerbot.py --burst 10        # longer spray before reset
    python tools/triggerbot.py --recoil-pull 80  # pull down harder per shot
    python tools/triggerbot.py --kill-grace 0.05 # stop firing sooner after a kill

Safety: END quits instantly, the run auto-stops after --max-seconds, the mouse
is only ever clicked while armed, and all buttons are force-released on exit.

NOTE: synthetic fire is VAC-bannable in real matchmaking. Offline / private /
with-friends only -- this is a prank-and-learning tool.
"""

import argparse
import ctypes
import os
import random
import sys
import threading
import time

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.capture.screen import ScreenCapture
from src.input import mouse
from src.utils.timer_setup import enable_high_resolution_timer
from src.vision.confirmation_filter import ConfirmationFilter
from src.vision.detector import YOLODetector

# Virtual-key codes for the arming key.
KEYS = {
    "rmb": 0x02,  # right mouse button
    "mouse4": 0x05,  # XBUTTON1 (side, back)
    "mouse5": 0x06,  # XBUTTON2 (side, forward) -- default, no game conflict
    "shift": 0x10,
    "alt": 0x12,
    "ctrl": 0x11,
    "c": 0x43,
}
END_VK = 0x23


class BoxTracker(threading.Thread):
    """Background capture + detection. Publishes the current enemy boxes.

    The snapshot is a single immutable tuple swapped atomically, so the main
    loop never sees a half-updated list.
    """

    def __init__(self, cfg, fps, confirm):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.fps = fps
        self.confirm = confirm
        self.running = True
        self.ready = threading.Event()
        self.cx = self.cy = 0
        self.screen_w = self.screen_h = 0
        self.boxes = ()  # tuple of (x1, y1, x2, y2)
        self.frames = 0
        self.n_enemies = 0
        self.infer_ms = 0.0

    def run(self):
        det_cfg = self.cfg["detection"]
        cap = ScreenCapture(monitor=self.cfg["display"]["monitor"], target_fps=self.fps)
        cap.start()
        det = YOLODetector(
            model_path=os.path.join(PROJECT_ROOT, det_cfg["model_path"]),
            input_size=det_cfg["input_size"],
            confidence_threshold=det_cfg["confidence_threshold"],
            nms_threshold=det_cfg["nms_threshold"],
            classes=det_cfg["classes"],
        )
        det.load()
        # Fire is higher-stakes than the aim nudge, so confirm a target over more
        # frames before we will shoot at it (default 3 vs the assist's 2).
        conf = ConfirmationFilter(min_confirm_frames=self.confirm, max_missing_frames=3)

        while self.running:
            frame = cap.grab()
            if frame is None:
                time.sleep(0.002)
                continue
            if not self.ready.is_set():
                self.screen_h, self.screen_w = frame.shape[:2]
                self.cx, self.cy = self.screen_w // 2, self.screen_h // 2
                self.ready.set()
            self.frames += 1

            dets = conf.update(det.detect(frame))
            self.infer_ms = det.inference_ms
            # Bodies only -- never trigger off a stray head box. Prefer the body
            # class, fall back to "anything that is not a head" (robust whether
            # the model is the old 2-class one or the new single-class "player").
            enemies = [d for d in dets if d.class_name == "ct_player"]
            if not enemies:
                enemies = [d for d in dets if not getattr(d, "is_head", False)]
            self.n_enemies = len(enemies)
            self.boxes = tuple((d.x1, d.y1, d.x2, d.y2) for d in enemies)
        cap.stop()


def crosshair_in_box(cx, cy, box, inset):
    """True if (cx, cy) is inside `box` after shrinking it by `inset` on each
    side. inset is a fraction of the box size, so edge grazes do not fire."""
    x1, y1, x2, y2 = box
    mx = (x2 - x1) * inset
    my = (y2 - y1) * inset
    return (x1 + mx) <= cx <= (x2 - mx) and (y1 + my) <= cy <= (y2 - my)


def spray_burst(count, interval, pull, on_target, kill_grace):
    """Hold fire and pull the view DOWN each shot to counter the AK's upward
    climb, then release so the in-game spray pattern resets.

    The pull eases in over the first two rounds (bullet one barely kicks) then
    holds steady, so the view tracks down progressively through the climb.

    Stops EARLY once `on_target()` has been false for `kill_grace` seconds, so a
    killed/vanished enemy only eats a bullet or two instead of the whole burst.
    Polled every few ms (not once per shot) so it reacts fast and the END kill
    switch stays responsive. Returns False if END was pressed."""
    user32 = ctypes.windll.user32
    step = 0.012  # poll interval (~1 bullet at 600rpm is 100ms, so this is tight)
    total = count * interval
    next_shot = interval  # time of the next recoil pull
    shot_i = 1
    lost_for = None  # seconds the target has been gone (None = currently on target)
    elapsed = 0.0
    mouse.mouse_down("left")
    try:
        while elapsed < total:
            if user32.GetAsyncKeyState(END_VK) & 0x8000:
                return False
            time.sleep(step)
            elapsed += step
            # Apply the recoil pull at each shot boundary.
            if elapsed >= next_shot and shot_i < count:
                ramp = min(1.0, (shot_i + 1) / 2.0)  # ease in over 2 shots
                dy = int(round(pull * ramp))
                if dy:
                    mouse.move_relative(0, dy)
                shot_i += 1
                next_shot += interval
            # Cut the burst short once the target has been gone long enough.
            if on_target():
                lost_for = None
            else:
                lost_for = step if lost_for is None else lost_for + step
                if lost_for >= kill_grace:
                    break
    finally:
        mouse.mouse_up("left")
    return True


def main():
    ap = argparse.ArgumentParser(description="Rung 3: triggerbot")
    ap.add_argument("--key", default="mouse5", choices=sorted(KEYS), help="Arming key")
    ap.add_argument(
        "--mode",
        default="hold",
        choices=["hold", "toggle", "always"],
        help="hold = while key held; toggle = tap to flip; always = always armed",
    )
    ap.add_argument(
        "--inset",
        type=float,
        default=0.15,
        help="Shrink the hit box by this fraction each side (higher = stricter)",
    )
    ap.add_argument(
        "--reaction-min", type=float, default=0.05, help="Min reaction delay before firing (s)"
    )
    ap.add_argument(
        "--reaction-max", type=float, default=0.12, help="Max reaction delay before firing (s)"
    )
    ap.add_argument("--burst", type=int, default=9, help="Bullets per burst (the AK climb portion)")
    ap.add_argument("--rpm", type=float, default=600.0, help="Fire rate (AK-47 = 600 rounds/min)")
    ap.add_argument(
        "--recoil-pull",
        type=float,
        default=128.0,
        help="Mouse counts pulled DOWN per shot to fight the climb",
    )
    ap.add_argument(
        "--kill-grace",
        type=float,
        default=0.10,
        help="Seconds to keep firing after the target vanishes (lower = stop sooner)",
    )
    ap.add_argument(
        "--cooldown", type=float, default=0.4, help="Min seconds between bursts (lets recoil reset)"
    )
    ap.add_argument(
        "--repeat",
        action="store_true",
        help="Keep firing while aligned (spray). Default = one tap per acquisition.",
    )
    ap.add_argument("--fps", type=int, default=144, help="Detection capture rate")
    ap.add_argument(
        "--confirm", type=int, default=3, help="Frames before a target is shootable (safety)"
    )
    ap.add_argument("--control-hz", type=float, default=250.0, help="Crosshair-check loop rate")
    ap.add_argument("--max-seconds", type=float, default=120.0, help="Auto-stop after N seconds")
    args = ap.parse_args()

    enable_high_resolution_timer()
    cfg = yaml.safe_load(open(os.path.join(PROJECT_ROOT, "config", "settings.yaml")))
    arm_vk = KEYS[args.key]
    user32 = ctypes.windll.user32

    tracker = BoxTracker(cfg, args.fps, args.confirm)
    tracker.start()
    print("[trigger] loading detector + capture...")
    if not tracker.ready.wait(timeout=30):
        print("[trigger] capture never produced a frame -- aborting.")
        return
    cx, cy = tracker.cx, tracker.cy
    print(f"[trigger] {tracker.screen_w}x{tracker.screen_h} crosshair ({cx},{cy})")
    how = {
        "hold": f"HOLD {args.key.upper()}",
        "toggle": f"TAP {args.key.upper()}",
        "always": "ARMED",
    }[args.mode]
    print(f"[trigger] running -- {how} to arm, END to quit (mode={args.mode})")

    period = 1.0 / args.control_hz
    toggled_on = False
    key_was_down = False
    aligned_since = None  # when continuous alignment began (None = not aligned)
    react_delay = 0.0  # rolled fresh each time alignment begins
    fired_this_onset = False
    last_fire = 0.0
    last = time.perf_counter()
    status_t = last
    start = last
    try:
        while True:
            if user32.GetAsyncKeyState(END_VK) & 0x8000:
                break
            if args.max_seconds and time.perf_counter() - start > args.max_seconds:
                print("\n[trigger] max-seconds reached, stopping.")
                break

            now = time.perf_counter()

            # Resolve whether we are armed this tick from the chosen mode.
            key_down = bool(user32.GetAsyncKeyState(arm_vk) & 0x8000)
            if args.mode == "always":
                armed = True
            elif args.mode == "toggle":
                if key_down and not key_was_down:
                    toggled_on = not toggled_on
                armed = toggled_on
            else:
                armed = key_down
            key_was_down = key_down

            # Is the crosshair sitting inside an enemy body right now?
            aligned = armed and any(crosshair_in_box(cx, cy, b, args.inset) for b in tracker.boxes)

            fired = False
            if aligned:
                if aligned_since is None:  # alignment just began -> roll reaction time
                    aligned_since = now
                    react_delay = random.uniform(args.reaction_min, args.reaction_max)
                    fired_this_onset = False
                ready = now - aligned_since >= react_delay
                cooled = now - last_fire >= args.cooldown
                allowed = args.repeat or not fired_this_onset
                if ready and cooled and allowed:
                    interval = 60.0 / args.rpm

                    def on_target():
                        return any(crosshair_in_box(cx, cy, b, args.inset) for b in tracker.boxes)

                    alive = spray_burst(
                        args.burst, interval, args.recoil_pull, on_target, args.kill_grace
                    )
                    last_fire = time.perf_counter()  # measure cooldown from burst END
                    fired_this_onset = True
                    fired = True
                    if not alive:  # END pressed mid-burst
                        break
            else:
                aligned_since = None  # lost alignment -> require re-acquire

            if now - status_t >= 0.5:
                if fired:
                    state = "FIRE "
                elif aligned:
                    state = "ONTGT"
                elif armed:
                    state = "ARMED"
                else:
                    state = "idle "
                print(
                    f"\r[trigger] {state} | enemies {tracker.n_enemies} | "
                    f"cap {tracker.frames} | infer {tracker.infer_ms:.0f}ms   ",
                    end="",
                    flush=True,
                )
                status_t = now

            last = now
            sleep_for = period - (time.perf_counter() - now)
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        tracker.running = False
        mouse.release_all_buttons()
        print("\n[trigger] stopped.")


if __name__ == "__main__":
    main()

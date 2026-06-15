"""Rung 1 (overlay version): draw threat highlights ON the real game.

A transparent, click-through, always-on-top window covering the screen. A
background thread captures + runs YOLO; the overlay paints a red box on each
detected enemy directly over CS2. Because it's click-through, your mouse and
keyboard pass straight through to the game -- you play normally, the boxes just
appear on top. No input routing, no focus fight.

REQUIREMENT: run CS2 in "Fullscreen Windowed" (borderless). A transparent
overlay cannot draw over true exclusive-fullscreen.

    python tools/threat_overlay.py            # 5 second countdown, then overlay
    python tools/threat_overlay.py --delay 8  # longer countdown

After launching, ALT-TAB into CS2 during the countdown so the overlay lands on
top of the game. Press END to quit. View-only: never touches mouse/keyboard.
"""

import argparse
import ctypes
import os
import sys
import threading
import time
import tkinter as tk

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.capture.screen import ScreenCapture
from src.vision.confirmation_filter import ConfirmationFilter
from src.vision.detector import YOLODetector

END_VK = 0x23
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020


class Detector(threading.Thread):
    """Background capture + detection; publishes the latest boxes."""

    def __init__(self, cfg):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.boxes: list[tuple[int, int, int, int, bool, float]] = []
        self.frames = 0  # capture frames seen -- proves the pipeline is feeding
        self.running = True

    def run(self):
        det_cfg = self.cfg["detection"]
        cap = ScreenCapture(
            monitor=self.cfg["display"]["monitor"], target_fps=self.cfg["display"]["capture_fps"]
        )
        cap.start()
        det = YOLODetector(
            model_path=os.path.join(PROJECT_ROOT, det_cfg["model_path"]),
            input_size=det_cfg["input_size"],
            confidence_threshold=det_cfg["confidence_threshold"],
            nms_threshold=det_cfg["nms_threshold"],
            classes=det_cfg["classes"],
        )
        det.load()
        conf = ConfirmationFilter()
        while self.running:
            frame = cap.grab()
            if frame is None:
                time.sleep(0.003)
                continue
            self.frames += 1
            dets = conf.update(det.detect(frame))
            self.boxes = [
                (
                    int(d.x1),
                    int(d.y1),
                    int(d.x2),
                    int(d.y2),
                    getattr(d, "is_head", False),
                    float(d.confidence),
                )
                for d in dets
            ]
        cap.stop()


def main():
    ap = argparse.ArgumentParser(description="Rung 1 overlay: on-game threat highlights")
    ap.add_argument(
        "--delay",
        type=float,
        default=5.0,
        help="Countdown (seconds) before the overlay appears -- tab into CS2 during it",
    )
    args = ap.parse_args()

    # Match overlay pixels to capture pixels regardless of Windows scaling.
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()

    cfg = yaml.safe_load(open(os.path.join(PROJECT_ROOT, "config", "settings.yaml")))
    # Start capture + model load now so it warms up while you tab into the game.
    worker = Detector(cfg)
    worker.start()

    # Countdown: switch to CS2 (Fullscreen-Windowed) before the overlay shows.
    for remaining in range(int(args.delay), 0, -1):
        print(f"[overlay] alt-tab into CS2... starting in {remaining}s")
        time.sleep(1)

    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.attributes("-transparentcolor", "black")
    root.config(bg="black")
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{sw}x{sh}+0+0")
    canvas = tk.Canvas(root, width=sw, height=sh, bg="black", highlightthickness=0)
    canvas.pack()

    root.update_idletasks()  # realize the window so the HWND + styles exist
    user32 = ctypes.windll.user32

    # Make the overlay click-through so input falls through to the game. Target
    # the real top-level window (GA_ROOT) -- for an overrideredirect window the
    # GetParent trick can return the wrong handle and the styles silently no-op.
    hwnd = user32.GetAncestor(root.winfo_id(), 2)  # GA_ROOT
    ex = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex | WS_EX_LAYERED | WS_EX_TRANSPARENT)
    # Changing the style can drop Tk's colour-key transparency, leaving an opaque
    # black sheet over the game. Re-assert the black colour key explicitly.
    LWA_COLORKEY = 0x1
    user32.SetLayeredWindowAttributes(hwnd, 0x000000, 0, LWA_COLORKEY)

    def redraw():
        if user32.GetAsyncKeyState(END_VK) & 0x8000:
            worker.running = False
            root.destroy()
            return
        canvas.delete("all")
        boxes = worker.boxes

        # Always-on proof-of-life HUD (draws even with zero detections). If you
        # see this but no enemy boxes -> detection found nothing. If you see
        # nothing at all -> the overlay itself isn't painting over the game.
        canvas.create_rectangle(8, 8, 360, 56, outline="#00FF00", width=2)
        canvas.create_text(
            16,
            16,
            anchor="nw",
            fill="#00FF00",
            text=f"OVERLAY ALIVE  cap={worker.frames}  boxes={len(boxes)}",
            font=("Consolas", 14, "bold"),
        )
        for x1, y1, x2, y2, head, c in boxes:
            colour = "#FFC800" if head else "#FF0000"  # head=orange, body=red
            canvas.create_rectangle(x1, y1, x2, y2, outline=colour, width=3)
            canvas.create_text(
                x1 + 2,
                y1 - 10,
                anchor="w",
                fill=colour,
                text=f"{c:.2f}",
                font=("Consolas", 11, "bold"),
            )
        if boxes:
            canvas.create_rectangle(2, 2, sw - 2, sh - 2, outline="#FF0000", width=8)
            canvas.create_text(
                20,
                20,
                anchor="nw",
                fill="#FF0000",
                text=f"THREAT x{len(boxes)}",
                font=("Consolas", 22, "bold"),
            )
        root.after(50, redraw)  # ~20 fps overlay refresh

    print("[overlay] running. CS2 must be Fullscreen-Windowed. END to quit.")
    root.after(50, redraw)
    root.mainloop()
    print("[overlay] stopped.")


if __name__ == "__main__":
    main()

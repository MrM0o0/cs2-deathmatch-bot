"""Main loop orchestrator for the CS2 Deathmatch Bot."""

import ctypes
import os
import random
import sys
import threading
import time

import yaml

# Hotkeys read globally via GetAsyncKeyState, so they work while CS2 has focus.
# END  -> kill the bot instantly and release all input.
# HOME -> toggle pause: the bot lets go of mouse/keyboard so you can take over,
#         press again to hand control back. The process keeps running.
PANIC_VK = 0x23
PANIC_KEY_NAME = "END"
PAUSE_VK = 0x24
PAUSE_KEY_NAME = "HOME"

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.aim.mouse_mover import MouseMover
from src.aim.recoil import RecoilCompensator
from src.aim.targeting import TargetingSystem
from src.brain.decision import Action, DecisionMaker
from src.brain.priorities import ThreatAssessor
from src.brain.state_machine import StateMachine
from src.capture.screen import ScreenCapture
from src.humanizer.mistakes import MistakeMaker
from src.humanizer.noise import NoiseGenerator
from src.humanizer.personality import load_personality
from src.humanizer.timing import ReactionTimer
from src.input import keyboard, mouse
from src.movement.bc_policy import BCMovementPolicy
from src.movement.explorer import WallFollower
from src.movement.navigator import NavigationController, WaypointGraph
from src.movement.stuck_detector import StuckDetector
from src.utils.debug_overlay import DebugOverlay
from src.utils.session_logger import SessionLogger
from src.vision.confirmation_filter import ConfirmationFilter
from src.vision.detector import Detection, YOLODetector
from src.vision.hud_reader import HUDReader
from src.vision.minimap import MinimapReader


def _round(value, ndigits: int = 1):
    """Round floats for compact logging; pass through None/other types."""
    return round(value, ndigits) if isinstance(value, (int, float)) else value


def load_config(path: str = "config/settings.yaml") -> dict:
    """Load main configuration."""
    config_path = os.path.join(PROJECT_ROOT, path)
    with open(config_path) as f:
        return yaml.safe_load(f)


class Bot:
    """Main bot orchestrator."""

    def __init__(self, personality_name: str | None = None):
        self.config = load_config()
        self.running = False

        # Personality
        pname = personality_name or self.config["bot"]["default_personality"]
        self.personality = load_personality(
            pname, os.path.join(PROJECT_ROOT, "config", "personalities")
        )
        print(f"[Bot] Loaded personality: {self.personality}")

        # Screen capture
        display = self.config["display"]
        self.capture = ScreenCapture(
            monitor=display["monitor"],
            target_fps=display["capture_fps"],
        )

        # Vision
        det_cfg = self.config["detection"]
        self.detector = YOLODetector(
            model_path=os.path.join(PROJECT_ROOT, det_cfg["model_path"]),
            input_size=det_cfg["input_size"],
            confidence_threshold=det_cfg["confidence_threshold"],
            nms_threshold=det_cfg["nms_threshold"],
            classes=det_cfg["classes"],
        )
        conf_cfg = self.config.get("confirmation", {})
        self.confirmation_filter = ConfirmationFilter(
            min_confirm_frames=conf_cfg.get("min_confirm_frames", 3),
            max_missing_frames=conf_cfg.get("max_missing_frames", 2),
            match_distance=conf_cfg.get("match_distance", 150.0),
            match_size_ratio=conf_cfg.get("match_size_ratio", 0.4),
        )
        self.hud_reader = HUDReader(self.config["regions"])
        mm = self.config["minimap"]
        self.minimap_reader = MinimapReader(
            mm["x"],
            mm["y"],
            mm["size"],
            sat_min=mm.get("sat_min", 150),
            val_min=mm.get("val_min", 190),
        )

        # Brain
        self.state_machine = StateMachine()
        self.decision_maker = DecisionMaker(self.personality)
        game = self.config["game"]
        self.threat_assessor = ThreatAssessor((game["crosshair_x"], game["crosshair_y"]))

        # Aim
        self.targeting = TargetingSystem(
            game["crosshair_x"],
            game["crosshair_y"],
            game["sensitivity"],
            game["m_yaw"],
            game["m_pitch"],
            self.personality.head_aim_chance,
        )
        self.mouse_mover = MouseMover(
            base_speed=self.personality.aim_speed,
            noise_amplitude=self.personality.tracking_error / 4,
        )
        self.recoil = RecoilCompensator(
            compensation_factor=self.personality.recoil_compensation,
        )

        # Movement
        nav_cfg = self.config["navigation"]
        self.explorer = WallFollower(
            wall_threshold=nav_cfg["wall_avoid_threshold"],
        )
        self.nav_graph = WaypointGraph()
        self.nav_controller = NavigationController(
            self.nav_graph,
            reach_dist=nav_cfg["waypoint_reach_dist"],
            turn_gain=nav_cfg.get("nav_turn_gain", 1.6),
            max_turn=nav_cfg.get("nav_max_turn", 22),
            invert_turn=nav_cfg.get("nav_invert_turn", False),
        )

        # Movement mode: "waypoint" (hand-written nav) or "bc" (learned policy).
        self.movement_mode = self.config["bot"].get("movement_mode", "waypoint")
        self.bc_policy = None
        if self.movement_mode == "bc":
            bc = self.config.get("bc_movement", {})
            self.bc_policy = BCMovementPolicy(
                os.path.join(PROJECT_ROOT, bc.get("model_path", "models/movement.onnx")),
                mild_turn=bc.get("mild_turn", 150),
                hard_turn=bc.get("hard_turn", 400),
                invert_turn=bc.get("invert_turn", False),
            )
        self.stuck_detector = StuckDetector(
            timeout=nav_cfg["stuck_timeout"],
        )

        # Humanizer
        p = self.personality
        self.reaction_timer = ReactionTimer(
            p.reaction_mean_ms,
            p.reaction_std_ms,
            p.reaction_min_ms,
            p.reaction_max_ms,
        )
        self.mistake_maker = MistakeMaker(
            p.overshoot_chance,
            p.overshoot_magnitude,
            p.tracking_error,
        )
        self.noise = NoiseGenerator()

        # Debug
        self.debug = None
        if self.config["bot"]["debug_overlay"]:
            self.debug = DebugOverlay(scale=0.5)

        # Session logging for after-the-fact diagnosis of a run.
        self.logger = SessionLogger(
            PROJECT_ROOT,
            enabled=self.config["bot"].get("debug_log", False),
        )
        self._last_nav_cmd: dict = {}

        # State
        self._is_firing = False
        self._movement_keys_held: set[str] = set()
        self._tick_count = 0
        self.paused = False

    def start(self) -> None:
        """Initialize all systems and start the main loop."""
        print("[Bot] Starting...")

        # Start screen capture
        backend = self.capture.start()
        print(f"[Bot] Screen capture: {backend}")

        # Load YOLO model
        try:
            self.detector.load()
        except Exception as e:
            print(f"[Bot] WARNING: Detector failed to load: {e}")
            print("[Bot] Running without detection (debug/movement only)")

        # Load the learned movement policy, or fall back to waypoint nav.
        if self.bc_policy is not None:
            try:
                self.bc_policy.load()
                print("[Bot] Movement: behavioural-cloning policy")
            except Exception as e:
                print(f"[Bot] WARNING: BC policy failed to load ({e}); using waypoint nav")
                self.bc_policy = None

        # Load waypoints if available
        map_name = "dust2"  # TODO: auto-detect map
        wp_path = os.path.join(PROJECT_ROOT, "config", "maps", f"{map_name}.json")
        if os.path.exists(wp_path):
            self.nav_graph.load(wp_path)
            print(f"[Bot] Loaded waypoints for {map_name}")

        self.running = True
        self._loop_start = time.perf_counter()
        self._max_run_seconds = self.config["bot"].get("max_run_seconds", 120)
        # Dedicated watchdog thread polls the panic key every 20ms, independent
        # of the (sometimes slow) main loop, and force-exits. Can't be starved.
        threading.Thread(target=self._panic_watchdog, daemon=True).start()
        print("[Bot] Ready.")
        print(
            f"[Bot] *** {PANIC_KEY_NAME} = STOP instantly | {PAUSE_KEY_NAME} = "
            f"pause/resume (take control) -- both work while CS2 is focused. ***"
        )
        if self._max_run_seconds:
            print(f"[Bot] Auto-stops after {self._max_run_seconds}s as a failsafe.")
        print(f"[Bot] Tick rate: {self.config['bot']['tick_rate']} Hz")

        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\n[Bot] Stopped by user.")
        finally:
            self.stop()

    def _release_input(self) -> None:
        """Let go of mouse and keyboard (used by pause and panic)."""
        for cleanup in (self._stop_firing, self._release_all_movement, keyboard.release_all):
            try:
                cleanup()
            except Exception:
                pass

    def _panic_watchdog(self) -> None:
        """Handle the global hotkeys: END = kill, HOME = pause/resume toggle.

        Runs in its own thread polling every 20ms, so a slow/blocked main loop
        can never delay it.
        """
        user32 = ctypes.windll.user32
        prev_pause_down = False
        while self.running:
            if user32.GetAsyncKeyState(PANIC_VK) & 0x8000:
                print(f"\n[Bot] PANIC ({PANIC_KEY_NAME}) -- force stopping NOW.")
                self.running = False
                self._release_input()
                try:
                    self.logger.close()
                except Exception:
                    pass
                os._exit(0)

            # Edge-detect HOME so a held key toggles once, not every poll.
            pause_down = bool(user32.GetAsyncKeyState(PAUSE_VK) & 0x8000)
            if pause_down and not prev_pause_down:
                self.paused = not self.paused
                if self.paused:
                    self._release_input()
                    print(
                        f"\n[Bot] PAUSED -- input released. {PAUSE_KEY_NAME} "
                        f"to resume, {PANIC_KEY_NAME} to quit."
                    )
                else:
                    print("\n[Bot] RESUMED.")
            prev_pause_down = pause_down

            time.sleep(0.02)

    def _should_abort(self) -> bool:
        """True if the panic key is down or the run time limit was hit."""
        # High bit set => key currently pressed (read globally, focus-independent)
        if ctypes.windll.user32.GetAsyncKeyState(PANIC_VK) & 0x8000:
            print(f"\n[Bot] PANIC key ({PANIC_KEY_NAME}) pressed -- stopping.")
            return True
        if self._max_run_seconds and (
            time.perf_counter() - self._loop_start > self._max_run_seconds
        ):
            print(f"\n[Bot] Run time limit ({self._max_run_seconds}s) reached -- stopping.")
            return True
        return False

    def _main_loop(self) -> None:
        """Main 30 Hz game loop."""
        tick_interval = 1.0 / self.config["bot"]["tick_rate"]

        while self.running:
            tick_start = time.perf_counter()
            self._tick_count += 1

            # 0. Panic / time-limit check BEFORE doing anything else this tick.
            if self._should_abort():
                self.running = False
                break

            # 0b. Paused (HOME): hold off all input/decisions, keep the loop alive.
            if self.paused:
                time.sleep(0.05)
                continue

            # 1. Capture frame
            frame = self.capture.grab()
            if frame is None:
                time.sleep(0.001)
                continue

            # 2. Run detection + confirmation filter
            raw_detections = self.detector.detect(frame)
            detections = self.confirmation_filter.update(raw_detections)

            # 3. Read HUD
            hud = self.hud_reader.read(frame)

            # 4. Check stuck
            is_moving = len(self._movement_keys_held) > 0
            is_stuck = self.stuck_detector.update(frame, is_moving)

            # 5. Prioritize targets
            enemies = self.threat_assessor.prioritize_targets(detections)

            # 6. Update state machine
            self.state_machine.update(
                is_alive=hud.is_alive,
                enemies_visible=len(enemies),
                health=hud.health,
                is_stuck=is_stuck,
                disengage_health=self.personality.disengage_health,
            )

            # 7. Make decision
            action = self.decision_maker.decide(
                state=self.state_machine.state,
                enemies=enemies,
                health=hud.health,
                ammo_clip=hud.ammo_clip,
                time_in_state=self.state_machine.time_in_state,
            )

            # 8. Execute action
            self._execute_action(action, frame, enemies)

            # 8b. Session logging (cheap; throttled frame saves)
            if self.logger.enabled:
                nav = self._last_nav_cmd
                self.logger.log_tick(
                    state=self.state_machine.state.name,
                    action=action.type,
                    enemies=len(enemies),
                    health=hud.health,
                    ammo=hud.ammo_clip,
                    alive=hud.is_alive,
                    stuck=is_stuck,
                    nav_heading=_round(nav.get("heading_deg")),
                    nav_yaw_err=_round(nav.get("yaw_error_deg")),
                    nav_turn=nav.get("turn_x"),
                    nav_route=nav.get("has_route"),
                    infer_ms=round(self.detector.inference_ms, 1),
                )
                self.logger.maybe_save_frame(
                    frame,
                    label=f"{self.state_machine.state.name} | {action.type} | "
                    f"en={len(enemies)} hp={hud.health}",
                )

            # 9. Debug overlay
            if self.debug:
                vis = self.debug.draw(
                    frame,
                    detections,
                    self.state_machine.state,
                    hud_info=str(hud),
                    inference_ms=self.detector.inference_ms,
                    extra_lines=[
                        f"Action: {action.type}",
                        f"Raw: {len(raw_detections)} | Confirmed: {len(detections)}",
                        f"Stuck recoveries: {self.stuck_detector.recovery_count}",
                    ],
                )
                if not self.debug.show(vis):
                    self.running = False

            # 10. Sleep remainder of tick
            elapsed = time.perf_counter() - tick_start
            sleep_time = tick_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _execute_action(self, action: Action, frame, enemies: list[Detection]) -> None:
        """Execute a decided action."""
        keybinds = self.config["keybinds"]

        if action.type == "wait":
            self._release_all_movement()
            self._stop_firing()

        elif action.type == "click":
            mouse.click("left")

        elif action.type == "engage":
            target = action.params.get("target")
            fire_mode = action.params.get("fire_mode", "spray")
            combat_move = action.params.get("combat_move")

            if target:
                self._aim_at_target(target)
                self._handle_fire_mode(fire_mode)
                self._handle_combat_movement(combat_move, keybinds)

        elif action.type == "reload":
            self._stop_firing()
            keyboard.key_press(keybinds["reload"])

        elif action.type == "roam":
            self._stop_firing()
            self._roam(frame, keybinds)

        elif action.type == "search":
            self._stop_firing()
            self._roam(frame, keybinds)

        elif action.type == "check_corner":
            direction = action.params.get("direction", "left")
            turn_amount = 8 if direction == "right" else -8
            mouse.move_relative(turn_amount, 0)

        elif action.type == "flee":
            self._stop_firing()
            self._release_all_movement()
            # Turn away and run
            enemy = action.params.get("enemy")
            if enemy:
                dx, dy, _ = self.targeting.get_aim_delta(enemy)
                mouse.move_relative(-dx // 4, 0)  # Turn away
            keyboard.hold_key(keybinds["forward"])
            self._movement_keys_held.add(keybinds["forward"])
            if random.random() < 0.3:
                keyboard.key_press(keybinds["jump"])

        elif action.type == "unstick":
            phase = action.params.get("phase", "backup")
            self._handle_unstick(phase, keybinds)

        elif action.type == "inspect_weapon":
            keyboard.key_press(keybinds["inspect"])

        elif action.type == "look_around":
            dx = random.randint(-50, 50)
            dy = random.randint(-15, 15)
            self.mouse_mover.move_to_delta(dx, dy, duration_ms=200)

        elif action.type == "jump":
            keyboard.key_press(keybinds["jump"])

    def _aim_at_target(self, target: Detection) -> None:
        """Aim at a detected enemy with humanization."""
        if not self.reaction_timer.is_ready():
            return

        dx, dy, dist = self.targeting.get_aim_delta(target)

        if dist < 30:
            # Already on target, just apply small correction
            self.mouse_mover.micro_correct(dx, dy)
            return

        # Apply aim error
        aim_x, aim_y = self.mistake_maker.apply_aim_error(dx, dy)

        # Decide overshoot
        if self.mistake_maker.should_overshoot():
            aim_x, aim_y = self.mistake_maker.overshoot_target(0, 0, aim_x, aim_y)
            # Main flick (overshoots)
            self.mouse_mover.move_to_delta(aim_x, aim_y)
            # Correction back to target
            correction_x = dx - aim_x
            correction_y = dy - aim_y
            time.sleep(self.personality.correction_delay_ms / 1000)
            self.mouse_mover.micro_correct(correction_x, correction_y)
        else:
            self.mouse_mover.move_to_delta(aim_x, aim_y)

        # Start new reaction timer for next target acquisition
        if dist > 200:
            self.reaction_timer.start_reaction()

    def _handle_fire_mode(self, fire_mode: str) -> None:
        """Handle firing based on fire mode."""
        if fire_mode == "tap":
            mouse.click("left", hold_ms=random.uniform(20, 50))
            self.recoil.reset()
            self._is_firing = False
        elif fire_mode == "spray":
            if not self._is_firing:
                mouse.mouse_down("left")
                self._is_firing = True
            self.recoil.apply()
        elif fire_mode == "burst_end":
            self._stop_firing()
            self.recoil.reset()

    def _stop_firing(self) -> None:
        """Stop firing if currently firing."""
        if self._is_firing:
            mouse.mouse_up("left")
            self._is_firing = False
            self.recoil.reset()

    def _handle_combat_movement(self, move: str | None, keybinds: dict) -> None:
        """Apply combat movement (strafing, crouching)."""
        if move == "crouch":
            keyboard.hold_key(keybinds["crouch"])
            self._movement_keys_held.add(keybinds["crouch"])
        elif move == "strafe_left":
            self._release_all_movement()
            keyboard.hold_key(keybinds["left"])
            self._movement_keys_held.add(keybinds["left"])
        elif move == "strafe_right":
            self._release_all_movement()
            keyboard.hold_key(keybinds["right"])
            self._movement_keys_held.add(keybinds["right"])

    def _roam(self, frame, keybinds: dict) -> None:
        """Handle roaming movement."""
        # Learned behavioural-cloning policy takes priority when loaded.
        if self.bc_policy is not None:
            cmd = self.bc_policy.act(frame)
            self._last_nav_cmd = cmd
            self._release_all_movement()
            for flag, bind in (
                ("forward", "forward"),
                ("back", "back"),
                ("left", "left"),
                ("right", "right"),
                ("crouch", "crouch"),
            ):
                if cmd.get(flag):
                    keyboard.hold_key(keybinds[bind])
                    self._movement_keys_held.add(keybinds[bind])
            if cmd["turn_x"] != 0:
                mouse.move_relative(cmd["turn_x"], 0)
            return

        # Use waypoint navigation if available
        if self.nav_controller.has_waypoints():
            pos, _ = self.minimap_reader.read(frame)
            cmd = self.nav_controller.update(pos)
            self._last_nav_cmd = cmd
            if cmd["has_route"]:
                self._release_all_movement()
                if cmd["forward"]:
                    keyboard.hold_key(keybinds["forward"])
                    self._movement_keys_held.add(keybinds["forward"])
                if cmd["back"]:
                    keyboard.hold_key(keybinds["back"])
                    self._movement_keys_held.add(keybinds["back"])
                if cmd["left"]:
                    keyboard.hold_key(keybinds["left"])
                    self._movement_keys_held.add(keybinds["left"])
                if cmd["right"]:
                    keyboard.hold_key(keybinds["right"])
                    self._movement_keys_held.add(keybinds["right"])
                if cmd["turn_x"] != 0:
                    mouse.move_relative(cmd["turn_x"], 0)
                return

        # Fallback: reactive wall-following
        movement = self.explorer.get_movement(frame)

        self._release_all_movement()

        if movement["forward"]:
            keyboard.hold_key(keybinds["forward"])
            self._movement_keys_held.add(keybinds["forward"])

        if movement["left"]:
            keyboard.hold_key(keybinds["left"])
            self._movement_keys_held.add(keybinds["left"])
        elif movement["right"]:
            keyboard.hold_key(keybinds["right"])
            self._movement_keys_held.add(keybinds["right"])

        if movement["turn_x"] != 0:
            mouse.move_relative(movement["turn_x"], 0)

    def _handle_unstick(self, phase: str, keybinds: dict) -> None:
        """Handle stuck recovery phases."""
        self._release_all_movement()

        if phase == "backup":
            keyboard.hold_key(keybinds["back"])
            self._movement_keys_held.add(keybinds["back"])
        elif phase == "turn":
            # Turn 90 degrees
            mouse.move_relative(random.choice([-60, 60]), 0)
        elif phase == "forward":
            keyboard.hold_key(keybinds["forward"])
            self._movement_keys_held.add(keybinds["forward"])
            self.stuck_detector.reset()

    def _release_all_movement(self) -> None:
        """Release all held movement keys."""
        for key in list(self._movement_keys_held):
            keyboard.release_key(key)
        self._movement_keys_held.clear()

    def stop(self) -> None:
        """Clean shutdown."""
        print("[Bot] Shutting down...")
        self._stop_firing()
        self._release_all_movement()
        keyboard.release_all()
        self.capture.stop()
        if self.debug:
            self.debug.cleanup()
        self.logger.close()
        print("[Bot] Stopped.")


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="CS2 Deathmatch Bot")
    parser.add_argument(
        "--personality",
        "-p",
        type=str,
        default=None,
        help="Personality profile (noob, average, tryhard)",
    )
    parser.add_argument("--no-debug", action="store_true", help="Disable debug overlay")
    args = parser.parse_args()

    bot = Bot(personality_name=args.personality)

    if args.no_debug:
        bot.debug = None

    bot.start()


if __name__ == "__main__":
    main()

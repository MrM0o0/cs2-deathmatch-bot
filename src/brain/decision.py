"""High-level decision making for the bot."""

import random
import time

from src.brain.state_machine import BotState, StateMachine
from src.vision.detector import Detection
from src.humanizer.personality import Personality


class Action:
    """A decision output that the main loop executes."""

    def __init__(self, action_type: str, **kwargs):
        self.type = action_type
        self.params = kwargs

    def __repr__(self) -> str:
        return f"Action({self.type}, {self.params})"


class DecisionMaker:
    """Makes high-level decisions based on game state and personality."""

    def __init__(self, personality: Personality):
        self.personality = personality
        self._last_fire_time = 0.0
        self._spray_count = 0
        self._current_target: Detection | None = None

    def decide(self, state: BotState, enemies: list[Detection],
               health: int, ammo_clip: int,
               time_in_state: float) -> Action:
        """Make a decision based on current state.

        Returns an Action to execute.
        """
        if state == BotState.DEAD:
            return self._decide_dead(time_in_state)
        elif state == BotState.FIGHTING:
            return self._decide_fighting(enemies, health, ammo_clip)
        elif state == BotState.SEARCHING:
            return self._decide_searching(time_in_state)
        elif state == BotState.RETREATING:
            return self._decide_retreating(enemies)
        elif state == BotState.STUCK:
            return self._decide_stuck(time_in_state)
        else:  # ROAMING
            return self._decide_roaming()

    def _decide_dead(self, time_in_state: float) -> Action:
        """Wait for respawn. In DM, respawn is automatic."""
        # Occasionally click mouse to respawn faster
        if time_in_state > 1.0 and random.random() < 0.1:
            return Action("click")
        return Action("wait")

    def _decide_fighting(self, enemies: list[Detection],
                         health: int, ammo_clip: int) -> Action:
        """Engage enemies in combat."""
        if not enemies:
            return Action("search")

        # Need to reload?
        if ammo_clip <= 0:
            self._spray_count = 0
            return Action("reload")

        # Select target (closest to crosshair)
        target = enemies[0]  # Pre-sorted by targeting system

        # Decide fire mode
        fire_mode = self._choose_fire_mode()

        # Combat movement
        move = None
        if self.personality.strafe_while_shooting:
            if random.random() < self.personality.crouch_spray_chance:
                move = "crouch"
            else:
                move = random.choice(["strafe_left", "strafe_right"])

        return Action("engage",
                      target=target,
                      fire_mode=fire_mode,
                      combat_move=move)

    def _choose_fire_mode(self) -> str:
        """Choose between tap, burst, or spray."""
        if random.random() < self.personality.tap_chance:
            self._spray_count = 0
            return "tap"

        burst_lo, burst_hi = self.personality.burst_length
        if self._spray_count >= random.randint(burst_lo, burst_hi):
            self._spray_count = 0
            return "burst_end"  # Stop firing briefly

        self._spray_count += 1
        return "spray"

    def _decide_searching(self, time_in_state: float) -> Action:
        """Search for enemies after losing sight."""
        # Check corners by rotating view
        if time_in_state < 1.0:
            return Action("check_corner", direction="left")
        elif time_in_state < 2.0:
            return Action("check_corner", direction="right")
        return Action("roam")

    def _decide_retreating(self, enemies: list[Detection]) -> Action:
        """Run away from danger."""
        if enemies:
            # Turn away from nearest enemy
            return Action("flee", enemy=enemies[0])
        return Action("roam")

    def _decide_stuck(self, time_in_state: float) -> Action:
        """Recover from being stuck."""
        if time_in_state < 1.0:
            return Action("unstick", phase="backup")
        elif time_in_state < 2.0:
            return Action("unstick", phase="turn")
        return Action("unstick", phase="forward")

    def _decide_roaming(self) -> Action:
        """Normal roaming behavior with idle actions."""
        p = self.personality

        # Random idle behaviors
        roll = random.random()
        if roll < p.inspect_chance:
            return Action("inspect_weapon")
        elif roll < p.inspect_chance + p.look_around_chance:
            return Action("look_around")
        elif roll < p.inspect_chance + p.look_around_chance + p.random_jump_chance:
            return Action("jump")

        return Action("roam")

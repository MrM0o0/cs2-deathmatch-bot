"""Finite State Machine for bot behavior."""

import time
from enum import Enum, auto


class BotState(Enum):
    DEAD = auto()        # Waiting to respawn
    ROAMING = auto()     # Moving around the map, no enemies visible
    SEARCHING = auto()   # Recently lost sight of enemy, checking corners
    FIGHTING = auto()    # Enemy detected, engaging in combat
    RETREATING = auto()  # Low health, trying to disengage
    STUCK = auto()       # Detected stuck, performing recovery


class StateMachine:
    """Manages bot state transitions with timing."""

    def __init__(self):
        self.state = BotState.DEAD
        self._state_start = time.perf_counter()
        self._prev_state = BotState.DEAD
        self._state_data: dict = {}

    @property
    def time_in_state(self) -> float:
        """Seconds spent in current state."""
        return time.perf_counter() - self._state_start

    @property
    def previous_state(self) -> BotState:
        return self._prev_state

    def transition(self, new_state: BotState, **data) -> None:
        """Transition to a new state."""
        if new_state == self.state:
            return
        self._prev_state = self.state
        self.state = new_state
        self._state_start = time.perf_counter()
        self._state_data = data

    def get_data(self, key: str, default=None):
        """Get state transition data."""
        return self._state_data.get(key, default)

    def update(self, is_alive: bool, enemies_visible: int,
               health: int, is_stuck: bool,
               disengage_health: int = 30) -> BotState:
        """Update state based on current game conditions.

        Returns the new state after evaluation.
        """
        # Death overrides everything
        if not is_alive:
            self.transition(BotState.DEAD)
            return self.state

        # Stuck detection overrides normal behavior
        if is_stuck and self.state != BotState.STUCK:
            self.transition(BotState.STUCK)
            return self.state

        # Stuck recovery - return to roaming after 3 seconds
        if self.state == BotState.STUCK and self.time_in_state > 3.0:
            self.transition(BotState.ROAMING)
            return self.state

        # Combat states
        if enemies_visible > 0:
            if health <= disengage_health:
                self.transition(BotState.RETREATING)
            else:
                self.transition(BotState.FIGHTING)
            return self.state

        # Transition from fighting -> searching when enemies disappear
        if self.state == BotState.FIGHTING:
            self.transition(BotState.SEARCHING,
                            search_start=time.perf_counter())
            return self.state

        # Search timeout -> roaming
        if self.state == BotState.SEARCHING and self.time_in_state > 3.0:
            self.transition(BotState.ROAMING)
            return self.state

        # Retreat -> roaming when safe
        if self.state == BotState.RETREATING and enemies_visible == 0:
            self.transition(BotState.ROAMING)
            return self.state

        # Dead -> roaming on respawn
        if self.state == BotState.DEAD:
            self.transition(BotState.ROAMING)
            return self.state

        return self.state

    def __repr__(self) -> str:
        return f"StateMachine({self.state.name}, {self.time_in_state:.1f}s)"

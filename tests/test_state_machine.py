"""Tests for the bot state machine."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.brain.state_machine import StateMachine, BotState


def test_initial_state():
    sm = StateMachine()
    assert sm.state == BotState.DEAD


def test_death_overrides():
    sm = StateMachine()
    sm.transition(BotState.FIGHTING)
    sm.update(is_alive=False, enemies_visible=5, health=0,
              is_stuck=False)
    assert sm.state == BotState.DEAD


def test_fighting_transition():
    sm = StateMachine()
    sm.update(is_alive=True, enemies_visible=0, health=100,
              is_stuck=False)
    assert sm.state == BotState.ROAMING

    sm.update(is_alive=True, enemies_visible=2, health=100,
              is_stuck=False)
    assert sm.state == BotState.FIGHTING


def test_retreat_on_low_health():
    sm = StateMachine()
    sm.transition(BotState.ROAMING)
    sm.update(is_alive=True, enemies_visible=1, health=20,
              is_stuck=False, disengage_health=30)
    assert sm.state == BotState.RETREATING


def test_stuck_detection():
    sm = StateMachine()
    sm.transition(BotState.ROAMING)
    sm.update(is_alive=True, enemies_visible=0, health=100,
              is_stuck=True)
    assert sm.state == BotState.STUCK


def test_previous_state():
    sm = StateMachine()
    sm.transition(BotState.ROAMING)
    sm.transition(BotState.FIGHTING)
    assert sm.previous_state == BotState.ROAMING


def test_no_self_transition():
    sm = StateMachine()
    sm.transition(BotState.ROAMING)
    start_time = sm._state_start
    sm.transition(BotState.ROAMING)  # Same state
    assert sm._state_start == start_time  # Timer shouldn't reset

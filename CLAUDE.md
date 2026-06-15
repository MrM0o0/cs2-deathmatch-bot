# Working in this repo

Standing rules and hard-won context for Claude Code sessions. Read this before
making changes.

## What this is

A vision-only CS2 deathmatch bot. North star: a bot that **reads as a real
player**, not a perfect aimbot. Detection and aim already work; **navigation is
the weak link under active work**. Pixels in, synthetic input out — no game
memory access.

## Run / check

```
python -m pytest            # the suite (43+ tests, pure logic only)
python -m ruff check src tests
python -m ruff format src tests
```

Tests cover pure logic (A* nav, heading math, minimap detection, input struct).
They do **not** drive the game — anything involving real capture/input must be
run in CS2 by the user, who then reports back or sends logs.

## Things that will bite you (learned in-game)

- **Detection must run on the GPU (DirectML), not CPU.** On this machine (AMD
  RX 6900 XT) GPU inference is ~9ms (~67 FPS pipeline) vs ~22ms on CPU (~34 FPS)
  -- detection is the shared spine, so CPU drags down every rung. The detector
  prefers `DmlExecutionProvider` automatically and prints a loud WARNING if it
  falls back to CPU. The usual cause of a CPU fallback is the plain
  `onnxruntime` package being installed alongside `onnxruntime-directml`: they
  share the import name and the CPU one wins. Fix: `pip uninstall onnxruntime`
  then `pip install --force-reinstall --no-deps onnxruntime-directml`.

- **Keyboard input via `SendInput`:** the `INPUT` struct union must be sized to
  its largest member (`MOUSEINPUT`), or `sizeof` is 32 not 40 on 64-bit and
  `SendInput` silently rejects every keypress (returns 0). `test_input.py`
  guards this. If WASD does nothing in-game but mouse works, suspect this.
- **The minimap player dot has no fixed colour.** CS2 reassigns it per match
  (cyan one game, yellow the next). `MinimapReader` finds it as the brightest,
  most-saturated blob (hue-agnostic) — never hardcode a hue.
- **Radar must be fixed, not rotating:** needs `cl_radar_rotate 0` AND
  `cl_radar_always_centered 0`, or the dot stays centred and position tracking
  is dead. Zooming in re-enables centring.
- **The dot moves slowly on the radar** (~5px/sec at whole-map zoom). Heading is
  derived from how the dot *moves*, so estimation needs a ~1s window and the bot
  must **never stop walking to turn** — turning in place freezes the heading
  estimate and causes an infinite spin (see `NavigationController.update`).

## Safety is mandatory

Anything that takes over input MUST keep the kill path working: `END` stops the
bot (watchdog thread, `os._exit`), `HOME` pauses it, and `max_run_seconds`
auto-stops. Never ship a change that can hold the mouse/keyboard without a
working `END`.

## Repo hygiene

- **Never commit** the dataset dirs (`merged_dataset*`, `CS2 Object detection*`,
  `models/training/`) or run logs (`logs/`, `tools/diagnostic_log.txt`) — all
  gitignored. Don't stage the user's uncommitted work (e.g. `aim_shoot_test.py`)
  unless they ask.
- Lint/format with ruff before committing; the pre-commit hook enforces it
  (`git config core.hooksPath .githooks` to enable).

## Commit style

State what changed and **why**, reference the symptom or log evidence when
fixing a bug, plain English, no marketing words. Keep diffs scoped to one
concern. End messages with the Co-Authored-By trailer.

# CS2 Deathmatch Bot

A screen-reading bot that plays Counter-Strike 2 deathmatch from pixels alone:
it sees the game through a YOLO detector, reads its own position off the
minimap, and drives the mouse and keyboard like a player would. No game memory
is read or written; everything is vision in, synthetic input out.

The goal is a bot that a spectator reads as a real player, not a perfect aimbot.
Detection and aim are solid; navigation is the part under active work.

## Where things live

```
src/
  capture/      screen grab (dxcam, mss fallback)
  vision/       YOLO detector, HUD reader, minimap (player-dot) reader
  brain/        state machine (ROAMING/FIGHTING/...), decisions, threat sort
  aim/          targeting, human-like mouse paths, recoil
  movement/     A* pathfinding + face-aware waypoint navigation
  humanizer/    reaction timing, aim mistakes, personality profiles
  input/        Win32 SendInput keyboard + mouse
  main.py       the orchestrator loop (capture -> detect -> decide -> act)

tools/          standalone helpers (record maps, calibrate, benchmark, tests)
config/
  settings.yaml main config (display, detection, minimap, navigation, keybinds)
  maps/         recorded waypoint graphs, one JSON per map
  personalities/ noob / average / tryhard behaviour profiles
tests/          pytest suite for the pure logic (nav, minimap, input struct, ...)
models/         the ONNX detection model
```

## Getting started

```
pip install -r requirements.txt          # runtime deps (Windows)
pip install ruff pytest                   # dev tooling
python -m pytest                          # run the suite
```

### One-time CS2 radar setup

The bot reads its position from the radar, so the radar must be a fixed map
with a moving dot (not the default rotating, player-centred one). These are
written to an `autoexec.cfg`; run them once in the console if needed:

```
cl_radar_rotate 0
cl_radar_always_centered 0
cl_radar_scale 0.4
```

### Per-map workflow

```
# 1. Confirm the bot can see your dot move (walk during the capture window):
python tools/record_waypoints.py --map dust2 --preview

# 2. Record a map by walking its routes (auto-saves after --record-secs):
python tools/record_waypoints.py --map dust2 --record-secs 90

# 3. Calibrate which way a mouse turn rotates the radar:
python tools/calibrate_nav.py --write

# 4. Run the bot:
python -m src.main
```

### Safety hotkeys (work while CS2 is focused)

| Key | Action |
|-----|--------|
| `END`  | Stop the bot instantly and release all input |
| `HOME` | Pause / resume — bot lets go of mouse + keyboard so you can take over |

A watchdog thread polls these every 20ms independent of the main loop, and the
bot also auto-stops after `bot.max_run_seconds` (default 120s).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the dev workflow, commit conventions,
and the lint/test gates. [CLAUDE.md](CLAUDE.md) holds the standing rules for
working in this repo.

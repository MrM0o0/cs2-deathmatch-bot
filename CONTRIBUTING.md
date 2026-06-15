# Contributing

## Setup

```
pip install -r requirements.txt     # runtime (Windows)
pip install ruff pytest             # dev tooling
git config core.hooksPath .githooks # enable the pre-commit gate
```

## Workflow

1. Branch off `main` for anything non-trivial.
2. Make the change. Keep it scoped to one concern.
3. Run the gates locally:
   ```
   python -m ruff check src tests
   python -m ruff format src tests
   python -m pytest
   ```
4. Commit (the pre-commit hook re-runs lint + tests on what you staged).
5. Open a PR. CI runs ruff + the test suite on Windows.

## What gets a test

Pure logic — pathfinding, heading/turn math, minimap detection, input structs.
Anything that needs the real game (capture, input, calibration) can't be unit
tested; verify it in CS2 and note what you observed in the PR or commit.

## Commit messages

- Say what changed and **why**, not only what.
- When fixing a bug, name the symptom or cite the log evidence.
- Plain English. No marketing words ("robust", "seamless", "powerful").
- One concern per commit.
- End with:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  ```

## Safety rule

Any change that drives the mouse or keyboard must preserve the kill path: `END`
stops the bot, `HOME` pauses it, `max_run_seconds` auto-stops. Don't merge
anything that can hold input without a working `END`.

## Conventions

- Python 3.13, ruff for lint + format (config in `pyproject.toml`, line length 100).
- Entry scripts (`src/main.py`, `tools/*`) insert the project root on `sys.path`
  before importing `src.*`; that's why `E402` is ignored.
- Config lives in `config/settings.yaml`; don't hardcode tunables.

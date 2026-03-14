"""Load and manage personality profiles from YAML config."""

import os
import yaml


class Personality:
    """A personality profile that controls all human-like behavior parameters."""

    def __init__(self, data: dict):
        self.name: str = data.get("name", "unknown")
        self.description: str = data.get("description", "")

        # Reaction timing
        r = data.get("reaction", {})
        self.reaction_mean_ms: float = r.get("mean_ms", 250)
        self.reaction_std_ms: float = r.get("std_ms", 50)
        self.reaction_min_ms: float = r.get("min_ms", 150)
        self.reaction_max_ms: float = r.get("max_ms", 500)

        # Aim parameters
        a = data.get("aim", {})
        self.aim_speed: float = a.get("base_speed", 6.0)
        self.overshoot_chance: float = a.get("overshoot_chance", 0.35)
        self.overshoot_magnitude: float = a.get("overshoot_magnitude", 1.3)
        self.micro_corrections: list[int] = a.get("micro_corrections", [1, 3])
        self.correction_delay_ms: float = a.get("correction_delay_ms", 40)
        self.head_aim_chance: float = a.get("head_aim_chance", 0.3)
        self.tracking_error: float = a.get("tracking_error", 8.0)

        # Spray control
        s = data.get("spray", {})
        self.max_spray_length: int = s.get("max_spray_length", 10)
        self.recoil_compensation: float = s.get("recoil_compensation", 0.5)
        self.burst_length: list[int] = s.get("burst_length", [3, 7])
        self.tap_chance: float = s.get("tap_chance", 0.3)

        # Movement
        m = data.get("movement", {})
        self.strafe_while_shooting: bool = m.get("strafe_while_shooting", True)
        self.crouch_spray_chance: float = m.get("crouch_spray_chance", 0.4)
        self.jump_frequency: float = m.get("jump_frequency", 0.05)
        self.walk_chance: float = m.get("walk_chance", 0.2)
        self.movement_noise: float = m.get("movement_noise", 0.15)

        # Idle behaviors
        i = data.get("idle", {})
        self.inspect_chance: float = i.get("inspect_chance", 0.01)
        self.look_around_chance: float = i.get("look_around_chance", 0.03)
        self.random_jump_chance: float = i.get("random_jump_chance", 0.01)
        self.pause_duration: list[float] = i.get("pause_duration", [0.3, 1.5])

        # Combat
        c = data.get("combat", {})
        self.engage_distance: float = c.get("engage_distance", 600)
        self.disengage_health: int = c.get("disengage_health", 30)
        self.reload_threshold: float = c.get("reload_threshold", 0.3)
        self.switch_target_delay_ms: float = c.get("switch_target_delay_ms", 350)

    def __repr__(self) -> str:
        return f"Personality({self.name}: {self.description})"


def load_personality(name: str, config_dir: str = "config/personalities") -> Personality:
    """Load a personality profile from YAML file."""
    path = os.path.join(config_dir, f"{name}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Personality profile not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return Personality(data)


def list_personalities(config_dir: str = "config/personalities") -> list[str]:
    """List available personality profile names."""
    if not os.path.isdir(config_dir):
        return []
    return [
        os.path.splitext(f)[0]
        for f in os.listdir(config_dir)
        if f.endswith(".yaml")
    ]

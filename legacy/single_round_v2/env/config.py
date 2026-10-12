"""Configuration loading for the paper-constrained combat environment."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import math
import yaml

from .models import AircraftSpec


ENVIRONMENT_VERSION = "2.3"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the active V2.3 environment configuration."""
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = {
        "environment_version", "simulation", "action", "aircraft", "arena",
        "scenario", "weapon", "reward", "observation", "blue_policy",
    }
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError(f"configuration must contain: {sorted(required)}")
    if str(config["environment_version"]) != ENVIRONMENT_VERSION:
        raise ValueError(
            f"environment_version must be {ENVIRONMENT_VERSION}, got "
            f"{config['environment_version']}"
        )
    return config


def aircraft_spec(config: dict[str, Any]) -> AircraftSpec:
    return AircraftSpec(**config["aircraft"])


def attack_cooldown_steps(config: dict[str, Any]) -> int:
    """Validate fixed firing cadence and convert seconds to exact integer steps."""
    attack = config.get("attack", {})
    if not isinstance(attack, dict) or attack.get("cadence") != "fixed_cooldown":
        raise ValueError("attack.cadence must be fixed_cooldown")
    if isinstance(attack.get("cooldown_seconds"), bool):
        raise ValueError("attack.cooldown_seconds must be a positive number, not bool")
    try:
        seconds = float(attack["cooldown_seconds"])
        dt = float(config["simulation"]["dt"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("attack.cooldown_seconds and simulation.dt must be specified") from error
    if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(dt) or dt <= 0:
        raise ValueError("cooldown_seconds and dt must be finite and positive")
    ratio = seconds / dt
    steps = round(ratio)
    if not math.isclose(ratio, steps, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("cooldown_seconds / dt must be an integer number of steps")
    if seconds != 1.0 or steps != 10:
        raise ValueError("formal cooldown must be 1.0 seconds / 10 steps")
    return steps


__all__ = ["ENVIRONMENT_VERSION", "aircraft_spec", "load_config", "attack_cooldown_steps"]

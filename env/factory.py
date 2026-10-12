"""Configuration-driven combat-environment construction."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_config
from .combat_env import MultiUAVCombatEnv


def make_combat_environment(config: str | Path | dict[str, Any]) -> MultiUAVCombatEnv:
    loaded = load_config(config) if not isinstance(config, dict) else config
    variant = loaded.get("environment_variant")
    if variant != "single_round_v1":
        raise ValueError(f"environment_variant must be 'single_round_v1', got {variant!r}")
    return MultiUAVCombatEnv(loaded)



__all__ = ["make_combat_environment"]

"""Configuration-driven combat-environment construction."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_config
from .combat_env import MultiUAVCombatEnv


def make_combat_environment(config: str | Path | dict[str, Any]) -> MultiUAVCombatEnv:
    loaded = load_config(config) if not isinstance(config, dict) else config
    variant = loaded.get("environment_variant")
    if variant not in {"single_round_v1", "single_round_6v6_v1"}:
        raise ValueError(f"unsupported environment_variant: {variant!r}")
    return MultiUAVCombatEnv(loaded)



__all__ = ["make_combat_environment"]

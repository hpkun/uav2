"""Public low-fidelity 4v4 combat-environment components."""

from .combat_env import MultiUAVCombatEnv
from .factory import make_combat_environment
from .geometry import EngagementGeometry, engagement_geometry
from .weapon import WeaponEnvelope
from .models import AircraftSpec, AircraftState, ControlCommand

__all__ = [
    "AircraftSpec", "AircraftState", "ControlCommand", "EngagementGeometry",
    "MultiUAVCombatEnv",
    "WeaponEnvelope", "engagement_geometry", "make_combat_environment",
]

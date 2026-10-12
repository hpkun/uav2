"""Explicit pursuit actor initialization, never a training-state resume."""
from pathlib import Path
import hashlib
import torch
from .trainer import MAPPO_IMPL_VERSION


def validate_actor_initialization(state, target_config, actor=None):
    if target_config.get("task_type") != "combat" or target_config.get("environment_variant") != "single_round_v3":
        raise RuntimeError("actor initialization is for fresh combat v3 only")
    extra = state.get("extra", {})
    expected = {"environment_version":"2.4", "environment_variant":"pursuit_v2",
                "task_type":"pursuit", "action_semantics":"direct_physical_control",
                "observation_dim":52, "action_dim":3, "num_agents":4}
    if state.get("algorithm") != "MAPPO" or state.get("mappo_impl_version") != MAPPO_IMPL_VERSION:
        raise RuntimeError("actor initialization requires compatible MAPPO implementation")
    for field,value in expected.items():
        if extra.get(field) != value:
            raise RuntimeError(f"actor initialization {field} mismatch: expected {value!r}, got {extra.get(field)!r}")
    parameters = state.get("actor")
    if not isinstance(parameters, dict) or not parameters:
        raise RuntimeError("source checkpoint actor parameters missing")
    if actor is not None:
        target = actor.state_dict()
        if target.keys() != parameters.keys() or any(target[k].shape != parameters[k].shape for k in target):
            raise RuntimeError("actor initialization network architecture incompatible")
        if any(not torch.isfinite(value).all() for value in parameters.values()):
            raise RuntimeError("source actor parameters are nonfinite")


def initialize_actor(runner, checkpoint):
    if runner.trainer.sampled_steps or runner.evaluation_history or runner.completed_records:
        raise RuntimeError("actor initialization requires fresh run state")
    path = Path(checkpoint).resolve()
    state = torch.load(path, map_location="cpu", weights_only=False)
    validate_actor_initialization(state, runner.env_config, runner.trainer.actor)
    runner.trainer.actor.load_state_dict(state["actor"], strict=True)
    runner.actor_initialization = {"source_checkpoint":str(path),
        "source_checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_task_type":"pursuit", "source_action_semantics":"direct_physical_control",
        "loaded_parameters":"actor_only"}
    return runner.actor_initialization

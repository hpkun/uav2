"""Single-round contract, deployment compatibility, and reset regressions."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from env.combat_env import MultiUAVCombatEnv
from env.factory import make_combat_environment
from env.fixed_policy import GroundAwareNearestTargetPursuitPolicy, NearestTargetPursuitPolicy
from algorithm.common.checkpoint import (
    evaluation_selection_key, validate_checkpoint_for_resume, validate_checkpoint_for_evaluation,
)
from algorithm.common.evaluator import evaluate, aggregate_episode_records
from algorithm.common.vector_env import ParallelVectorEnv
from algorithm.madsac.evaluation import aggregate_madsac_records
from algorithm.madsac.protocol import validate_madsac_config, validate_madsac_checkpoint
from algorithm.mappo.runner import MAPPOTrainingRunner
from algorithm.mappo.evaluation import evaluate_mappo_checkpoint

ROOT = Path(__file__).resolve().parents[1]
def load(name):
    return yaml.safe_load((ROOT / 'configs' / name).read_text(encoding='utf-8'))
def config():
    return load('single_round_environment.yaml')
def zeros():
    return np.zeros((4, 3), np.float32)


def test_factory_and_reset_contract():
    env = make_combat_environment(config())
    assert type(env) is MultiUAVCombatEnv
    observation, info = env.reset(12)
    assert observation.shape == (4, 52) and env.action_dim == 3
    assert len(env.red) == len(env.blue) == 4
    assert env.red_alive_mask.sum() == env.blue_alive_mask.sum() == 4
    assert info['environment_variant'] == 'single_round_v2'
    assert env.max_steps == 1000
    assert isinstance(env.fixed_policy, GroundAwareNearestTargetPursuitPolicy)


@pytest.mark.parametrize('variant', ['single_round_v1', 'single_round_6v6_v1', 'persistent_wave_v1', 'persistent_wave_v2', 'direct_v2_3', None, 'unknown'])
def test_factory_and_constructor_fail_closed(variant):
    cfg = config()
    if variant is None:
        del cfg['environment_variant']
    else:
        cfg['environment_variant'] = variant
    with pytest.raises(ValueError, match='environment_variant'):
        make_combat_environment(cfg)
    with pytest.raises(ValueError, match='environment_variant'):
        MultiUAVCombatEnv(cfg)


@pytest.mark.parametrize('red_dead,blue_dead,reason', [
    (False, True, 'red_win'), (True, False, 'blue_win'),
    (True, True, 'draw_mutual_destruction'),
])
def test_elimination_terminates_without_replacement(red_dead, blue_dead, reason):
    env = make_combat_environment(config())
    env.reset(15)
    for side, dead in ((env.red, red_dead), (env.blue, blue_dead)):
        if dead:
            for state in side: state.alive = False
    before = [state.as_array().copy() for state in env.blue]
    _, _, terminated, truncated, info = env.step(zeros())
    assert terminated and not truncated
    assert info['termination_reason'] == reason
    assert info['red_win'] == (reason == 'red_win')
    assert info['blue_win'] == (reason == 'blue_win')
    assert info['draw'] == (reason == 'draw_mutual_destruction')
    if blue_dead:
        assert env.blue_alive_mask.sum() == 0
        assert all(np.array_equal(old, state.as_array()) for old, state in zip(before, env.blue))
    with pytest.raises(RuntimeError, match='reset'):
        env.step(zeros())
    assert not {'wave_index', 'waves_cleared', 'spawned_next_wave', 'per_wave_metrics'} & info.keys()


def test_actual_1000_step_unresolved_timeout(monkeypatch):
    env = make_combat_environment(config())
    env.reset(17)
    # Freeze only movement to keep this unresolved; all 1000 step calls execute.
    monkeypatch.setattr(env, '_advance', lambda *_: np.zeros(4, np.float32))
    for step in range(1, 1001):
        _, _, terminated, truncated, info = env.step(zeros(), zeros())
        assert not terminated
        assert truncated == (step == 1000)
    assert info['termination_reason'] == 'red_failure_timeout'
    assert info['episode_length'] == 1000
    assert not info['red_win'] and not info['blue_win'] and not info['draw']


def test_vector_terminal_masks_and_new_episode_seeds():
    cfg = config()
    # A deterministic Red boundary loss via real worker steps, no test command.
    cfg['scenario']['center_radius'] = 4990
    cfg['scenario']['formation_offsets'] = [0, 0, 0, 0]
    with ParallelVectorEnv(2, cfg, base_seed=700) as vector:
        vector.reset()
        for _ in range(500):
            result = vector.step_batch(np.zeros((2, 4, 3), np.float32))
            done = result.terminated | result.truncated
            if done.any(): break
        assert done.any()
        for index in np.flatnonzero(done):
            assert vector.episode_indices[index] == 1
            assert vector.last_reset_seeds[index] == 702 + index
            assert vector.current_alive_masks[index].sum() == 4
            assert vector.current_observations[index].shape == (4, 52)
            assert result.infos[index]['episode_length'] < 1000
        assert len(vector.used_training_seeds) == 2 + int(done.sum())
        assert not hasattr(vector, 'export_curriculum_states')
        assert not hasattr(vector, 'restore_curriculum_states')


def test_ground_guard_risk_override_and_episode_reset():
    env = make_combat_environment(config())
    env.reset(19)
    policy = env.fixed_policy
    env.blue[0].z = -1
    env.blue[0].theta = -np.pi / 6
    actions = policy.team_actions(env.blue, env.red)
    assert policy.last_override_mask[0]
    assert actions[0, 1] > 0
    assert policy.override_steps > 0 and policy.activation_count > 0
    _, _, _, _, info = env.step(zeros())
    assert info['blue_ground_guard_override_steps'] > 0
    _, reset_info = env.reset(20)
    for key in ('decision_steps', 'override_steps', 'activations', 'activation_ratio', 'max_duration_steps'):
        assert reset_info[f'blue_ground_guard_{key}'] == 0
    assert not reset_info['blue_ground_guard_override_mask'].any()


def test_policy_types_explicit_and_invalid():
    cfg = config()
    cfg['blue_policy']['type'] = 'nearest_target'
    assert type(make_combat_environment(cfg).fixed_policy) is NearestTargetPursuitPolicy
    cfg['blue_policy']['type'] = 'invalid'
    with pytest.raises(ValueError, match='blue_policy.type'):
        make_combat_environment(cfg)


def test_evaluation_metrics_equal_for_both_algorithms():
    class ZeroActor:
        def act(self, *_args, **_kwargs): return zeros()
    cfg = config()
    mappo = evaluate(ZeroActor(), cfg, [10000000])
    env = make_combat_environment(cfg)
    env.reset(10000000)
    total = np.zeros(4)
    while True:
        _, reward, terminated, truncated, info = env.step(zeros())
        total += reward
        if terminated or truncated: break
    records = [dict(info, episode_return=float(total.sum()), mean_agent_episode_return=float(total.mean()))]
    madsac = aggregate_madsac_records(records)
    assert mappo == madsac == aggregate_episode_records(records)
    assert not any('wave' in key or key in {'Q2', 'Q3'} for key in mappo)
    assert all(f'average_episode_r{i}_total' in madsac for i in range(1, 5))


def test_best_selection_priority():
    def key(win, ret, loss):
        return evaluation_selection_key(dict(win_rate=win, average_return=ret, average_red_loss=loss))
    assert key(.6, -100, 4) > key(.5, 100, 0)
    assert key(.5, 11, 4) > key(.5, 10, 0)
    assert key(.5, 10, 1) > key(.5, 10, 2)


@pytest.mark.parametrize('variant', ['single_round_v1', 'single_round_6v6_v1', 'persistent_wave_v1', 'persistent_wave_v2', 'direct_v2_3', None])
def test_old_checkpoint_resume_and_evaluation_rejected(tmp_path, variant):
    cfg, algorithm = config(), load('mappo.yaml')
    runner = MAPPOTrainingRunner(cfg, algorithm, num_envs=1, total_sampled_steps=4,
                                 device='cuda', output_dir=tmp_path, smoke=True)
    try:
        path = tmp_path / 'checkpoint.pt'
        runner.save_checkpoint(path)
        state = torch.load(path, map_location='cpu', weights_only=False)
        state['extra']['environment_variant'] = variant
        for validate in (validate_checkpoint_for_resume, validate_checkpoint_for_evaluation):
            with pytest.raises(RuntimeError, match='environment_variant'):
                validate(state, cfg, algorithm)
    finally:
        runner.vector.close()


def test_formal_mappo_evaluation_rejects_incomplete_metadata(tmp_path):
    cfg, algorithm = config(), load('mappo.yaml')
    runner = MAPPOTrainingRunner(cfg, algorithm, num_envs=1, total_sampled_steps=4,
                                 device='cuda', output_dir=tmp_path, smoke=True)
    try:
        path = tmp_path / 'checkpoint.pt'
        runner.save_checkpoint(path)
        state = torch.load(path, map_location='cpu', weights_only=False)
        del state['extra']['training_seed']
        torch.save(state, path)
        with pytest.raises(RuntimeError, match='incomplete'):
            evaluate_mappo_checkpoint(path, algorithm, cfg, 'cuda', [10000000])
    finally:
        runner.vector.close()


@pytest.mark.parametrize('change', ['variant', 'max_steps', 'dimensions'])
def test_madsac_protocol_rejects_wrong_task(change):
    cfg, algorithm = config(), load('madsac_single_round_3m.yaml')
    validate_madsac_config(cfg, algorithm)
    if change == 'variant': cfg['environment_variant'] = 'persistent_wave_v2'
    if change == 'max_steps': cfg['simulation']['max_steps'] = 3000
    if change == 'dimensions': algorithm['network']['observation_dim'] = 54
    with pytest.raises(RuntimeError): validate_madsac_config(cfg, algorithm)


def test_madsac_checkpoint_rejects_old_environment():
    from algorithm.common.protocol import config_sha256
    from algorithm.madsac.trainer import MADSAC_IMPL_VERSION
    cfg, algorithm = config(), load('madsac_single_round_3m.yaml')
    extra = dict(training_seed=5301, environment_version='2.3',
                 environment_variant='persistent_wave_v2', observation_dim=52,
                 action_dim=3, num_agents=4, environment_config_sha256=config_sha256(cfg),
                 algorithm_config_sha256=config_sha256(algorithm))
    state = dict(algorithm='madsac', implementation_version=MADSAC_IMPL_VERSION,
                 extra=extra, formal_exact_resume_supported=False, replay_buffer_included=False)
    with pytest.raises(RuntimeError, match='environment_variant'):
        validate_madsac_checkpoint(state, cfg, algorithm)

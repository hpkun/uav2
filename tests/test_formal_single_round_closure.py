"""Formal identity, strict continuation, actual final kill, and commit audit regressions."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from algorithm.common.protocol import validate_single_round_environment_config
from algorithm.common.checkpoint import validate_checkpoint_for_resume, validate_checkpoint_for_evaluation
from algorithm.common.evaluator import evaluate
from algorithm.mappo.runner import MAPPOTrainingRunner
from algorithm.madsac.protocol import validate_madsac_config
from algorithm.madsac.evaluation import evaluate_madsac
from env.combat_env import MultiUAVCombatEnv
from env.models import AircraftState
from tools import audit_single_round as audit

ROOT = Path(__file__).resolve().parents[1]
def config(name='single_round_environment.yaml'):
    return yaml.safe_load((ROOT / 'configs' / name).read_text(encoding='utf-8'))


@pytest.mark.parametrize('path,value', [
    (('environment_variant',), 'invalid'), (('environment_version',), '2.2'),
    (('simulation', 'max_steps'), 3000), (('simulation', 'dt'), .2),
    (('scenario', 'team_size'), 5), (('observation', 'include_own_fire_ready'), True),
    (('action_dim',), 4),
])
@pytest.mark.parametrize('smoke', [False, True])
def test_both_formal_protocols_fail_fast(path, value, smoke, tmp_path):
    cfg = config()
    target = cfg
    for key in path[:-1]: target = target[key]
    target[path[-1]] = value
    with pytest.raises(RuntimeError, match='formal single-round'):
        MAPPOTrainingRunner(cfg, config('mappo.yaml'), output_dir=tmp_path, smoke=smoke)
    assert not list(tmp_path.iterdir())  # Rejected before output, GPU, or worker allocation.
    with pytest.raises(RuntimeError, match='formal single-round'):
        validate_madsac_config(cfg, config('madsac_single_round_3m.yaml'))
    with pytest.raises(RuntimeError, match='formal single-round'):
        evaluate(None, cfg, [10000000])
    with pytest.raises(RuntimeError, match='formal single-round'):
        evaluate_madsac(None, cfg, [44000000])


def test_bottom_level_short_timeout_remains_legal():
    cfg = config()
    cfg['simulation']['max_steps'] = 1
    env = MultiUAVCombatEnv(cfg)
    env.reset(123)
    _, _, terminated, truncated, _ = env.step(np.zeros((4, 3)), np.zeros((4, 3)))
    assert not terminated and truncated
    with pytest.raises(RuntimeError, match='max_steps'):
        validate_single_round_environment_config(cfg)


@pytest.mark.parametrize('field,value', [('observation_dim', 53), ('action_dim', 4), ('num_agents', 5)])
def test_network_dimensions_remain_strict(field, value, tmp_path):
    for method in ('mappo.yaml', 'madsac_single_round_3m.yaml'):
        algorithm = config(method)
        algorithm['network'][field] = value
        if method == 'mappo.yaml':
            with pytest.raises(ValueError, match='dimension mismatch'):
                MAPPOTrainingRunner(config(), algorithm, output_dir=tmp_path)
        else:
            with pytest.raises(RuntimeError, match='dimension mismatch'):
                validate_madsac_config(config(), algorithm)


@pytest.fixture(scope='module')
def checkpoint(tmp_path_factory):
    runner = MAPPOTrainingRunner(config(), config('mappo.yaml'), num_envs=1,
        total_sampled_steps=4, device='cuda', output_dir=tmp_path_factory.mktemp('identity'), smoke=True)
    try:
        path = runner.output_dir / 'identity.pt'
        runner.save_checkpoint(path)
        return torch.load(path, map_location='cpu', weights_only=False)
    finally:
        runner.vector.close()


def validate(state):
    validate_checkpoint_for_resume(state, config(), config('mappo.yaml'),
        expected_training_seed=0, expected_num_envs=1, expected_smoke=True)


def test_valid_checkpoint_accepted(checkpoint): validate(checkpoint)


@pytest.mark.parametrize('field', ['environment_version', 'environment_variant',
    'environment_config_sha256', 'algorithm_config_sha256', 'training_seed',
    'training_num_envs', 'training_gamma', 'training_smoke'])
@pytest.mark.parametrize('mutation', ['missing', 'tampered'])
def test_resume_required_fields_fail_closed(checkpoint, field, mutation):
    state = deepcopy(checkpoint)
    if mutation == 'missing': del state['extra'][field]
    else: state['extra'][field] = 'tampered'
    with pytest.raises(RuntimeError): validate(state)


@pytest.mark.parametrize('field', ['environment_config_sha256', 'algorithm_config_sha256'])
def test_evaluation_fingerprints_fail_closed(checkpoint, field):
    state = deepcopy(checkpoint)
    del state['extra'][field]
    with pytest.raises(RuntimeError, match=field):
        validate_checkpoint_for_evaluation(state, config(), config('mappo.yaml'))


def test_cli_rejects_missing_checkpoint_metadata_with_valid_snapshots(checkpoint, tmp_path):
    state = deepcopy(checkpoint)
    del state['extra']['training_gamma']
    path = tmp_path / 'checkpoint.pt'
    torch.save(state, path)
    for name, cfg in (('env_config.yaml', config()), ('algorithm_config.yaml', config('mappo.yaml'))):
        (tmp_path / name).write_text(yaml.safe_dump(cfg), encoding='utf-8')
    import json
    (tmp_path / 'run_config.json').write_text(json.dumps(dict(seed=0, num_envs=1,
        total_sampled_steps=4, device='cuda', smoke=True)))
    result = subprocess.run([sys.executable, str(ROOT / 'algorithm/train_mappo.py'),
        '--resume', str(path), '--total-sampled-steps', '8'], cwd=ROOT,
        capture_output=True, text=True, timeout=60)
    assert result.returncode != 0 and 'training_gamma' in result.stderr
    assert not (tmp_path / 'training_metrics.jsonl').exists()


def test_actual_last_blue_kill_ends_episode_without_replacement(monkeypatch):
    env = MultiUAVCombatEnv(config())
    env.reset(123)
    env.red = [AircraftState(0, 0, -3000, 225, 0, 0)] + [
        AircraftState(0, 0, -3000, 225, 0, 0, False) for _ in range(3)]
    env.blue = [AircraftState(1000, 0, -3000, 225, 0, 0)] + [
        AircraftState(1000, 0, -3000, 225, 0, 0, False) for _ in range(3)]
    assert env.blue_alive_mask.sum() == 1
    # Existing resolution hook only; weapon parameters and window gate are untouched.
    monkeypatch.setattr(type(env.weapon), 'attempt_hit', lambda *_: True)
    observation, _, terminated, truncated, info = env.step(np.zeros((4, 3)), np.zeros((4, 3)))
    assert terminated and not truncated
    assert info['blue_survivors'] == 0 and info['termination_reason'] == 'red_win'
    assert info['red_attack_kills'] == 1 and info['red_step_weapon_hits'] == 1
    assert len(env.blue) == 4 and not env.blue_alive_mask.any()
    expected = env._observations()
    assert np.array_equal(observation, expected)
    assert not expected[:, 28:52].any()  # All opponent slots dead, no replacement observation.
    assert env.blue[0].x == pytest.approx(1022.5)
    with pytest.raises(RuntimeError, match='episode is finished'):
        env.step(np.zeros((4, 3)))


def test_audit_baseline_and_dirty_index_independence(tmp_path, monkeypatch):
    assert audit.BASE_COMMIT == '8d09df1fd1e6d2425b712d0c336e9525cd313c4c'
    name = 'env/dynamics.py'
    path = tmp_path / name
    path.parent.mkdir(); path.write_bytes(audit.baseline_content(ROOT, name))
    monkeypatch.setattr(audit, 'baseline_blobs', lambda _: audit_baseline)
    audit_baseline = {name: audit.blob_hash(path)}
    audit.compare_frozen_files(tmp_path, [name])
    path.write_bytes(b'changed and staged or unstaged')
    with pytest.raises(RuntimeError, match='baseline'):
        audit.compare_frozen_files(tmp_path, [name])


def test_changed_files_uses_committed_baseline_to_head(monkeypatch):
    calls = []
    def command(args, **kwargs):
        calls.append(args)
        return 'M\tenv/combat_env.py\nR100\tenv/persistent_env.py\tlegacy/env/persistent_env.py\n'
    monkeypatch.setattr(audit.subprocess, 'check_output', command)
    rows = audit.changed_files(ROOT)
    assert calls == [['git', 'diff', '--name-status', '-M', audit.BASE_COMMIT, 'HEAD']]
    assert rows[1]['original_path'] == 'env/persistent_env.py'
    assert audit.legacy_original('README_original.md') == 'README.md'
    assert audit.legacy_original('env/persistent_env.py') == 'env/persistent_env.py'


def test_archived_source_matches_baseline():
    blobs = audit.baseline_blobs(ROOT)
    for original in ('env/persistent_env.py', 'algorithm/modular_mappo/runner.py'):
        assert audit.blob_hash(ROOT / 'legacy' / original) == blobs[original]


def test_committed_clean_clone_audit_is_repeatable_and_ignores_staged_changes(tmp_path):
    import json
    clone = tmp_path / 'clone'
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=clone, stderr=subprocess.STDOUT)
    subprocess.run(['git', 'clone', '--shared', str(ROOT), str(clone)], check=True,
                   capture_output=True, timeout=60)
    # Commit only inside this disposable clone to simulate this patch after commit.
    for folder in ('algorithm', 'env', 'tests', 'tools'):
        for path in (ROOT / folder).rglob('*.py'):
            if '__pycache__' in path.parts: continue
            destination = clone / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    git('add', 'algorithm', 'env', 'tests', 'tools')
    git('-c', 'user.name=Protocol Test', '-c', 'user.email=protocol-test@example.invalid',
        'commit', '-m', 'Temporary protocol snapshot for reproducibility test')
    assert not git('status', '--porcelain').strip()
    command = [sys.executable, str(clone / 'tools/audit_single_round.py')]
    completed = subprocess.run(command, cwd=clone, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    first = json.loads((clone / 'outputs/single_round_audit/single_round_audit.json').read_text())
    subprocess.run(command, cwd=clone, check=True, capture_output=True, timeout=60)
    second = json.loads((clone / 'outputs/single_round_audit/single_round_audit.json').read_text())
    assert first == second and first['archived_files_verified'] == 435
    assert first['baseline_commit'] == audit.BASE_COMMIT
    assert not git('status', '--porcelain').strip()
    # Even staging a changed frozen file cannot redefine the baseline.
    path = clone / 'env/dynamics.py'
    path.write_bytes(path.read_bytes() + b'\n# deliberate staged change\n')
    git('add', 'env/dynamics.py')
    with pytest.raises(RuntimeError, match='baseline'):
        audit.compare_frozen_files(clone, ['env/dynamics.py'])

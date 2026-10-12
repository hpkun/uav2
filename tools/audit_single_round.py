"""Reproducible source, frozen-foundation, and CUDA checkpoint acceptance audit."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import re
import subprocess
import sys
import argparse
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from algorithm.mappo.evaluation import evaluate_mappo_checkpoint
from algorithm.madsac.protocol import validate_madsac_checkpoint
from algorithm.madsac.trainer import MADSACTrainer
from algorithm.madsac.evaluation import evaluate_madsac
from algorithm.common.protocol import validate_single_round_environment_config

BASE_COMMIT = '8d09df1fd1e6d2425b712d0c336e9525cd313c4c'
TEAM_SIZE_REFACTORS = ('observation', 'scenario', 'fixed_policy')


def baseline_blobs(root):
    rows = subprocess.check_output(['git', 'ls-tree', '-r', BASE_COMMIT], cwd=root, text=True)
    return {line.split('\t')[1]: line.split('\t')[0].split()[2]
            for line in rows.splitlines() if line.split()[1] == 'blob'}


def baseline_content(root, path):
    return subprocess.check_output(['git', 'show', f'{BASE_COMMIT}:{path}'], cwd=root)


def blob_hash(path):
    data = path.read_bytes().replace(b'\r\n', b'\n')
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()


def compare_frozen_files(root, paths):
    blobs = baseline_blobs(root)
    for path in paths:
        if blob_hash(root / path) != blobs[path]:
            raise RuntimeError(f'frozen source differs from baseline: {path}')


def changed_files(root):
    """Committed baseline → HEAD diff; independent of pending or staged edits."""
    lines = subprocess.check_output(
        ['git', 'diff', '--name-status', '-M', BASE_COMMIT, 'HEAD'], cwd=root, text=True)
    return [dict(status=parts[0], path=parts[-1],
                 original_path=parts[1] if len(parts) == 3 else '')
            for parts in (line.split('\t') for line in lines.splitlines())]


def legacy_original(relative):
    return 'README.md' if relative == 'README_original.md' else relative


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/single_round_audit')
    args = parser.parse_args()
    if args.checkpoint_dir and not torch.cuda.is_available():
        raise RuntimeError('CUDA mandatory for checkpoint audit')
    report = args.output_dir
    report.mkdir(parents=True, exist_ok=True)
    forbidden = re.compile(r'persistent_wave|persistent_waves|wave_index|waves_cleared|wave_cleared|spawned_next_wave|spawn_next_wave|clear_wave_|per_wave_metrics|wave_context|inter_wave|wave_entry|wave_specific|\bQ2\b|\bQ3\b')
    runtime = [p for folder in ('env', 'algorithm', 'configs')
               for p in (ROOT / folder).rglob('*') if p.suffix in ('.py', '.yaml')]
    runtime.append(ROOT / 'README.md')
    findings = []
    for path in runtime:
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if forbidden.search(line):
                findings.append(dict(path=path.relative_to(ROOT).as_posix(), line=number, text=line))
    require(not findings, f'runtime keyword findings: {findings}')
    for path in runtime + list((ROOT / 'tests').glob('*.py')) + list((ROOT / 'tools').glob('*.py')):
        if path.suffix == '.py': ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    index = baseline_blobs(ROOT)
    cadence_baseline = json.loads((ROOT / 'reports/cadence_v2_baseline.json').read_text(encoding='utf-8'))
    for path, expected in cadence_baseline['frozen_files'].items():
        require(blob_hash(ROOT / path) == expected, f'cadence-only frozen source changed: {path}')
    cadence_archived = []
    for row in cadence_baseline['new_archives']:
        require(blob_hash(ROOT / row['archived']) == row['git_blob_sha1'], f'v1 archive changed: {row["archived"]}')
        cadence_archived.append(row)
    frozen = ('control', 'dynamics', 'integrator', 'weapon', 'reward', 'observation',
              'scenario', 'fixed_policy', 'models', 'geometry', 'math_utils')
    team_size_refactors = []
    for name in frozen:
        path = f'env/{name}.py'
        if name in TEAM_SIZE_REFACTORS:
            team_size_refactors.append(dict(path=path, baseline_blob=index[path],
                current_blob=blob_hash(ROOT / path), changed=blob_hash(ROOT / path) != index[path],
                reason='Explicit user-requested 6v6 support; 4v4 regression tests retained'))
            continue
        compare_frozen_files(ROOT, [path])
    for method in ('mappo', 'madsac'):
        for name in ('networks', 'trainer'):
            path = f'algorithm/{method}/{name}.py'
            compare_frozen_files(ROOT, [path])
    archived = []
    for path in sorted((ROOT / 'legacy').rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts: continue
        original = path.relative_to(ROOT / 'legacy').as_posix()
        if original.startswith('single_round_v1/'):
            require(path.relative_to(ROOT).as_posix() in {row['archived'] for row in cadence_archived},
                    f'unmapped new archive: {original}')
            continue
        if original == 'README.md': continue  # Archive index, newly authored.
        original = legacy_original(original)
        if original not in index:
            raise RuntimeError(f'legacy file has no baseline mapping: {original}')
        if original in index:
            require(blob_hash(path) == index[original], f'archive content changed: {original}')
            archived.append(dict(original=original, archived=path.relative_to(ROOT).as_posix(),
                                 git_blob_sha1=index[original]))
    (report / 'legacy_inventory.json').write_text(json.dumps(archived + cadence_archived, indent=2), encoding='utf-8')
    originals = {row['original'] for row in archived}
    for original in index:
        if original.startswith(('env/', 'algorithm/', 'configs/', 'tools/', 'tests/')):
            if not (ROOT / original).exists() and original not in originals:
                raise RuntimeError(f'missing archive: {original}')
    base = args.checkpoint_dir
    cfg = yaml.safe_load((ROOT / 'configs/single_round_environment.yaml').read_text())
    validate_single_round_environment_config(cfg)
    old = yaml.safe_load(baseline_content(ROOT, 'configs/persistent_wave_v2_environment.yaml'))
    require('persistent_waves' not in cfg, 'obsolete task configuration remains')
    require(cfg['blue_policy']['type'] == 'ground_aware_nearest_target', 'Blue policy type mismatch')
    for section in ('action', 'aircraft', 'arena', 'scenario', 'weapon', 'reward', 'observation'):
        require(cfg[section] == old[section], f'frozen config section changed: {section}')
    require(cfg['simulation']['dt'] == old['simulation']['dt'] == .1, 'dt mismatch')
    require(cfg['blue_policy']['ground_avoidance'] == old['blue_policy']['ground_avoidance'], 'Blue guard mismatch')
    require(cfg['blue_policy']['desired_speed'] == old['blue_policy']['desired_speed'], 'Blue desired_speed mismatch')
    previous = yaml.safe_load((ROOT / 'legacy/single_round_v1/configs/single_round_environment.yaml').read_text())
    expected_config = {**previous, 'environment_variant': 'single_round_v2',
                       'attack': {'cadence': 'fixed_cooldown', 'cooldown_seconds': 1.0}}
    require(cfg == expected_config, 'v2 config changed beyond cadence/variant')
    combat_source = (ROOT / 'env/combat_env.py').read_text(encoding='utf-8')
    for obsolete in ('FireState', '.armed', 'red_fire_states', 'blue_fire_states', '_entry_attempts'):
        require(obsolete not in combat_source, f'formal combat still depends on {obsolete}')
    algo = yaml.safe_load((ROOT / 'configs/madsac_single_round_3m.yaml').read_text())
    require(algo == yaml.safe_load(baseline_content(ROOT, 'configs/madsac_persistent_wave_v2_3m.yaml')), 'MADSAC config mismatch')
    mappo, modes = None, None
    if args.checkpoint_dir:
        algo = yaml.safe_load((ROOT / 'configs/mappo.yaml').read_text())
        result = evaluate_mappo_checkpoint(base / 'mappo/latest.pt', algo, cfg, 'cuda', [10000000])
        require(result['protocol_complete'] and result['evaluation_environment_variant'] == 'single_round_v2', 'checkpoint protocol mismatch')
        mappo = {k: result[k] for k in ('average_return', 'win_rate', 'average_episode_length', 'protocol_complete')}
        algo = yaml.safe_load((ROOT / 'configs/madsac_single_round_3m.yaml').read_text())
        state = torch.load(base / 'madsac/latest.pt', map_location='cpu', weights_only=False)
        validate_madsac_checkpoint(state, cfg, algo)
        n, t, i = algo['network'], algo['training'], algo['implementation']
        trainer = MADSACTrainer(n['observation_dim'], n['action_dim'], n['num_agents'],
                                state['extra']['network_architecture']['hidden_dim'], n['attention_heads'],
                                t['actor_learning_rate'], t['critic_learning_rate'], t['gamma'],
                                t['alpha'], t['tau'], t['policy_delay'], 'cuda', t['seed'],
                                i['actor_activation'], i['critic_activation'], i['log_std_min'], i['log_std_max'])
        trainer.load_for_evaluation(base / 'madsac/latest.pt')
        modes = {mode: evaluate_madsac(trainer, cfg, [44000000], mode) for mode in ('deterministic', 'stochastic')}
        for metrics in (result, *modes.values()):
            require(not any(forbidden.search(key) for key in metrics), 'obsolete evaluation metric')
    acceptance = dict(baseline_commit=BASE_COMMIT, current_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(), frozen_comparison_scope='baseline to working tree; Git index ignored', changed_files_scope='baseline to HEAD only', runtime_files_scanned=len(runtime), runtime_keyword_findings=findings,
                      static_python_parse='passed', frozen_foundations_unchanged=[n for n in frozen if n not in TEAM_SIZE_REFACTORS],
                      team_size_refactors=team_size_refactors,
                      cadence_frozen_files_verified=len(cadence_baseline['frozen_files']),
                      cadence_archived_files_verified=len(cadence_archived),
                      formal_protocol='single_round_v2, 52/3/4, cooldown=1.0s=10steps',
                      only_formal_combat_cadence_changed='exit-to-rearm -> fixed 1.0 s per-aircraft cooldown',
                      no_formal_armed_dependency=True,
                      frozen_config_sections_unchanged=True, archived_files_verified=len(archived),
                      baseline_networks_and_trainers_unchanged=True,
                      madsac_hyperparameters_unchanged=True,
                      cuda_device=torch.cuda.get_device_name(0) if args.checkpoint_dir else None, mappo_checkpoint_evaluation=mappo,
                      madsac_checkpoint_evaluation=modes)
    (report / 'single_round_audit.json').write_text(json.dumps(acceptance, indent=2), encoding='utf-8')
    with (report / 'changed_files.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['status', 'path', 'original_path'])
        writer.writeheader(); writer.writerows(changed_files(ROOT))
    print(json.dumps({k: v for k, v in acceptance.items() if k != 'madsac_checkpoint_evaluation'}, indent=2))


if __name__ == '__main__': main()

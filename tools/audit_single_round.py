"""Immutable pre-cadence formal audit and separate historical archive provenance."""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json
import re
import subprocess
import sys
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from algorithm.common.protocol import validate_single_round_environment_config
from algorithm.mappo.evaluation import evaluate_mappo_checkpoint
from algorithm.madsac.protocol import validate_madsac_checkpoint
from algorithm.madsac.trainer import MADSACTrainer
from algorithm.madsac.evaluation import evaluate_madsac

BASE_COMMIT = '8d09df1fd1e6d2425b712d0c336e9525cd313c4c'
PRE_CADENCE_COMMIT = '89fd4c54d07205bd0952eb972cf133b4a41349f7'
ARCHIVE_SNAPSHOT_COMMIT = 'b80337fd7aad8a37865f10e3f2cb6451d2775b96'
V2_COMMIT = '81d623e1ecafbf7df91aa2d9dc1fc19cf885f2a9'
V3_CLOSURE_BASELINE = 'e05130e4a20d81799762996f9ea14a61867c8339'
FROZEN_FILES = tuple(f'env/{name}.py' for name in (
    'dynamics', 'integrator', 'weapon', 'reward', 'observation',
    'scenario', 'fixed_policy', 'models', 'geometry', 'math_utils')) + (
    'algorithm/common/vector_env.py',
    'algorithm/mappo/networks.py', 'algorithm/mappo/trainer.py',
    'algorithm/madsac/networks.py', 'algorithm/madsac/trainer.py', 'algorithm/madsac/evaluation.py')
FORBIDDEN = re.compile(r'persistent_wave|persistent_waves|wave_index|waves_cleared|wave_cleared|spawned_next_wave|spawn_next_wave|clear_wave_|per_wave_metrics|wave_context|inter_wave|wave_entry|wave_specific|\bQ2\b|\bQ3\b')


def require(condition, message):
    if not condition: raise RuntimeError(message)


def baseline_blobs(root, commit=BASE_COMMIT):
    rows = subprocess.check_output(['git', 'ls-tree', '-r', commit], cwd=root, text=True)
    return {line.split('\t')[1]: line.split('\t')[0].split()[2]
            for line in rows.splitlines() if line.split()[1] == 'blob'}


def baseline_content(root, path, commit=BASE_COMMIT):
    return subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=root)


def blob_hash(path):
    data = path.read_bytes().replace(b'\r\n', b'\n')
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()


def compare_frozen_files(root, paths=FROZEN_FILES):
    blobs = baseline_blobs(root, V2_COMMIT)
    result = []
    for path in paths:
        current = blob_hash(root / path)
        require(current == blobs[path], f'frozen source differs from pre-cadence baseline: {path}')
        result.append(dict(path=path, baseline_blob=blobs[path], current_blob=current))
    return result


def changed_files(root):
    """Committed pre-cadence → HEAD changes; no dirty-index or worktree inference."""
    lines = subprocess.check_output(['git', 'diff', '--name-status', '-M', PRE_CADENCE_COMMIT, 'HEAD'],
                                    cwd=root, text=True)
    return [dict(status=parts[0], path=parts[-1], original_path=parts[1] if len(parts) == 3 else '')
            for parts in (line.split('\t') for line in lines.splitlines())]


def legacy_original(relative):
    return 'README.md' if relative == 'README_original.md' else relative


def verify_archives(root):
    historical = baseline_blobs(root)
    snapshots = baseline_blobs(root, ARCHIVE_SNAPSHOT_COMMIT)
    rows = []
    for path in sorted((root / 'legacy').rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts: continue
        relative = path.relative_to(root / 'legacy').as_posix()
        if relative in ('README.md', 'single_round_v2/README.md'): continue  # Authored archive indexes.
        archived = path.relative_to(root).as_posix()
        if relative.startswith('pursuit_v1/'):
            original = relative[len('pursuit_v1/'):]
            expected = baseline_blobs(root,V3_CLOSURE_BASELINE)[original]
            source = V3_CLOSURE_BASELINE
        elif relative.startswith('single_round_v2/'):
            original = relative[len('single_round_v2/'):]
            expected = baseline_blobs(root, V2_COMMIT)[original]
            source = V2_COMMIT
        elif relative.startswith('single_round_v1/'):
            require(archived in snapshots, f'archive absent from immutable snapshot: {archived}')
            original, expected, source = relative[len('single_round_v1/'):], snapshots[archived], ARCHIVE_SNAPSHOT_COMMIT
        else:
            original = legacy_original(relative)
            require(original in historical, f'unmapped historical archive: {archived}')
            expected, source = historical[original], BASE_COMMIT
        require(blob_hash(path) == expected, f'archived source changed: {archived}')
        rows.append(dict(original=original, archived=archived, provenance_commit=source, git_blob_sha1=expected))
    for prefix, blobs, archived_prefix in (
        ('', historical, None), ('legacy/single_round_v1/', snapshots, 'legacy/single_round_v1/')):
        verified = {row['archived'] for row in rows}
        if archived_prefix:
            for path in blobs:
                if path.startswith(prefix): require(path in verified, f'missing archive snapshot: {path}')
        else:
            originals = {row['original'] for row in rows if row['provenance_commit'] == BASE_COMMIT}
            for path in blobs:
                if path.startswith(('env/', 'algorithm/', 'configs/', 'tools/', 'tests/')) and not (root / path).exists():
                    require(path in originals, f'missing historical source archive: {path}')
    return rows


def exact_config_delta(root):
    previous = yaml.safe_load(baseline_content(root, 'configs/single_round_environment.yaml', V2_COMMIT))
    current = yaml.safe_load((root / 'configs/single_round_environment.yaml').read_text(encoding='utf-8'))
    allowed = {'environment_version', 'environment_variant', 'task_type', 'action_semantics', 'action', 'blue_policy'}
    require(set(current) == set(previous) | {'task_type', 'action_semantics'}, 'unexpected v3 configuration keys')
    for key in set(previous) - allowed:
        require(current[key] == previous[key], f'frozen combat configuration changed: {key}')
    validate_single_round_environment_config(current)
    for name in ('mappo.yaml', 'madsac_single_round_3m.yaml'):
        old = yaml.safe_load(baseline_content(root, f'configs/{name}', V2_COMMIT))
        new = yaml.safe_load((root / 'configs' / name).read_text(encoding='utf-8'))
        require(old == new, f'formal algorithm config changed: {name}')
    return current


def direct_path_audit(root):
    closure_blobs = baseline_blobs(root,V3_CLOSURE_BASELINE)
    require(blob_hash(root/'env/control.py') == closure_blobs['env/control.py'], 'direct decoder changed since v3 acceptance')
    old_config = yaml.safe_load(baseline_content(root,'configs/single_round_environment.yaml',V3_CLOSURE_BASELINE))
    new_config = yaml.safe_load((root/'configs/single_round_environment.yaml').read_text(encoding='utf-8'))
    old_config['blue_policy']['ground_avoidance']['consider_commanded_pitch'] = True
    require(new_config == old_config, 'combat config changed beyond commanded-descent guard identity')
    old_policy = ast.parse(baseline_content(root,'env/direct_policy.py',V3_CLOSURE_BASELINE).decode())
    new_policy = ast.parse((root/'env/direct_policy.py').read_text(encoding='utf-8'))
    find_guidance = lambda t: next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name=='action_toward')
    require(ast.dump(find_guidance(old_policy))==ast.dump(find_guidance(new_policy)), 'direct guidance core changed')
    current = ast.parse((root/'env/combat_env.py').read_text(encoding='utf-8'))
    previous = ast.parse(baseline_content(root, 'env/combat_env.py', V2_COMMIT).decode())
    methods = lambda tree: {n.name:n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
    a,b = methods(current),methods(previous)
    for name in ('step', '_attack_attempts', '_resolve_combat', '_resolve_noncombat_losses'):
        require(ast.dump(a[name]) == ast.dump(b[name]), f'frozen combat method changed: {name}')
    decoder_tree = ast.parse((root/'env/control.py').read_text(encoding='utf-8'))
    decoder = methods(decoder_tree)['normalized_action_to_direct_control']
    red_path = ast.unparse(a['_advance']) + ast.unparse(decoder)
    for token in ('heading_time_constant','pitch_time_constant','speed_time_constant','ManeuverTarget',
                  'action_to_target','target_to_control','action_to_control','raw_nz','normal_load_max'):
        require(token not in red_path, f'legacy controller in direct Red path: {token}')
    require('normalized_action_to_direct_control' in ast.unparse(a['_advance']), 'missing direct decoder')
    # Legacy controller definitions remain exactly the same, including function bodies.
    old_control = ast.parse(baseline_content(root,'env/control.py',V2_COMMIT).decode())
    old_defs = {n.name:ast.dump(n) for n in old_control.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    new_defs = {n.name:ast.dump(n) for n in decoder_tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    require(all(new_defs[k] == v for k,v in old_defs.items()), 'historical controller changed')
    return dict(red_path='linear decoder -> ControlCommand -> unchanged RK4/dynamics',
                legacy_controller_excluded=True, frozen_combat_methods_verified=4)


def structural_audit(root=ROOT):
    frozen = compare_frozen_files(root)
    archives = verify_archives(root)
    exact_config_delta(root)
    direct = direct_path_audit(root)
    runtime = [p for folder in ('env', 'algorithm', 'configs') for p in (root / folder).rglob('*')
               if p.suffix in ('.py', '.yaml')] + [root / 'README.md']
    findings = [dict(path=p.relative_to(root).as_posix(), line=i, text=line)
                for p in runtime for i, line in enumerate(p.read_text(encoding='utf-8').splitlines(), 1)
                if FORBIDDEN.search(line)]
    require(not findings, f'runtime keyword findings: {findings}')
    six_pattern = re.compile(r'6v6|78D|\(4,\s*6\)|\(num_envs,\s*self\.team_size\).*ones')
    require(not any(six_pattern.search(p.read_text(encoding='utf-8')) for p in runtime if p.name != 'README.md'),
            'active team-size generalization remains')
    for path in runtime + list((root / 'tests').glob('*.py')) + list((root / 'tools').glob('*.py')):
        if path.suffix == '.py': ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for folder in ('env', 'algorithm'):
        for path in (root / folder).rglob('*.py'):
            if path == root / 'env/weapon.py': continue  # Frozen historical class definition is intentionally retained.
            text = path.read_text(encoding='utf-8')
            require(not any(term in text for term in ('FireState', '.armed', '_entry_attempts', 'red_fire_states', 'blue_fire_states')),
                    f'old firing dependency remains: {path}')
    return dict(historical_archive_baseline=BASE_COMMIT, baseline_commit=BASE_COMMIT,
                pre_cadence_commit=PRE_CADENCE_COMMIT, v2_baseline_commit=V2_COMMIT, direct_path_audit=direct, archive_snapshot_commit=ARCHIVE_SNAPSHOT_COMMIT,
                current_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                frozen_comparison_scope='immutable v2 commit to working tree; index ignored',
                changed_files_scope='pre-cadence commit to HEAD only', frozen_files=frozen,
                frozen_files_verified=len(frozen), archived_files_verified=sum(r['provenance_commit']==BASE_COMMIT for r in archives),
                cadence_archived_files_verified=sum(r['provenance_commit']==ARCHIVE_SNAPSHOT_COMMIT for r in archives),
                legacy_inventory=archives, runtime_files_scanned=len(runtime), runtime_keyword_findings=findings,
                no_active_six_vs_six=True, no_formal_armed_dependency=True, exact_config_delta='version/variant/task/action identity + direct action + Blue guidance only',
                formal_protocol='single_round_v3 / 4v4 / 52D / 3D / max_steps=1000 / cooldown=1.0s=10steps',
                combat_cadence_unchanged='fixed per-aircraft 1.0 s cooldown',
                static_python_parse='passed')


def checkpoint_audit(base):
    require(torch.cuda.is_available(), 'CUDA mandatory for checkpoint audit')
    cfg = exact_config_delta(ROOT)
    algo = yaml.safe_load((ROOT / 'configs/mappo.yaml').read_text())
    mappo = evaluate_mappo_checkpoint(base / 'mappo/latest.pt', algo, cfg, 'cuda', [10000000])
    require(mappo['protocol_complete'] and mappo['evaluation_environment_variant']=='single_round_v3', 'checkpoint identity mismatch')
    algo = yaml.safe_load((ROOT / 'configs/madsac_single_round_3m.yaml').read_text())
    state = torch.load(base / 'madsac/latest.pt', map_location='cpu', weights_only=False)
    validate_madsac_checkpoint(state, cfg, algo)
    n, t, i = algo['network'], algo['training'], algo['implementation']
    trainer = MADSACTrainer(52, 3, 4, state['extra']['network_architecture']['hidden_dim'], n['attention_heads'],
        t['actor_learning_rate'], t['critic_learning_rate'], t['gamma'], t['alpha'], t['tau'], t['policy_delay'],
        'cuda', t['seed'], i['actor_activation'], i['critic_activation'], i['log_std_min'], i['log_std_max'])
    trainer.load_for_evaluation(base / 'madsac/latest.pt')
    modes = {mode: evaluate_madsac(trainer, cfg, [44000000], mode) for mode in ('deterministic', 'stochastic')}
    return dict(cuda_device=torch.cuda.get_device_name(0), mappo=mappo, madsac=modes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/single_round_audit')
    parser.add_argument('--checkpoint-dir', type=Path)
    args = parser.parse_args()
    result = structural_audit()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'legacy_inventory.json').write_text(json.dumps(result.pop('legacy_inventory'), indent=2))
    with (args.output_dir / 'changed_files.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['status', 'path', 'original_path'])
        writer.writeheader(); writer.writerows(changed_files(ROOT))
    if args.checkpoint_dir: result['checkpoint_audit'] = checkpoint_audit(args.checkpoint_dir)
    (args.output_dir / 'single_round_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('frozen_files', 'checkpoint_audit')}, indent=2))


if __name__ == '__main__': main()

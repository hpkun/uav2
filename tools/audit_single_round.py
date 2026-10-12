"""Reproducible source, frozen-foundation, and CUDA checkpoint acceptance audit."""
from pathlib import Path
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
from algorithm.mappo.evaluation import evaluate_mappo_checkpoint
from algorithm.madsac.protocol import validate_madsac_checkpoint
from algorithm.madsac.trainer import MADSACTrainer
from algorithm.madsac.evaluation import evaluate_madsac


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA mandatory for checkpoint audit')
    report = ROOT / 'reports'
    report.mkdir(exist_ok=True)
    forbidden = re.compile(r'persistent_wave|persistent_waves|wave_index|waves_cleared|wave_cleared|spawned_next_wave|spawn_next_wave|clear_wave_|per_wave_metrics|wave_context|inter_wave|wave_entry|wave_specific|\bQ2\b|\bQ3\b')
    runtime = [p for folder in ('env', 'algorithm', 'configs')
               for p in (ROOT / folder).rglob('*') if p.suffix in ('.py', '.yaml')]
    runtime.append(ROOT / 'README.md')
    findings = []
    for path in runtime:
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if forbidden.search(line):
                findings.append(dict(path=path.relative_to(ROOT).as_posix(), line=number, text=line))
    assert not findings, findings
    for path in runtime + list((ROOT / 'tests').glob('*.py')) + list((ROOT / 'tools').glob('*.py')):
        if path.suffix == '.py': ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    index = {}
    for line in subprocess.check_output(['git', 'ls-files', '-s'], cwd=ROOT, text=True).splitlines():
        metadata, path = line.split('\t')
        index[path] = metadata.split()[1]
    def blob_hash(path):
        # Git's Windows text checkout may use CRLF while its index stores LF.
        data = path.read_bytes().replace(b'\r\n', b'\n')
        return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
    frozen = ('control', 'dynamics', 'integrator', 'weapon', 'reward', 'observation',
              'scenario', 'fixed_policy', 'models', 'geometry', 'math_utils')
    for name in frozen:
        path = f'env/{name}.py'
        assert blob_hash(ROOT / path) == index[path], f'foundation changed: {path}'
    for method in ('mappo', 'madsac'):
        for name in ('networks', 'trainer'):
            path = f'algorithm/{method}/{name}.py'
            assert blob_hash(ROOT / path) == index[path], f'algorithm foundation changed: {path}'
    archived = []
    for path in sorted((ROOT / 'legacy').rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts: continue
        original = path.relative_to(ROOT / 'legacy').as_posix()
        if original == 'README.md': continue  # Archive index, newly authored.
        if original == 'README_original.md': original = 'README.md'
        if original in index:
            assert blob_hash(path) == index[original], f'archive content changed: {original}'
            archived.append(dict(original=original, archived=path.relative_to(ROOT).as_posix(),
                                 git_blob_sha1=index[original]))
    (report / 'legacy_inventory.json').write_text(json.dumps(archived, indent=2), encoding='utf-8')
    base = ROOT / 'outputs/single_round_acceptance_final'
    cfg = yaml.safe_load((ROOT / 'configs/single_round_environment.yaml').read_text())
    old = yaml.safe_load((ROOT / 'legacy/configs/persistent_wave_v2_environment.yaml').read_text())
    for section in ('action', 'aircraft', 'arena', 'scenario', 'weapon', 'reward', 'observation'):
        assert cfg[section] == old[section], f'frozen config section changed: {section}'
    assert cfg['simulation']['dt'] == old['simulation']['dt'] == .1
    assert cfg['blue_policy']['ground_avoidance'] == old['blue_policy']['ground_avoidance']
    assert cfg['blue_policy']['desired_speed'] == old['blue_policy']['desired_speed']
    algo = yaml.safe_load((ROOT / 'configs/mappo.yaml').read_text())
    result = evaluate_mappo_checkpoint(base / 'mappo/latest.pt', algo, cfg, 'cuda', [10000000])
    assert result['protocol_complete'] and result['evaluation_environment_variant'] == 'single_round_v1'
    mappo = {k: result[k] for k in ('average_return', 'win_rate', 'average_episode_length', 'protocol_complete')}
    algo = yaml.safe_load((ROOT / 'configs/madsac_single_round_3m.yaml').read_text())
    assert algo == yaml.safe_load((ROOT / 'legacy/configs/madsac_persistent_wave_v2_3m.yaml').read_text())
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
        assert not any(forbidden.search(key) for key in metrics)
    acceptance = dict(runtime_files_scanned=len(runtime), runtime_keyword_findings=findings,
                      static_python_parse='passed', frozen_foundations_unchanged=list(frozen),
                      frozen_config_sections_unchanged=True, archived_files_verified=len(archived),
                      baseline_networks_and_trainers_unchanged=True,
                      madsac_hyperparameters_unchanged=True,
                      cuda_device=torch.cuda.get_device_name(0), mappo_checkpoint_evaluation=mappo,
                      madsac_checkpoint_evaluation=modes)
    (report / 'single_round_audit.json').write_text(json.dumps(acceptance, indent=2), encoding='utf-8')
    changed = subprocess.check_output(['git', 'diff', '--name-only'], cwd=ROOT, text=True).splitlines()
    added = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=ROOT, text=True).splitlines()
    with (report / 'changed_files.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['status', 'path', 'archive_destination'])
        for path in sorted(changed):
            destination = f'legacy/{path}' if (ROOT / 'legacy' / path).is_file() and not (ROOT / path).exists() else ''
            writer.writerow(['archived' if destination else 'modified', path, destination])
        for path in sorted(added): writer.writerow(['added', path, ''])
    print(json.dumps({k: v for k, v in acceptance.items() if k != 'madsac_checkpoint_evaluation'}, indent=2))


if __name__ == '__main__': main()

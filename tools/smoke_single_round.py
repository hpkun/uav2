"""CUDA training smoke for both baselines with real episode/reset evidence."""
from pathlib import Path
import json
import argparse
import sys
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from algorithm.mappo.runner import MAPPOTrainingRunner
from algorithm.madsac.runner import MADSACTrainingRunner
from env.factory import make_combat_environment


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA mandatory for smoke tests')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/single_round_acceptance_final')
    root = parser.parse_args().output_dir
    cfg = yaml.safe_load((ROOT / 'configs/single_round_environment.yaml').read_text())
    evidence = {}
    for name, runner_type, filename in (
        ('mappo', MAPPOTrainingRunner, 'mappo.yaml'),
        ('madsac', MADSACTrainingRunner, 'madsac_single_round_3m.yaml'),
    ):
        algo = yaml.safe_load((ROOT / 'configs' / filename).read_text())
        output = root / name
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f'refusing nonempty smoke output: {output}')
        runner = runner_type(cfg, algo, num_envs=1, total_sampled_steps=1200,
                             device='cuda', output_dir=output, smoke=True)
        resets = []
        original_step = runner.vector.step_batch
        def verified_step(actions, auto_reset=True):
            result = original_step(actions, auto_reset=auto_reset)
            if auto_reset and bool((result.terminated | result.truncated)[0]):
                seed = int(runner.vector.last_reset_seeds[0])
                reference = make_combat_environment(cfg)
                expected, reset_info = reference.reset(seed)
                assert reference.red_alive_mask.sum() == reference.blue_alive_mask.sum() == 4
                assert torch.equal(torch.from_numpy(result.observations[0]), torch.from_numpy(expected))
                assert not torch.equal(torch.from_numpy(result.transition_next_observations[0]), torch.from_numpy(expected))
                resets.append(dict(new_seed=seed, new_red_alive=4, new_blue_alive=4,
                                   new_episode_length=reference.steps,
                                   terminal_reason=result.infos[0]['termination_reason'],
                                   terminal_red_survivors=result.infos[0]['red_survivors'],
                                   terminal_blue_survivors=result.infos[0]['blue_survivors']))
            return result
        runner.vector.step_batch = verified_step
        if name == 'mappo':
            runner.rollout_steps = 64
            runner.evaluation_seeds = [10000000]
            runner.evaluation_interval = runner.next_evaluation = 1200
        print(runner.start_log_line(), flush=True)
        summary = runner.run()
        assert runner.completed_records, f'{name}: no completed episode'
        assert runner.vector.episode_indices[0] > 0
        assert len(runner.vector.used_training_seeds) == runner.vector.episode_indices[0] + 1
        assert len(runner.vector.used_training_seeds) == len(set(runner.vector.used_training_seeds))
        assert runner.vector.worker_environment_classes == ['MultiUAVCombatEnv']
        assert runner.vector.worker_fixed_policy_classes == ['GroundAwareNearestTargetPursuitPolicy']
        assert not any('wave' in key for row in runner.completed_records for key in row)
        (output / 'run_summary.json').write_text(json.dumps(summary, indent=2))
        (output / 'env_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        (output / 'algorithm_config.yaml').write_text(yaml.safe_dump(algo, sort_keys=False))
        evidence[name] = {
            'device': runner.device, 'sampled_steps': runner.trainer.sampled_steps,
            'completed_episodes': len(runner.completed_records),
            'episode_lengths': [r['episode_length'] for r in runner.completed_records],
            'terminal_reasons': [r['termination_reason'] for r in runner.completed_records],
            'episode_indices': runner.vector.episode_indices.tolist(),
            'unique_reset_seeds': sorted(runner.vector.used_training_seeds),
            'verified_new_episode_resets': resets,
            'worker_environment_classes': runner.vector.worker_environment_classes,
            'worker_fixed_policy_classes': runner.vector.worker_fixed_policy_classes,
            'best_selection': runner.best_evaluation,
            'optimization_metrics': runner.last_metrics,
            'max_steps': cfg['simulation']['max_steps'],
        }
        print(json.dumps({k: v for k, v in evidence[name].items()
                          if k not in {'best_selection', 'optimization_metrics'}}), flush=True)
    (root / 'evidence.json').write_text(json.dumps(evidence, indent=2))


if __name__ == '__main__': main()

"""Bounded CUDA smokes and synthetic log-size acceptance in a fresh directory."""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,sys
import numpy as np
import torch,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from algorithm.mappo.runner import MAPPOTrainingRunner
from algorithm.madsac.runner import MADSACTrainingRunner
from algorithm.mappo.evaluation import evaluate_mappo_checkpoint
from algorithm.common.protocol import config_sha256
from env.pursuit_env import PursuitEnv
from tools.compact_training_metrics import compact_metrics


def read(name):return yaml.safe_load((ROOT/'configs'/name).read_text())

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/logging_acceptance')
    root=parser.parse_args().output_dir
    if root.exists() and any(root.iterdir()):raise RuntimeError(f'refusing nonempty output: {root}')
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory; no fallback')
    root.mkdir(parents=True,exist_ok=True)
    evidence={'cuda_device':torch.cuda.get_device_name(0),'smokes':{}}
    for name,runner_type,env_name,alg_name in (
        ('combat',MAPPOTrainingRunner,'single_round_environment.yaml','mappo.yaml'),
        ('pursuit',MAPPOTrainingRunner,'pursuit_environment.yaml','mappo.yaml'),
        ('madsac',MADSACTrainingRunner,'single_round_environment.yaml','madsac_single_round_3m.yaml')):
        env,alg=read(env_name),read(alg_name)
        runner=runner_type(env,alg,num_envs=1,total_sampled_steps=128,device='cuda',seed=79,
            output_dir=root/name,smoke=True)
        runner.run()
        records=[json.loads(line) for line in (root/name/'training_metrics.jsonl').read_text().splitlines()]
        assert [r['sampled_steps'] for r in records]==[128]
        assert all(np.isfinite(value) for value in runner.last_metrics.values())
        evaluation=None
        if name!='madsac':evaluation=evaluate_mappo_checkpoint(root/name/'latest.pt',alg,env,'cuda',[10000000])
        evidence['smokes'][name]={'sampled_steps':128,'rows':1,
            'training_metrics_bytes':(root/name/'training_metrics.jsonl').stat().st_size,
            'optimization_records':len((root/name/'optimization_metrics.jsonl').read_text().splitlines()),
            'evaluation':evaluation or runner.evaluation_history[-1],
            'environment_config_sha256':config_sha256(env),'algorithm_config_sha256':config_sha256(alg)}
        print(f'[LOGGING SMOKE] {name}: 128 CUDA transitions, final snapshot present',flush=True)
    # Actual pursuit schema with 24 environments, synthetic counters only (no training).
    env=PursuitEnv(read('pursuit_environment.yaml'));env.reset(79)
    _,reward,_,_,info=env.step(np.tile([0,.2,0],(4,1)))
    result=SimpleNamespace(rewards=np.tile(reward,(24,1)),infos=[info]*24)
    logger=MAPPOTrainingRunner.__new__(MAPPOTrainingRunner)
    logger.env_config=env.config;logger.console_interval=20000
    logger.trainer=SimpleNamespace(sampled_steps=0);logger.last_metrics={}
    logger.output_dir=root/'synthetic';logger.output_dir.mkdir()
    logger.initialize_training_metrics_logging()
    for step in range(24,3000001,24):
        logger.trainer.sampled_steps=step
        logger.record_training_metrics_transition(result,[],logger._write_step_metrics)
    logger.finalize_training_metrics_logging(logger._write_step_metrics)
    path=logger.output_dir/'training_metrics.jsonl'
    records=[json.loads(line) for line in path.read_text().splitlines()]
    assert len(records)==150 and records[-1]['sampled_steps']==3000000
    representative=records[0]
    old_base=len((json.dumps(representative)+'\n').encode())-len(str(representative['sampled_steps']))
    old_bytes=sum(old_base+len(str(step)) for step in range(24,3000001,24))
    evidence['synthetic_3m']={'num_envs':24,'interval':20000,'old_rows':125000,'new_rows':150,
        'row_reduction_fraction':1-150/125000,'new_bytes':path.stat().st_size,
        'old_bytes_representative_schema_estimate':old_bytes}
    # Offline smoke touches only a new synthetic file, never an existing run.
    source=root/'offline_synthetic.jsonl'
    with source.open('w') as stream:
        total=0
        for index in range(1000):
            total+=24+(index%17)
            stream.write(json.dumps({'sampled_steps':total,'diagnostic':[1,2,3]})+'\n')
    evidence['offline_compactor']=compact_metrics(source)
    (root/'evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps(evidence['synthetic_3m'],indent=2))

if __name__=='__main__':main()

"""Frozen v3.31 audit and bounded v3.32 environment/MAPPO CUDA acceptance."""
from pathlib import Path
import argparse,ast,json,subprocess,sys,hashlib
from unittest.mock import patch
import numpy as np
import torch,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from env.config import load_config,environment_dimensions
from env.factory import make_combat_environment
from env.models import AircraftState
from algorithm.mappo.runner import MAPPOTrainingRunner
from algorithm.mappo.evaluation import evaluate_mappo_checkpoint
from algorithm.common.checkpoint import validate_checkpoint_for_evaluation,validate_checkpoint_for_resume
BASELINE='9f6dc576367a071c2b784ee1ada37ea822000e11'


def frozen_audit():
    tracked=subprocess.check_output(['git','ls-tree','-r','--name-only',BASELINE],cwd=ROOT,text=True).splitlines()
    frozen=[name for name in tracked if name.startswith(('env/','configs/','algorithm/'))
            and name not in ('env/config.py','env/combat_env.py','algorithm/mappo/runner.py','algorithm/common/evaluator.py')]
    for name in frozen:
        old=subprocess.check_output(['git','show',f'{BASELINE}:{name}'],cwd=ROOT)
        assert old==(ROOT/name).read_bytes().replace(b'\r\n',b'\n'),f'frozen source changed: {name}'
    a=load_config(ROOT/'configs/combat_environment_v331.yaml')
    b=load_config(ROOT/'configs/combat_environment_v332.yaml')
    back=dict(b);back.pop('red_direct_control');back['environment_version']='3.31'
    assert back==a
    old_alg=yaml.safe_load((ROOT/'configs/mappo_4v4_v331.yaml').read_text())
    new_alg=yaml.safe_load((ROOT/'configs/mappo_4v4_v332.yaml').read_text())
    new_alg['network']['observation_dim']=45;assert old_alg==new_alg
    for folder in ('env','algorithm','tools','tests'):
        for file in (ROOT/folder).rglob('*.py'):ast.parse(file.read_text(encoding='utf-8'),filename=str(file))
    compile_count=0
    for file in [ROOT/f'env/{name}.py' for name in ('v332_config','v332_control','v332_observation','v332_transition','combat_v332')]:
        compile(file.read_text(encoding='utf-8'),str(file),'exec');compile_count+=1
    old_core=ast.parse((ROOT/'env/v330_transition.py').read_text())
    new_core=ast.parse((ROOT/'env/v332_transition.py').read_text())
    old_fn=next(n for n in old_core.body if isinstance(n,ast.FunctionDef))
    new_fn=next(n for n in new_core.body if isinstance(n,ast.FunctionDef))
    for node in ast.walk(new_fn):
        if isinstance(node,ast.Attribute) and node.attr=='_advance_red_direct':node.attr='_advance'
    assert ast.dump(old_fn)==ast.dump(new_fn)
    old_wrapper=ast.parse((ROOT/'env/combat_v330.py').read_text())
    new_wrapper=ast.parse((ROOT/'env/combat_v332.py').read_text())
    step=lambda t:next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name=='step')
    assert ast.dump(step(old_wrapper))==ast.dump(step(new_wrapper))
    return dict(baseline=BASELINE,frozen_files_verified=len(frozen),static_ast='passed',
                compiled_new_environment_modules=compile_count,
                transition_difference='only Red advance call',reward_info_wrapper_ast='identical')


def environment_smoke():
    e=make_combat_environment(ROOT/'configs/combat_environment_v332.yaml');obs,info=e.reset(18)
    assert obs.shape==(4,47) and environment_dimensions(e.config)==(47,3,4)
    old=make_combat_environment(ROOT/'configs/combat_environment_v331.yaml');old.reset(18)
    actions=np.random.default_rng(19).uniform(-1,1,(4,3))
    e._advance(e.blue,actions);old._advance(old.blue,actions)
    assert np.array_equal(np.stack([s.as_array() for s in e.blue]),np.stack([s.as_array() for s in old.blue]))
    e.reset(18)
    def state(x,alive=True):return AircraftState(x,0,-3000,225,0,0,alive)
    e.red=[state(0)]+[state(0,False) for _ in range(3)]
    e.blue=[state(500)]+[state(500,False) for _ in range(3)]
    attempts=[]
    with patch.object(e,'attempt_hit_for_pair',return_value=False):
        for number in range(1,26):
            obs,reward,terminated,truncated,info=e.step(np.zeros((4,3)),np.zeros((4,3)))
            assert obs.shape==(4,47) and not terminated and not truncated
            assert np.isfinite(reward).all()
            if info['red_step_fire_attempts']:attempts.append(number)
    assert attempts==[1,11,21],attempts
    return dict(observation_shape=list(obs.shape),blue_step_bitwise_equal_to_v331=True,
                continuous_window_attempt_steps=attempts,weapon_range_m=e.weapon.range_max,
                trim_control=e.red_control_states[0].tolist())


def cuda_smoke(root):
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory, no fallback')
    cfg=load_config(ROOT/'configs/combat_environment_v332.yaml')
    algo=yaml.safe_load((ROOT/'configs/mappo_4v4_v332.yaml').read_text())
    output=root/'mappo_cuda_smoke'
    if output.exists() and any(output.iterdir()):raise RuntimeError(f'refusing nonempty smoke directory: {output}')
    runner=MAPPOTrainingRunner(cfg,algo,num_envs=2,total_sampled_steps=512,device='cuda',seed=31,
                               output_dir=output,smoke=True)
    print(runner.start_log_line(),flush=True)
    summary=runner.run()
    assert runner.trainer.sampled_steps==512
    assert runner.trainer.ppo_update_count==64
    assert all(np.isfinite(v) for v in runner.last_metrics.values())
    assert runner.vector.worker_environment_classes==['CombatEnvironmentV332']*2
    records=[json.loads(line) for line in (output/'training_metrics.jsonl').read_text().splitlines()]
    assert len(records)==64 and records[-1]['sampled_steps']==512
    assert all('mean_abs_mu' in row and 'mean_abs_xi_hat' not in row and 'mean_xi_hat' in row for row in records)
    assert not any('red_mean_abs_high_level_action_psi' in row for row in records)
    evaluation=evaluate_mappo_checkpoint(output/'latest.pt',algo,cfg,'cuda',[10000000])
    state=torch.load(output/'latest.pt',map_location='cpu',weights_only=False)
    old=dict(state);old['extra']=dict(state['extra']);old['extra']['environment_version']='3.31'
    rejections={}
    for name,check in (('resume',validate_checkpoint_for_resume),('evaluation',validate_checkpoint_for_evaluation)):
        try:check(old,cfg,algo)
        except RuntimeError as error:rejections[name]=str(error)
        else:raise AssertionError('old v3.31 identity accepted')
    validate_checkpoint_for_resume(state,cfg,algo)
    validate_checkpoint_for_evaluation(state,cfg,algo)
    for name,value in (('env_config.yaml',cfg),('algorithm_config.yaml',algo)):
        (output/name).write_text(yaml.safe_dump(value,sort_keys=False),encoding='utf-8')
    (output/'run_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return dict(cuda_device=torch.cuda.get_device_name(0),sampled_steps=512,ppo_updates=64,
                observation_dim=47,action_dim=3,num_agents=4,training_metrics_records=len(records),
                completed_episodes=len(runner.completed_records),evaluation=evaluation,
                legacy_v331_rejections=rejections,checkpoint_metadata=state['extra'],
                final_optimization_metrics=runner.last_metrics)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/v332_acceptance')
    parser.add_argument('--cuda-smoke',action='store_true');args=parser.parse_args()
    result={'frozen_audit':frozen_audit(),'environment_smoke':environment_smoke()}
    if args.cuda_smoke:result['cuda_smoke']=cuda_smoke(args.output_dir)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'evidence.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

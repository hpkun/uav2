"""Bounded CUDA acceptance: combat, pursuit, and pursuit actor-initialized combat."""
from pathlib import Path
import sys, json, argparse
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from algorithm.mappo.runner import MAPPOTrainingRunner
from algorithm.mappo.actor_initialization import initialize_actor,validate_actor_initialization
from algorithm.mappo.evaluation import evaluate_mappo_checkpoint
from algorithm.common.checkpoint import validate_checkpoint_for_evaluation,validate_checkpoint_for_resume


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/pursuit_v2_acceptance')
    parser.add_argument('--init-only',action='store_true')
    args=parser.parse_args()
    root=args.output_dir
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory; no fallback')
    evidence=json.loads((root/'evidence.json').read_text()) if args.init_only else {'cuda_device':torch.cuda.get_device_name(0)}
    algo=yaml.safe_load((ROOT/'configs/mappo.yaml').read_text())
    if not args.init_only:
        evidence['old_checkpoint_incompatibility']={}
        combat_cfg=yaml.safe_load((ROOT/'configs/single_round_environment.yaml').read_text())
        pursuit_cfg=yaml.safe_load((ROOT/'configs/pursuit_environment.yaml').read_text())
        for old_name,cfg in (('combat',combat_cfg),('pursuit',pursuit_cfg)):
            path=ROOT/'outputs/direct_control_acceptance'/old_name/'latest.pt'
            if not path.exists():
                evidence['old_checkpoint_incompatibility'][old_name]={'available':False}
                continue
            old_state=torch.load(path,map_location='cpu',weights_only=False)
            rejected={}
            for label,check in (('evaluation',validate_checkpoint_for_evaluation),('resume',validate_checkpoint_for_resume)):
                try:check(old_state,cfg,algo)
                except RuntimeError as error:rejected[label]=str(error)
                else:raise AssertionError(f'old {old_name} checkpoint accepted')
            if old_name=='pursuit':
                try:validate_actor_initialization(old_state,combat_cfg)
                except RuntimeError as error:rejected['actor_initialization']=str(error)
                else:raise AssertionError('pursuit_v1 actor accepted')
            evidence['old_checkpoint_incompatibility'][old_name]={'available':True,'checkpoint':str(path),**rejected}
    source=root/'pursuit/latest.pt' if args.init_only else None
    tasks=(('initialized_combat',False,202),) if args.init_only else (('combat',False,0),('pursuit',True,101),('initialized_combat',False,202))
    for name,pursuit,seed in tasks:
        cfg=yaml.safe_load((ROOT/'configs'/('pursuit_environment.yaml' if pursuit else 'single_round_environment.yaml')).read_text())
        output=root/name
        if output.exists() and any(output.iterdir()):raise RuntimeError(f'nonempty smoke directory: {output}')
        runner=MAPPOTrainingRunner(cfg,algo,num_envs=1,total_sampled_steps=1200,
                                  device='cuda',seed=seed,output_dir=output,smoke=True)
        runner.rollout_steps=64
        runner.evaluation_seeds=[10000000]
        runner.next_evaluation=runner.evaluation_interval=1200
        before_initialization=None
        if name=='initialized_combat':
            try:
                critic={k:v.clone() for k,v in runner.trainer.critic.state_dict().items()}
                source_state=torch.load(source,map_location='cpu',weights_only=False)
                initialize_actor(runner,source)
                assert all(torch.equal(v,runner.trainer.actor.state_dict()[k].cpu()) for k,v in source_state['actor'].items())
                assert all(torch.equal(v,runner.trainer.critic.state_dict()[k]) for k,v in critic.items())
                assert runner.trainer.sampled_steps==runner.trainer.ppo_update_count==0
                assert not runner.trainer.actor_optimizer.state and not runner.trainer.critic_optimizer.state
                assert not runner.evaluation_history and not runner.completed_records
                before_initialization={'actor_exact_copy':True,'critic_unchanged':True,
                    'optimizer_empty':True,'sampled_steps':0,'evaluation_history_empty':True}
                try:validate_checkpoint_for_evaluation(source_state,cfg,algo)
                except RuntimeError as error: evidence['pursuit_normal_combat_load_rejected']=str(error)
                else:raise AssertionError('normal pursuit -> combat loading accepted')
            except BaseException:
                runner.vector.close();raise
        print(runner.start_log_line(),flush=True)
        summary=runner.run()
        if pursuit:
            assert len(runner.completed_records)==1 and runner.completed_records[0]['episode_length']==1000
            assert summary['average_episode_length']==1000
        assert runner.trainer.sampled_steps==1200 and runner.trainer.ppo_update_count>0
        assert all(np.isfinite(v) for v in runner.last_metrics.values())
        checkpoint=output/'latest.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        identity={k:state['extra'][k] for k in ('environment_version','environment_variant','task_type','action_semantics')}
        assert identity['environment_version']=='2.4' and identity['action_semantics']=='direct_physical_control'
        old=dict(state);old['extra']=dict(state['extra']);old['extra'].update(environment_version='2.3',environment_variant='single_round_v2',action_semantics='delta_command')
        try:validate_checkpoint_for_evaluation(old,cfg,algo)
        except RuntimeError as error:rejection=str(error)
        else:raise AssertionError('old v2 accepted')
        evaluation=evaluate_mappo_checkpoint(checkpoint,algo,cfg,'cuda',[10000001])
        evidence[name]={'sampled_steps':1200,'identity':identity,'optimizer_updates':runner.trainer.ppo_update_count,
            'completed_episodes':len(runner.completed_records),'episode_indices':runner.vector.episode_indices.tolist(),
            'unique_reset_seeds':sorted(runner.vector.used_training_seeds),
            'worker_class':runner.vector.worker_environment_classes,
            'worker_policy':runner.vector.worker_fixed_policy_classes,
            'optimization_metrics':runner.last_metrics,'checkpoint_evaluation':evaluation,
            'old_v2_rejection':rejection,'initialization_verification':before_initialization,
            'actor_initialization':runner.actor_initialization}
        for file,value in (('env_config.yaml',cfg),('algorithm_config.yaml',algo)):
            (output/file).write_text(yaml.safe_dump(value,sort_keys=False),encoding='utf-8')
        (output/'run_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        if pursuit:source=checkpoint
        print(f'[SMOKE] {name}: 1200 CUDA transitions; strict checkpoint evaluation passed',flush=True)
        root.mkdir(parents=True,exist_ok=True)
        (root/'evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')

if __name__=='__main__':main()

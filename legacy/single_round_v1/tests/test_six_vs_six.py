"""Six-aircraft protocol and complete MAPPO path regressions."""
from pathlib import Path
from copy import deepcopy
import numpy as np
import pytest
import torch
import yaml
from env.factory import make_combat_environment
from env.observation import observation_dim_from_config
from algorithm.common.protocol import validate_single_round_environment_config
from algorithm.common.checkpoint import validate_checkpoint_for_evaluation
from algorithm.common.vector_env import ParallelVectorEnv
from algorithm.mappo.runner import MAPPOTrainingRunner
from algorithm.mappo.evaluation import evaluate_mappo_checkpoint
from tools.audit_single_round import baseline_content

ROOT=Path(__file__).resolve().parents[1]
def config(name): return yaml.safe_load((ROOT/'configs'/name).read_text(encoding='utf-8'))
def environment(): return config('single_round_6v6_environment.yaml')
def algorithm(): return config('mappo_6v6_3m.yaml')


def test_six_reset_observation_and_last_slot():
    cfg=environment(); validate_single_round_environment_config(cfg)
    env=make_combat_environment(cfg); obs,info=env.reset(17)
    assert len(env.red)==len(env.blue)==6 and obs.shape==(6,78)
    assert env.red_alive_mask.sum()==env.blue_alive_mask.sum()==6
    assert observation_dim_from_config(cfg['observation'])==78
    assert info['blue_ground_guard_override_mask'].shape==(6,)
    env.red[5].alive=False; env.blue[5].alive=False
    obs=env._observations()
    assert not obs[5].any() and not obs[0,72:78].any()
    with pytest.raises(ValueError,match='shape'):
        env.step(np.zeros((4,3)))


def test_six_last_enemy_and_mutual_termination(monkeypatch):
    env=make_combat_environment(environment()); env.reset(1)
    for state in env.blue: state.alive=False
    _,reward,terminated,truncated,info=env.step(np.zeros((6,3)))
    assert reward.shape==(6,) and terminated and not truncated and info['red_win']
    assert len(env.blue)==6 and info['blue_survivors']==0
    env.reset(2)
    for state in env.red+env.blue: state.alive=False
    assert env.step(np.zeros((6,3)))[4]['draw']


def test_six_vector_auto_reset_preserves_terminal_observation():
    cfg=environment(); cfg['simulation']['max_steps']=1
    with ParallelVectorEnv(2,cfg,base_seed=700) as vector:
        assert vector.reset().shape==(2,6,78)
        result=vector.step_batch(np.zeros((2,6,3)))
        assert result.truncated.all() and vector.current_alive_masks.shape==(2,6)
        assert vector.last_reset_seeds.tolist()==[702,703]
        assert not np.array_equal(result.observations,result.transition_next_observations)
        assert np.all(vector.current_alive_masks==1)


@pytest.mark.parametrize('section,value',[('team_size',4),('max_steps',3000)])
def test_six_protocol_mismatch_rejected(section,value):
    cfg=environment()
    cfg['scenario' if section=='team_size' else 'simulation'][section]=value
    with pytest.raises(RuntimeError): validate_single_round_environment_config(cfg)


def test_six_ground_guard_and_base_four_policy_math_unchanged():
    cfg=environment(); env=make_combat_environment(cfg); env.reset(11)
    env.blue[5].z=-1; env.blue[5].theta=-.4
    actions=env.fixed_policy.team_actions(env.blue,env.red)
    assert actions.shape==(6,3) and env.fixed_policy.last_override_mask[5]
    assert actions[5,1]>0
    env.reset(12); assert not env.fixed_policy.last_override_mask.any()
    assert env.fixed_policy.override_steps==0
    # Compare the full original policy implementation, not a mirrored formula.
    namespace={'__name__':'env._baseline_policy','__package__':'env'}
    exec(compile(baseline_content(ROOT,'env/fixed_policy.py'),'baseline','exec'),namespace)
    original=namespace['GroundAwareNearestTargetPursuitPolicy'](cfg['blue_policy'],cfg['action'],cfg['aircraft'])
    ours=type(env.fixed_policy)(cfg['blue_policy'],cfg['action'],cfg['aircraft'])
    assert np.array_equal(original.team_actions(env.blue[:4],env.red[:4]),
                          ours.team_actions(env.blue[:4],env.red[:4]))


def test_six_cuda_ppo_checkpoint_and_config_mismatch(tmp_path):
    runner=MAPPOTrainingRunner(environment(),algorithm(),num_envs=2,total_sampled_steps=8,
                               device='cuda',output_dir=tmp_path,smoke=True)
    try:
        rollout=runner.collect_rollout(4)
        assert rollout.observations.shape==(4,2,6,78)
        metrics=runner.trainer.update(rollout)
        assert all(np.isfinite(value) for value in metrics.values())
        path=tmp_path/'latest.pt'; runner.save_checkpoint(path)
        state=torch.load(path,map_location='cpu',weights_only=False)
        validate_checkpoint_for_evaluation(state,environment(),algorithm())
        with pytest.raises(RuntimeError):
            validate_checkpoint_for_evaluation(state,config('single_round_environment.yaml'),config('mappo.yaml'))
    finally: runner.vector.close()


def test_four_and_six_frozen_combat_parameters_match():
    four=config('single_round_environment.yaml'); six=environment()
    for section in ('simulation','action','aircraft','arena','weapon','reward','blue_policy'):
        assert four[section]==six[section]
    for key,value in four['scenario'].items():
        if key not in ('team_size','formation_offsets'): assert six['scenario'][key]==value
    a=config('mappo.yaml'); b=algorithm()
    for key,value in a['training'].items():
        if key!='total_sampled_steps': assert b['training'][key]==value
    assert b['network']['actor_hidden_layers']==a['network']['actor_hidden_layers']

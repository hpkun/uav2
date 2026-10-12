"""Analytical physical controls, independent pursuit, and explicit transfer contracts."""
from pathlib import Path
from copy import deepcopy
import numpy as np
import pytest
import torch
import yaml
from env.control import normalized_action_to_direct_control as decode
from env.dynamics import PointMassDynamics
from env.models import AircraftState, ControlCommand
from env.factory import make_combat_environment
from env.integrator import RK4Integrator
from env.config import aircraft_spec
from env.direct_policy import DirectPursuitPolicy
from algorithm.common.protocol import validate_single_round_environment_config
from algorithm.common.checkpoint import validate_checkpoint_for_resume, validate_checkpoint_for_evaluation
from algorithm.mappo.runner import MAPPOTrainingRunner
from algorithm.mappo.actor_initialization import initialize_actor, validate_actor_initialization
from tools.audit_single_round import direct_path_audit, compare_frozen_files
ROOT = Path(__file__).resolve().parents[1]

def config(pursuit=False):
    name = 'pursuit_environment.yaml' if pursuit else 'single_round_environment.yaml'
    return yaml.safe_load((ROOT/'configs'/name).read_text(encoding='utf-8'))

def algorithm():
    return yaml.safe_load((ROOT/'configs/mappo.yaml').read_text(encoding='utf-8'))

@pytest.mark.parametrize('action,physical', [([-1,0,0],[-2,0,0]), ([1,0,0],[2,0,0]),
    ([0,-1,0],[0,-5,0]),([0,1,0],[0,5,0]),([0,0,-1],[0,0,-np.pi/3]),
    ([0,0,1],[0,0,np.pi/3]),([0,0,0],[0,0,0]),([4,-9,2],[2,-5,np.pi/3]),
    ([0,.2,0],[0,1,0])])
def test_exact_decoder(action, physical):
    c = decode(action)
    np.testing.assert_allclose([c.nx,c.nz,c.phi],physical,rtol=0,atol=1e-15)

@pytest.mark.parametrize('action', [[0,0,float('nan')], [0,0,float('inf')], [0,0], [[0,0,0]]])
def test_invalid_decoder_input(action):
    with pytest.raises(ValueError):decode(action)

@pytest.mark.parametrize('T,L,beta', [(0,1,0),(0,5,0),(0,-5,0),
    (0,5,np.pi/3),(0,5,-np.pi/3),(2,0,0),(-2,0,0),(0,0,0)])
def test_analytical_derivatives(T,L,beta):
    state = AircraftState(0,0,-3000,225,0,0)
    derivative = PointMassDynamics().derivatives(state,ControlCommand(T,L,beta))
    expected = [9.81*T,9.81/225*(L*np.cos(beta)-1),9.81*L*np.sin(beta)/225]
    np.testing.assert_allclose(derivative[3:],expected,atol=1e-14,rtol=0)
    if (T,L,beta)==(0,1,0):assert np.array_equal(derivative[3:],np.zeros(3))
    if (T,L,beta)==(0,0,0):assert derivative[4]==pytest.approx(-9.81/225,abs=1e-15)

def test_red_and_blue_share_decoder_and_previous_roll_slot(monkeypatch):
    import env.combat_env as module
    env=make_combat_environment(config());env.reset(31)
    called=[]
    def traced(action):called.append(np.array(action));return decode(action)
    monkeypatch.setattr(module,'normalized_action_to_direct_control',traced)
    a=np.tile([.1,.2,.3],(4,1))
    observation,_,_,_,_=env.step(a,a)
    assert len(called)==8 and observation.shape==(4,52)
    assert env.red_last_executed_phi==pytest.approx(np.full(4,.3*np.pi/3))
    assert observation[:,4]==pytest.approx(np.full(4,.2))

def test_blue_nearest_alive_target_deterministic_and_legal():
    cfg=config();p=DirectPursuitPolicy(cfg['blue_policy'])
    own=AircraftState(0,0,-3000,225,0,0)
    targets=[AircraftState(0,10,-3000,225,0,0,False),AircraftState(1000,0,-3000,225,0,0)]
    a=p.action(own,targets)
    np.testing.assert_array_equal(a,p.action(own,targets))
    control=decode(a)
    assert control.phi==0 and control.nz==pytest.approx(1)
    assert control.nx==pytest.approx(25/(2*9.81),rel=1e-7)
    for s in [own, AircraftState(0,0,-3000,150,-np.pi/3,np.pi),AircraftState(0,0,-3000,300,np.pi/3,-np.pi)]:
        assert np.abs(p.action(s,targets)).max()<=1

def test_blue_recovery_explicit_four_seconds_prevents_simple_dive():
    cfg=config();p=DirectPursuitPolicy(cfg['blue_policy'])
    own=AircraftState(0,0,-450,225,-np.pi/6,0)
    risk,time=p.ground_risk(own,own.theta)
    assert risk and time==pytest.approx(4.0,abs=1e-14)
    a=p.action(own,[AircraftState(1000,0,-20,225,0,0)])
    assert a[1]==1 and a[2]==0
    dynamics=PointMassDynamics();integrator=RK4Integrator(.1);spec=aircraft_spec(cfg)
    minimum=own.altitude
    for _ in range(100):
        own=integrator.step(own,decode(p.action(own,[AircraftState(1000,0,-450,225,0,0)])),dynamics,spec)
        minimum=min(minimum,own.altitude)
    assert minimum>0
    assert 'pitch_time_constant' not in vars(p)

def test_pursuit_seed_reward_and_no_weapons():
    cfg=config(True);a=make_combat_environment(cfg);b=make_combat_environment(deepcopy(cfg))
    np.testing.assert_array_equal(a.reset(37)[0],b.reset(37)[0])
    assert not hasattr(a,'weapon') and not hasattr(a,'red_next_fire_step')
    first_waypoints=None
    for step in range(55):
        x=a.step(np.tile([0,.2,0],(4,1)));y=b.step(np.tile([0,.2,0],(4,1)))
        np.testing.assert_array_equal(x[0],y[0]);np.testing.assert_array_equal(x[1],y[1])
        info=x[4]
        np.testing.assert_allclose(x[1],-info['raw_nearest_distance_m']*.001,rtol=1e-7)
        assert not any('kill' in k or 'fire' in k or k.startswith('r1') for k in info)
        if step==0:first_waypoints=a.fixed_policy.waypoints.copy()
        if step<50:np.testing.assert_array_equal(a.fixed_policy.waypoints,first_waypoints)
        if step==50:assert not np.array_equal(a.fixed_policy.waypoints,first_waypoints)
    c=make_combat_environment(config(True));assert not np.array_equal(c.reset(38)[0],a._observations())
    assert info['mean_abs_L']==pytest.approx(1,rel=1e-7)
    assert info['fraction_abs_L_near_limit']==0

def test_pursuit_explicit_scale_and_exact_horizon(monkeypatch):
    cfg=config(True)
    env=make_combat_environment(cfg);env.reset(41)
    monkeypatch.setattr(env,'_advance',lambda *_:np.zeros(4))
    for step in range(1,1001):
        _,r,term,trunc,info=env.step(np.zeros((4,3)))
        assert not term and trunc==(step==1000)
        np.testing.assert_allclose(r,-info['raw_nearest_distance_m']*.001,rtol=1e-7)
    with pytest.raises(RuntimeError):env.step(np.zeros((4,3)))

@pytest.mark.parametrize('field,value', [('environment_version','2.3'),('environment_variant','single_round_v2'),
    ('task_type','combat'),('action_semantics','delta_command'),('observation_dim',53),('num_agents',6)])
def test_actor_init_rejects_incompatible_identity(field,value):
    from algorithm.mappo.trainer import MAPPO_IMPL_VERSION
    extra=dict(environment_version='2.4',environment_variant='pursuit_v2',task_type='pursuit',
        action_semantics='direct_physical_control',observation_dim=52,action_dim=3,num_agents=4)
    extra[field]=value
    with pytest.raises(RuntimeError):validate_actor_initialization(dict(algorithm='MAPPO',mappo_impl_version=MAPPO_IMPL_VERSION,extra=extra),config())

def test_actor_only_initialization_and_strict_task_resume(tmp_path):
    source=MAPPOTrainingRunner(config(True),algorithm(),num_envs=1,total_sampled_steps=4,
        device='cuda',seed=19,output_dir=tmp_path/'pursuit',smoke=True)
    target=MAPPOTrainingRunner(config(),algorithm(),num_envs=1,total_sampled_steps=4,
        device='cuda',seed=20,output_dir=tmp_path/'combat',smoke=True)
    try:
        source.trainer.sampled_steps=999
        path=tmp_path/'pursuit/source.pt';source.save_checkpoint(path)
        critic={k:v.clone() for k,v in target.trainer.critic.state_dict().items()}
        initialize_actor(target,path)
        assert all(torch.equal(v,target.trainer.actor.state_dict()[k]) for k,v in source.trainer.actor.state_dict().items())
        assert all(torch.equal(v,target.trainer.critic.state_dict()[k]) for k,v in critic.items())
        assert target.trainer.sampled_steps==0 and not target.evaluation_history
        assert not target.trainer.actor_optimizer.state and not target.trainer.critic_optimizer.state
        state=torch.load(path,map_location='cpu',weights_only=False)
        validate_checkpoint_for_resume(state,config(True),algorithm(),
            expected_training_seed=19,expected_num_envs=1,expected_smoke=True)
        validate_checkpoint_for_evaluation(state,config(True),algorithm())
        for field in ('task_type','action_semantics'):
            incomplete=deepcopy(state);del incomplete['extra'][field]
            with pytest.raises(RuntimeError,match=field):
                validate_checkpoint_for_evaluation(incomplete,config(True),algorithm())
        for validator in (validate_checkpoint_for_resume,validate_checkpoint_for_evaluation):
            with pytest.raises(RuntimeError,match='task_type'):validator(state,config(),algorithm())
        state['extra']['environment_version']='2.3'
        with pytest.raises(RuntimeError,match='environment_version'):validate_checkpoint_for_evaluation(state,config(),algorithm())
        state['extra']['environment_version']='2.4';state['actor']['mean.weight']=torch.zeros(3,17)
        with pytest.raises(RuntimeError,match='architecture'):validate_actor_initialization(state,config(),target.trainer.actor)
    finally:
        source.vector.close();target.vector.close()

def test_frozen_physics_and_direct_red_path():
    assert len(compare_frozen_files(ROOT))==16
    assert direct_path_audit(ROOT)['frozen_combat_methods_verified']==4

@pytest.mark.parametrize('variant', ['single_round_v1','single_round_v2'])
def test_old_configs_fail_closed(variant):
    cfg=config();cfg['environment_variant']=variant
    with pytest.raises(ValueError):make_combat_environment(cfg)
    with pytest.raises(RuntimeError):validate_single_round_environment_config(cfg)


def test_cli_rejects_resume_with_actor_init_before_run_allocation(tmp_path):
    import subprocess, sys
    output=tmp_path/'should_not_exist'
    result=subprocess.run([sys.executable,str(ROOT/'algorithm/train_mappo.py'),
        '--resume','missing.pt','--init-actor-checkpoint','missing.pt','--output-dir',str(output)],
        capture_output=True,text=True,timeout=30)
    assert result.returncode!=0 and 'cannot be combined' in result.stderr
    assert not output.exists()


def test_pursuit_diagnostics_never_modify_reward():
    cfg=config(True);different=deepcopy(cfg);different['diagnostics']['near_limit_fraction']=.1
    a=make_combat_environment(cfg);b=make_combat_environment(different)
    a.reset(91);b.reset(91)
    actions=np.tile([.2,.2,.2],(4,1))
    x=a.step(actions);y=b.step(actions)
    np.testing.assert_array_equal(x[1],y[1])
    assert x[4]['fraction_abs_T_near_limit']==0 and y[4]['fraction_abs_T_near_limit']==1

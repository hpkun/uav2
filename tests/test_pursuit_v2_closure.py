"""Fixed-horizon reward-hacking regressions and waypoint/guard identities."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from env.models import AircraftState
from env.control import normalized_action_to_direct_control
from env.pursuit_env import PursuitEnv,validate_pursuit_config
from env.direct_policy import DirectPursuitPolicy
from env.factory import make_combat_environment
from algorithm.mappo.runner import MAPPOTrainingRunner
from algorithm.mappo.actor_initialization import validate_actor_initialization
from algorithm.common.checkpoint import validate_checkpoint_for_resume,validate_checkpoint_for_evaluation
from algorithm.common.protocol import config_sha256,validate_single_round_environment_config
from algorithm.mappo.trainer import MAPPO_IMPL_VERSION
ROOT=Path(__file__).resolve().parents[1]

def config(combat=False):
    return yaml.safe_load((ROOT/'configs'/('single_round_environment.yaml' if combat else 'pursuit_environment.yaml')).read_text())

def trim():return np.tile([0,.2,0],(4,1))

def test_ground_and_boundary_crossing_keep_all_masks_and_negative_reward_for_full_horizon():
    env=PursuitEnv(config());env.reset(103)
    env.red[0]=AircraftState(0,0,-.1,225,-.5,0)
    env.red[1]=AircraftState(4999,0,-2500,225,0,0)
    for step in range(1,1001):
        _,reward,terminated,truncated,info=env.step(trim())
        assert not terminated and truncated==(step==1000)
        np.testing.assert_array_equal(env.red_alive_mask,np.ones(4))
        np.testing.assert_array_equal(env.blue_alive_mask,np.ones(4))
        np.testing.assert_array_equal(info['nearest_distance_valid_mask'],np.ones(4,dtype=bool))
        assert np.isfinite(info['raw_nearest_distance_m']).all() and (info['raw_nearest_distance_m']>=0).all()
        assert (reward<0).all()
        np.testing.assert_allclose(reward,-info['raw_nearest_distance_m']*.001,rtol=1e-7)
        if step==1:
            assert env.red[0].altitude<0 and np.hypot(env.red[1].x,env.red[1].y)>5000
            assert info['red_ground_violation_steps']==info['red_boundary_violation_steps']==1
    assert info['red_ground_violation_steps']>=1000
    assert info['red_boundary_violation_steps']>=1000
    assert info['red_ground_violation_agents']>=1 and info['red_boundary_violation_agents']>=1
    assert info['red_ground_violation_rate']==info['red_ground_violation_steps']/4000
    assert not any('loss_count' in key for key in info)
    with pytest.raises(RuntimeError,match='reset'):env.step(trim())

def test_opponent_violations_are_only_diagnostics(monkeypatch):
    env=PursuitEnv(config());env.reset(15)
    env.blue[0]=AircraftState(5001,0,.1,225,-.5,0)
    monkeypatch.setattr(env.fixed_policy,'team_actions',lambda *_:trim())
    for step in range(1,5):
        _,reward,terminated,truncated,info=env.step(trim())
        assert not terminated and not truncated and (reward<0).all()
        assert info['opponent_ground_violation_steps']==step
        assert info['opponent_boundary_violation_steps']==step
        assert info['opponent_ground_violation_agents']==info['opponent_boundary_violation_agents']==1
        assert env.blue_alive_mask.sum()==4

def test_two_km_is_exactly_minus_two_and_diagnostics_do_not_change_reward(monkeypatch):
    env=PursuitEnv(config());env.reset(5)
    env.red=[AircraftState(0,0,-3000,225,0,0) for _ in range(4)]
    env.blue=[AircraftState(2000,0,-3000,225,0,0) for _ in range(4)]
    monkeypatch.setattr(env,'_advance',lambda *_:np.zeros(4))
    _,reward,_,_,info=env.step(trim())
    np.testing.assert_array_equal(reward,np.full(4,-2.))
    np.testing.assert_array_equal(info['raw_nearest_distance_m'],np.full(4,2000.))
    np.testing.assert_array_equal(info['pursuit_reward'],np.full(4,-2.))

@pytest.mark.parametrize('scale',[1.,.01,0.,-.001,.00100001,None])
def test_formal_scale_is_exactly_point_zero_zero_one(scale):
    cfg=config();cfg['reward']['distance_reward_scale']=scale
    with pytest.raises(RuntimeError,match='0.001'):validate_pursuit_config(cfg)

def test_pursuit_best_uses_mean_then_final_distance_not_return(tmp_path):
    runner=MAPPOTrainingRunner.__new__(MAPPOTrainingRunner)
    runner.env_config=config();runner.output_dir=tmp_path;runner.best_evaluation=None
    saves=[];runner.save_checkpoint=lambda path:saves.append(path)
    a=dict(average_return=-1.,mean_nearest_distance=3000.,final_nearest_distance=1000.,minimum_nearest_distance=0.)
    b=dict(average_return=-10.,mean_nearest_distance=2000.,final_nearest_distance=1500.,minimum_nearest_distance=100.)
    assert runner._consider_best_evaluation(a)
    assert runner._consider_best_evaluation(b)
    assert not runner._consider_best_evaluation(a)
    c={**b,'average_return':-100.,'final_nearest_distance':1400.}
    assert runner._consider_best_evaluation(c)
    assert runner.best_evaluation==c and len(saves)==3
    runner.env_config=config(True)
    assert runner._evaluation_key(dict(win_rate=.5,average_return=5.,average_red_loss=1.))==(.5,5.,-1.)

def test_complete_waypoint_and_trajectory_seed_reproducibility():
    envs=[PursuitEnv(config()) for _ in range(3)]
    states=[env.reset(seed)[0] for env,seed in zip(envs,[104,104,105])]
    np.testing.assert_array_equal(states[0],states[1]);assert not np.array_equal(states[0],states[2])
    previous=None;different=False
    for step in range(1000):
        for env in envs:env.step(trim())
        a,b,c=envs
        np.testing.assert_array_equal(a.fixed_policy.waypoints,b.fixed_policy.waypoints)
        np.testing.assert_array_equal(a.fixed_policy.desired_speeds,b.fixed_policy.desired_speeds)
        np.testing.assert_array_equal(np.stack([s.as_array() for s in a.blue]),np.stack([s.as_array() for s in b.blue]))
        different |= not np.array_equal(a.fixed_policy.waypoints,c.fixed_policy.waypoints)
        waypoints=a.fixed_policy.waypoints
        assert np.hypot(waypoints[:,0],waypoints[:,1]).max()<=2500
        assert (waypoints[:,2]>=2200).all() and (waypoints[:,2]<=3800).all()
        assert (a.fixed_policy.desired_speeds>=200).all() and (a.fixed_policy.desired_speeds<=250).all()
        if step%50==0:
            if previous is not None:assert not np.array_equal(waypoints,previous)
            previous=waypoints.copy()
        else:np.testing.assert_array_equal(waypoints,previous)
    assert different and envs[0].fixed_policy.update_count==20

def test_opponent_movement_is_rk4_integration_without_teleport():
    env=PursuitEnv(config());env.reset(41)
    original=[s.copy() for s in env.blue]
    env.step(trim())
    expected=[env.integrator.step(s,normalized_action_to_direct_control(a),env.dynamics,env.spec)
              for s,a in zip(original,env.opponent_actions)]
    np.testing.assert_array_equal(np.stack([s.as_array() for s in env.blue]),np.stack([s.as_array() for s in expected]))

@pytest.mark.parametrize('theta,altitude,target_altitude,target_x,expected',[
    (-.5,100.,200.,1000.,True), # current descent despite upward command
    (0.,100.,10.,100.,True),  # anticipation before any actual descent
    (0.,3000.,2900.,10000.,False), # safe downward command
    (0.,100.,200.,1000.,False), # climb cannot add downward risk
])
def test_blue_current_and_commanded_descent_guard(theta,altitude,target_altitude,target_x,expected):
    p=DirectPursuitPolicy(config(True)['blue_policy'])
    own=AircraftState(0,0,-altitude,225,theta,0)
    target=AircraftState(target_x,0,-target_altitude,225,0,0)
    pitch=np.arctan2(target_altitude-altitude,target_x)
    risk,_=p.ground_risk(own,pitch)
    action,override=p._action_and_override(own,[target])
    assert risk==override==expected
    if expected:assert action[1]==1. and action[2]==0.

@pytest.mark.parametrize('value',[False,None,1])
def test_combat_requires_true_commanded_pitch_identity(value):
    cfg=config(True);cfg['blue_policy']['ground_avoidance']['consider_commanded_pitch']=value
    with pytest.raises(RuntimeError,match='consider_commanded_pitch'):validate_single_round_environment_config(cfg)
    with pytest.raises(ValueError,match='consider_commanded_pitch'):make_combat_environment(cfg)

def state_for(cfg):
    return {'algorithm':'MAPPO','mappo_impl_version':MAPPO_IMPL_VERSION,'actor':{'dummy':torch.zeros(1)},
        'extra':{'environment_version':'2.4','environment_variant':cfg['environment_variant'],
        'task_type':cfg['task_type'],'action_semantics':'direct_physical_control','observation_dim':52,'action_dim':3,'num_agents':4}}

def test_pursuit_v1_rejected_from_resume_evaluation_and_actor_initialization():
    cfg=config();old=state_for(cfg);old['extra']['environment_variant']='pursuit_v1'
    algo=yaml.safe_load((ROOT/'configs/mappo.yaml').read_text())
    for check in (validate_checkpoint_for_resume,validate_checkpoint_for_evaluation):
        with pytest.raises(RuntimeError,match='environment_variant'):check(old,cfg,algo)
    with pytest.raises(RuntimeError,match='environment_variant'):validate_actor_initialization(old,config(True))
    old_config=deepcopy(cfg);old_config['environment_variant']='pursuit_v1'
    with pytest.raises(RuntimeError,match='pursuit_v2'):validate_pursuit_config(old_config)
    with pytest.raises(ValueError,match='environment_variant'):make_combat_environment(old_config)

def test_old_combat_v3_fingerprint_rejected_from_resume_and_evaluation():
    cfg=config(True);old_config=deepcopy(cfg)
    del old_config['blue_policy']['ground_avoidance']['consider_commanded_pitch']
    state=state_for(cfg);state['extra']['environment_config_sha256']=config_sha256(old_config)
    algo=yaml.safe_load((ROOT/'configs/mappo.yaml').read_text())
    assert config_sha256(old_config)!=config_sha256(cfg)
    for check in (validate_checkpoint_for_resume,validate_checkpoint_for_evaluation):
        with pytest.raises(RuntimeError,match='environment_config_sha256'):check(state,cfg,algo)

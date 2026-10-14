"""Version-isolated Red overload controls with frozen Blue and battle mechanics."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import ast,subprocess
import numpy as np
import pytest
import yaml
from env.factory import make_combat_environment
from env.combat_env import MultiUAVCombatEnv
from env.combat_v332 import CombatEnvironmentV332
from env.config import load_config,validate_config,environment_dimensions
from env.v332_config import RED_DIRECT_CONTROL
from env.v332_control import (normalized_action_to_target_v332 as decode,rate_limit_control_v332 as limit,
    executed_control_to_command_v332 as command,normalized_control_features_v332 as features)
from env.math_utils import wrap_angle
from env.models import AircraftState,ControlCommand
from env.dynamics import PointMassDynamics
from env.v329_control import action_to_control_v329
from algorithm.common.vector_env import ParallelVectorEnv
from algorithm.common.checkpoint import validate_checkpoint_for_resume,validate_checkpoint_for_evaluation
ROOT=Path(__file__).resolve().parents[1]
BASELINE='9f6dc576367a071c2b784ee1ada37ea822000e11'

def cfg(version='332'):return load_config(ROOT/f'configs/combat_environment_v{version}.yaml')
def env(version='332'):return make_combat_environment(cfg(version))

@pytest.mark.parametrize('action,target',[([0,0,0],[0,0,1]),([1,1,1],[np.pi,2,8]),
    ([-1,-1,-1],[-np.pi,-1,0]),([.5,-.4,-.5],[np.pi/2,-.4,.5]),([-.5,.4,.5],[-np.pi/2,.8,4.5]),
    ([3,-9,4],[np.pi,-1,8])])
def test_decoder_exact_order_and_trim(action,target):np.testing.assert_array_equal(decode(action),target)

@pytest.mark.parametrize('action',[[np.nan,0,0],[0,np.inf,0],[0,0],[[0,0,0]]])
def test_decoder_rejects_invalid(action):
    with pytest.raises(ValueError):decode(action)

@pytest.mark.parametrize('sign',[1,-1])
def test_first_rate_limited_step(sign):
    executed,flags=limit([0,0,1],decode(np.full(3,sign)),.1,RED_DIRECT_CONTROL)
    np.testing.assert_allclose(executed,[sign*np.pi/20,sign*.2,1+sign*.2],rtol=0,atol=1e-15)
    assert flags.all()

@pytest.mark.parametrize('sign',[1,-1])
def test_roll_shortest_path_across_seam(sign):
    previous=[sign*np.deg2rad(179),0,1];target=[-sign*np.deg2rad(179),0,1]
    executed,flags=limit(previous,target,.1,RED_DIRECT_CONTROL)
    assert wrap_angle(executed[0]-previous[0])==pytest.approx(sign*np.deg2rad(2),abs=1e-14)
    assert executed[0]==pytest.approx(target[0],abs=1e-14) and not flags.any()

def test_many_steps_respect_ranges_and_circular_rates():
    rng=np.random.default_rng(71);previous=np.array([0.,0.,1.])
    for _ in range(300):
        current,_=limit(previous,decode(rng.uniform(-1,1,3)),.1,RED_DIRECT_CONTROL)
        assert abs(wrap_angle(current[0]-previous[0]))<=np.pi/20+1e-14
        assert np.abs(current[1:]-previous[1:]).max()<=.2+1e-14
        assert -np.pi<=current[0]<=np.pi and -1<=current[1]<=2 and 0<=current[2]<=8
        previous=current

@pytest.mark.parametrize('control',[[0,0,1],[.7,1.3,4.2],[-2.1,-.4,7.5]])
def test_direct_command_and_analytical_dynamics_no_gravity_double_scaling(control):
    mu,xi_hat,xi_bar=control;c=command(control)
    assert c==ControlCommand(nx=xi_hat,nz=xi_bar,phi=mu)
    theta=.0 if control==[0,0,1] else .23
    s=AircraftState(0,0,-3000,225,theta,.4)
    g=9.81
    expected=[g*(xi_hat-np.sin(theta)),g/225*(xi_bar*np.cos(mu)-np.cos(theta)),g*xi_bar*np.sin(mu)/(225*np.cos(theta))]
    np.testing.assert_allclose(PointMassDynamics().derivatives(s,c)[3:],expected,rtol=0,atol=1e-14)
    if control==[0,0,1]:np.testing.assert_array_equal(expected,[0.,0.,0.])

@pytest.mark.parametrize('key',list(RED_DIRECT_CONTROL))
def test_red_config_missing_unknown_drift_rejected(key):
    c=cfg();del c['red_direct_control'][key]
    with pytest.raises(ValueError,match='red_direct_control'):validate_config(c)
    c=cfg();c['red_direct_control']['unknown']=1
    with pytest.raises(ValueError,match='unknown'):validate_config(c)
    c=cfg();c['red_direct_control'][key]='wrong' if key=='semantics' else RED_DIRECT_CONTROL[key]+.1
    with pytest.raises(ValueError,match=key):validate_config(c)

@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf'),True,'2.0',None])
def test_red_config_rejects_nonfinite_nonnumeric(value):
    c=cfg();c['red_direct_control']['normal_load_rate_max']=value
    with pytest.raises(ValueError):validate_config(c)

def test_config_and_constructor_isolation():
    c=cfg();baseline=deepcopy(c);baseline.pop('red_direct_control');baseline['environment_version']='3.31'
    assert baseline==cfg('331') and environment_dimensions(c)==(47,3,4)
    assert isinstance(env(),CombatEnvironmentV332)
    assert type(MultiUAVCombatEnv(ROOT/'configs/combat_environment_v332.yaml')) is CombatEnvironmentV332
    with pytest.raises(ValueError):CombatEnvironmentV332(cfg('331'))
    with pytest.raises(ValueError):validate_config({**cfg('331'),'red_direct_control':RED_DIRECT_CONTROL.copy()})
    assert cfg()['action']['controller']['normal_load_max']==10 and cfg()['red_direct_control']['normal_load_max']==8
    old=yaml.safe_load((ROOT/'configs/mappo_4v4_v331.yaml').read_text())
    new=yaml.safe_load((ROOT/'configs/mappo_4v4_v332.yaml').read_text());new['network']['observation_dim']=45
    assert new==old

def test_blue_low_level_exactly_matches_331_and_red_bypasses_controller(monkeypatch):
    import env.combat_v329 as old_module
    a,b=env('331'),env();a.reset(8);b.reset(8)
    actions=np.array([[1,1,1],[-1,-1,-1],[.2,-.3,.4],[0,0,0]],dtype=float)
    original=[s.copy() for s in a.blue]
    targets=[action_to_control_v329(s,x,a.config['action'],a.spec,a.dynamics.gravity) for s,x in zip(original,actions)]
    captured=[]
    def traced(s,x,*args):
        c=action_to_control_v329(s,x,*args);captured.append(c);return c
    monkeypatch.setattr(old_module,'action_to_control_v329',traced)
    a._advance(a.blue,actions);b._advance(b.blue,actions)
    assert captured[:4]==captured[4:]==targets
    np.testing.assert_array_equal(np.stack([s.as_array() for s in a.blue]),np.stack([s.as_array() for s in b.blue]))
    captured.clear();b._advance_red_direct(b.red,actions)
    assert not captured and '_advance' not in CombatEnvironmentV332.__dict__
    old_red=[s.copy() for s in a.red];a._advance(a.red,actions)
    assert len(captured)==4
    assert not np.array_equal(np.stack([s.as_array() for s in a.red]),np.stack([s.as_array() for s in b.red]))

def test_full_transition_invokes_old_controller_only_for_blue(monkeypatch):
    import env.combat_v329 as module
    e=env();e.reset(15);calls=[];original=module.action_to_control_v329
    def traced(*args):calls.append(args[0]);return original(*args)
    monkeypatch.setattr(module,'action_to_control_v329',traced)
    before_blue=list(e.blue)
    obs,r,_,_,info=e.step(np.ones((4,3)),np.zeros((4,3)))
    assert len(calls)==4 and all(s is t for s,t in zip(calls,before_blue))
    assert obs.shape==(4,47) and r.shape==(4,)
    assert info['mean_abs_mu']==pytest.approx(np.pi/20)
    assert info['roll_rate_limit_fraction']==1

@pytest.mark.parametrize('physical,normalized',[
    ([0,0,1],[0,0,0]),([np.pi,2,8],[1,1,1]),([-np.pi,-1,0],[-1,-1,-1]),([.5,-.2,.5],[.5/np.pi,-.2,-.5])])
def test_control_observation_inverse(physical,normalized):np.testing.assert_allclose(features(physical),normalized,rtol=0,atol=1e-15)

def test_observation_shift_geometry_dead_slots_and_reset():
    a,b=env('331'),env();old,_=a.reset(17);new,_=b.reset(17)
    np.testing.assert_array_equal(old[:,:4],new[:,:4]);np.testing.assert_array_equal(old[:,5:7],new[:,7:9])
    np.testing.assert_array_equal(old[:,7:],new[:,9:]);assert not new[:,4:7].any()
    b.red_control_states[0]=[np.pi,2,8];np.testing.assert_array_equal(b._observations()[0,4:7],[1,1,1])
    b.red[1].alive=False;b.blue[2].alive=False
    obs=b._observations();assert not obs[1].any() and not obs[0,9:15].any() and not obs[:,37:42].any()
    before=b.red_control_states[1].copy();state=b.red[1].as_array().copy()
    b._advance_red_direct(b.red,np.ones((4,3)))
    np.testing.assert_array_equal(b.red_control_states[1],before);np.testing.assert_array_equal(b.red[1].as_array(),state)
    b.reset(17);np.testing.assert_array_equal(b.red_control_states,np.tile([0,0,1],(4,1)))

def test_instances_and_real_vector_control_states_do_not_leak():
    a,b=env(),env();a.reset(10);b.reset(10);a._advance_red_direct(a.red,np.ones((4,3)))
    assert not np.shares_memory(a.red_control_states,b.red_control_states)
    np.testing.assert_array_equal(b.red_control_states,np.tile([0,0,1],(4,1)))
    with ParallelVectorEnv(2,cfg(),11) as vector:
        obs=vector.reset();assert obs.shape==(2,4,47)
        actions=np.zeros((2,4,3));actions[0]=1
        result=vector.step_batch(actions)
        np.testing.assert_allclose(result.observations[0,:,4:7],np.tile([.05,.1,.2/7],(4,1)),atol=1e-8,rtol=0)
        np.testing.assert_array_equal(result.observations[1,:,4:7],np.zeros((4,3)))
        vector.episode_indices+=1 # Fresh episode seeds under the frozen vector protocol.
        obs=vector.reset();assert not obs[:,:,4:7].any()

def test_transition_core_frozen_except_red_advance_and_old_sources_untouched():
    old=ast.parse((ROOT/'env/v330_transition.py').read_text());new=ast.parse((ROOT/'env/v332_transition.py').read_text())
    a=next(n for n in old.body if isinstance(n,ast.FunctionDef));b=next(n for n in new.body if isinstance(n,ast.FunctionDef))
    count=0
    for n in ast.walk(b):
        if isinstance(n,ast.Attribute) and n.attr=='_advance_red_direct':n.attr='_advance';count+=1
    assert count==1 and ast.dump(a)==ast.dump(b)
    for name in ('env/dynamics.py','env/integrator.py','env/v329_control.py','env/v310_policy.py',
                 'env/combat_v331.py','env/combat_v330.py','env/v330_transition.py','env/v330_observation.py',
                 'configs/combat_environment_v331.yaml','configs/mappo_4v4_v331.yaml',
                 'algorithm/mappo/trainer.py','algorithm/mappo/networks.py'):
        assert (ROOT/name).read_bytes().replace(b'\r\n',b'\n')==subprocess.check_output(['git','show',f'{BASELINE}:{name}'],cwd=ROOT)

def test_old_complete_checkpoint_rejected_by_version():
    from algorithm.mappo.trainer import MAPPO_IMPL_VERSION
    algorithm=yaml.safe_load((ROOT/'configs/mappo_4v4_v332.yaml').read_text())
    state=dict(algorithm='MAPPO',mappo_impl_version=MAPPO_IMPL_VERSION,extra={'environment_version':'3.31','observation_dim':45,'action_dim':3,'num_agents':4})
    for validate in (validate_checkpoint_for_resume,validate_checkpoint_for_evaluation):
        with pytest.raises(RuntimeError,match='environment_version'):validate(state,cfg(),algorithm)


def test_action_metric_labels_v332_are_physical_and_preserve_values():
    from env.v332_control import label_action_metrics_v332
    metrics=dict(policy_log_std_heading=-.5,policy_sigma_pitch=.6,
        policy_log_std_mean_v=-.4,action_abs_gt_0_9_fraction_psi=.1,blue_mean_abs_high_level_action_psi=.2)
    renamed=label_action_metrics_v332(metrics)
    assert renamed==dict(policy_log_std_a_mu=-.5,policy_sigma_a_t=.6,
        policy_log_std_mean_a_n=-.4,action_abs_gt_0_9_fraction_a_mu=.1,blue_mean_abs_high_level_action_psi=.2)

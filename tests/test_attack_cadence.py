"""Exact step cadence, per-aircraft independence, symmetry and frozen semantics."""
from pathlib import Path
from copy import deepcopy
import numpy as np
import pytest
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.factory import make_combat_environment
from env.models import AircraftState
from env.config import attack_cooldown_steps
from env.reward import paper_state_reward_components
from algorithm.common.protocol import validate_single_round_environment_config
from algorithm.common.checkpoint import validate_checkpoint_for_evaluation,validate_checkpoint_for_resume
from algorithm.common.evaluator import evaluate
from algorithm.mappo.runner import MAPPOTrainingRunner
from algorithm.madsac.protocol import validate_madsac_config
from algorithm.madsac.evaluation import evaluate_madsac

ROOT=Path(__file__).resolve().parents[1]
def cfg(): return yaml.safe_load((ROOT/'configs/single_round_environment.yaml').read_text())
def state(x=0,psi=0,alive=True): return AircraftState(x,0,-3000,225,0,psi,alive)
def team(first): return [first]+[state(alive=False) for _ in range(3)]
def setup(monkeypatch,hit=False,head_on=False):
 env=MultiUAVCombatEnv(cfg());env.reset(7)
 env.red=team(state());env.blue=team(state(1000,np.pi if head_on else 0))
 monkeypatch.setattr(env,'_advance',lambda *_:np.zeros(4,np.float32))
 monkeypatch.setattr(type(env.weapon),'attempt_hit',lambda *_:hit)
 return env
def step(env): return env.step(np.zeros((4,3)),np.zeros((4,3)))


def test_continuous_window_exact_25_step_cadence(monkeypatch):
 env=setup(monkeypatch);fired=[]
 for number in range(1,26):
  obs,_,terminated,truncated,info=step(env)
  assert obs.shape==(4,52) and not terminated and not truncated
  if info['red_step_fire_attempts']: fired.append(number)
  assert info['red_step_cooldown_blocked']==(0 if number in (1,11,21) else 1)
  if number==10: assert env.red_next_fire_step[0]==11 and not info['red_step_fire_attempts']
 assert fired==[1,11,21] and info['red_fire_attempts']==3
 assert info['red_cooldown_blocked_agent_steps']==22
 assert env.cooldown_steps==10 and env.red_next_fire_step[0]==31


def test_exit_reentry_does_not_rearm_early(monkeypatch):
 env=setup(monkeypatch);assert step(env)[4]['red_step_fire_attempts']==1
 for number in range(2,12):
  env.blue[0].x=5000 if number<=5 else 1000
  # Keep the deliberately ineligible target alive inside this unit fixture.
  monkeypatch.setattr(env,'_resolve_noncombat_losses',lambda:([],[],[],[]))
  info=step(env)[4]
  assert info['red_step_fire_attempts']==(1 if number==11 else 0)
  if number<=5: assert info['red_step_cooldown_blocked']==0


def test_switch_target_cannot_bypass_cooldown(monkeypatch):
 env=setup(monkeypatch);env.blue[1]=state(1200)
 assert step(env)[4]['red_step_fire_attempts']==1
 env.blue[0].alive=False
 for number in range(2,12):
  assert step(env)[4]['red_step_fire_attempts']==(1 if number==11 else 0)


@pytest.mark.parametrize('hit',[False,True])
def test_hit_and_miss_both_start_cooldown(monkeypatch,hit):
 env=setup(monkeypatch,hit)
 info=step(env)[4]
 assert env.red_next_fire_step[0]==11
 assert info['red_step_fire_attempts']==1 and info['red_step_weapon_hits']==int(hit)
 if hit: assert info['red_win'] and info['blue_survivors']==0


def test_reset_clears_cooldown_and_first_encounter_is_immediate(monkeypatch):
 env=setup(monkeypatch);step(env);env.reset(10)
 assert env.red_next_fire_step.dtype==np.int64 and not env.red_next_fire_step.any()
 assert not env.blue_next_fire_step.any()
 env.red=team(state());env.blue=team(state(1000))
 assert step(env)[4]['red_step_fire_attempts']==1


def test_ready_without_target_stays_ready(monkeypatch):
 env=setup(monkeypatch)
 monkeypatch.setattr(env,'_resolve_noncombat_losses',lambda:([],[],[],[]))
 env.blue[0].x=5000
 for _ in range(15): assert step(env)[4]['red_step_fire_attempts']==0
 assert not env.red_next_fire_step.any()
 env.blue[0].x=1000
 assert step(env)[4]['red_step_fire_attempts']==1 and env.red_next_fire_step[0]==26


def test_expired_cooldown_without_target_waits_without_postponing_clock(monkeypatch):
 env=setup(monkeypatch);step(env)
 monkeypatch.setattr(env,'_resolve_noncombat_losses',lambda:([],[],[],[]))
 env.blue[0].x=5000
 for _ in range(14): assert step(env)[4]['red_step_fire_attempts']==0
 assert env.red_next_fire_step[0]==11
 env.blue[0].x=1000
 assert step(env)[4]['red_step_fire_attempts']==1 and env.red_next_fire_step[0]==26


def test_nearest_eligible_target_selection_preserved(monkeypatch):
 env=setup(monkeypatch);env.blue[0].x=1200;env.blue[1]=state(800)
 env.steps=1
 assert env._attack_attempts(env.red,env.blue,env.red_next_fire_step,'red')==[(0,1,False)]


def test_dead_aircraft_never_attempt_and_aircraft_clocks_independent(monkeypatch):
 env=setup(monkeypatch);env.red[1]=state(0,alive=True)
 env.red_next_fire_step[0]=11
 info=step(env)[4]
 assert info['red_step_fire_attempts']==1
 assert env.red_next_fire_step.tolist()==[11,11,0,0]


def test_red_blue_symmetry_in_continuous_window(monkeypatch):
 env=setup(monkeypatch,head_on=True);red=[];blue=[]
 for number in range(1,26):
  info=step(env)[4]
  if info['red_step_fire_attempts']:red.append(number)
  if info['blue_step_fire_attempts']:blue.append(number)
 assert red==blue==[1,11,21]
 assert np.array_equal(env.red_next_fire_step,env.blue_next_fire_step)


def test_simultaneous_hits_preserve_both_attempts_and_rewards(monkeypatch):
 env=setup(monkeypatch,hit=True,head_on=True)
 _,reward,terminated,truncated,info=step(env)
 assert terminated and not truncated and info['draw']
 assert info['red_step_fire_attempts']==info['blue_step_fire_attempts']==1
 assert info['red_step_weapon_hits']==info['blue_step_weapon_hits']==1
 assert info['red_attack_kills']==info['blue_attack_kills']==1
 assert env.red_next_fire_step[0]==env.blue_next_fire_step[0]==11
 assert reward[0]==0  # +10 kill and -10 own loss; no cadence reward added.


@pytest.mark.parametrize('seconds',[0,-1,.95,2,float('nan')])
def test_invalid_cooldown_rejected(seconds):
 config=cfg();config['attack']['cooldown_seconds']=seconds
 with pytest.raises(ValueError):MultiUAVCombatEnv(config)
 with pytest.raises(RuntimeError):validate_single_round_environment_config(config)


@pytest.mark.parametrize('cadence',['entry_triggered','invalid',None])
def test_invalid_cadence_rejected(cadence):
 config=cfg();config['attack']['cadence']=cadence
 with pytest.raises(ValueError):MultiUAVCombatEnv(config)
 with pytest.raises(RuntimeError):validate_single_round_environment_config(config)


def test_non_integer_cooldown_mapping_rejected():
 config=cfg();config['simulation']['dt']=.3
 with pytest.raises(ValueError,match='integer'):attack_cooldown_steps(config)


def test_old_variant_rejected_by_every_formal_entry(tmp_path):
 config=cfg();config['environment_variant']='single_round_v1'
 algo=yaml.safe_load((ROOT/'configs/mappo.yaml').read_text())
 madsac=yaml.safe_load((ROOT/'configs/madsac_single_round_3m.yaml').read_text())
 with pytest.raises(ValueError):make_combat_environment(config)
 for action in (
  lambda:MAPPOTrainingRunner(config,algo,output_dir=tmp_path),
  lambda:validate_madsac_config(config,madsac),
  lambda:evaluate(None,config,[10000000]),
  lambda:evaluate_madsac(None,config,[44000000]),
 ):
  with pytest.raises(RuntimeError,match='environment_variant'):action()


def test_cooldown_state_not_in_observation_and_reward_math_unchanged(monkeypatch):
 env=setup(monkeypatch);before=env._observations().copy()
 env.red_next_fire_step[:]=999;env.blue_next_fire_step[:]=333
 assert np.array_equal(before,env._observations()) and before.shape==(4,52)
 assert not hasattr(env,'red_fire_states') and not hasattr(env,'_entry_attempts')
 old=yaml.safe_load((ROOT/'legacy/single_round_v1/configs/single_round_environment.yaml').read_text())
 assert cfg()['reward']==old['reward'] and cfg()['weapon']==old['weapon']
 namespace={'__name__':'env._baseline_reward','__package__':'env'}
 # The immutable original reward source is byte-identical, independently checked by the audit.
 from tools.audit_single_round import baseline_content
 exec(compile(baseline_content(ROOT,'env/reward.py'),'baseline_reward','exec'),namespace)
 for distance in (1000,3000,5000):
  red=team(state());blue=team(state(distance))
  current=paper_state_reward_components(red,blue,cfg()['reward'])
  original=namespace['paper_state_reward_components'](red,blue,old['reward'])
  for key in ('r3','r4'):assert np.array_equal(current[key],original[key])

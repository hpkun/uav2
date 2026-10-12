"""25 real simulation steps in a continuous fire window with deterministic misses."""
from pathlib import Path
import json,sys
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from env.combat_env import MultiUAVCombatEnv
from env.models import AircraftState

def main():
 env=MultiUAVCombatEnv();env.reset(123)
 def state(x,alive=True):return AircraftState(x,0,-3000,225,0,0,alive)
 env.red=[state(0)]+[state(0,False) for _ in range(3)]
 env.blue=[state(1000)]+[state(1000,False) for _ in range(3)]
 attempts=[]
 with patch.object(type(env.weapon),'attempt_hit',return_value=False):
  for number in range(1,26):
   _,_,terminated,truncated,info=env.step(np.zeros((4,3)),np.zeros((4,3)))
   if terminated or truncated:raise RuntimeError('cadence fixture unexpectedly ended')
   if info['red_step_fire_attempts']:attempts.append(number)
 if attempts!=[1,11,21]:raise RuntimeError(f'incorrect cadence: {attempts}')
 result=dict(variant=env.environment_variant,dt=env.dt,cooldown_seconds=1.0,cooldown_steps=env.cooldown_steps,
             attempt_steps=attempts,red_attempts=info['red_fire_attempts'],
             cooldown_blocked_agent_steps=info['red_cooldown_blocked_agent_steps'],observation_shape=list(env._observations().shape))
 output=ROOT/'outputs/cadence_v3_validation';output.mkdir(parents=True,exist_ok=True)
 (output/'cadence.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2))

if __name__=='__main__':main()

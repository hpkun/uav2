"""Full deterministic 100-seed x 1000-step waypoint safety validation."""
from pathlib import Path
import sys,json,argparse,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from env.config import load_config
from env.pursuit_env import PursuitEnv


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/pursuit_v2_waypoint_validation.json')
    args=parser.parse_args()
    cfg=load_config(ROOT/'configs/pursuit_environment.yaml')
    rows=[];start=time.monotonic()
    for seed in range(100):
        env=PursuitEnv(cfg);env.reset(seed)
        min_altitude=float('inf');max_radius=0.
        for step in range(1,1001):
            _,reward,terminated,truncated,info=env.step(np.tile([0,.2,0],(4,1)))
            assert not terminated and truncated==(step==1000)
            assert np.array_equal(env.red_alive_mask,np.ones(4)) and np.array_equal(env.blue_alive_mask,np.ones(4))
            assert np.isfinite(reward).all()
            min_altitude=min(min_altitude,min(s.altitude for s in env.blue))
            max_radius=max(max_radius,max(np.hypot(s.x,s.y) for s in env.blue))
        rows.append({'seed':seed,'steps':env.steps,'minimum_opponent_altitude_m':min_altitude,
            'maximum_opponent_radius_m':max_radius,
            **{k:v for k,v in info.items() if k.startswith('opponent_') and 'violation' in k}})
        if (seed+1)%10==0:print(f'[WAYPOINT] {seed+1}/100 full episodes; elapsed={time.monotonic()-start:.1f}s',flush=True)
    result={'environment_variant':'pursuit_v2','seeds':100,'steps_per_seed':1000,'total_steps':100000,
        'opponent_ground_violation_steps':sum(r['opponent_ground_violation_steps'] for r in rows),
        'opponent_boundary_violation_steps':sum(r['opponent_boundary_violation_steps'] for r in rows),
        'opponent_ground_violation_agents':sum(r['opponent_ground_violation_agents'] for r in rows),
        'opponent_boundary_violation_agents':sum(r['opponent_boundary_violation_agents'] for r in rows),
        'minimum_opponent_altitude_m':min(r['minimum_opponent_altitude_m'] for r in rows),
        'maximum_opponent_radius_m':max(r['maximum_opponent_radius_m'] for r in rows),'episodes':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    assert result['opponent_ground_violation_steps']==result['opponent_boundary_violation_steps']==0,result
    print(json.dumps({k:v for k,v in result.items() if k!='episodes'},indent=2),flush=True)

if __name__=='__main__':main()

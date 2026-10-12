"""Record and render one deterministic MAPPO match without altering combat semantics."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from algorithm.common.checkpoint import validate_checkpoint_for_evaluation
from algorithm.common.protocol import config_sha256, validate_single_round_environment_config
from algorithm.mappo.factory import build_mappo_trainer
from env.combat_env import MultiUAVCombatEnv
from env.factory import make_combat_environment


class RecordingEnvironment(MultiUAVCombatEnv):
    """Observe existing attempts and kill credits; resolve them through the base code."""
    def _attack_attempts(self, attackers, targets, next_fire_step, side):
        attempts = super()._attack_attempts(attackers, targets, next_fire_step, side)
        self.recorded_attempts[side] = attempts
        return attempts

    def _resolve_combat(self, red_attempts, blue_attempts):
        result = super()._resolve_combat(red_attempts, blue_attempts)
        self.recorded_kills = result
        return result


def record(run, output, seed):
    checkpoint = run / 'checkpoint_3000000.pt'
    cfg = yaml.safe_load((run / 'env_config.yaml').read_text(encoding='utf-8'))
    algo = yaml.safe_load((run / 'algorithm_config.yaml').read_text(encoding='utf-8'))
    validate_single_round_environment_config(cfg)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    validate_checkpoint_for_evaluation(state, cfg, algo)
    extra = state['extra']
    for key, expected in dict(training_seed=0, training_num_envs=24,
        training_total_sampled_steps=3000000, training_smoke=False, training_gamma=.99).items():
        if extra.get(key) != expected: raise RuntimeError(f'checkpoint identity mismatch: {key}')
    if state['sampled_steps'] != 3000000: raise RuntimeError('not the final 3M model')
    trainer = build_mappo_trainer(algo, 'cuda', int(extra['effective_hidden_dim']))
    trainer.load(checkpoint)
    env = RecordingEnvironment(cfg)
    reference = make_combat_environment(cfg)
    observation, _ = env.reset(seed)
    reference.reset(seed)
    trace = {'red': [], 'blue': [], 'red_alive': [], 'blue_alive': [], 'returns': []}
    events, attempts, summaries = [], [], []
    returns = np.zeros(env.team_size, dtype=np.float64)
    def frame():
        for side in ('red', 'blue'):
            trace[side].append(np.stack([s.as_array() for s in getattr(env, side)]))
            trace[f'{side}_alive'].append(getattr(env, f'{side}_alive_mask').copy())
        trace['returns'].append(returns.copy())
    frame()
    while True:
        before = {side: getattr(env, f'{side}_alive_mask').copy() for side in ('red', 'blue')}
        actions = trainer.act(observation, env.red_alive_mask, deterministic=True)
        env.recorded_attempts = {}
        observation, reward, terminated, truncated, info = env.step(actions)
        comparison = reference.step(actions)
        if not np.array_equal(observation, comparison[0]) or not np.array_equal(reward, comparison[1]) or (terminated, truncated) != comparison[2:4]:
            raise RuntimeError('recording instrumentation changed the environment transition')
        for side in ('red', 'blue'):
            if not np.array_equal(np.stack([s.as_array() for s in getattr(env, side)]),
                                  np.stack([s.as_array() for s in getattr(reference, side)])):
                raise RuntimeError('instrumented state mismatch')
        returns += reward
        frame()
        for side in ('red', 'blue'):
            for attacker, target, hit in env.recorded_attempts[side]:
                attempts.append(dict(step=env.steps, time=env.steps * env.dt, side=side,
                                     attacker=attacker+1, target=target+1, hit=bool(hit)))
            for index in np.flatnonzero(before[side] > getattr(env, f'{side}_alive_mask')):
                aircraft = getattr(env, side)[index]
                cause = 'boundary' if np.hypot(aircraft.x, aircraft.y) > env.arena_radius else 'ground' if aircraft.altitude <= 0 else 'attack'
                credits = env.recorded_kills[1 if side == 'red' else 0].get(int(index), [])
                events.append(dict(step=env.steps, time=env.steps * env.dt, side=side,
                    aircraft=int(index)+1, cause=cause, attackers=[x+1 for x in credits],
                    position=aircraft.as_array()[:3].tolist()))
        summaries.append({key: info[key] for key in (
            'red_survivors', 'blue_survivors', 'red_attack_kills', 'blue_attack_kills',
            'red_fire_attempts', 'blue_fire_attempts', 'blue_ground_guard_override_steps')})
        if terminated or truncated: break
    data = {key: np.stack(values) for key, values in trace.items()}
    np.savez_compressed(output / 'trace.npz', **data)
    metadata = dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        checkpoint_sampled_steps=state['sampled_steps'], episode_seed=seed, deterministic=True,
        environment_variant=cfg['environment_variant'], dt=env.dt, max_steps=env.max_steps,
        observation_dim=env.observation_dim, action_dim=3, num_agents=env.team_size, blue_policy=type(env.fixed_policy).__name__,
        episode_length=env.steps, simulation_duration=env.steps*env.dt,
        team_return=float(returns.sum()), agent_returns=returns.tolist(), events=events,
        attempts=attempts, frame_statistics=summaries, recording_matches_base_environment=True,
        environment_config_sha256=config_sha256(cfg), algorithm_config_sha256=config_sha256(algo),
        result={key: value for key, value in info.items() if isinstance(value, (bool, str, int, float))})
    (output / 'match.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in metadata.items() if k not in ('frame_statistics','attempts')}, indent=2), flush=True)
    return data, metadata


def render(data, meta, output, ffmpeg):
    matplotlib.rcParams['animation.ffmpeg_path'] = str(ffmpeg)
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor='#101827')
    ax = fig.add_axes([.04,.20,.46,.69], projection='3d', facecolor='#f4f6fa')
    top = fig.add_axes([.55,.37,.32,.52], facecolor='#f4f6fa')
    ax.view_init(elev=25, azim=-55)
    ax.set(xlim=(-5200,5200), ylim=(-5200,5200), zlim=(0,max(3500,float(max(-data['red'][...,2].min(),-data['blue'][...,2].min())+100))),
           xlabel='X (m)', ylabel='Y (m)', zlabel='Altitude (m)')
    top.set(xlim=(-5200,5200), ylim=(-5200,5200), xlabel='X (m)', ylabel='Y (m)', title='Top view')
    top.set_aspect('equal'); top.grid(alpha=.25)
    top.title.set_color('white')
    top.xaxis.label.set_color('white'); top.yaxis.label.set_color('white')
    top.tick_params(colors='white')
    angle=np.linspace(0,2*np.pi,150)
    ax.plot(5000*np.cos(angle),5000*np.sin(angle),np.zeros_like(angle),color='#a0aab8',lw=1)
    top.plot(5000*np.cos(angle),5000*np.sin(angle),color='#a0aab8',lw=1)
    agents = meta['num_agents']
    title = fig.text(.04,.95,f'MAPPO | final 3M model | single-round {agents}v{agents}',color='white',fontsize=18,weight='bold')
    fig.text(.04,.025,f"seed={meta['episode_seed']} | deterministic Red | GroundAware Blue | 4x playback | tracks: actual states",color='#b8c5d8',fontsize=10)
    status = fig.text(.55,.27,'',color='white',fontsize=12,family='monospace',va='top')
    event_label=fig.text(.04,.13,'',color='#ffd166',fontsize=12,va='top')
    artists = {}
    shot_links = [top.plot([], [], color='#d69200', lw=1.6, linestyle='--')[0] for _ in range(8)]
    for side,color in [('red','#e84a5f'),('blue','#3388ee')]:
        artists[side]=[]
        for index in range(agents):
            line=ax.plot([],[],[],color=color,lw=1.3,alpha=.8)[0]
            marker=ax.plot([],[],[],marker='o',color=color,ms=5,linestyle='None')[0]
            path=top.plot([],[],color=color,lw=1.1,alpha=.7)[0]
            dot=top.plot([],[],marker='o',color=color,ms=5,linestyle='None')[0]
            label=top.text(0,0,f'{side[0].upper()}{index+1}',color=color,fontsize=9)
            artists[side].append((line,marker,path,dot,label))
    def update(frame):
        time=frame*meta['dt']
        for side in ('red','blue'):
            for index,(line,marker,path,dot,label) in enumerate(artists[side]):
                xyz=data[side][:frame+1,index,:3].copy(); xyz[:,2]*=-1
                alive=bool(data[f'{side}_alive'][frame,index])
                line.set_data(xyz[:,0],xyz[:,1]); line.set_3d_properties(xyz[:,2])
                marker.set_data([xyz[-1,0]],[xyz[-1,1]]); marker.set_3d_properties([xyz[-1,2]])
                marker.set_marker('o' if alive else 'x'); marker.set_alpha(1 if alive else .6)
                path.set_data(xyz[:,0],xyz[:,1]); dot.set_data([xyz[-1,0]],[xyz[-1,1]])
                offset = [(-450,200), (120,100), (-350,-200), (200,300), (-500,400), (300,-400)][index]
                dot.set_marker('o' if alive else 'x'); label.set_position((xyz[-1,0]+offset[0],xyz[-1,1]+offset[1]))
                label.set_alpha(1 if alive else .4)
        row=meta['frame_statistics'][frame-1] if frame else dict(red_attack_kills=0,blue_attack_kills=0,red_fire_attempts=0,blue_fire_attempts=0)
        text=f"Sim time: {time:5.1f}s / {meta['simulation_duration']:.1f}s\nAlive      Red {int(data['red_alive'][frame].sum())}/{agents}   Blue {int(data['blue_alive'][frame].sum())}/{agents}\nAttack kills  {row['red_attack_kills']}          {row['blue_attack_kills']}\nFire attempts {row['red_fire_attempts']}          {row['blue_fire_attempts']}\nTeam return: {float(data['returns'][frame].sum()):.2f}"
        if frame==len(data['red'])-1: text+='\nResult: '+meta['result']['termination_reason']
        status.set_text(text)
        recent=[event for event in meta['events'] if event['time']<=time][-3:]
        event_label.set_text('\n'.join(f"{e['time']:5.1f}s  {e['side'][0].upper()}{e['aircraft']} lost: {e['cause']}" + (f" (by {('Blue' if e['side']=='red' else 'Red')} {e['attackers']})" if e['attackers'] else '') for e in recent))
        current_attempts = [a for a in meta['attempts'] if 0 <= time-a['time'] <= .8][-8:]
        for index, link in enumerate(shot_links):
            if index >= len(current_attempts):
                link.set_data([], [])
                continue
            attempt = current_attempts[index]
            opposite = 'blue' if attempt['side'] == 'red' else 'red'
            start = data[attempt['side']][attempt['step'], attempt['attacker']-1, :2]
            end = data[opposite][attempt['step'], attempt['target']-1, :2]
            link.set_data([start[0], end[0]], [start[1], end[1]])
            link.set_color('#d69200' if attempt['hit'] else '#999999')
    frames=list(range(0,len(data['red']),2))
    if frames[-1]!=len(data['red'])-1: frames.append(len(data['red'])-1)
    encoders = subprocess.check_output([str(ffmpeg), '-hide_banner', '-encoders'], text=True)
    codec = 'libx264' if 'libx264' in encoders else 'h264_mf'
    writer=FFMpegWriter(fps=20,codec=codec,bitrate=2400,
        extra_args=['-pix_fmt','yuv420p','-movflags','+faststart'])
    with writer.saving(fig,str(output/'match.mp4'),dpi=100):
        for number,frame in enumerate(frames):
            update(frame); writer.grab_frame()
            if number%100==0: print(f'[RENDER] {number}/{len(frames)} frames',flush=True)
        for _ in range(60): writer.grab_frame()
    update(len(data['red'])-1); fig.savefig(output/'preview.png',dpi=100,facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--episode-seed',type=int,default=10000000)
    parser.add_argument('--ffmpeg',type=Path,required=True)
    args=parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for checkpoint recording')
    if not args.ffmpeg.is_file(): raise FileNotFoundError(args.ffmpeg)
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise RuntimeError('output must be empty')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    data,metadata=record(args.run_dir,args.output_dir,args.episode_seed)
    render(data,metadata,args.output_dir,args.ffmpeg)


if __name__=='__main__': main()

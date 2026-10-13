"""Threshold, schema, resume/rollback, streaming and unchanged math regressions."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from algorithm.common.training_metrics_logging import TrainingMetricsSnapshots,last_valid_metrics_step
from algorithm.mappo.runner import MAPPOTrainingRunner
from algorithm.madsac.runner import MADSACTrainingRunner
from algorithm.train_mappo import prepare_resume_rollback
from tools.compact_training_metrics import compact_metrics


def steps(path):
    return [json.loads(line)['sampled_steps'] for line in path.read_text().splitlines()]

class Harness(TrainingMetricsSnapshots):
    def __init__(self,path,interval=20000):
        self.output_dir=path;path.mkdir(exist_ok=True)
        self.console_interval=interval;self.trainer=SimpleNamespace(sampled_steps=0)
        self.initialize_training_metrics_logging()
    def write(self,result,rows):
        with (self.output_dir/'training_metrics.jsonl').open('a') as stream:
            stream.write(json.dumps({'sampled_steps':self.trainer.sampled_steps,'real_result':result},separators=(',',':'))+'\n')


@pytest.mark.parametrize('total',[60000,60024,65016])
def test_24_env_thresholds_independent_and_final_unique(tmp_path,total):
    h=Harness(tmp_path);h.next_console_log=999999
    for step in range(24,total+1,24):
        h.trainer.sampled_steps=step
        h.record_training_metrics_transition(step,[],h.write)
    h.finalize_training_metrics_logging(h.write);h.finalize_training_metrics_logging(h.write)
    logged=steps(tmp_path/'training_metrics.jsonl')
    assert logged==sorted(set(logged)) and logged[-1]==total
    for row,threshold in zip(logged[:3],[20000,40000,60000]):assert threshold<=row<threshold+24
    assert h.next_console_log==999999 and len(logged)<=4
    assert json.loads((tmp_path/'training_metrics.jsonl').read_text().splitlines()[-1])['real_result']==total


def test_three_million_synthetic_has_150_rows(tmp_path):
    h=Harness(tmp_path)
    for step in range(24,3000001,24):
        h.trainer.sampled_steps=step;h.record_training_metrics_transition(step,[],h.write)
    h.finalize_training_metrics_logging(h.write)
    assert len(steps(tmp_path/'training_metrics.jsonl'))==150
    assert 1-150/125000>.998


def test_old_high_frequency_resume_missing_invalid_tail_and_rollback(tmp_path):
    h=Harness(tmp_path)
    path=tmp_path/'training_metrics.jsonl'
    original=''.join(json.dumps({'sampled_steps':s})+'\n' for s in range(24,1009,24))
    path.write_text(original)
    h.trainer.sampled_steps=1008;h.restore_training_metrics_logging()
    assert h.last_training_metrics_step==1008 and h.next_training_metrics_log==20000
    for step in range(1032,21001,24):
        h.trainer.sampled_steps=step;h.record_training_metrics_transition(step,[],h.write)
    h.finalize_training_metrics_logging(h.write)
    assert path.read_text().startswith(original)
    assert steps(path)[-2:]==[20016,21000]
    assert len(steps(path))==44
    path.write_text(''.join(json.dumps({'sampled_steps':s})+'\n' for s in [1480000,1500008,1520000]))
    checkpoint=tmp_path/'checkpoint.pt'
    prepare_resume_rollback(tmp_path,checkpoint,1500000)
    h.trainer.sampled_steps=1500000;h.restore_training_metrics_logging()
    assert steps(path)==[1480000] and h.last_training_metrics_step==1480000
    assert h.next_training_metrics_log==1520000
    assert last_valid_metrics_step(tmp_path/'absent') is None
    path.write_text('{"sampled_steps":24}\ninvalid\n{"sampled_steps":true}\n')
    assert last_valid_metrics_step(path)==24


@pytest.mark.parametrize('kind',['combat','pursuit','madsac'])
def test_existing_snapshot_schemas_and_compact_encoding(tmp_path,kind):
    from algorithm.common.pursuit_metrics import DIAGNOSTICS
    runner=(MADSACTrainingRunner if kind=='madsac' else MAPPOTrainingRunner).__new__(
        MADSACTrainingRunner if kind=='madsac' else MAPPOTrainingRunner)
    runner.output_dir=tmp_path;runner.trainer=SimpleNamespace(sampled_steps=20016,vector_steps=834)
    runner.last_metrics={'actor_loss':1.};runner.env_config={'task_type':kind}
    info={key:0. for key in DIAGNOSTICS}
    info.update(raw_nearest_distance_m=np.ones(4),pursuit_reward=-np.ones(4)*.001,nearest_distance_valid_mask=np.ones(4,dtype=bool))
    for side in ('red','blue'):
        for name in ('fire_window_pairs','step_fire_attempts','step_weapon_hits'):info[f'{side}_{name}']=0
    for name in ('r1','r2','r3','r4'):info[f'{name}_rewards']=np.zeros(4)
    result=SimpleNamespace(rewards=np.zeros((1,4)),infos=[info])
    if kind=='madsac':runner.replay=[];runner._write_training_metrics(result,[])
    else:runner._write_step_metrics(result,[])
    text=(tmp_path/'training_metrics.jsonl').read_text();record=json.loads(text)
    assert text==json.dumps(record,separators=(',',':'))+'\n'
    if kind=='pursuit':
        assert all(k in record for k in ('raw_nearest_distance_m','pursuit_reward','nearest_distance_valid_mask','mean_nearest_distance'))
    elif kind=='combat':assert all(k in record for k in ('mean_r1_reward','win_rate','red_step_fire_attempts','actor_loss'))
    else:assert all(k in record for k in ('replay_size','red_loss','completed_episodes_this_step'))


def test_compactor_streaming_first_threshold_final_original_untouched(tmp_path,monkeypatch):
    path=tmp_path/'training_metrics.jsonl';values=np.cumsum([17+(i%29) for i in range(1000)]).tolist()
    original=''.join(json.dumps({'sampled_steps':s,'array':[1,2,3]})+'\n' for s in values)
    path.write_text(original)
    def forbidden(*args,**kwargs):raise AssertionError('whole file read forbidden')
    monkeypatch.setattr(Path,'read_text',forbidden)
    result=compact_metrics(path,interval=2000)
    with path.open() as stream:assert stream.read()==original
    with Path(result['output']).open() as stream:logged=[json.loads(line)['sampled_steps'] for line in stream]
    expected=[values[0]]
    threshold=2000
    for s in values[1:]:
        if s>=threshold:expected.append(s);threshold=(s//2000+1)*2000
    if expected[-1]!=values[-1]:expected.append(values[-1])
    assert logged==expected and logged==sorted(set(logged))
    assert result['input_rows']==1000 and result['output_rows']==len(expected)
    assert result['first_sampled_steps']==values[0] and result['last_sampled_steps']==values[-1]


def test_compactor_replace_backup_and_active_file_refusal(tmp_path,monkeypatch):
    import tools.compact_training_metrics as module
    path=tmp_path/'training_metrics.jsonl'
    original=''.join(json.dumps({'sampled_steps':s})+'\n' for s in [24,48,72,20016,20040])
    path.write_text(original)
    result=compact_metrics(path,replace=True)
    assert Path(result['backup']).read_text()==original and steps(path)==[24,20016,20040]
    path.write_text(original)
    real_identity=module.file_identity;calls=0
    def growing(p):
        nonlocal calls
        calls+=1
        if calls==2:
            with p.open('a') as stream:stream.write('{"sampled_steps":30000}\n')
        return real_identity(p)
    monkeypatch.setattr(module,'file_identity',growing)
    with pytest.raises(RuntimeError,match='active log'):compact_metrics(path,replace=True)
    assert path.read_text()==original+'{"sampled_steps":30000}\n'
    assert not list(tmp_path.glob('.*.tmp'))


@pytest.mark.parametrize('record',[{}, {'sampled_steps':True},{'sampled_steps':1.5},{'sampled_steps':-1},[]])
def test_compactor_invalid_steps_preserves_original(tmp_path,record):
    path=tmp_path/'training_metrics.jsonl';path.write_text(json.dumps(record)+'\n')
    with pytest.raises(ValueError,match='integer'):compact_metrics(path)
    assert not (tmp_path/'training_metrics.compact.jsonl').exists()


def test_compactor_rejects_duplicates_and_overwrite(tmp_path):
    path=tmp_path/'training_metrics.jsonl';path.write_text('{"sampled_steps":24}\n{"sampled_steps":24}\n')
    with pytest.raises(ValueError,match='increasing'):compact_metrics(path)
    with pytest.raises(ValueError,match='distinct'):compact_metrics(path,path)


def recursive_equal(a,b):
    import torch
    if isinstance(a,torch.Tensor):assert torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:recursive_equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):recursive_equal(x,y)
    else:assert a==b


def legacy_logging(runner,writer):
    def record(result,rows,_writer):
        runner.last_step_result,runner.last_completed_rows=result,rows
        writer(result,rows)
        runner.last_training_metrics_step=runner.trainer.sampled_steps
    runner.record_training_metrics_transition=record


@pytest.mark.parametrize('kind',['combat','pursuit','madsac'])
def test_cuda_logging_ab_training_state_and_evaluation_exactly_equal(tmp_path,kind):
    import yaml,torch,copy
    from algorithm.common.evaluator import evaluate
    from algorithm.madsac.evaluation import evaluate_madsac
    root=Path(__file__).resolve().parents[1]
    env=yaml.safe_load((root/'configs'/('pursuit_environment.yaml' if kind=='pursuit' else 'single_round_environment.yaml')).read_text())
    alg=yaml.safe_load((root/'configs'/('madsac_single_round_3m.yaml' if kind=='madsac' else 'mappo.yaml')).read_text())
    runner_type=MADSACTrainingRunner if kind=='madsac' else MAPPOTrainingRunner
    results=[]
    for mode in ('high_frequency','snapshots'):
        runner=runner_type(env,alg,num_envs=1,total_sampled_steps=64,device='cuda',seed=73,
                           output_dir=tmp_path/mode,smoke=True)
        if mode=='high_frequency':
            legacy_logging(runner,runner._write_training_metrics if kind=='madsac' else runner._write_step_metrics)
        runner.run()
        state=copy.deepcopy(runner.trainer.checkpoint_state())
        if kind=='madsac':
            evaluation=evaluate_madsac(runner.trainer,env,[44000000],runner.evaluation_mode,runner.evaluation_policy_seed)
            replay={key:(value[:runner.replay.size].copy() if isinstance(value,np.ndarray)
                          else copy.deepcopy(value.bit_generator.state) if key=='rng' else copy.deepcopy(value))
                    for key,value in runner.replay.__dict__.items()}
        else:evaluation=evaluate(runner.trainer,env,[10000000]);replay=None
        results.append((state,len(runner.completed_records),evaluation,replay,runner.vector.used_training_seeds,
                        (tmp_path/mode/'optimization_metrics.jsonl').read_bytes()))
        assert steps(tmp_path/mode/'training_metrics.jsonl')[-1]==64
    for a,b in zip(results[0],results[1]):
        if isinstance(a,set):assert a==b
        else:recursive_equal(a,b)
    assert len(steps(tmp_path/'high_frequency/training_metrics.jsonl'))==64
    assert steps(tmp_path/'snapshots/training_metrics.jsonl')==[64]


def test_cuda_resume_old_high_frequency_log_appends_sparse_final(tmp_path):
    import yaml
    root=Path(__file__).resolve().parents[1]
    env=yaml.safe_load((root/'configs/pursuit_environment.yaml').read_text())
    alg=yaml.safe_load((root/'configs/mappo.yaml').read_text())
    first=MAPPOTrainingRunner(env,alg,num_envs=1,total_sampled_steps=128,device='cuda',seed=74,output_dir=tmp_path,smoke=True)
    legacy_logging(first,first._write_step_metrics)
    first.run()
    path=tmp_path/'training_metrics.jsonl';original=path.read_bytes()
    second=MAPPOTrainingRunner(env,alg,num_envs=1,total_sampled_steps=256,device='cuda',seed=74,output_dir=tmp_path,smoke=True)
    second.console_interval=100 # Accelerate only the logger's smoke thresholds.
    second.resume(tmp_path/'latest.pt')
    assert second.last_training_metrics_step==128 and second.next_training_metrics_log==200
    second.run()
    assert path.read_bytes().startswith(original)
    logged=steps(path)
    assert logged[-2:]==[200,256] and len(logged)==130 and logged==sorted(set(logged))

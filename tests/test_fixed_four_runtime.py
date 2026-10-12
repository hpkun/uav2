"""Strict fixed 4v4 runtime and immutable pre-cadence source regressions."""
from pathlib import Path
import inspect
from copy import deepcopy
import numpy as np
import pytest
import yaml
from env.observation import observation_dim_from_config, build_team_observations, BASE_OBSERVATION_DIM, OBSERVATION_DIM
from env.scenario import random_combat_states
from env.fixed_policy import GroundAwareNearestTargetPursuitPolicy
from env.factory import make_combat_environment
from algorithm.common.evaluator import episode_return_metrics
from algorithm.mappo.runner import MAPPOTrainingRunner
from algorithm.madsac.protocol import validate_madsac_config
from tools import audit_single_round as audit

ROOT=Path(__file__).resolve().parents[1]
def load(name):return yaml.safe_load((ROOT/'configs'/name).read_text(encoding='utf-8'))
def cfg():return load('single_round_environment.yaml')


def test_observation_fixed_52_and_six_slots_rejected():
    config=cfg();env=make_combat_environment(config);observation,_=env.reset(17)
    assert BASE_OBSERVATION_DIM==OBSERVATION_DIM==52
    assert observation_dim_from_config(config['observation'])==52
    assert observation_dim_from_config({**config['observation'],'team_size':6})==52
    assert observation.shape==(4,52)
    six=env.red+[env.red[0].copy(),env.red[1].copy()]
    with pytest.raises(FloatingPointError):
        build_team_observations(six,six,config['observation'])


def test_scenario_and_factory_reject_six_aircraft():
    config=cfg();config['scenario']['team_size']=6
    config['scenario']['formation_offsets']=[-750,-450,-150,150,450,750]
    with pytest.raises(ValueError):random_combat_states(np.random.default_rng(0),**config['scenario'])
    with pytest.raises(ValueError):make_combat_environment(config)


def test_ground_aware_fixed_interface_and_masks():
    assert 'team_size' not in inspect.signature(GroundAwareNearestTargetPursuitPolicy).parameters
    env=make_combat_environment(cfg());env.reset(0)
    for name in ('_previous_override','_current_duration','last_override_mask'):
        assert getattr(env.fixed_policy,name).shape==(4,)
    assert env.red_next_fire_step.shape==env.blue_next_fire_step.shape==(4,)
    assert (env.team_size,env.observation_dim,env.action_dim)==(4,52,3)


def test_formal_mappo_and_madsac_accept_only_52_3_4(tmp_path):
    config=cfg();mappo=load('mappo.yaml');madsac=load('madsac_single_round_3m.yaml')
    validate_madsac_config(config,madsac)
    runner=MAPPOTrainingRunner(config,mappo,num_envs=1,total_sampled_steps=4,
                               device='cuda',output_dir=tmp_path,smoke=True)
    runner.vector.close()
    for algorithm in (mappo,madsac):
        algorithm['network']['observation_dim']=78;algorithm['network']['num_agents']=6
    with pytest.raises(ValueError,match='dimension mismatch'):
        MAPPOTrainingRunner(config,mappo,output_dir=tmp_path)
    with pytest.raises(RuntimeError,match='dimension mismatch'):
        validate_madsac_config(config,madsac)


def test_return_metrics_reject_six_agent_vector():
    assert episode_return_metrics(np.ones(4))==(4.,1.)
    with pytest.raises(ValueError):episode_return_metrics(np.ones(6))


def test_no_active_six_config_and_legacy_isolated():
    for path in (ROOT/'configs').glob('*.yaml'):
        config=yaml.safe_load(path.read_text(encoding='utf-8'))
        if 'scenario' in config:assert config['scenario']['team_size']==4
        if 'network' in config:assert config['network']['num_agents']==4
    assert (ROOT/'legacy/single_round_v1/configs/single_round_6v6_environment.yaml').is_file()
    assert not (ROOT/'tests/test_six_vs_six.py').exists()
    import sys
    assert not any(name.startswith('legacy.') for name in sys.modules)


def test_all_frozen_sources_match_immutable_v2_commit():
    assert audit.V2_COMMIT=='81d623e1ecafbf7df91aa2d9dc1fc19cf885f2a9'
    result=audit.compare_frozen_files(ROOT)
    assert len(result)==16 and all(row['baseline_blob']==row['current_blob'] for row in result)
    assert audit.direct_path_audit(ROOT)['legacy_controller_excluded']


def test_exact_configuration_delta_from_committed_v2():
    current=audit.exact_config_delta(ROOT)
    previous=yaml.safe_load(audit.baseline_content(ROOT,'configs/single_round_environment.yaml',audit.V2_COMMIT))
    assert previous['environment_variant']=='single_round_v2'
    assert current['environment_variant']=='single_round_v3'
    allowed={'environment_version','environment_variant','action','blue_policy'}
    for key in set(previous)-allowed:assert previous[key]==current[key]
    assert set(current)-set(previous)=={'task_type','action_semantics'}


def test_clean_clone_regression_has_no_snapshot_commit_or_worktree_copy():
    import ast
    parsed=ast.parse((ROOT/'tests/test_formal_single_round_closure.py').read_text(encoding='utf-8'))
    function=next(node for node in parsed.body if isinstance(node,ast.FunctionDef)
        and node.name=='test_committed_clean_clone_audit_is_repeatable_and_ignores_staged_changes')
    calls=[node for node in ast.walk(function) if isinstance(node,ast.Call)]
    for node in calls:
        if isinstance(node.func,ast.Name) and node.func.id=='git' and node.args:
            assert not isinstance(node.args[0],ast.Constant) or node.args[0].value!='commit'
        if isinstance(node.func,ast.Attribute):
            assert node.func.attr not in ('write_text', 'copy2', 'copytree')

"""Strict Red overload protocol on an otherwise unchanged v3.31 baseline."""
from copy import deepcopy
from numbers import Real
import math
from .v331_config import validate_v331

RED_DIRECT_CONTROL = dict(
    semantics='lin_direct_overload_rate_limited',
    roll_min=-math.pi,roll_max=math.pi,roll_rate_max=math.pi/2,
    tangential_load_min=-1.,tangential_load_max=2.,tangential_load_rate_max=2.,
    normal_load_min=0.,normal_load_max=8.,normal_load_rate_max=2.,
    trim_roll=0.,trim_tangential_load=0.,trim_normal_load=1.)


def validate_v332(config):
    if not isinstance(config,dict) or config.get('environment_version')!='3.32':
        raise ValueError('v3.32 requires environment_version 3.32')
    red=config.get('red_direct_control')
    if not isinstance(red,dict) or set(red)!=set(RED_DIRECT_CONTROL):
        raise ValueError('v3.32 red_direct_control missing/unknown fields')
    for name,expected in RED_DIRECT_CONTROL.items():
        value=red[name]
        if name!='semantics' and (isinstance(value,bool) or not isinstance(value,Real) or not math.isfinite(value)):
            raise ValueError(f'v3.32 red_direct_control.{name} must be finite numeric')
        if value!=expected:
            raise ValueError(f'v3.32 red_direct_control.{name} must equal {expected!r}')
    baseline=deepcopy(config)
    baseline.pop('red_direct_control')
    baseline['environment_version']='3.31'
    validate_v331(baseline)
    return config

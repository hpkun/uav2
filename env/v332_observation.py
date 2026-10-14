"""47D Red observation; frozen ally/enemy blocks shifted by two slots."""
import numpy as np
from .v330_observation import build_team_observations_v330
from .v332_control import normalized_control_features_v332

OBSERVATION_DIM=47


def build_team_observations_v332(team,opponents,config,red_control_states):
    controls=np.asarray(red_control_states,dtype=float)
    if controls.shape!=(4,3) or not np.isfinite(controls).all():
        raise ValueError('v3.32 red_control_states must be finite shape (4,3)')
    baseline=build_team_observations_v330(team,opponents,config)
    result=np.zeros((4,OBSERVATION_DIM),dtype=baseline.dtype)
    result[:,:4]=baseline[:,:4]
    result[:,7:9]=baseline[:,5:7]
    result[:,9:]=baseline[:,7:]
    for index,state in enumerate(team):
        if state.alive:result[index,4:7]=normalized_control_features_v332(controls[index])
    return result

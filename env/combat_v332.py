"""v3.31 battle mechanics with a Red-only overload and circular rate limiter."""
import numpy as np
from .combat_v331 import CombatEnvironmentV331
from .v332_control import TRIM, normalized_action_to_target_v332,rate_limit_control_v332,executed_control_to_command_v332
from .v332_observation import build_team_observations_v332
from .v332_transition import step_core
from .config import load_config
from .v332_config import validate_v332


class CombatEnvironmentV332(CombatEnvironmentV331):
    environment_version='3.32'

    def __init__(self,config):
        loaded=load_config(config) if not isinstance(config,dict) else config
        validate_v332(loaded)
        super().__init__(loaded)
        self.red_control_states=np.tile(TRIM,(self.team_size,1))
        self.red_direct_control_diagnostics={}

    def reset(self,seed=None):
        self.red_control_states[:]=TRIM
        self.red_direct_control_diagnostics={}
        return super().reset(seed)

    def _observations(self):
        return build_team_observations_v332(self.red,self.blue,self.config['observation'],self.red_control_states)

    def _advance_red_direct(self,states,actions):
        phis=np.zeros(self.team_size,dtype=float)
        executed_rows=[];rate_rows=[]
        for index,state in enumerate(states):
            if not state.alive:continue
            target=normalized_action_to_target_v332(actions[index])
            executed,limited=rate_limit_control_v332(self.red_control_states[index],target,self.dt,self.config['red_direct_control'])
            self.red_control_states[index]=executed
            phis[index]=executed[0]
            executed_rows.append(executed);rate_rows.append(limited)
            control=executed_control_to_command_v332(executed)
            states[index]=self.integrator.step(state,control,self.dynamics,self.spec)
        rows=np.asarray(executed_rows,dtype=float).reshape(-1,3)
        rates=np.asarray(rate_rows,dtype=float).reshape(-1,3)
        n=len(rows)
        mean=lambda x:float(np.mean(x)) if n else 0.
        self.red_direct_control_diagnostics=dict(agent_steps=n,
            mean_abs_mu=mean(np.abs(rows[:,0])),mean_xi_hat=mean(rows[:,1]),mean_xi_bar=mean(rows[:,2]),
            roll_rate_limit_fraction=mean(rates[:,0]),tangential_rate_limit_fraction=mean(rates[:,1]),normal_rate_limit_fraction=mean(rates[:,2]),
            roll_near_limit_fraction=mean(np.abs(rows[:,0])>=.99*np.pi),
            xi_hat_near_min_fraction=mean(rows[:,1]<=-.97),xi_hat_near_max_fraction=mean(rows[:,1]>=1.97),
            xi_bar_near_min_fraction=mean(rows[:,2]<=.08),xi_bar_near_max_fraction=mean(rows[:,2]>=7.92))
        return phis

    def _info(self,*args,**kwargs):
        info=super()._info(*args,**kwargs)
        info['red_direct_control_diagnostics']=dict(self.red_direct_control_diagnostics)
        info.update({key:value for key,value in self.red_direct_control_diagnostics.items() if key!='agent_steps'})
        return info

    def step(self,red_actions,blue_actions=None):
        # Same wrapper order as v3.26 -> v3.9/37/36/35 -> frozen v3.0 core.
        for i,own in enumerate(self.blue):
            target=self.fixed_policy.nearest_target_index(own,self.red) if own.alive else None
            self.blue_step_target_indices[i]=-1 if target is None else target
        automatic=blue_actions is None
        if automatic:blue_actions=self.fixed_policy.team_actions(self.blue,self.red)
        obs,_,terminated,truncated,info=step_core(self,red_actions,blue_actions)
        info.update({f'blue_{name}_returns':v for name,v in self.blue_returns.items()},
                    blue_boundary_return_episode=self.blue_returns['boundary']>0)
        if automatic:info.update(self.fixed_policy.diagnostics())
        info['tactical_total_rewards']=info.pop('tactical_rewards');info['pbrs_enabled']=False
        info['individual_event_rewards']=info['r1_rewards'].copy()
        info['boundary_rewards']=info['r2_rewards'].copy()
        info['tactical_rewards']=info['r4_rewards'].copy()
        info['safety_rewards']=info['safe_rewards'].copy()
        for name,source in (('individual_event','r1'),('boundary','r2'),('tactical','r4'),('safety','safe')):
            info[f'episode_{name}_rewards']=info[f'episode_{source}_rewards'].copy()
            info[f'episode_{name}_total']=info[f'episode_{source}_total']
        for i in range(1,5):info[f'madsac_r{i}_total']=info[f'episode_r{i}_total']
        paper=sum((info[f'r{i}_rewards'] for i in range(1,5)),np.zeros(self.team_size,dtype=float))
        extended=paper+info['safety_rewards']
        self.episode_paper_rewards+=paper;self.episode_extended_rewards+=extended
        self._reward_diagnostics(info,paper,extended);info['local_rewards']=extended.copy()
        info['blue_pursuit_target_indices']=self.blue_step_target_indices.copy()
        info['controller_diagnostics']={side:dict(row) for side,row in self.controller_diagnostics.items()}
        info['red_firing_geometry_agent_steps']=dict(self.firing_geometry_diagnostics)
        return obs,extended,terminated,truncated,info

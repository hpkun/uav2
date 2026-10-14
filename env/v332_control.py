"""Red-only [a_mu,a_t,a_n] decoder and independent executed-control limiter."""
import numpy as np
from .math_utils import wrap_angle
from .models import ControlCommand

TRIM = np.array([0.,0.,1.])


def normalized_action_to_target_v332(action):
    a=np.asarray(action,dtype=float)
    if a.shape!=(3,) or not np.isfinite(a).all():
        raise ValueError('v3.32 Red action must be finite shape (3,)')
    a_mu,a_t,a_n=np.clip(a,-1.,1.)
    return np.array([np.pi*a_mu,a_t if a_t<0 else 2*a_t,
                     1+a_n if a_n<0 else 1+7*a_n],dtype=float)


def rate_limit_control_v332(previous,target,dt,config):
    previous=np.asarray(previous,dtype=float);target=np.asarray(target,dtype=float)
    if previous.shape!=(3,) or target.shape!=(3,) or not np.isfinite([previous,target]).all():
        raise ValueError('v3.32 control states must be finite shape (3,)')
    if not np.isfinite(dt) or dt<=0:
        raise ValueError('dt must be finite and positive')
    raw_error=target[0]-previous[0]
    error=wrap_angle(raw_error)
    # Exactly pi has two shortest paths: retain the requested turn direction.
    if abs(abs(error)-np.pi)<=1e-12:
        error=float(np.copysign(np.pi,raw_error))
    limits=dt*np.array([config['roll_rate_max'],config['tangential_load_rate_max'],config['normal_load_rate_max']])
    errors=np.array([error,target[1]-previous[1],target[2]-previous[2]])
    limited=np.abs(errors)>limits+1e-12
    executed=previous+np.clip(errors,-limits,limits)
    executed[0]=wrap_angle(executed[0])
    executed[1]=np.clip(executed[1],config['tangential_load_min'],config['tangential_load_max'])
    executed[2]=np.clip(executed[2],config['normal_load_min'],config['normal_load_max'])
    return executed,limited


def executed_control_to_command_v332(executed):
    mu,xi_hat,xi_bar=executed
    return ControlCommand(nx=float(xi_hat),nz=float(xi_bar),phi=float(mu))


def normalized_control_features_v332(executed):
    mu,xi_hat,xi_bar=executed
    return np.array([mu/np.pi,xi_hat if xi_hat<0 else xi_hat/2,
                     xi_bar-1 if xi_bar<1 else (xi_bar-1)/7])


def red_control_metric_values_v332(infos):
    """Replicate per-agent step values for existing rollout weighted aggregation."""
    values={}
    for info in infos:
        row=info.get('red_direct_control_diagnostics',{})
        n=row.get('agent_steps',0)
        if n:
            for key,value in row.items():
                if key!='agent_steps':values.setdefault(key,[]).extend([value]*n)
    return values


def label_action_metrics_v332(metrics):
    """Rename legacy generic trainer diagnostic labels, without changing values."""
    renamed={}
    for key,value in metrics.items():
        name=key
        if key.startswith(('policy_log_std_','policy_sigma_','action_abs_gt_')):
            for old,new in (('heading','a_mu'),('pitch','a_t'),('speed','a_n'),('psi','a_mu'),('theta','a_t'),('v','a_n')):
                if key.endswith('_'+old):
                    name=key[:-len(old)]+new
                    break
        renamed[name]=value
    return renamed

"""Independent weapon-free pursuit task with raw negative-distance rewards."""
from pathlib import Path
import numpy as np
from .config import load_config, aircraft_spec
from .direct_protocol import validate_direct_action_config
from .models import AircraftState
from .dynamics import PointMassDynamics
from .integrator import RK4Integrator
from .combat_env import MultiUAVCombatEnv
from .waypoint_policy import SeededRandomWaypointDirectPolicy


def validate_pursuit_config(config):
    validate_direct_action_config(config)
    if (config.get("environment_version"), config.get("environment_variant"), config.get("task_type")) != ("2.4", "pursuit_v2", "pursuit"):
        raise RuntimeError("pursuit requires 2.4/pursuit_v2/pursuit identity")
    if config["simulation"] != {"dt": .1, "max_steps": 1000} or config["scenario"]["team_size"] != 4:
        raise RuntimeError("pursuit requires 4 agents, dt=.1, project horizon=1000")
    from .observation import observation_dim_from_config
    if observation_dim_from_config(config["observation"]) != 52:
        raise RuntimeError("pursuit requires observation_dim=52")
    if any(key in config for key in ("weapon", "attack")):
        raise RuntimeError("pursuit must have no weapon or attack configuration")
    if config["reward"] != {"distance_reward_scale":0.001}:
        raise RuntimeError("pursuit_v2 distance_reward_scale must be exactly 0.001")
    s = config["scenario"]
    if not 0 < s["initial_radius_m"] < config["arena"]["radius"] or not 0 < s["altitude_min_m"] <= s["altitude_max_m"]:
        raise ValueError("illegal pursuit initial position range")
    a = config["aircraft"]
    if not a["v_min"] <= s["speed_min_m_s"] <= s["speed_max_m_s"] <= a["v_max"] or not 0 <= s["pitch_abs_max_rad"] <= min(-a["theta_min"], a["theta_max"]):
        raise ValueError("illegal pursuit initial state envelope")
    p = config["opponent_policy"]
    if p["type"] != "seeded_random_waypoint_direct" or type(p["waypoint_interval_steps"]) is not int or p["waypoint_interval_steps"] != 50:
        raise ValueError("formal opponent requires seeded random waypoints every 50 steps")
    if not 0 < p["waypoint_radius_m"] < p["boundary_avoidance"]["activation_radius_m"] < config["arena"]["radius"]:
        raise ValueError("waypoint/safety radii must be inside the arena")
    if not 0 < p["waypoint_altitude_min_m"] <= p["waypoint_altitude_max_m"]:
        raise ValueError("invalid waypoint altitude range")
    if not a["v_min"] <= p["desired_speed_min_m_s"] <= p["desired_speed_max_m_s"] <= a["v_max"]:
        raise ValueError("invalid waypoint speed range")
    if not a["v_min"] <= p["boundary_avoidance"]["desired_speed_m_s"] <= a["v_max"] or p["boundary_avoidance"]["recovery_altitude_m"] <= 0:
        raise ValueError("invalid opponent safety guidance")
    SeededRandomWaypointDirectPolicy(p)
    if not 0 < config["diagnostics"]["near_limit_fraction"] <= 1:
        raise ValueError("invalid diagnostic near-limit threshold")


class PursuitEnv:
    """Shares only integration and observation helpers with combat, never attacks."""
    team_size, observation_dim, action_dim = 4, 52, 3
    environment_version, environment_variant = "2.4", "pursuit_v2"
    _advance = MultiUAVCombatEnv._advance
    _observations = MultiUAVCombatEnv._observations
    red_alive_mask = MultiUAVCombatEnv.red_alive_mask
    blue_alive_mask = MultiUAVCombatEnv.blue_alive_mask

    def __init__(self, config):
        self.config = load_config(config) if isinstance(config, (str, Path)) else config
        validate_pursuit_config(self.config)
        self.spec = aircraft_spec(self.config)
        self.dt, self.max_steps = .1, 1000
        self.arena_radius = float(self.config["arena"]["radius"])
        self.dynamics, self.integrator = PointMassDynamics(), RK4Integrator(self.dt)
        self.fixed_policy = SeededRandomWaypointDirectPolicy(self.config["opponent_policy"])
        self._episode_done = True

    def _initial_states(self):
        s = self.config["scenario"]
        states = []
        for _ in range(4):
            radius = s["initial_radius_m"] * np.sqrt(self.rng.random())
            angle = self.rng.uniform(-np.pi, np.pi)
            states.append(AircraftState(radius*np.cos(angle), radius*np.sin(angle),
                -self.rng.uniform(s["altitude_min_m"], s["altitude_max_m"]),
                self.rng.uniform(s["speed_min_m_s"], s["speed_max_m_s"]),
                self.rng.uniform(-s["pitch_abs_max_rad"], s["pitch_abs_max_rad"]),
                self.rng.uniform(-np.pi, np.pi)))
        return states

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.red, self.blue = self._initial_states(), self._initial_states()
        self.red_last_executed_phi = np.zeros(4, dtype=np.float32)
        self.blue_last_executed_phi = np.zeros(4, dtype=np.float32)
        self.steps, self._episode_done = 0, False
        self.distance_sum, self.distance_count, self.distance_min = 0., 0, float('inf')
        self.speed_sum, self.state_count = 0., 0
        self.control_sum, self.control_near = np.zeros(3), np.zeros(3)
        self.clamp_counts = np.zeros(4)
        self.violation_steps = {f"{side}_{kind}_violation_steps":0
            for side in ("red","opponent") for kind in ("ground","boundary")}
        self.violation_masks = {key:np.zeros(4,dtype=bool) for key in self.violation_steps}
        self.fixed_policy.reset()
        self.opponent_actions = np.zeros((4,3))
        self.last_valid_distances = self._distances()
        return self._observations(), self._info(self._distances(), np.zeros(4))

    def _distances(self):
        self._validate_states()
        red = np.stack([s.as_array()[:3] for s in self.red])
        blue = np.stack([s.as_array()[:3] for s in self.blue])
        return np.linalg.norm(red[:,None,:]-blue[None,:,:],axis=2).min(axis=1)

    def _validate_states(self):
        if len(self.red)!=4 or len(self.blue)!=4:
            raise RuntimeError("pursuit_v2 requires four Red and four opponents")
        if not all(s.alive for s in self.red+self.blue):
            raise RuntimeError("pursuit_v2 aircraft must stay alive; loss semantics forbidden")
        if not all(np.isfinite(s.as_array()).all() for s in self.red+self.blue):
            raise FloatingPointError("nonfinite pursuit aircraft state")

    def step(self, actions):
        if self._episode_done:
            raise RuntimeError("episode done; reset required")
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (4,3) or not np.isfinite(actions).all():
            raise ValueError("actions must be finite with shape (4,3)")
        actions = np.clip(actions, -1, 1)
        self._validate_states()
        alive_before = self.red_alive_mask.astype(bool)
        self.opponent_actions = self.fixed_policy.team_actions(self.blue,self.rng,self.steps)
        self.red_last_executed_phi = self._advance(self.red, actions)
        self.blue_last_executed_phi = self._advance(self.blue, self.opponent_actions)
        self.steps += 1
        for side, states in (("red", self.red), ("opponent", self.blue)):
            for kind,mask in (("ground",np.array([s.altitude<=0 for s in states])),
                              ("boundary",np.array([np.hypot(s.x,s.y)>self.arena_radius for s in states]))):
                key = f"{side}_{kind}_violation_steps"
                self.violation_steps[key] += int(mask.sum())
                self.violation_masks[key] |= mask
        distances = self._distances()
        reward = -distances * self.config["reward"]["distance_reward_scale"]
        valid = self.red_alive_mask.astype(bool) & bool(self.blue_alive_mask.any())
        if valid.any():
            self.last_valid_distances[valid] = distances[valid]
            self.distance_sum += distances[valid].sum()
            self.distance_count += int(valid.sum())
            self.distance_min = min(self.distance_min, distances[valid].min())
        for index in np.flatnonzero(alive_before):
            s = self.red[index]
            self.state_count += 1
            self.speed_sum += s.v
            self.clamp_counts += [s.v == self.spec.v_min, s.v == self.spec.v_max,
                                  s.theta == self.spec.theta_min, s.theta == self.spec.theta_max]
            self.control_sum += np.abs(actions[index]) * [2,5,np.pi/3]
            self.control_near += np.abs(actions[index]) >= self.config["diagnostics"]["near_limit_fraction"]
        terminated = False
        truncated = self.steps == self.max_steps
        self._episode_done = terminated or truncated
        return self._observations(), reward.astype(np.float32), bool(terminated), bool(truncated), self._info(distances,reward)

    def _info(self, distances, reward):
        mask = self.red_alive_mask.astype(bool) & bool(self.blue_alive_mask.any())
        count = max(self.state_count,1)
        return {"environment_version":"2.4", "environment_variant":"pursuit_v2",
            "task_type":"pursuit", "action_semantics":"direct_physical_control",
            "episode_length":self.steps, "red_alive_mask":self.red_alive_mask,
            "blue_alive_mask":self.blue_alive_mask, "raw_nearest_distance_m":distances.copy(),
            "pursuit_reward":reward.copy(), "nearest_distance_valid_mask":mask,
            "mean_nearest_distance":self.distance_sum/max(self.distance_count,1),
            "final_nearest_distance":float(self.last_valid_distances.mean()),
            "minimum_nearest_distance":self.distance_min if self.distance_count else 0.,
            **self.violation_steps,
            **{key.replace("_steps","_agents"):int(mask.sum()) for key,mask in self.violation_masks.items()},
            **{key.replace("_steps","_rate"):value/max(4*self.steps,1) for key,value in self.violation_steps.items()},
            "mean_speed":self.speed_sum/count,
            **{name:float(value/count) for name,value in zip(
                ("speed_min_hit_ratio","speed_max_hit_ratio","pitch_min_hit_ratio","pitch_max_hit_ratio"),self.clamp_counts)},
            **{f"mean_abs_{name}":float(value/count) for name,value in zip(("T","L","beta"),self.control_sum)},
            **{f"fraction_abs_{name}_near_limit":float(value/count) for name,value in zip(("T","L","beta"),self.control_near)}}

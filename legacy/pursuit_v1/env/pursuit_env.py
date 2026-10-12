"""Independent weapon-free pursuit task with raw negative-distance rewards."""
from pathlib import Path
import numpy as np
from .config import load_config, aircraft_spec
from .direct_protocol import validate_direct_action_config
from .models import AircraftState
from .dynamics import PointMassDynamics
from .integrator import RK4Integrator
from .combat_env import MultiUAVCombatEnv


def validate_pursuit_config(config):
    validate_direct_action_config(config)
    if (config.get("environment_version"), config.get("environment_variant"), config.get("task_type")) != ("2.4", "pursuit_v1", "pursuit"):
        raise RuntimeError("pursuit requires 2.4/pursuit_v1/pursuit identity")
    if config["simulation"] != {"dt": .1, "max_steps": 1000} or config["scenario"]["team_size"] != 4:
        raise RuntimeError("pursuit requires 4 agents, dt=.1, project horizon=1000")
    from .observation import observation_dim_from_config
    if observation_dim_from_config(config["observation"]) != 52:
        raise RuntimeError("pursuit requires observation_dim=52")
    if any(key in config for key in ("weapon", "attack")):
        raise RuntimeError("pursuit must have no weapon or attack configuration")
    if set(config["reward"]) != {"distance_reward_scale"} or not np.isfinite(config["reward"]["distance_reward_scale"]) or config["reward"]["distance_reward_scale"] <= 0:
        raise RuntimeError("pursuit reward must be negative distance with explicit positive scale")
    s = config["scenario"]
    if not 0 < s["initial_radius_m"] < config["arena"]["radius"] or not 0 < s["altitude_min_m"] <= s["altitude_max_m"]:
        raise ValueError("illegal pursuit initial position range")
    a = config["aircraft"]
    if not a["v_min"] <= s["speed_min_m_s"] <= s["speed_max_m_s"] <= a["v_max"] or not 0 <= s["pitch_abs_max_rad"] <= min(-a["theta_min"], a["theta_max"]):
        raise ValueError("illegal pursuit initial state envelope")
    p = config["opponent_policy"]
    if p["type"] != "seeded_random_direct" or not isinstance(p["maneuver_interval_steps"], int) or p["maneuver_interval_steps"] <= 0:
        raise ValueError("invalid random opponent maneuver interval")
    for key in ("normalized_thrust_range", "normalized_lift_range", "normalized_roll_range"):
        if not -1 <= p[key][0] <= p[key][1] <= 1:
            raise ValueError("opponent control ranges must be normalized")
    if not 0 < config["diagnostics"]["near_limit_fraction"] <= 1:
        raise ValueError("invalid diagnostic near-limit threshold")


class SeededRandomDirectPolicy:
    def __init__(self, config):
        self.config = config

    def team_actions(self, rng):
        return np.stack([rng.uniform(*self.config[key], size=4) for key in
                         ("normalized_thrust_range", "normalized_lift_range", "normalized_roll_range")], axis=1)


class PursuitEnv:
    """Shares only integration and observation helpers with combat, never attacks."""
    team_size, observation_dim, action_dim = 4, 52, 3
    environment_version, environment_variant = "2.4", "pursuit_v1"
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
        self.fixed_policy = SeededRandomDirectPolicy(self.config["opponent_policy"])
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
        self.ground_loss_count, self.boundary_loss_count = 0, 0
        self.blue_ground_loss_count, self.blue_boundary_loss_count = 0, 0
        self.opponent_actions = np.zeros((4,3))
        self.last_valid_distances = self._distances()
        return self._observations(), self._info(self._distances(), np.zeros(4))

    def _distances(self):
        targets = [s.as_array()[:3] for s in self.blue if s.alive]
        return np.asarray([min(np.linalg.norm(s.as_array()[:3]-t) for t in targets)
                           if s.alive and targets else 0. for s in self.red], dtype=float)

    def step(self, actions):
        if self._episode_done:
            raise RuntimeError("episode done; reset required")
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (4,3) or not np.isfinite(actions).all():
            raise ValueError("actions must be finite with shape (4,3)")
        actions = np.clip(actions, -1, 1)
        alive_before = self.red_alive_mask.astype(bool)
        actions[~alive_before] = 0
        if self.steps % self.fixed_policy.config["maneuver_interval_steps"] == 0:
            self.opponent_actions = self.fixed_policy.team_actions(self.rng)
        self.red_last_executed_phi = self._advance(self.red, actions)
        self.blue_last_executed_phi = self._advance(self.blue, self.opponent_actions)
        self.steps += 1
        for side, states in (("red", self.red), ("blue", self.blue)):
            for state in states:
                if state.alive:
                    ground = state.altitude <= 0
                    boundary = np.hypot(state.x,state.y) > self.arena_radius
                    if ground or boundary:
                        state.alive = False
                        prefix = "" if side == "red" else "blue_"
                        field = prefix + ("ground_loss_count" if ground else "boundary_loss_count")
                        setattr(self, field, getattr(self, field)+1)
        distances = self._distances()
        # Dead UAVs receive zero under the existing alive-mask architecture.
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
        terminated = not self.red_alive_mask.any() or not self.blue_alive_mask.any()
        truncated = not terminated and self.steps >= self.max_steps
        self._episode_done = terminated or truncated
        return self._observations(), reward.astype(np.float32), bool(terminated), bool(truncated), self._info(distances,reward)

    def _info(self, distances, reward):
        mask = self.red_alive_mask.astype(bool) & bool(self.blue_alive_mask.any())
        count = max(self.state_count,1)
        return {"environment_version":"2.4", "environment_variant":"pursuit_v1",
            "task_type":"pursuit", "action_semantics":"direct_physical_control",
            "episode_length":self.steps, "red_alive_mask":self.red_alive_mask,
            "blue_alive_mask":self.blue_alive_mask, "raw_nearest_distance_m":distances.copy(),
            "pursuit_reward":reward.copy(), "nearest_distance_valid_mask":mask,
            "mean_nearest_distance":self.distance_sum/max(self.distance_count,1),
            "final_nearest_distance":float(self.last_valid_distances.mean()),
            "minimum_nearest_distance":self.distance_min if self.distance_count else 0.,
            "ground_loss_count":self.ground_loss_count, "boundary_loss_count":self.boundary_loss_count,
            "blue_ground_loss_count":self.blue_ground_loss_count, "blue_boundary_loss_count":self.blue_boundary_loss_count,
            "mean_speed":self.speed_sum/count,
            **{name:float(value/count) for name,value in zip(
                ("speed_min_hit_ratio","speed_max_hit_ratio","pitch_min_hit_ratio","pitch_max_hit_ratio"),self.clamp_counts)},
            **{f"mean_abs_{name}":float(value/count) for name,value in zip(("T","L","beta"),self.control_sum)},
            **{f"fraction_abs_{name}_near_limit":float(value/count) for name,value in zip(("T","L","beta"),self.control_near)}}

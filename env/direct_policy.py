"""Deterministic Blue guidance; inverse guidance belongs only to this opponent."""
import numpy as np
from .fixed_policy import GroundAwareNearestTargetPursuitPolicy, NearestTargetPursuitPolicy
from .math_utils import wrap_angle
from .dynamics import PointMassDynamics


class DirectPursuitPolicy(GroundAwareNearestTargetPursuitPolicy):
    """Reuse diagnostic accounting, with explicit physical guidance and recovery."""
    def __init__(self, config, action_config=None, aircraft_config=None):
        self.config = config
        self.gravity = PointMassDynamics().gravity
        self.recovery_time_seconds = float(config["ground_avoidance"]["recovery_time_seconds"])
        self.downward_speed_epsilon = float(config["ground_avoidance"]["downward_speed_epsilon"])
        for key in ("speed_response_time_seconds", "heading_response_time_seconds",
                    "pitch_response_time_seconds", "max_heading_rate_rad_s", "max_pitch_rate_rad_s"):
            if not np.isfinite(config[key]) or config[key] <= 0:
                raise ValueError(f"blue_policy.{key} must be finite and positive")
        if not np.isfinite(self.recovery_time_seconds) or self.recovery_time_seconds <= 0 or not 0 < config["ground_avoidance"]["recovery_lift"] <= 5:
            raise ValueError("invalid ground recovery parameters")
        if not np.isfinite(self.downward_speed_epsilon) or self.downward_speed_epsilon < 0:
            raise ValueError("downward_speed_epsilon must be finite and nonnegative")
        if config["desired_speed"] != 250:
            raise ValueError("Blue desired_speed must remain 250 m/s")
        self.reset_diagnostics()

    def ground_risk(self, own):
        downward = max(-own.v * np.sin(own.theta), 0.0)
        if downward <= self.downward_speed_epsilon:
            return False, None
        time = own.altitude / downward
        # Include the exact threshold despite one-ULP division/trigonometric rounding.
        return bool(time <= np.nextafter(self.recovery_time_seconds, np.inf)), float(time)

    def action_toward(self, own, desired_heading, desired_elevation, desired_speed):
        c = self.config
        thrust = np.sin(own.theta) + (desired_speed - own.v) / (c["speed_response_time_seconds"] * self.gravity)
        heading_rate = np.clip(wrap_angle(desired_heading - own.psi) / c["heading_response_time_seconds"],
                               -c["max_heading_rate_rad_s"], c["max_heading_rate_rad_s"])
        pitch_rate = np.clip((desired_elevation - own.theta) / c["pitch_response_time_seconds"],
                             -c["max_pitch_rate_rad_s"], c["max_pitch_rate_rad_s"])
        vertical = np.cos(own.theta) + own.v * pitch_rate / self.gravity
        lateral = own.v * np.cos(own.theta) * heading_rate / self.gravity
        # Physical roll limit may prevent exact requested rates; keep controls legal.
        roll = np.clip(np.arctan2(lateral, vertical), -np.pi/3, np.pi/3)
        lift = np.clip(vertical * np.cos(roll) + lateral * np.sin(roll), -5, 5)
        return np.asarray([np.clip(thrust, -2, 2)/2, lift/5, roll/(np.pi/3)], dtype=np.float32)

    def _action_and_override(self, own, targets):
        if not own.alive:
            return np.zeros(3, dtype=np.float32), False
        index = NearestTargetPursuitPolicy.nearest_target_index(own, targets)
        if index is None:
            heading, pitch = own.psi, 0.0
        else:
            target = targets[index]
            dx, dy = target.x-own.x, target.y-own.y
            heading = np.arctan2(dy, dx)
            pitch = np.arctan2(own.z-target.z, np.hypot(dx, dy))
        action = self.action_toward(own, heading, pitch, self.config["desired_speed"])
        override, _ = self.ground_risk(own)
        if override:
            action[1] = self.config["ground_avoidance"]["recovery_lift"] / 5.0
            action[2] = 0.0
        return action, override

    def action(self, own, targets):
        return self._action_and_override(own, targets)[0]

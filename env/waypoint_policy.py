"""Episode-seeded waypoint opponent; all movement uses the direct flight model."""
import numpy as np
from .direct_policy import DirectPursuitPolicy


class SeededRandomWaypointDirectPolicy:
    # Reuse exactly the combat Blue physical guidance and commanded-descent guard.
    action_toward = DirectPursuitPolicy.action_toward
    ground_risk = DirectPursuitPolicy.ground_risk

    def __init__(self, config):
        self.config = config
        # Validate the shared physical guidance using a temporary combat-compatible speed.
        guidance = DirectPursuitPolicy({**config, "desired_speed":250.})
        self.gravity = guidance.gravity
        self.recovery_time_seconds = guidance.recovery_time_seconds
        self.downward_speed_epsilon = guidance.downward_speed_epsilon
        self.waypoints = np.zeros((4,3))
        self.desired_speeds = np.zeros(4)

    def reset(self):
        self.waypoints.fill(0.)
        self.desired_speeds.fill(0.)
        self.update_count = 0
        self.last_boundary_override_mask = np.zeros(4,dtype=bool)

    def team_actions(self, team, rng, step):
        c = self.config
        if step % c["waypoint_interval_steps"] == 0:
            radii = c["waypoint_radius_m"] * np.sqrt(rng.random(4))
            angles = rng.uniform(-np.pi,np.pi,4)
            self.waypoints[:,0] = radii*np.cos(angles)
            self.waypoints[:,1] = radii*np.sin(angles)
            self.waypoints[:,2] = rng.uniform(c["waypoint_altitude_min_m"],c["waypoint_altitude_max_m"],4)
            self.desired_speeds[:] = rng.uniform(c["desired_speed_min_m_s"],c["desired_speed_max_m_s"],4)
            self.update_count += 1
        rows=[]
        safety=c["boundary_avoidance"]
        for index,own in enumerate(team):
            x,y,altitude = self.waypoints[index]
            speed = self.desired_speeds[index]
            override = np.hypot(own.x,own.y) >= safety["activation_radius_m"]
            self.last_boundary_override_mask[index] = override
            if override:
                x,y,altitude,speed = 0.,0.,safety["recovery_altitude_m"],safety["desired_speed_m_s"]
            dx,dy = x-own.x,y-own.y
            heading = np.arctan2(dy,dx)
            pitch = np.arctan2(altitude-own.altitude,np.hypot(dx,dy))
            action = self.action_toward(own,heading,pitch,speed)
            risk,_ = self.ground_risk(own,pitch)
            if risk:
                action[1] = c["ground_avoidance"]["recovery_lift"]/5
                action[2] = 0.
            rows.append(action)
        return np.stack(rows)

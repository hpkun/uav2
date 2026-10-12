"""Pursuit-only diagnostics; no fabricated combat outcomes or reward terms."""
import numpy as np

DIAGNOSTICS = ("mean_nearest_distance", "final_nearest_distance", "minimum_nearest_distance",
    "red_ground_violation_steps", "red_boundary_violation_steps",
    "opponent_ground_violation_steps", "opponent_boundary_violation_steps",
    "red_ground_violation_agents", "red_boundary_violation_agents",
    "opponent_ground_violation_agents", "opponent_boundary_violation_agents",
    "red_ground_violation_rate", "red_boundary_violation_rate",
    "opponent_ground_violation_rate", "opponent_boundary_violation_rate",
    "mean_speed", "speed_min_hit_ratio", "speed_max_hit_ratio", "pitch_min_hit_ratio", "pitch_max_hit_ratio",
    "mean_abs_T", "mean_abs_L", "mean_abs_beta", "fraction_abs_T_near_limit",
    "fraction_abs_L_near_limit", "fraction_abs_beta_near_limit")


def aggregate_pursuit_records(records):
    if not records:
        raise ValueError("pursuit evaluation records must not be empty")
    mean = lambda key: float(np.mean([r[key] for r in records]))
    return {"task_type":"pursuit", "average_return":mean("episode_return"),
        "average_agent_return":mean("mean_agent_episode_return"),
        "average_episode_length":mean("episode_length"), "evaluation_episodes":len(records),
        **{key:mean(key) for key in DIAGNOSTICS}}


def pursuit_step_metrics(result, sampled_steps, last_metrics):
    return {"sampled_steps":sampled_steps, "task_type":"pursuit",
        "mean_step_reward":float(np.mean(result.rewards)),
        "raw_nearest_distance_m":[info["raw_nearest_distance_m"].tolist() for info in result.infos],
        "pursuit_reward":[info["pursuit_reward"].tolist() for info in result.infos],
        "nearest_distance_valid_mask":[info["nearest_distance_valid_mask"].tolist() for info in result.infos],
        **{key:float(np.mean([info[key] for info in result.infos])) for key in DIAGNOSTICS},
        **last_metrics}

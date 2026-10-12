"""Strict direct-control action identity shared by both tasks."""
import math


def validate_direct_action_config(config):
    expected = {"mode": "direct_physical_control", "thrust": {"min": -2., "max": 2.},
                "lift": {"min": -5., "max": 5.},
                "roll": {"min": -math.pi/3, "max": math.pi/3}}
    if config.get("action") != expected or config.get("action_semantics") != "direct_physical_control":
        raise ValueError("direct_physical_control requires exact T/L/beta ranges; legacy controller incompatible")

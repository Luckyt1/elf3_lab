"""Shared GL-165 disturbance settings for Isaac training and native playback."""

VELOCITY_ONLY = "root_velocity_delta"
MIXED_PUSH = "root_velocity_delta_or_torso_force"


def torso_force_parameters(cfg, iteration):
    progress = min(1., max(0., iteration / cfg.full_iteration))
    force = cfg.initial_force_n + progress * (cfg.final_force_n - cfg.initial_force_n)
    duration = cfg.initial_duration_s + progress * (cfg.final_duration_s - cfg.initial_duration_s)
    return force, duration

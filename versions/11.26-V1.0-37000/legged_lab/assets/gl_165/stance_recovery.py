"""Observable GL-165 standing-reference transitions, shared by Isaac and the exported actor."""

import math


DEFAULT_STANCE_RECOVERY = {
    "enter_speed_m_s": 0.35,
    "step_speed_m_s": 0.16,
    "enter_tilt_rad": 0.15,
    "enter_tilt_rate_rad_s": 0.7,
    "settle_speed_m_s": 0.12,
    "settle_tilt_rad": 0.13,
    "settle_tilt_rate_rad_s": 0.45,
    "foot_load_n": 40.0,
    "settle_s": 0.4,
    "release_grace_s": 2.0,
}


def stance_is_quiet(cfg, bilateral, speed_sq, tilt_cos, tilt_rate_sq):
    return (bilateral & (speed_sq <= cfg["settle_speed_m_s"]**2)
            & (tilt_cos >= math.cos(cfg["settle_tilt_rad"]))
            & (tilt_rate_sq <= cfg["settle_tilt_rate_rad_s"]**2))


def advance_stance_reference(cfg, dt, recovering, hold_ticks, stable_ticks, external,
                             speed_sq, tilt_cos, tilt_rate_sq, left_load, right_load):
    """Advance once per control tick. Operates elementwise on scalar or batched signals."""
    bilateral = (left_load >= cfg["foot_load_n"]) & (right_load >= cfg["foot_load_n"])
    uneven = (bilateral == False) | (tilt_cos < math.cos(cfg["enter_tilt_rad"]))
    destabilized = (speed_sq >= cfg["enter_speed_m_s"]**2) | (
        uneven & ((speed_sq >= cfg["step_speed_m_s"]**2) |
                  (tilt_rate_sq >= cfg["enter_tilt_rate_rad_s"]**2)))
    trigger = external | destabilized
    active = recovering | trigger
    refresh = external | (destabilized & (recovering == False))
    hold = (hold_ticks - 1) * (hold_ticks > 0)
    hold = hold + (round(cfg["release_grace_s"] / dt) - hold) * refresh
    quiet = stance_is_quiet(cfg, bilateral, speed_sq, tilt_cos, tilt_rate_sq)
    stable = (stable_ticks + 1) * (active & quiet & (external == False))
    relock = active & (hold == 0) & (stable >= max(1, math.ceil(cfg["settle_s"] / dt)))
    return active & (relock == False), hold, stable * (relock == False), relock


def advance_foot_reference(gathering, stable_ticks, pelvis_recovering, pelvis_relocked,
                           quiet_unforced, gap_in_band, settle_steps):
    """Count consecutive in-band double-support frames and lock the two footprints once."""
    pending = (gathering | pelvis_relocked) & (pelvis_recovering == False)
    stable = (stable_ticks + 1) * ((pelvis_recovering | pending) & quiet_unforced & gap_in_band)
    relock = pending & (stable >= settle_steps)
    return pending & (relock == False), stable * (relock == False), relock


def advance_gap_timer(outside_ticks, push_recovery, pushed, recovering, in_range):
    """Suspend the standing band through a real push and its recovery. A voluntary step keeps the timer active."""
    push_recovery = pushed | (push_recovery & recovering)
    active = ~push_recovery
    outside_ticks = (outside_ticks + 1) * (active & ~in_range)
    return outside_ticks, push_recovery, active

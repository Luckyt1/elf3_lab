"""Exercise the GL-165 quiet stance rules in a real vectorized Isaac scene."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from legged_lab.assets.gl_165.disturbances import MIXED_PUSH, VELOCITY_ONLY
from legged_lab.envs.gl_165.stand_cfg import Gl165StandEnvCfg, TorsoForceCfg
from legged_lab.envs.gl_165.stand_env import Gl165StandEnv


@torch.inference_mode()
def check():
    cfg = Gl165StandEnvCfg()
    cfg.scene.num_envs = 16
    cfg.scene.seed = 42
    env = Gl165StandEnv(cfg, headless=True)
    env.learning_iteration = 2000
    assert env.no_push_env_count == 4 and env.max_episode_length == 6000
    assert env.get_observations()[0].shape == (16, 540)
    names = env.reward_manager.active_terms
    expected = {"feet_drift": -10., "feet_slide": -.5, "support": 1., "feet_clearance": -1.,
                "quiet_sole_gap": -4., "quiet_foot_stagger": -4., "quiet_foot_slide": -2., "quiet_target_curvature": -8.,
                "torso_sway": -1.}
    for name, weight in expected.items():
        assert env.reward_manager.get_term_cfg(name).weight == weight
    assert not {"feet_width", "feet_inward_slip", "feet_stagger"}.intersection(names)
    actions = torch.zeros(16, 15, device=env.device)
    peaks = torch.zeros(16, device=env.device)
    push = env.event_manager.get_term_cfg("push_robot")
    assert cfg.push.mode == VELOCITY_ONLY and cfg.push.torso_force is None and env.torso_force is None
    assert push.interval_range_s == (6., 12.) and not push.is_global_time
    assert not cfg.push.curriculum_enabled and env.push_scale() == 1.
    for _ in range(20):
        env.step(actions)
        error = (env.arm_motion.target - env.arm_motion.nominal).abs().amax(dim=-1)
        peaks = torch.maximum(peaks, error)
    assert peaks[:2].eq(0).all() and peaks[2:4].gt(0).all()
    env.push_applied.zero_()
    ids = torch.arange(16, device=env.device)
    before = env.robot.data.root_vel_w.clone()
    push.func(env, ids)
    received = env.push_applied.clone()
    assert not received[:4].any() and received[4:].all()
    torch.testing.assert_close(env.robot.data.root_vel_w - before, env.push_delta_w)
    assert env.push_delta_w[:4].count_nonzero() == 0 and env.push_delta_w[:, 2:].count_nonzero() == 0
    xy = env.push_delta_w[4:, :2]
    assert xy.abs().max() <= .80001 and (xy.abs().amax(dim=0) > .4).all()
    env.reset(ids)
    env.push_applied.zero_()
    env.push_applied[1] = True
    gaps = torch.full((16,), .08, device=env.device)
    gaps[:2] = .15
    gaps[2] = .02
    stagger = torch.full_like(gaps, .06)
    with patch("legged_lab.envs.gl_165.stand_env.sole_inner_gap", return_value=gaps), \
         patch("legged_lab.envs.gl_165.stand_env.signed_sole_stagger", return_value=stagger):
        for tick in range(1, 101):
            env.sim_step_counter += cfg.sim.decimation
            done, timeout = env.check_reset()
            assert env.gap_outside_ticks[:2].tolist() == [tick, 0]
            assert env.failure_reasons["foot_gap"][0].item() == (tick == 100)
        assert done[0] and not timeout[0] and not env.failure_reasons["foot_gap"][1]
        env.reward_manager.compute(env.step_dt)
        values = env.reward_manager._step_reward * env.step_dt
        gap, align = (values[:, names.index(k)] for k in ("quiet_sole_gap", "quiet_foot_stagger"))
        torch.testing.assert_close(gap[:2], gap.new_tensor([-.04, 0.]))
        torch.testing.assert_close(align[:2], align.new_tensor([-.04, 0.]))
        assert values[2, names.index("feet_clearance")] < 0
        fz = env.contact_sensor.data.net_forces_w[:, env.feet_cfg.body_ids, 2]
        support = .02 * ((fz > 150.).all(dim=1) & env.feet_should_be_planted)
        torch.testing.assert_close(values[:, names.index("support")], support)
        env.push_applied.zero_()
        env.sim_step_counter += cfg.sim.decimation
        env.check_reset()
        assert not env.quiet_gap_active[1] and env.gap_outside_ticks[1] == 0
    env.reset(ids[:1])
    assert env.gap_outside_ticks[0] == 0 and not env.planted[0].any() and not env.quiet_stance_valid[0]
    saved = env.controller_parameters
    restored = Gl165StandEnvCfg()
    restored.restore_push_profile({"push_mode": MIXED_PUSH, "torso_force_curriculum": TorsoForceCfg().to_dict()})
    assert restored.push.interval_s == (8., 12.) and restored.push.curriculum_enabled
    assert restored.push.velocity_ranges["yaw"] == (-.2, .2) and restored.push.torso_force.final_duration_s == .8
    restored.restore_standing_profile(saved)
    assert restored.push.velocity_profile() == saved["velocity_push"] == cfg.push.velocity_profile()
    assert restored.push.mode == VELOCITY_ONLY and restored.push.torso_force is None
    assert saved["nominal_sole_gap_m"] == .08 and restored.nominal_stance == cfg.nominal_stance
    assert restored.reward.torso_sway.params == cfg.reward.torso_sway.params
    assert restored.quiet_foot_stance == cfg.quiet_foot_stance and restored.interpolate_arm_targets
    assert restored.scene.max_episode_length_s == 120 and restored.reward.support.weight == 1.
    # Advance the actual per-environment event timers for 30 s without integrating the scene.
    env.reset(ids)
    times = [[] for _ in range(16)]
    deltas = []
    for tick in range(1, 1501):
        env.push_applied.zero_()
        env.event_manager.apply(mode="interval", dt=env.step_dt)
        active = env.push_applied.nonzero().flatten()
        if active.numel():
            for index in active.tolist():
                times[index].append(tick * env.step_dt)
            deltas.append(env.push_delta_w[active].clone())
    intervals = [b-a for events in times[4:] for a, b in zip(events, events[1:])]
    samples = torch.cat(deltas)
    assert all(not events for events in times[:4]) and all(len(events) >= 2 for events in times[4:])
    assert all(6.-1e-4 <= s <= 12. + env.step_dt for s in intervals)
    assert len(set(round(events[0], 2) for events in times[4:])) > 1
    assert samples[:, 2:].count_nonzero() == 0 and samples[:, :2].abs().max() <= .80001
    return dict(passed=True, tested_envs=16, fixed_arms=2, moving_arms_without_push=2,
                velocity_push=saved["velocity_push"], actual_velocity_events=len(samples),
                observed_intervals_s=[min(intervals), max(intervals)],
                max_abs_delta_xy_m_s=samples[:, :2].abs().amax(dim=0).tolist(),
                max_abs_delta_z_and_angular=samples[:, 2:].abs().max().item(),
                pushed_envs=int(received.sum()), gap_failure_tick=100, quiet_fore_aft_step_at_6cm=-.04,
                quiet_gap_step_at_15cm=-.04, enabled_weights=expected)


try:
    result = check()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
finally:
    app.close(wait_for_replicator=False)

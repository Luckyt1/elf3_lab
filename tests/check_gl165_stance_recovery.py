"""Exercise the full Isaac observation, reward and termination paths for disturbed standing."""
import argparse
import json
from pathlib import Path
import traceback
from types import SimpleNamespace
from unittest.mock import patch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--balancing-actor", type=Path, required=True, help="A 310-input actor for the settled test fixture")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from isaaclab.managers import RewardTermCfg
from legged_lab.assets.gl_165.model import LEGACY_PHYSICS_PROFILE
from legged_lab.assets.gl_165.disturbances import MIXED_PUSH
from legged_lab.envs.gl_165 import rewards
from legged_lab.envs.gl_165.stand_cfg import Gl165StandEnvCfg, TorsoForceCfg
from legged_lab.envs.gl_165.stand_env import Gl165StandEnv


def prior_load_profile(cfg):
    """Configuration used by the 13:59 run, before the quiet stance rules."""
    cfg.restore_standing_profile({
        "physics_profile": cfg.physics_profile, "stance_recovery": cfg.stance_recovery,
        "observe_arm_motion": False, "observe_standing_state": False,
        "push_mode": MIXED_PUSH, "torso_force_curriculum": TorsoForceCfg().to_dict(),
        "min_foot_clearance_m": .05, "foot_clearance_reward_weight": -25., "min_reset_foot_clearance_m": .0581,
        "feet_drift_reward_weight": 0., "loaded_sole_slide": {"weight": -.5, "foot_load_n": 40.},
        "quiet_foot_contact": dict(weight=-2., post_push_recovery_s=2., **cfg.reward.feet_lift.params),
        "foot_load_shortfall": {"minimum_n": 150., "weight": -1., "post_push_recovery_s": 2.},
        "foot_gathering": {"minimum_m": .05, "maximum_m": .12, "scale_m": .10, "weight": -1.5,
                           "pelvis_tolerance_m": .10, "all_phases": True},
    })


def check_clearance_profiles():
    cfg = Gl165StandEnvCfg()
    assert cfg.quiet_foot_stance["minimum_m"] == .0581 and cfg.scene.max_episode_length_s == 120.
    assert cfg.reward.support.func is rewards.both_feet_loaded and cfg.reward.quiet_foot_stagger.weight == -4.
    assert cfg.reward.quiet_target_curvature.weight == -8. and cfg.interpolate_arm_targets
    prior_load_profile(cfg)
    assert cfg.quiet_foot_stance is None and cfg.reward.quiet_sole_gap is None and cfg.scene.max_episode_length_s == 20.
    assert cfg.reward.quiet_target_curvature is None and not cfg.interpolate_arm_targets
    support = cfg.reward.support
    assert support.func is rewards.foot_load_shortfall and support.weight == -1. and support.params == {"minimum_n": 150.}
    recovery = cfg.stance_recovery
    assert cfg.reward.feet_drift is None and cfg.reward.feet_slide.weight == -.5
    upper = cfg.reward.feet_width
    assert upper.params == {"maximum_m": .12, "scale_m": .10, "all_phases": True}
    assert upper.weight == -1.5
    sample = SimpleNamespace(sole_gap_m=torch.tensor([.05, .12, .17, .24, .24]),
                             recovering=torch.tensor([False, False, False, False, True]))
    step_dt = cfg.sim.dt * cfg.sim.decimation
    torch.testing.assert_close(upper.func(sample, **upper.params) * upper.weight * step_dt,
                               torch.tensor([0., 0., -.00375, -.021, -.021]))
    cfg.set_max_foot_clearance(0.)
    assert cfg.reward.feet_width is None
    gaps = SimpleNamespace(sole_gap_m=torch.tensor([.04, .045, .049, .05, .0581]))
    new_cost = torch.tensor([-.02, -.005, -.0002, 0., 0.])
    legacy = {"min_foot_clearance_m": .05}
    for profile, expected in ((None, new_cost), (legacy, new_cost / 25),
                              ({**legacy, "foot_clearance_reward_weight": -25.}, new_cost)):
        if profile is not None:
            cfg.restore_standing_profile(profile)
        cfg.set_min_foot_clearance(.05)
        term = cfg.reward.feet_clearance
        step_dt = cfg.sim.dt * cfg.sim.decimation
        torch.testing.assert_close(term.func(gaps, **term.params) * term.weight * step_dt, expected, atol=1e-8, rtol=1e-5)
    cfg.restore_standing_profile({})
    assert cfg.reward.support.func is rewards.both_feet_loaded and cfg.reward.support.weight == 1.
    cfg.restore_standing_profile({"foot_load_shortfall": {"minimum_n": 120., "weight": -.5, "post_push_recovery_s": 1.0}})
    assert cfg.reward.support.func is rewards.foot_load_shortfall and cfg.reward.support.weight == -.5
    assert cfg.reward.support.params == {"minimum_n": 120.} and cfg.foot_recovery_s == 1.0
    assert cfg.reward.feet_clearance is None
    assert cfg.reward.feet_width is None
    assert cfg.reward.feet_drift.weight == -10.
    cfg.restore_standing_profile({"feet_drift_reward_weight": 0.})
    assert cfg.reward.feet_drift is None and cfg.reward.feet_slide.weight == -.5
    gathered = {"maximum_m": .12, "scale_m": .10, "weight": -1., "pelvis_tolerance_m": .10}
    cfg.restore_standing_profile({"stance_recovery": recovery, "foot_gathering": gathered})
    historical = cfg.reward.feet_width
    assert historical.weight == -1.
    assert historical.func(sample, **historical.params)[-1] == 0
    cfg.set_max_foot_clearance(.12)
    assert cfg.reward.feet_width.params == upper.params
    cfg.restore_standing_profile({"stance_recovery": recovery, "foot_gathering": {**gathered, "all_phases": True}})
    assert cfg.reward.feet_width.params == upper.params
    assert cfg.reward.feet_width.weight == -1.
    cfg.set_max_foot_clearance_weight(-1.5)
    assert cfg.reward.feet_width.weight == upper.weight and cfg.reward.feet_width.params == upper.params


def check():
    check_clearance_profiles()
    cfg = Gl165StandEnvCfg()
    prior_load_profile(cfg)
    assert cfg.push.torso_force.final_duration_s == .8
    historical_push = {**cfg.push.torso_force.to_dict(), "final_duration_s": 1.5}
    cfg.restore_push_profile({"push_mode": cfg.push.mode, "torso_force_curriculum": historical_push})
    assert cfg.push.torso_force.final_duration_s == 1.5
    cfg.set_final_torso_force_duration(.8)
    cfg.set_actor_observations(standing_state=True)
    cfg.set_physics_profile(LEGACY_PHYSICS_PROFILE)
    cfg.arm_motion.database = str(Path(cfg.arm_motion.database).with_name("arm_disturbance_overhead_xml_5ms_v5.npz"))
    # This balancing actor uses the legacy assets and a separate footprint reference.
    cfg.reward.feet_drift = RewardTermCfg(func=rewards.feet_drift, weight=-10.)
    cfg.device, cfg.scene.num_envs, cfg.scene.seed = args.device, 5, 42
    cfg.arm_motion.enabled = False
    env = Gl165StandEnv(cfg, headless=True)
    assert env.controller_parameters["foot_load_shortfall"] == {"minimum_n": 150., "weight": -1., "post_push_recovery_s": 2.}
    support_term = env.reward_manager.active_terms.index("support")
    actor = torch.jit.load(str(args.balancing_actor), map_location=env.device).eval()
    obs, _ = env.get_observations()
    # Settle onto real contacts, then displace the reference as if the feet have stepped.
    for _ in range(80):
        obs, _, _, _ = env.step(actor(obs))
    origin = env.robot.data.root_pos_w[:, :2].clone()
    env.stand_anchor_xy[:, 0] = origin[:, 0] - .75
    env.feet_anchor_xy[:, :, 0] -= .75
    old_yaw = env.stand_anchor_yaw.clone()
    env.recovering.zero_()
    env.recovery_hold.zero_()
    env.recovery_stable.zero_()
    env.push_applied.zero_()
    env.push_applied[0] = True
    ids = torch.tensor([1], device=env.device)
    env.torso_force.start(ids, cfg.push.torso_force.final_force_n, cfg.push.torso_force.final_duration_s)
    assert env.torso_force.remaining[1] == 160
    env.robot.write_root_velocity_to_sim(
        torch.tensor([[.55, 0., 0., 0., 0., 0.]], device=env.device), torch.tensor([2], device=env.device))
    env.check_reset()
    assert env.recovering[:3].all() and not env.recovering[3:].any(), (
        env.recovering, env.robot.data.root_lin_vel_w, env.robot.data.projected_gravity_b)
    assert not env.failure_reasons["drift"][:3].any() and env.failure_reasons["drift"][3:].all()
    assert not env.feet_should_be_planted[:3].any()
    assert rewards.keep_position(env)[:3].count_nonzero() == 0
    assert rewards.feet_drift(env)[:3].count_nonzero() == 0 and rewards.feet_drift(env)[3] > 0
    a, c = env.compute_observations()
    assert a[:3].reshape(3, 5, 62)[:, :, -5:-3].count_nonzero() == 0
    assert c[:3].reshape(3, 5, 73)[:, :, -11:-9].count_nonzero() == 0
    # The settled sample uses actual bilateral ground loads. One environment is still pushed.
    for _ in range(105):
        env.push_applied.zero_()
        env.check_reset()
    assert not env.recovering[0] and env.recovering[1] and env.recovering[2]
    torch.testing.assert_close(env.stand_anchor_xy[0], env.robot.data.root_pos_w[0, :2])
    torch.testing.assert_close(env.feet_anchor_xy[0], env.robot.data.body_pos_w[0, env.feet_body_ids, :2])
    torch.testing.assert_close(old_yaw, env.stand_anchor_yaw)
    assert rewards.keep_position(env)[0] == 1 and rewards.feet_drift(env)[0] == 0
    # Exercise the recovery masks on real bilateral contacts. Vary only the gap measurement.
    dt_ticks = env.feet_settle_steps
    anchor = env.stand_anchor_xy[0].clone()
    foot_sample = env.robot.data.body_pos_w[0, env.feet_body_ids, :2].clone()
    env.feet_anchor_xy[0] = foot_sample - .5
    gap = env.sole_gap_m.clone()
    gap[0] = .22
    width_term = env.reward_manager.active_terms.index("feet_width")

    def advance_checked_frame():
        env.sim_step_counter += cfg.sim.decimation
        env.check_reset()

    def check_wide_step(recovering):
        advance_checked_frame()
        assert bool(env.recovering[0]) == recovering
        env.reward_manager.compute(env.step_dt)
        loads = env.contact_sensor.data.net_forces_w[:, env.feet_cfg.body_ids, 2]
        expected_support = -(1. - loads.clamp(0., 150.) / 150.).sum(dim=1) * env.feet_should_be_planted
        torch.testing.assert_close(env.reward_manager._step_reward[:, support_term], expected_support)
        step_reward = env.reward_manager._step_reward[0, width_term] * env.step_dt
        torch.testing.assert_close(step_reward, step_reward.new_tensor(-.015))

    with patch("legged_lab.envs.gl_165.stand_env.sole_inner_gap", return_value=gap):
        check_wide_step(False)
        env.push_applied[0] = True
        check_wide_step(True)
        env.push_applied[0] = False
        check_wide_step(True)
        assert rewards.feet_width_excess(env, .12, .10)[0] == 0
        for _ in range(105):
            advance_checked_frame()
        check_wide_step(False)
        assert env.gathering_feet[0] and not env.recovering[0] and not env.feet_should_be_planted[0]
        torch.testing.assert_close(env.stand_anchor_xy[0], anchor)
        torch.testing.assert_close(env.feet_anchor_xy[0], foot_sample - .5)
        assert rewards.feet_width_excess(env, .12, .10)[0] > 0
        assert rewards.feet_drift(env)[0] == 0 and rewards.both_feet_loaded(env)[0] == 0
        env.stand_anchor_xy[0, 0] += .08
        assert rewards.keep_position(env)[0] == 1
        env.push_applied[0] = True
        advance_checked_frame()
        assert env.recovering[0] and not env.gathering_feet[0]
        env.push_applied[0] = False
        for _ in range(105):
            advance_checked_frame()
        assert env.gathering_feet[0]
        gap[0] = .04
        for _ in range(dt_ticks):
            advance_checked_frame()
        assert env.gathering_feet[0] and rewards.feet_clearance_shortfall(env, env.min_foot_clearance_m)[0] > 0
        gap[0] = .08
        for _ in range(dt_ticks - 1):
            advance_checked_frame()
        assert env.gathering_feet[0]
        advance_checked_frame()
        assert not env.gathering_feet[0] and env.feet_should_be_planted[0]
        torch.testing.assert_close(env.feet_anchor_xy[0], foot_sample)
    assert not env.gathering_feet.any()
    base_pose = env.robot.data.root_state_w[1:2, :7].clone()
    base_pose[:, 2] -= .4
    env.robot.write_root_pose_to_sim(base_pose, ids)
    env.check_reset()
    assert env.failure_reasons["height"][1] and not env.failure_reasons["drift"][1]
    env.reset(ids)
    assert not env.recovering[1] and env.recovery_hold[1] == 0 and env.recovering[2]
    assert not env.gathering_feet[1] and env.feet_stable_ticks[1] == 0
    return {"passed": True, "displacement_m": .75, "tested": [
        "velocity event", "sustained force", "unsignalled velocity", "quiet drift termination",
        "old XY hidden in every noisy actor and critic frame", "position and foot rewards",
        "bilateral re-lock and unchanged heading", "wide gap releases footprints only", "pelvis deadband",
        "new push interrupts gathering", "continuous 5-12 cm double-support foot lock",
        "low height during recovery", "partial reset", "current and historical sole-gap penalty at 5 cm",
        "12 cm all-phase limit and historical settled-only limit", "per-foot load penalty, phase mask and legacy support bonus"]}


try:
    with torch.inference_mode():
        result = check()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
except Exception:
    args.output.write_text(json.dumps({"passed": False, "error": traceback.format_exc()}, indent=2) + "\n")
    traceback.print_exc()
    raise
finally:
    app.close(wait_for_replicator=False)

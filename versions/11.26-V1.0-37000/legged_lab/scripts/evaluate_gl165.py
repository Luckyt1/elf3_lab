"""Finite GL-165 evaluation, with nominal PD control or a trained PPO checkpoint."""

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint", type=Path, help="PPO model file; omit to test nominal PD standing.")
parser.add_argument("--arm-database", type=Path, help="Override the GL-165 arm disturbance database.")
parser.add_argument("--min-foot-clearance-m", type=float,
                    help="Evaluate the signed lateral gap of the soles against this soft minimum in metres.")
parser.add_argument("--no_push", action="store_true", help="Disable disturbances for the standing baseline.")
parser.add_argument("--no_arms", action="store_true", help="Hold arms at the nominal pose instead of playing disturbances.")
parser.add_argument("--push-delta", type=float, nargs=3, metavar=("DX", "DY", "DYAW"),
                    help="Use fixed world-frame root velocity increments (m/s, m/s, rad/s).")
parser.add_argument("--push_delay", type=float, help="Set time to the first disturbance after startup (s).")
parser.add_argument("--torso-force-iteration", type=int,
                    help="Test only sustained torso pushes at this curriculum iteration.")
parser.add_argument("--diagnose-joints", action="store_true", help="Record policy actions, targets and actual waist/leg motion.")
parser.add_argument("--output", type=Path, default=Path("logs/stand_gl165/evaluation.json"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from rsl_rl.runners import OnPolicyRunner

from legged_lab.envs.gl_165.stand_cfg import Gl165StandAgentCfg, Gl165StandEnvCfg, TorsoForceCfg
from legged_lab.envs.gl_165.stand_env import Gl165StandEnv
from legged_lab.envs.gl_165.rewards import feet_heading_error, yaw_from_quat
from legged_lab.assets.gl_165.disturbances import MIXED_PUSH, VELOCITY_ONLY, torso_force_parameters


def verify_collision_filters(env):
    """Check both the source and a clone, including relationship remapping."""
    import omni.usd
    from pxr import PhysxSchema, Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    pairs = [(joint.find("parent").get("link"), joint.find("child").get("link"))
             for joint in ET.parse(env.physics_asset.urdf).getroot().findall("joint")] + env.physics_asset.parameters["nested_collision_pairs"]
    count = 0
    for index in sorted({0, env.num_envs - 1}):
        root = stage.GetPrimAtPath(f"/World/envs/env_{index}/Robot")
        prims = list(Usd.PrimRange(root))
        roots = [p for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        if not roots or not all(PhysxSchema.PhysxArticulationAPI(p).GetEnabledSelfCollisionsAttr().Get() for p in roots):
            raise RuntimeError(f"Self-collision is disabled in environment {index}")
        bodies = {p.GetName(): p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)}
        for first, second in pairs:
            targets = UsdPhysics.FilteredPairsAPI(bodies[first]).GetFilteredPairsRel().GetTargets()
            if bodies[second].GetPath() not in targets:
                raise RuntimeError(f"Missing or incorrectly remapped collision exclusion: env {index}, {first}, {second}")
            count += 1
    return {"self_collisions_enabled": True, "verified_mounting_pair_relationships": count}


def evaluate():
    if args.steps <= 0 or args.num_envs <= 0:
        raise ValueError("--steps and --num_envs must be positive")
    cfg = Gl165StandEnvCfg()
    cfg.device = args.device
    cfg.scene.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.noise.add_noise = False
    if args.checkpoint:
        cfg.restore_for_playback(args.checkpoint, args.arm_database)
    if args.torso_force_iteration is not None:
        if args.torso_force_iteration < 0 or args.push_delta is not None or args.no_push:
            raise ValueError("Select one push test and a nonnegative torso-force iteration")
        cfg.push.mode = MIXED_PUSH
        if cfg.push.torso_force is None:
            cfg.push.torso_force = TorsoForceCfg()
        cfg.push.torso_force.event_probability = 1.0
    cfg.push.enabled = not args.no_push
    cfg.push.curriculum_enabled = False
    cfg.arm_motion.enabled = not args.no_arms
    cfg.arm_motion.curriculum_enabled = False
    if args.arm_database is not None:
        cfg.arm_motion.database = str(args.arm_database.resolve(strict=True))
    if args.min_foot_clearance_m is not None:
        cfg.set_min_foot_clearance(args.min_foot_clearance_m)
    if args.push_delta is not None:
        cfg.push.mode = VELOCITY_ONLY
        if not all(math.isfinite(value) for value in args.push_delta):
            raise ValueError("--push-delta requires three finite velocity increments")
        cfg.push.velocity_ranges = {axis: (v, v) for axis, v in zip(("x", "y", "yaw"), args.push_delta)}
    subsequent_intervals = cfg.push.interval_s
    if args.push_delay is not None:
        if not math.isfinite(args.push_delay) or args.push_delay < 0:
            raise ValueError("--push_delay must be finite and non-negative")
        cfg.push.interval_s = (args.push_delay, args.push_delay)
    env = Gl165StandEnv(cfg, args.headless)
    if args.push_delay is not None and cfg.push.enabled:
        # EventManager has sampled its initial timers. Future events use the training interval.
        env.event_manager.get_term_cfg("push_robot").interval_range_s = subsequent_intervals
    cfg.push.interval_s = subsequent_intervals
    collision_config = verify_collision_filters(env)
    policy = None
    if args.checkpoint:
        agent = Gl165StandAgentCfg()
        agent.device = cfg.device
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=cfg.device)
        runner.load(str(args.checkpoint.resolve()), load_optimizer=False)
        policy = runner.get_inference_policy(device=cfg.device)
        env.learning_iteration = runner.current_learning_iteration
    if args.torso_force_iteration is not None:
        env.learning_iteration = args.torso_force_iteration
    obs, extras = env.get_observations()
    failures = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    timeouts = torch.zeros_like(failures)
    height_sum = 0.0
    push_events = torch.zeros((), device=env.device, dtype=torch.long)
    force_events = torch.zeros_like(push_events)
    peak_force = torch.zeros((), device=env.device)
    max_abs_delta = torch.zeros(6, device=env.device)
    max_delta_xy = torch.zeros((), device=env.device)
    failure_reasons = {name: 0 for name in ("contact", "height", "tilt", "drift")}
    joint_trace = []
    stance_trace = []
    sole_gaps = []
    hip_roll_ids = env.robot.find_joints(["lhip_x_joint", "rhip_x_joint"], preserve_order=True)[0]
    hip_yaw_ids = env.robot.find_joints(["lhip_z_joint", "rhip_z_joint"], preserve_order=True)[0]
    knee_ids = env.robot.find_bodies(["lsmallleg_y_link", "rsmallleg_y_link"], preserve_order=True)[0]
    with torch.inference_mode():
        for _ in range(args.steps):
            actions = policy(obs) if policy else torch.zeros(env.num_envs, env.num_actions, device=env.device)
            if args.diagnose_joints:
                clipped = actions if env.clip_actions is None else actions.clamp(-env.clip_actions, env.clip_actions)
                requested_target = env.robot.data.default_joint_pos[:, env.policy_joint_ids] + clipped * env.action_scale
                limits = env.robot.data.soft_joint_pos_limits[:, env.policy_joint_ids]
                target = requested_target.clamp(limits[..., 0], limits[..., 1])
            obs, reward, done, extras = env.step(actions)
            if env.min_foot_clearance_m:
                sole_gaps.append(env.sole_gap_m)
            root_yaw = yaw_from_quat(env.robot.data.root_quat_w)
            side_direction = torch.stack((-root_yaw.sin(), root_yaw.cos()), dim=-1)
            feet = env.robot.data.body_pos_w[:, env.feet_body_ids, :2]
            knees = env.robot.data.body_pos_w[:, knee_ids, :2]
            stance_trace.append({
                "waist_angle_deg": torch.rad2deg(env.robot.data.joint_pos[:, env.waist_joint_ids]),
                "waist_target_deg": torch.rad2deg(env.robot.data.joint_pos_target[:, env.waist_joint_ids]),
                "feet_yaw_deg": torch.rad2deg(feet_heading_error(env)),
                "hip_roll_offset_deg": torch.rad2deg(
                    env.robot.data.joint_pos[:, hip_roll_ids] - env.robot.data.default_joint_pos[:, hip_roll_ids]),
                "hip_yaw_offset_deg": torch.rad2deg(
                    env.robot.data.joint_pos[:, hip_yaw_ids] - env.robot.data.default_joint_pos[:, hip_yaw_ids]),
                "feet_separation_m": ((feet[:, 0] - feet[:, 1]) * side_direction).sum(dim=-1),
                "knees_separation_m": ((knees[:, 0] - knees[:, 1]) * side_direction).sum(dim=-1),
                "valid": ((env.episode_length_buf >= 35) & ~done).clone(),
            })
            if args.diagnose_joints:
                joint_trace.append({
                    "raw_action": actions.clone(), "clipped_action": clipped.clone(), "target_rad": target.clone(),
                    "sent_target_rad": env.robot.data.joint_pos_target[:, env.policy_joint_ids].clone(),
                    "joint_limit_clamped": (target - requested_target).abs() > 1e-6,
                    "actual_rad": env.robot.data.joint_pos[:, env.policy_joint_ids].clone(),
                    "velocity_rad_s": env.robot.data.joint_vel[:, env.policy_joint_ids].clone(),
                    "torque_nm": env.robot.data.applied_torque[:, env.policy_joint_ids].clone(),
                    "settled": (env.episode_length_buf >= 35).clone(), "done": done.clone(),
                })
            for value in (obs, extras["observations"]["critic"], reward, env.robot.data.root_state_w):
                if not torch.isfinite(value).all():
                    raise RuntimeError("Non-finite observations, rewards or robot state.")
            failures += (done & ~extras["time_outs"]).long()
            timeouts += extras["time_outs"].long()
            for name, flags in env.failure_reasons.items():
                failure_reasons[name] += flags.sum().item()
            height_sum += (env.robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]).mean().item()
            push_events += env.push_applied.sum()
            if env.torso_force is not None:
                force_events += env.torso_force.started.sum()
                peak_force = torch.maximum(peak_force, env.torso_force.local_forces[:, env.torso_body_id].norm(dim=-1).max())
            max_abs_delta = torch.maximum(max_abs_delta, env.push_delta_w.abs().amax(dim=0))
            max_delta_xy = torch.maximum(max_delta_xy, env.push_delta_w[:, :2].norm(dim=-1).max())
    result = {
        "controller": str(args.checkpoint) if args.checkpoint else "nominal_pd",
        "num_envs": env.num_envs,
        "seconds_per_env": args.steps * env.step_dt,
        "push_enabled": cfg.push.enabled,
        "arm_motion_enabled": cfg.arm_motion.enabled,
        "arm_database_sha256": env.arm_motion.sha256,
        "arm_database": str(env.arm_motion.path),
        "arm_motion_profile": env.arm_motion.metadata.get("profile", "random"),
        "arm_segment_duration_s": env.arm_motion.metadata.get("segment_duration_s", 1.0),
        "collision_configuration": collision_config,
        "failure_resets": failures.sum().item(),
        "timeout_resets": timeouts.sum().item(),
        "failure_reasons": failure_reasons,
        "fraction_envs_without_failure": (failures == 0).float().mean().item(),
        "mean_base_height_m": height_sum / args.steps,
        "push_mode": cfg.push.mode,
        "push_interval_s": cfg.push.interval_s,
        "first_push_delay_override_s": args.push_delay,
        "push_velocity_ranges": cfg.push.velocity_ranges,
        "push_events": push_events.item(),
        "max_delta_xy_m_s": max_delta_xy.item(),
        "max_abs_linear_delta_world_m_s": max_abs_delta[:3].tolist(),
        "max_abs_angular_delta_world_rad_s": max_abs_delta[3:].tolist(),
        "actor_obs_shape": list(obs.shape),
        "critic_obs_shape": list(extras["observations"]["critic"].shape),
        "num_actions": env.num_actions,
        "action_clip_enabled": env.clip_actions is not None,
        "clip_actions": env.clip_actions,
        "floating_base": not env.robot.is_fixed_base,
        "finite": True,
    }
    if env.torso_force is not None:
        force, duration = torso_force_parameters(cfg.push.torso_force, env.learning_iteration)
        result["torso_push"] = dict(iteration=env.learning_iteration, scheduled_force_n=force,
                                    scheduled_duration_s=duration, events=force_events.item(),
                                    max_applied_force_n=peak_force.item())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if sole_gaps:
        gap = torch.stack(sole_gaps).cpu()
        result["foot_clearance"] = {
            "measurement": "signed lateral gap between sole sphere edges in the pelvis heading frame",
            "sample_selection": "all control steps before reset",
            "samples": gap.numel(),
            "threshold_m": env.min_foot_clearance_m,
            "min_m": gap.min().item(),
            "below_min_fraction": (gap < env.min_foot_clearance_m).float().mean().item(),
            "projected_overlap_fraction": (gap < 0).float().mean().item(),
        }
    stance = {key: torch.stack([row[key] for row in stance_trace]).cpu() for key in stance_trace[0]}
    valid = stance["valid"]
    result["standing_pose"] = {"sample_selection": "after_0_7_s_excluding_resets", "samples": int(valid.sum())}
    result["waist_posture"] = {
        "joint_names": [env.robot.joint_names[index] for index in env.waist_joint_ids],
        "reference": "joint_zero",
        "new_command_limit": None,
        "zero_reward_weight": cfg.reward.waist_zero.weight,
        "sample_selection": "after_0_7_s_excluding_resets", "samples": int(valid.sum()),
    }
    if bool(valid.any()):
        waist = stance["waist_angle_deg"][valid]
        waist_target = stance["waist_target_deg"][valid]
        result["waist_posture"].update({
            "actual_mean_deg": waist.mean(dim=0).tolist(),
            "actual_abs_mean_deg": waist.abs().mean(dim=0).tolist(),
            "actual_abs_p95_deg": waist.abs().quantile(.95, dim=0).tolist(),
            "actual_abs_max_deg": waist.abs().amax(dim=0).tolist(),
            "target_abs_mean_deg": waist_target.abs().mean(dim=0).tolist(),
            "target_abs_max_deg": waist_target.abs().amax(dim=0).tolist(),
            "all_axes_within_3deg_fraction": float((waist.abs() <= 3).all(dim=-1).float().mean()),
            "all_axes_within_5deg_fraction": float((waist.abs() <= 5).all(dim=-1).float().mean()),
        })
        yaw = stance["feet_yaw_deg"][valid]
        inward = torch.stack((-yaw[:, 0], yaw[:, 1]), dim=-1).clamp(min=0.0)
        result["standing_pose"].update({
            "feet_order": ["left", "right"],
            "feet_heading_reference": "pelvis_forward",
            "feet_heading_mean_deg": yaw.mean(dim=0).tolist(),
            "feet_heading_abs_mean_deg": yaw.abs().mean(dim=0).tolist(),
            "feet_heading_abs_p95_deg": yaw.abs().quantile(.95, dim=0).tolist(),
            "toe_in_mean_deg": inward.mean(dim=0).tolist(),
            "toe_in_over_5deg_fraction": (inward > 5).float().mean(dim=0).tolist(),
            "both_feet_within_5deg_fraction": float((yaw.abs() <= 5).all(dim=-1).float().mean()),
            "hip_roll_offset_abs_mean_deg": stance["hip_roll_offset_deg"][valid].abs().mean(dim=0).tolist(),
            "hip_yaw_offset_abs_mean_deg": stance["hip_yaw_offset_deg"][valid].abs().mean(dim=0).tolist(),
            "feet_separation_mean_m": float(stance["feet_separation_m"][valid].mean()),
            "knees_separation_mean_m": float(stance["knees_separation_m"][valid].mean()),
        })
    if joint_trace:
        trace = {key: torch.stack([row[key] for row in joint_trace]).cpu() for key in joint_trace[0]}
        trace["joint_names"] = [env.robot.joint_names[index] for index in env.policy_joint_ids]
        trace_path = args.output.with_suffix(".joints.pt")
        torch.save(trace, trace_path)
        mask = trace["settled"] & ~trace["done"]
        settled_count = int(mask.sum())
        if not bool(mask.any()):
            mask = ~trace["done"]
        if not bool(mask.any()):
            mask = torch.ones_like(trace["done"])
        raw, clipped = trace["raw_action"][mask], trace["clipped_action"][mask]
        target, actual = trace["target_rad"][mask], trace["actual_rad"][mask]
        velocity, torque = trace["velocity_rad_s"][mask], trace["torque_nm"][mask]
        target_error = float((trace["sent_target_rad"] - trace["target_rad"]).abs().max())
        if target_error > 1e-5:
            raise RuntimeError(f"Actuator targets differ from policy-derived targets: {target_error} rad")
        result["joint_diagnostics"] = {
            "trace": str(trace_path), "samples_after_0_7_s": settled_count,
            "sample_selection": "after_0_7_s" if settled_count else "early_episode_fallback",
            "clip_actions": env.clip_actions,
            "max_action_clipping_change": float((trace["raw_action"] - trace["clipped_action"]).abs().max()),
            "max_actuator_target_error_rad": target_error,
            "joints": [{
                "name": name, "action_scale_rad": float(env.action_scale[index]),
                "raw_action_min": float(raw[:, index].min()), "raw_action_max": float(raw[:, index].max()),
                "saturation_fraction": (float((raw[:, index].abs() >= env.clip_actions).float().mean())
                                        if env.clip_actions is not None else 0.0),
                "joint_limit_fraction": float(trace["joint_limit_clamped"][mask][:, index].float().mean()),
                "clipped_action_std": float(clipped[:, index].std(unbiased=False)),
                "target_range_deg": float(torch.rad2deg(target[:, index].max() - target[:, index].min())),
                "actual_range_deg": float(torch.rad2deg(actual[:, index].max() - actual[:, index].min())),
                "mean_abs_velocity_deg_s": float(torch.rad2deg(velocity[:, index].abs().mean())),
                "mean_abs_tracking_error_deg": float(torch.rad2deg((actual[:, index] - target[:, index]).abs().mean())),
                "mean_abs_torque_nm": float(torque[:, index].abs().mean()),
            } for index, name in enumerate(trace["joint_names"])],
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    env.sim.clear_all_callbacks()
    env.sim.clear_instance()


if __name__ == "__main__":
    try:
        evaluate()
    finally:
        simulation_app.close(wait_for_replicator=False)

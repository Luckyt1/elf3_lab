"""Standing environment with base velocity changes and sustained torso loads."""

import math

import torch
from isaaclab.managers import EventTermCfg
from isaaclab.utils.math import quat_rotate_inverse

from legged_lab.assets.gl_165.gl_165 import make_gl165_cfg
from legged_lab.assets.gl_165.stance_recovery import advance_foot_reference, advance_stance_reference, stance_is_quiet
from legged_lab.assets.gl_165.disturbances import MIXED_PUSH, VELOCITY_ONLY, torso_force_parameters
from legged_lab.assets.gl_165.model import (
    ARM_JOINT_NAMES, CONTROLLER_PROFILE, FEET, JOINT_PHYSICS, PHYSICS_PARAMETERS,
    POLICY_JOINT_PATTERN, TORSO,
    joint_control_parameters, motion_asset_signature,
)
from legged_lab.envs.base.base_env import BaseEnv

from .pushes import TorsoForce, apply_mixed_push, apply_velocity_push, update_foot_plant_mask, velocity_push_scale
from .arm_motion import ArmMotionDatabase
from .rewards import feet_heading_error, heading_error, sole_height_bounds, sole_inner_gap, yaw_from_quat
from .scene import Gl165SceneCfg


class Gl165StandEnv(BaseEnv):
    scene_cfg_type = Gl165SceneCfg

    def __init__(self, cfg, headless):
        cfg.scene.robot = make_gl165_cfg()
        event = {VELOCITY_ONLY: apply_velocity_push, MIXED_PUSH: apply_mixed_push}[cfg.push.mode]
        cfg.domain_rand.events.push_robot = EventTermCfg(
            func=event, mode="interval", interval_range_s=cfg.push.interval_s,
            is_global_time=False,
        ) if cfg.push.enabled else None
        self.learning_iteration = 0
        super().__init__(cfg, headless)
        if self.robot.is_fixed_base or self.robot.num_joints != 29 or self.num_actions != 15:
            raise RuntimeError("GL-165 XML requires a floating base, 29 joints and 15 waist/leg policy actions.")
        self.controller_parameters = self._verify_controller_parameters()
        print(f"[GL-165] Verified {CONTROLLER_PROFILE}: 29 PD gains/armatures, 15 named action scales; "
              f"head fixed at zero; physics {self.physics_dt * 1000:g} ms, "
              f"policy {self.step_dt * 1000:g} ms.", flush=True)
        print(f"[GL-165] Action clipping: {self.clip_actions}; scaled targets use joint limits.", flush=True)

    def init_buffers(self):
        self.stand_anchor_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.stand_anchor_yaw = torch.zeros(self.num_envs, device=self.device)
        self.feet_anchor_xy = torch.zeros(self.num_envs, 2, 2, device=self.device)
        self.recovering = None
        if self.cfg.stance_recovery is not None:
            self.recovering = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            self.recovery_hold = torch.zeros_like(self.recovering, dtype=torch.long)
            self.recovery_stable = torch.zeros_like(self.recovery_hold)
            self.stance_relocked = torch.zeros_like(self.recovering)
        self.feet_body_ids = self.robot.find_bodies(FEET, preserve_order=True)[0]
        clearance = self.cfg.reward.feet_clearance
        self.min_foot_clearance_m = clearance.params["minimum_m"] if clearance is not None else 0.0
        width = self.cfg.reward.feet_width
        self.max_foot_clearance_m = width.params["maximum_m"] if width is not None else None
        self.gathering_feet = None
        if width is not None:
            if (self.recovering is None or not self.min_foot_clearance_m < self.max_foot_clearance_m < math.inf
                    or not 0 < width.params["scale_m"] < math.inf
                    or not 0 <= self.cfg.gather_pelvis_tolerance_m < math.inf):
                raise ValueError("Invalid stance recovery, foot gap bounds, cost scale or pelvis tolerance")
            self.gathering_feet = torch.zeros_like(self.recovering)
            self.feet_stable_ticks = torch.zeros_like(self.recovery_stable)
            self.feet_settle_steps = max(1, math.ceil(self.cfg.stance_recovery["settle_s"] / self.step_dt))
        self.foot_recovery_until = None
        self.feet_should_be_planted = None
        if self.cfg.reward.feet_lift is not None:
            self.foot_recovery_until = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.feet_should_be_planted = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            self.foot_recovery_steps = round(self.cfg.foot_recovery_s / self.physics_dt)
        if self.min_foot_clearance_m or width is not None or self.foot_recovery_until is not None:
            contacts = PHYSICS_PARAMETERS["foot_contacts"]
            self.sole_spheres = torch.tensor([
                [[*p["position_m"], p["radius_m"]] for p in contacts[foot]] for foot in FEET
            ], device=self.device)
        self.torso_body_id = self.robot.find_bodies(TORSO)[0][0]
        self.torso_force = (TorsoForce(self.robot, self.torso_body_id, self.physics_dt)
                            if self.cfg.push.enabled and self.cfg.push.mode == MIXED_PUSH else None)
        self.push_delta_w = torch.zeros(self.num_envs, 6, device=self.device)
        self.push_applied = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.policy_joint_ids = self.robot.find_joints(POLICY_JOINT_PATTERN)[0]
        self.waist_joint_ids = self.robot.find_joints(
            ["waist_x_joint", "waist_y_joint", "waist_z_joint"], preserve_order=True,
        )[0]
        self.arm_joint_ids = self.robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)[0]
        self.arm_motion = ArmMotionDatabase(
            self.cfg.arm_motion.database, ARM_JOINT_NAMES,
            self.robot.data.default_joint_pos[:, self.arm_joint_ids], motion_asset_signature(),
            self.num_envs, self.device, self.step_dt,
        )
        # Persist the exact dataset identity in every run's env.yaml.
        self.cfg.arm_motion.database_sha256 = self.arm_motion.sha256
        print(f"[GL-165] Arm database: {self.arm_motion.clips.shape[0]} clips, "
              f"SHA256={self.arm_motion.sha256}", flush=True)
        super().init_buffers()
        scales = []
        for index in self.policy_joint_ids:
            name = self.robot.joint_names[index]
            scale = joint_control_parameters(name)[2]
            if scale is None:
                raise ValueError(f"No policy action scale for {name}.")
            scales.append(scale)
        self.action_scale = torch.tensor(scales, device=self.device)
        # Save the effective vector and its actual simulator order, not the
        # unused BaseEnv scalar default, in env.yaml.
        self.cfg.robot.action_scale = scales
        self.cfg.robot.policy_joint_names = [self.robot.joint_names[i] for i in self.policy_joint_ids]
        self.effort_limits = self.robot.data.joint_effort_limits.clamp(min=1.0)

    def _verify_controller_parameters(self):
        expected = [joint_control_parameters(name) for name in self.robot.joint_names]
        for field, column in (("joint_stiffness", 0), ("joint_damping", 1)):
            reference = torch.tensor([row[column] for row in expected], device=self.device)
            actual = getattr(self.robot.data, field)
            if not torch.allclose(actual, reference.expand_as(actual), atol=1e-5, rtol=1e-6):
                raise RuntimeError(f"GL-165 effective {field} does not match the confirmed controller.")
        for field, key in (("joint_armature", "armature_kg_m2"),
                           ("joint_effort_limits", "effort_limit_nm"),
                           ("joint_vel_limits", "velocity_limit_rad_s")):
            reference = torch.tensor([JOINT_PHYSICS[n][key] for n in self.robot.joint_names], device=self.device)
            actual = getattr(self.robot.data, field)
            if not torch.allclose(actual, reference.expand_as(actual), atol=1e-7, rtol=1e-5):
                raise RuntimeError(f"GL-165 runtime {field} differs from the XML physics profile")
        actor_frame, critic_frame = self.compute_current_observations()
        return {
            "profile": CONTROLLER_PROFILE,
            "observe_arm_motion": self.cfg.observe_arm_motion,
            "observe_standing_state": self.cfg.observe_standing_state,
            "actor_obs_history_length": self.cfg.robot.actor_obs_history_length,
            "critic_obs_history_length": self.cfg.robot.critic_obs_history_length,
            "stance_recovery": self.cfg.stance_recovery,
            "actor_observation_frame_dim": actor_frame.shape[-1],
            "critic_observation_frame_dim": critic_frame.shape[-1],
            "push_mode": self.cfg.push.mode,
            "torso_force_curriculum": self.cfg.push.torso_force.to_dict() if self.cfg.push.mode == MIXED_PUSH else None,
            "min_foot_clearance_m": self.min_foot_clearance_m,
            **({"foot_gathering": dict(**self.cfg.reward.feet_width.params, weight=self.cfg.reward.feet_width.weight,
                                       pelvis_tolerance_m=self.cfg.gather_pelvis_tolerance_m)}
               if self.gathering_feet is not None else {}),
            "quiet_foot_contact": (dict(weight=self.cfg.reward.feet_lift.weight,
                                        post_push_recovery_s=self.cfg.foot_recovery_s,
                                        **self.cfg.reward.feet_lift.params)
                                   if self.cfg.reward.feet_lift is not None else None),
            "all_environment_gains_verified": True,
            "head_control": "two fixed joints at zero, as authored in the user-supplied XML",
            "physics_profile": PHYSICS_PARAMETERS["profile"],
            "source_xml_sha256": PHYSICS_PARAMETERS["source_xml_sha256"],
            "all_environment_armature_and_limits_verified": True,
            "armature_kg_m2": self.robot.data.joint_armature[0].tolist(),
            "joint_names": self.robot.joint_names,
            "kp": self.robot.data.joint_stiffness[0].tolist(),
            "kd": self.robot.data.joint_damping[0].tolist(),
            "default_joint_position_rad": self.robot.data.default_joint_pos[0].tolist(),
            "effort_limits_nm": self.robot.data.joint_effort_limits[0].tolist(),
            "velocity_limits_rad_s": self.robot.data.joint_vel_limits[0].tolist(),
            "soft_position_limits_rad": self.robot.data.soft_joint_pos_limits[0].tolist(),
            "policy_joint_names": self.cfg.robot.policy_joint_names,
            "action_scale_rad": self.action_scale.tolist(),
            "clip_actions": self.clip_actions,
            "physics_dt_s": self.physics_dt, "control_dt_s": self.step_dt,
            "arm_database_sha256": self.arm_motion.sha256,
            "arm_segment_duration_s": self.arm_motion.metadata.get("segment_duration_s", 1.0),
            "asset_signature": motion_asset_signature(),
        }

    def compute_current_observations(self):
        actor, critic = super().compute_current_observations()
        displacement = torch.zeros(self.num_envs, 3, device=self.device)
        displacement[:, :2] = self.robot.data.root_pos_w[:, :2] - self.stand_anchor_xy
        if self.recovering is not None:
            displacement = displacement * (~self.recovering[:, None])
        local_displacement = quat_rotate_inverse(self.robot.data.root_quat_w, displacement)
        yaw = heading_error(self)
        height = self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        # Heading stays fixed across a recovery. XY is relative to the current settled stance.
        standing = torch.cat([
            self.robot.data.root_lin_vel_b * self.obs_scales.lin_vel,
            local_displacement[:, :2], yaw.sin()[:, None], yaw.cos()[:, None], height[:, None],
        ], dim=-1)
        # Privileged only. A reset observation belongs to the next episode.
        push_delta = self.push_delta_w * (self.episode_length_buf > 0).unsqueeze(1)
        if self.cfg.observe_arm_motion:
            # Historical policies used this block before the standing observations.
            arm_state = [
                (self.robot.data.joint_pos - self.robot.data.default_joint_pos)[:, self.arm_joint_ids],
                self.robot.data.joint_vel[:, self.arm_joint_ids] * self.obs_scales.joint_vel,
                self.arm_motion.target - self.arm_motion.nominal,
                (self.arm_motion.frame / (self.arm_motion.clips.shape[1] - 1))[:, None],
            ]
            return torch.cat([actor, *arm_state, standing], -1), torch.cat([critic, *arm_state, standing, push_delta], -1)
        if self.cfg.observe_standing_state:
            actor = torch.cat([actor, standing], -1)
        return actor, torch.cat([critic, standing, push_delta], -1)

    def compute_observations(self):
        actor, critic = super().compute_observations()
        if self.recovering is not None:
            mask = self.recovering[:, None, None]
            if self.cfg.observe_standing_state:
                actor.reshape(self.num_envs, self.cfg.robot.actor_obs_history_length, -1)[:, :, -5:-3].masked_fill_(mask, 0.)
            critic.reshape(self.num_envs, self.cfg.robot.critic_obs_history_length, -1)[:, :, -11:-9].masked_fill_(mask, 0.)
        return actor, critic

    def init_obs_buffer(self):
        super().init_obs_buffer()
        if self.add_noise and self.cfg.observe_standing_state:
            if self.cfg.observe_arm_motion:
                arm_offset = 9 + 3 * self.num_actions
                self.noise_scale_vec[arm_offset:arm_offset + 14] = self.cfg.noise.noise_scales.joint_pos
                self.noise_scale_vec[arm_offset + 14:arm_offset + 28] = (
                    self.cfg.noise.noise_scales.joint_vel * self.obs_scales.joint_vel)
            self.noise_scale_vec[-8:-5] = 0.05 * self.obs_scales.lin_vel
            self.noise_scale_vec[-5:-3] = 0.005
            self.noise_scale_vec[-1] = 0.005

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return
        if self.recovering is not None:
            self.recovering[env_ids] = False
            self.recovery_hold[env_ids] = 0
            self.recovery_stable[env_ids] = 0
            self.stance_relocked[env_ids] = False
        if self.gathering_feet is not None:
            self.gathering_feet[env_ids] = False
            self.feet_stable_ticks[env_ids] = 0
        self.arm_motion.reset(env_ids)
        if self.foot_recovery_until is not None:
            self.foot_recovery_until[env_ids] = 0
        if self.torso_force is not None:
            self.torso_force.reset(env_ids)
        super().reset(env_ids)
        self.stand_anchor_xy[env_ids] = self.robot.data.root_pos_w[env_ids, :2]
        self.stand_anchor_yaw[env_ids] = yaw_from_quat(self.robot.data.root_quat_w[env_ids])
        self.feet_anchor_xy[env_ids] = self.robot.data.body_pos_w[env_ids][:, self.feet_body_ids, :2]

    def arm_scale(self):
        cfg = self.cfg.arm_motion
        if not cfg.enabled:
            return 0.0
        if not cfg.curriculum_enabled:
            return 1.0
        progress = min(1.0, max(0.0, (self.learning_iteration - cfg.warmup_iterations) /
                               max(1, cfg.full_motion_iteration - cfg.warmup_iterations)))
        return cfg.initial_scale + (1 - cfg.initial_scale) * progress

    def process_actions(self, actions):
        target = self.robot.data.default_joint_pos.clone()
        target[:, self.policy_joint_ids] += actions * self.action_scale
        target[:, self.arm_joint_ids] = self.arm_motion.step(self.arm_scale())
        limits = self.robot.data.soft_joint_pos_limits
        return target.clamp(limits[..., 0], limits[..., 1])

    def push_scale(self):
        return velocity_push_scale(self.cfg.push, self.sim_step_counter * self.physics_dt)

    def apply_external_forces(self):
        if self.torso_force is not None:
            self.torso_force.apply(self.robot)

    def step(self, actions):
        self.push_delta_w.zero_()
        self.push_applied.zero_()
        if self.torso_force is not None:
            self.torso_force.started.zero_()
        obs, reward, done, extras = super().step(actions)
        extras["time_outs"] = self.time_out_buf
        extras.setdefault("log", {})["Push/scale"] = self.push_scale()
        extras["log"]["Push/events_per_step"] = self.push_applied.sum()
        extras["log"]["Push/max_delta_xy_m_s"] = self.push_delta_w[:, :2].norm(dim=-1).max()
        extras["log"]["Push/max_abs_delta_yaw_rad_s"] = self.push_delta_w[:, 5].abs().max()
        if self.torso_force is not None:
            force, duration = torso_force_parameters(self.cfg.push.torso_force, self.learning_iteration)
            extras["log"]["TorsoPush/force_N"] = force
            extras["log"]["TorsoPush/duration_s"] = duration
            extras["log"]["TorsoPush/events_per_step"] = self.torso_force.started.sum()
            extras["log"]["TorsoPush/active_fraction"] = (self.torso_force.remaining > 0).float().mean()
        extras["log"]["Arm/scale"] = self.arm_scale()
        extras["log"]["Arm/tracking_error_rad"] = (
            self.robot.data.joint_pos[:, self.arm_joint_ids] - self.arm_motion.target).abs().mean()
        feet_yaw = torch.rad2deg(feet_heading_error(self))
        extras["log"]["Stance/feet_heading_abs_deg"] = feet_yaw.abs().mean()
        # Positive yaw is outward on the left foot and inward on the right.
        inward = torch.stack((-feet_yaw[:, 0], feet_yaw[:, 1]), dim=-1).clamp(min=0.0)
        extras["log"]["Stance/toe_in_deg"] = inward.mean()
        extras["log"]["Stance/both_feet_within_5deg"] = (feet_yaw.abs() <= 5.0).all(dim=-1).float().mean()
        if self.min_foot_clearance_m:
            extras["log"]["Stance/sole_gap_below_min_fraction"] = (self.sole_gap_m < self.min_foot_clearance_m).float().mean()
        if self.gathering_feet is not None:
            extras["log"]["Stance/gathering_feet_fraction"] = self.gathering_feet.float().mean()
            extras["log"]["Stance/gap_above_max_after_balance_fraction"] = self.wide_after_balance_fraction
        if self.recovering is not None:
            extras["log"]["Stance/recovering_fraction"] = self.recovering.float().mean()
            extras["log"]["Stance/relocks_per_step"] = self.stance_relocked.sum()
        if self.foot_recovery_until is not None:
            quiet = self.feet_should_be_planted
            count = quiet.sum().clamp_min(1)
            lifted = self.sole_height_m > self.cfg.reward.feet_lift.params["tolerance_m"]
            tipped = (self.sole_high_m - self.sole_height_m) > self.cfg.reward.feet_lift.params["edge_tolerance_m"]
            extras["log"]["FootContact/quiet_fraction"] = quiet.float().mean()
            extras["log"]["FootContact/quiet_any_foot_lift_fraction"] = (quiet & lifted.any(dim=1)).sum() / count
            extras["log"]["FootContact/quiet_both_feet_lift_fraction"] = (quiet & lifted.all(dim=1)).sum() / count
            extras["log"]["FootContact/quiet_any_foot_tipped_fraction"] = (quiet & tipped.any(dim=1)).sum() / count
        waist_deg = torch.rad2deg(self.robot.data.joint_pos[:, self.waist_joint_ids]).abs()
        for index, axis in enumerate(("x", "y", "z")):
            extras["log"][f"Waist/{axis}_abs_deg"] = waist_deg[:, index].mean()
        extras["log"]["Waist/all_axes_within_3deg"] = (waist_deg <= 3.0).all(dim=-1).float().mean()
        for name, flags in self.failure_reasons.items():
            extras["log"][f"Failure/{name}_fraction"] = flags.float().mean()
        return obs, reward, done, extras

    def update_stance_reference(self):
        data = self.robot.data
        push = self.push_applied
        if self.torso_force is not None:
            push = push | self.torso_force.started | (self.torso_force.remaining > 0)
        loads = self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, 2]
        cfg = self.cfg.stance_recovery
        speed_sq = data.root_lin_vel_w[:, :2].square().sum(-1)
        tilt_cos = -data.projected_gravity_b[:, 2]
        tilt_rate_sq = data.root_ang_vel_b[:, :2].square().sum(-1)
        self.recovering, self.recovery_hold, self.recovery_stable, self.stance_relocked = advance_stance_reference(
            cfg, self.step_dt, self.recovering, self.recovery_hold, self.recovery_stable,
            push, speed_sq, tilt_cos, tilt_rate_sq, loads[:, 0], loads[:, 1])
        self.stand_anchor_xy = torch.where(self.stance_relocked[:, None], data.root_pos_w[:, :2], self.stand_anchor_xy)
        lock_feet = self.stance_relocked
        if self.gathering_feet is not None:
            bilateral = (loads >= cfg["foot_load_n"]).all(dim=1)
            quiet = stance_is_quiet(cfg, bilateral, speed_sq, tilt_cos, tilt_rate_sq) & ~push
            wide = self.sole_gap_m > self.max_foot_clearance_m
            gap_in_band = (self.sole_gap_m >= self.min_foot_clearance_m) & ~wide
            self.gathering_feet, self.feet_stable_ticks, lock_feet = advance_foot_reference(
                self.gathering_feet, self.feet_stable_ticks, self.recovering, self.stance_relocked,
                quiet, gap_in_band, self.feet_settle_steps)
            balanced = ~self.recovering
            self.wide_after_balance_fraction = (wide & balanced).sum() / balanced.sum().clamp_min(1)
        self.feet_anchor_xy = torch.where(lock_feet[:, None, None],
                                          data.body_pos_w[:, self.feet_body_ids, :2], self.feet_anchor_xy)

    def check_reset(self):
        if self.min_foot_clearance_m or self.gathering_feet is not None:
            # Geometry, footprint locking and rewards all use the same pre-reset frame.
            self.sole_gap_m = sole_inner_gap(self)
        if self.recovering is not None:
            self.update_stance_reference()
        if self.foot_recovery_until is not None:
            self.sole_height_m, self.sole_high_m = sole_height_bounds(self)
            self.feet_should_be_planted = update_foot_plant_mask(
                self.foot_recovery_until, self.sim_step_counter, self.foot_recovery_steps,
                self.push_applied, self.torso_force,
            )
            if self.recovering is not None:
                self.feet_should_be_planted &= ~self.recovering
            if self.gathering_feet is not None:
                self.feet_should_be_planted &= ~self.gathering_feet
        contact_force = self.contact_sensor.data.net_forces_w_history[:, :, self.termination_contact_cfg.body_ids]
        contact_failure = (contact_force.norm(dim=-1).amax(dim=1) > 1.0).any(dim=-1)
        timed_out = self.episode_length_buf >= self.max_episode_length
        height = self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        tilt = self.robot.data.projected_gravity_b[:, 2] > -math.cos(self.cfg.max_tilt_rad)
        drift = (self.robot.data.root_pos_w[:, :2] - self.stand_anchor_xy).norm(dim=-1) > self.cfg.max_drift_m
        if self.recovering is not None:
            drift &= ~self.recovering
        self.failure_reasons = {
            "contact": contact_failure,
            "height": height < self.cfg.min_height,
            "tilt": tilt,
            "drift": drift,
        }
        failed = contact_failure | self.failure_reasons["height"] | tilt | self.failure_reasons["drift"]
        # Simultaneous falls and time limits are failures, not bootstrapped timeouts.
        return failed | timed_out, timed_out & ~failed

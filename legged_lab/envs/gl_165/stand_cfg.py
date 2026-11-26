"""Flat-ground, zero-command PPO standing task for GL-165."""

import hashlib
import json
import math
from copy import deepcopy

from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from pathlib import Path

from legged_lab.assets.gl_165.model import (
    CONTROL_DT, CURRENT_PHYSICS_PROFILE, FEET, LEGACY_PHYSICS_PROFILE, PHYSICS_DT,
    POLICY_JOINT_PATTERN, STAND_HEIGHT, WIDE_STANCE, load_physics,
)
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg

from . import rewards
from legged_lab.assets.gl_165.disturbances import MIXED_PUSH, VELOCITY_ONLY
from legged_lab.assets.gl_165.stance_recovery import DEFAULT_STANCE_RECOVERY


# The signed forward offset of the two sole centres is zero in the GL-165 reference pose.
GL165_QUIET_STANCE = {
    "minimum_m": 0.0581, "maximum_m": 0.12, "outside_timeout_s": 2.0,
    "gap_scale_m": 0.03, "gap_weight": -4.0, "plant_load_n": 40.0, "release_load_n": 5.0,
    "speed_scale_m_s": 0.05, "drift_scale_m": 0.02, "stance_scale_m": 0.06, "slide_weight": -2.0,
    "fore_aft": {"target_m": 0.0, "tolerance_m": 0.03, "scale_m": 0.03, "weight": -4.0},
    "training_episode_s": 120.0, "no_push_env_fraction": 0.25,
}


def quiet_inward_slip_term(weight=-1.0, full_gap_m=0.0581, fade_distance_m=0.0081,
                           speed_scale_m_s=0.01, foot_load_n=DEFAULT_STANCE_RECOVERY["foot_load_n"],
                           patch_tolerance_m=0.001):
    return RewTerm(func=rewards.quiet_inward_slip, weight=weight, params={
        "full_gap_m": full_gap_m, "fade_distance_m": fade_distance_m, "speed_scale_m_s": speed_scale_m_s,
        "foot_load_n": foot_load_n, "patch_tolerance_m": patch_tolerance_m,
    })


def feet_width_term(maximum_m=0.12, scale_m=0.10, weight=-1.5, *, all_phases=True):
    params = {"maximum_m": maximum_m, "scale_m": scale_m}
    if all_phases:
        params["all_phases"] = True
    return RewTerm(func=rewards.feet_width_excess, weight=weight, params=params)


def foot_load_shortfall_term(minimum_n=150.0, weight=-1.0):
    if not 0 < minimum_n < math.inf or not -math.inf < weight < 0:
        raise ValueError("Foot load shortfall requires a positive finite load and a negative finite weight")
    return RewTerm(func=rewards.foot_load_shortfall, weight=weight, params={"minimum_n": minimum_n})


def _recorded_feet_drift_term(recorded):
    # Before this weight was recorded, every GL-165 run used -10 for footprint drift.
    weight = recorded.get("feet_drift_reward_weight", -10.0)
    return RewTerm(func=rewards.feet_drift, weight=weight) if weight else None


@configclass
class TorsoForceCfg:
    initial_force_n: float = 20.0
    final_force_n: float = 100.0
    initial_duration_s: float = 0.3
    final_duration_s: float = 0.8
    full_iteration: int = 20000
    event_probability: float = 0.5


@configclass
class PushCfg:
    mode: str = VELOCITY_ONLY
    enabled: bool = True
    interval_s: tuple = (6.0, 12.0)
    # World-frame root COM linear velocity (m/s) and yaw velocity (rad/s).
    velocity_ranges: dict = {"x": (-0.8, 0.8), "y": (-0.8, 0.8), "yaw": (0.0, 0.0)}
    curriculum_enabled: bool = False
    curriculum_duration_s: float = 0.0
    initial_scale: float = 1.0
    torso_force: TorsoForceCfg | None = None

    def velocity_profile(self):
        return {
            "interval_s": list(self.interval_s),
            "velocity_ranges": {axis: list(bounds) for axis, bounds in self.velocity_ranges.items()},
            "curriculum": ({"initial_scale": self.initial_scale, "duration_s": self.curriculum_duration_s}
                           if self.curriculum_enabled else None),
        }


@configclass
class ArmMotionCfg:
    enabled: bool = True
    database: str = str(Path(__file__).parent / "datasets/arm_disturbance_overhead_261123_2_5ms_stance8cm.npz")
    database_sha256: str = ""
    curriculum_enabled: bool = True
    initial_scale: float = 0.02
    warmup_iterations: int = 100
    full_motion_iteration: int = 1500


@configclass
class StandRewardCfg:
    alive = RewTerm(func=mdp.alive, weight=1.0)
    upright = RewTerm(func=rewards.torso_upright, weight=3.0)
    torso_sway: RewTerm | None = RewTerm(func=rewards.torso_sway, weight=-1.0,
                                          params={"rate_scale_rad_s": 0.5, "ramp_s": 1.0})
    height = RewTerm(func=rewards.standing_height, weight=2.0, params={"target_height": STAND_HEIGHT})
    stationary = RewTerm(func=rewards.stationary, weight=2.0)
    position = RewTerm(func=rewards.keep_position, weight=1.5)
    heading = RewTerm(func=rewards.keep_heading, weight=-1.0)
    support = RewTerm(func=rewards.both_feet_loaded, weight=1.0)
    feet_drift: RewTerm | None = RewTerm(func=rewards.feet_drift, weight=-10.0)
    feet_lift: RewTerm | None = RewTerm(func=rewards.quiet_feet_lift, weight=-2.0, params={
        "tolerance_m": 0.005, "scale_m": 0.02, "edge_tolerance_m": 0.015, "edge_fraction": 0.25,
    })
    feet_clearance: RewTerm | None = RewTerm(
        func=rewards.feet_clearance_shortfall, weight=-1.0, params={"minimum_m": 0.03},
    )
    feet_inward_slip: RewTerm | None = None
    feet_stagger: RewTerm | None = None
    feet_width: RewTerm | None = None
    quiet_sole_gap: RewTerm | None = None
    quiet_foot_stagger: RewTerm | None = None
    quiet_foot_slide: RewTerm | None = None
    quiet_target_curvature: RewTerm | None = None
    feet_heading = RewTerm(func=rewards.feet_heading, weight=-10.0,
                          params={"tolerance_rad": math.radians(3.0)})
    feet_slide = RewTerm(func=mdp.feet_slide, weight=-0.5,
                         params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=FEET),
                                 "asset_cfg": SceneEntityCfg("robot", body_names=FEET)})
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    pose = RewTerm(func=rewards.nominal_pose, weight=-0.3)
    waist_zero = RewTerm(func=rewards.waist_zero, weight=-5.0)
    effort = RewTerm(func=rewards.normalized_effort, weight=-0.05)
    joint_velocity = RewTerm(func=mdp.joint_vel_l2, weight=-0.005,
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=POLICY_JOINT_PATTERN)})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    terminated = RewTerm(func=mdp.is_terminated, weight=-50.0)


@configclass
class Gl165StandEnvCfg(BaseEnvCfg):
    physics_profile: str = CURRENT_PHYSICS_PROFILE
    nominal_stance: str | None = WIDE_STANCE
    reward = StandRewardCfg()
    push = PushCfg()
    arm_motion = ArmMotionCfg()
    observe_arm_motion: bool = False
    observe_standing_state: bool = False
    stance_recovery: dict | None = DEFAULT_STANCE_RECOVERY.copy()
    quiet_foot_stance: dict | None = None
    interpolate_arm_targets: bool = False
    min_height: float = 0.75
    max_tilt_rad: float = 0.65
    max_drift_m: float = 0.60
    foot_recovery_s: float = 2.0
    sole_slide_foot_load_n: float = DEFAULT_STANCE_RECOVERY["foot_load_n"]
    # Start with the reference sole gap so initial settling has room above the 5 cm reward threshold.
    min_reset_foot_clearance_m: float = 0.0581
    gather_pelvis_tolerance_m: float = 0.10

    def set_actor_observations(self, *, standing_state, arm_motion=False):
        if arm_motion and not standing_state:
            raise ValueError("The historical arm observation profile requires standing observations")
        self.observe_standing_state = standing_state
        self.observe_arm_motion = arm_motion
        self.robot.actor_obs_history_length = 5 if standing_state else 10

    def _restore_feet_slide(self, recorded):
        slide = recorded.get("loaded_sole_slide")
        if slide is None:
            self.reward.feet_slide = RewTerm(
                func=mdp.feet_slide, weight=-0.5,
                params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=FEET),
                        "asset_cfg": SceneEntityCfg("robot", body_names=FEET)},
            )
        else:
            self.sole_slide_foot_load_n = slide["foot_load_n"]
            self.reward.feet_slide = RewTerm(func=rewards.loaded_sole_slide, weight=slide["weight"])

    def _restore_support(self, recorded):
        shortfall = recorded.get("foot_load_shortfall")
        if shortfall is None:
            self.reward.support = RewTerm(func=rewards.both_feet_loaded, weight=1.0)
        else:
            self.reward.support = foot_load_shortfall_term(shortfall["minimum_n"], shortfall["weight"])
            self.foot_recovery_s = shortfall["post_push_recovery_s"]

    def set_quiet_stance(self, settings, *, interpolate_arms=False, curvature_weight=0.0):
        if not math.isfinite(curvature_weight) or curvature_weight > 0:
            raise ValueError("Target curvature weight must be finite and nonpositive")
        self.quiet_foot_stance = deepcopy(settings)
        self.interpolate_arm_targets = interpolate_arms
        self.reward.quiet_target_curvature = (
            RewTerm(func=rewards.quiet_target_curvature, weight=curvature_weight) if curvature_weight else None
        )
        if settings is None:
            self.reward.quiet_sole_gap = self.reward.quiet_foot_stagger = self.reward.quiet_foot_slide = None
            self.scene.max_episode_length_s = 20.0
            return
        stagger = settings["fore_aft"]
        if not (0 < settings["minimum_m"] < settings["maximum_m"] < math.inf
                and 0 <= settings["release_load_n"] < settings["plant_load_n"] < math.inf
                and 0 <= settings["no_push_env_fraction"] <= 1 and math.isfinite(stagger["target_m"])
                and all(0 < value < math.inf for value in (
                    settings["outside_timeout_s"], settings["training_episode_s"], settings["gap_scale_m"],
                    settings["speed_scale_m_s"], settings["drift_scale_m"], settings["stance_scale_m"],
                    stagger["scale_m"], stagger["tolerance_m"]))
                and all(-math.inf < weight < 0 for weight in (settings["gap_weight"], settings["slide_weight"], stagger["weight"]))):
            raise ValueError("Invalid quiet stance limits, reward scales or training scenario")
        self.reward.quiet_sole_gap = RewTerm(func=rewards.quiet_sole_gap, weight=settings["gap_weight"], params={
            "minimum_m": settings["minimum_m"], "maximum_m": settings["maximum_m"], "scale_m": settings["gap_scale_m"]})
        self.reward.quiet_foot_stagger = RewTerm(func=rewards.quiet_foot_stagger, weight=stagger["weight"],
                                               params={k: stagger[k] for k in ("tolerance_m", "scale_m")})
        self.reward.quiet_foot_slide = RewTerm(func=rewards.quiet_foot_slide, weight=settings["slide_weight"],
                                             params={k: settings[k] for k in ("speed_scale_m_s", "drift_scale_m", "stance_scale_m")})
        self.scene.max_episode_length_s = settings["training_episode_s"]

    def use_xmz_standing(self):
        self.reward.feet_inward_slip = self.reward.feet_stagger = self.reward.feet_width = None
        self.reward.feet_drift = _recorded_feet_drift_term({})
        self._restore_feet_slide({})
        self._restore_support({})
        self.set_min_foot_clearance(0.03, weight=-1.0)
        self.set_quiet_stance(GL165_QUIET_STANCE, interpolate_arms=True, curvature_weight=-8.0)

    def restore_standing_profile(self, recorded):
        self.set_quiet_stance(recorded.get("quiet_foot_stance"),
                              interpolate_arms=recorded.get("interpolate_arm_targets", False),
                              curvature_weight=recorded.get("quiet_target_curvature_weight", 0.0))
        self.set_physics_profile(recorded.get("physics_profile", LEGACY_PHYSICS_PROFILE), recorded.get("nominal_stance"))
        sway = recorded.get("torso_sway")
        self.reward.torso_sway = (RewTerm(func=rewards.torso_sway, weight=sway["weight"],
                                         params={k: sway[k] for k in ("rate_scale_rad_s", "ramp_s")})
                                  if sway is not None else None)
        if sway is not None:
            self.foot_recovery_s = sway["post_push_recovery_s"]
        self.stance_recovery = recorded.get("stance_recovery")
        self.reward.feet_drift = _recorded_feet_drift_term(recorded)
        self._restore_feet_slide(recorded)
        self._restore_support(recorded)
        stagger = recorded.get("post_push_feet_stagger")
        self.reward.feet_stagger = (RewTerm(func=rewards.post_push_feet_stagger, weight=stagger["weight"],
                                           params={k: stagger[k] for k in ("tolerance_m", "scale_m")})
                                    if stagger is not None else None)
        # All GL-165 checkpoints written before this field used the 525-input policy.
        self.set_actor_observations(standing_state=recorded.get("observe_standing_state", True),
                                    arm_motion=recorded.get("observe_arm_motion", True))
        self.restore_push_profile(recorded)
        minimum = recorded.get("min_foot_clearance_m", 0.0)
        # Checkpoints predating the recorded weight used -1 for every enabled gap penalty.
        weight = recorded.get("foot_clearance_reward_weight", -1.0) if minimum else None
        self.set_min_foot_clearance(minimum, weight=weight)
        self.min_reset_foot_clearance_m = recorded.get("min_reset_foot_clearance_m", 0.0)
        slip = recorded.get("quiet_inward_slip")
        self.reward.feet_inward_slip = quiet_inward_slip_term(**slip) if slip is not None else None
        gather = recorded.get("foot_gathering")
        if gather is None:
            self.reward.feet_width = None
        else:
            self.gather_pelvis_tolerance_m = gather["pelvis_tolerance_m"]
            self.reward.feet_width = feet_width_term(
                gather["maximum_m"], gather["scale_m"], gather["weight"], all_phases=gather.get("all_phases", False),
            )
        saved = recorded.get("quiet_foot_contact")
        if saved is None:
            self.reward.feet_lift = None
        else:
            self.foot_recovery_s = saved["post_push_recovery_s"]
            params = {key: saved[key] for key in ("tolerance_m", "scale_m", "edge_tolerance_m", "edge_fraction")}
            self.reward.feet_lift = RewTerm(func=rewards.quiet_feet_lift, weight=saved["weight"], params=params)

    def restore_for_playback(self, checkpoint, arm_database=None):
        checkpoint = Path(checkpoint).resolve()
        profile = checkpoint.parent / "params/controller_parameters.json"
        if profile.is_file():
            recorded = json.loads(profile.read_text())
        else:
            import torch

            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            recorded = saved.get("standing_profile")
            if recorded is None:
                weights = saved["model_state_dict"]
                widths = (weights["actor.0.weight"].shape[1], weights["critic.0.weight"].shape[1])
                if widths not in ((310, 365), (525, 580)):
                    raise ValueError(f"Unsupported GL-165 actor/critic input dimensions {widths}")
                self.stance_recovery = None
                self.set_quiet_stance(None)
                self.set_physics_profile(LEGACY_PHYSICS_PROFILE)
                self.reward.torso_sway = None
                self.reward.feet_drift = _recorded_feet_drift_term({})
                self._restore_feet_slide({})
                self._restore_support({})
                self.reward.feet_stagger = None
                self.reward.feet_width = None
                self.reward.feet_inward_slip = None
                self.set_min_foot_clearance(0.0)
                self.min_reset_foot_clearance_m = 0.0
                self.set_actor_observations(standing_state=True, arm_motion=widths == (525, 580))
                self.restore_push_profile({"push_mode": MIXED_PUSH, "torso_force_curriculum": {"final_duration_s": 1.5}})
                self.arm_motion.database = str(arm_database or Path(__file__).parent / "datasets/arm_disturbance_overhead_xml_5ms_v5.npz")
                return
        self.restore_standing_profile(recorded)
        if arm_database is not None:
            self.arm_motion.database = str(Path(arm_database).resolve(strict=True))
            return
        # The same actor width was used with different arm timings. Prefer the run's own copy.
        expected = recorded.get("arm_database_sha256")
        copies = sorted((checkpoint.parent / "params/arm_motion").glob("*.npz"))
        for candidate in [*copies, *sorted((Path(__file__).parent / "datasets").glob("*.npz"))]:
            if (expected and hashlib.sha256(candidate.read_bytes()).hexdigest() == expected) or (
                    not expected and len(copies) == 1 and candidate == copies[0]):
                self.arm_motion.database = str(candidate)
                return
        if expected:
            raise FileNotFoundError(f"Missing recorded GL-165 arm database with SHA256 {expected}")
        self.arm_motion.database = str(Path(__file__).parent / "datasets/arm_disturbance_overhead_xml_5ms_v5.npz")

    def restore_push_profile(self, recorded):
        self.push.mode = recorded.get("push_mode", VELOCITY_ONLY)
        # Older GL-165 snapshots recorded only the mode and the torso-force schedule.
        velocity = recorded.get("velocity_push", {
            "interval_s": (8.0, 12.0),
            "velocity_ranges": {"x": (-0.8, 0.8), "y": (-0.8, 0.8), "yaw": (-0.2, 0.2)},
            "curriculum": {"initial_scale": 0.05, "duration_s": 4800.0},
        })
        self.push.interval_s = tuple(velocity["interval_s"])
        self.push.velocity_ranges = {axis: tuple(bounds) for axis, bounds in velocity["velocity_ranges"].items()}
        curriculum = velocity["curriculum"]
        self.push.curriculum_enabled = curriculum is not None
        self.push.initial_scale = curriculum["initial_scale"] if curriculum is not None else 1.0
        self.push.curriculum_duration_s = curriculum["duration_s"] if curriculum is not None else 0.0
        self.push.torso_force = None
        if self.push.mode == MIXED_PUSH:
            self.push.torso_force = TorsoForceCfg()
            self.push.torso_force.from_dict({"final_duration_s": 1.5, **recorded["torso_force_curriculum"]})

    def set_final_torso_force_duration(self, duration_s):
        if self.push.mode != MIXED_PUSH:
            raise ValueError("A torso-force duration requires a profile with sustained torso pushes")
        if not math.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError("The final torso-force duration must be finite and positive (seconds)")
        self.push.torso_force.final_duration_s = duration_s

    def set_physics_profile(self, profile, nominal_stance=None):
        physics = load_physics(profile, nominal_stance).parameters
        self.physics_profile = profile
        self.nominal_stance = nominal_stance
        self.sim.dt = physics["physics_dt_s"]
        self.sim.decimation = round(physics["control_dt_s"] / self.sim.dt)
        self.reward.height.params["target_height"] = physics["stand_height_m"]

    def set_min_foot_clearance(self, minimum_m=None, *, weight=None):
        current = self.reward.feet_clearance
        if minimum_m is None:
            minimum_m = current.params["minimum_m"] if current is not None else 0.0
        if not math.isfinite(minimum_m) or minimum_m < 0:
            raise ValueError("Minimum foot clearance must be finite and nonnegative (metres)")
        if minimum_m == 0:
            if weight is not None:
                raise ValueError("A foot clearance penalty weight requires a positive minimum foot clearance")
            self.reward.feet_clearance = None
            return
        if weight is None:
            weight = current.weight if current is not None else -1.0
        if not math.isfinite(weight) or weight >= 0:
            raise ValueError("Foot clearance penalty weight must be finite and negative")
        self.reward.feet_clearance = RewTerm(
            func=rewards.feet_clearance_shortfall, weight=weight, params={"minimum_m": minimum_m},
        )

    def set_max_foot_clearance(self, maximum_m):
        """Set an all-phase upper bound, including when starting from an older profile."""
        if not math.isfinite(maximum_m) or maximum_m < 0.0:
            raise ValueError("Maximum foot clearance must be finite and nonnegative (metres)")
        if maximum_m == 0.0:
            self.reward.feet_width = None
            return
        lower = self.reward.feet_clearance
        if lower is not None and maximum_m <= lower.params["minimum_m"]:
            raise ValueError("Maximum foot clearance must exceed the minimum gap")
        if self.reward.feet_width is None:
            self.reward.feet_width = feet_width_term(maximum_m)
        else:
            self.reward.feet_width.params.update(maximum_m=maximum_m, all_phases=True)

    def set_max_foot_clearance_weight(self, weight):
        if self.reward.feet_width is None:
            raise ValueError("A foot-width penalty weight requires a positive maximum foot clearance")
        if not math.isfinite(weight) or weight >= 0:
            raise ValueError("The foot-width penalty weight must be finite and negative")
        self.reward.feet_width.weight = weight

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = PHYSICS_DT
        self.sim.decimation = round(CONTROL_DT / PHYSICS_DT)
        if not math.isclose(self.sim.dt * self.sim.decimation, CONTROL_DT):
            raise ValueError("XML physics timestep must divide the policy control period")
        self.scene.num_envs = 4096
        self.scene.env_spacing = 2.5
        self.set_quiet_stance(GL165_QUIET_STANCE, interpolate_arms=True, curvature_weight=-8.0)
        self.scene.terrain_type = "plane"
        self.scene.terrain_generator = None
        self.scene.height_scanner.enable_height_scan = False
        self.set_actor_observations(standing_state=self.observe_standing_state, arm_motion=self.observe_arm_motion)
        self.robot.critic_obs_history_length = 5
        self.robot.feet_body_names = FEET
        self.robot.terminate_contacts_body_names = [
            "base_link", "body_y_link", "head_y_link", "[lr]hip_y_link", "[lr]bigleg_.*",
            "[lr]smallleg_y_link", "[lr]shouder_.*", "[lr]bigarm_z_link",
            "[lr]smallarm_.*", "[lr]wrist_.*",
        ]
        # Apply the policy output directly; joint-space limits are handled in process_actions.
        self.normalization.clip_actions = None
        self.normalization.obs_scales.joint_vel = 0.1
        self.commands.rel_standing_envs = 1.0
        self.commands.rel_heading_envs = 0.0
        self.commands.heading_command = False
        self.commands.debug_vis = False
        self.commands.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.ranges.heading = (0.0, 0.0)
        self.noise.noise_scales.ang_vel = 0.1
        self.noise.noise_scales.joint_vel = 0.5
        events = self.domain_rand.events
        events.add_base_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names="base_link")
        events.add_base_mass.params["mass_distribution_params"] = (-0.5, 0.5)
        events.physics_material.params["static_friction_range"] = (0.7, 1.0)
        events.physics_material.params["dynamic_friction_range"] = (0.6, 0.9)
        events.reset_base.params["pose_range"] = {
            "x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.1, 0.1),
        }
        events.reset_base.params["velocity_range"] = {
            "x": (-0.03, 0.03), "y": (-0.03, 0.03), "yaw": (-0.05, 0.05),
        }
        events.reset_robot_joints.func = mdp.reset_joints_by_offset
        events.reset_robot_joints.params = {"position_range": (-0.015, 0.015), "velocity_range": (-0.05, 0.05)}
        # Gl165StandEnv installs the requested interval event after CLI overrides.
        events.push_robot = None
        self.domain_rand.action_delay.enable = False


@configclass
class Gl165StandAgentCfg(BaseAgentCfg):
    runner_class_name = "OnPolicyRunner"
    experiment_name = "stand_gl165"
    logger = "tensorboard"
    max_iterations = 80000
    save_interval = 100

    def __post_init__(self):
        super().__post_init__()
        self.policy.init_noise_std = 0.3
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]
        self.algorithm.entropy_coef = 0.003

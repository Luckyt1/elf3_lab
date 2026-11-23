"""Flat-ground, zero-command PPO standing task for GL-165."""

import hashlib
import json
import math

from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from pathlib import Path

from legged_lab.assets.gl_165.model import CONTROL_DT, FEET, PHYSICS_DT, POLICY_JOINT_PATTERN, STAND_HEIGHT, portable_digest
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg

from . import rewards
from legged_lab.assets.gl_165.disturbances import MIXED_PUSH, VELOCITY_ONLY
from legged_lab.assets.gl_165.stance_recovery import DEFAULT_STANCE_RECOVERY


@configclass
class TorsoForceCfg:
    initial_force_n: float = 20.0
    final_force_n: float = 100.0
    initial_duration_s: float = 0.3
    final_duration_s: float = 1.5
    full_iteration: int = 20000
    event_probability: float = 0.5


@configclass
class PushCfg:
    mode: str = MIXED_PUSH
    enabled: bool = True
    interval_s: tuple = (8.0, 12.0)
    # World-frame root COM linear velocity (m/s) and yaw velocity (rad/s).
    velocity_ranges: dict = {"x": (-0.8, 0.8), "y": (-0.8, 0.8), "yaw": (-0.2, 0.2)}
    curriculum_enabled: bool = True
    curriculum_duration_s: float = 4800.0
    initial_scale: float = 0.05
    torso_force = TorsoForceCfg()


@configclass
class ArmMotionCfg:
    enabled: bool = True
    database: str = str(Path(__file__).parent / "datasets/arm_disturbance_overhead_xml_5ms_v5.npz")
    database_sha256: str = ""
    curriculum_enabled: bool = True
    initial_scale: float = 0.02
    warmup_iterations: int = 100
    full_motion_iteration: int = 1500


@configclass
class StandRewardCfg:
    alive = RewTerm(func=mdp.alive, weight=1.0)
    upright = RewTerm(func=rewards.torso_upright, weight=3.0)
    height = RewTerm(func=rewards.standing_height, weight=2.0, params={"target_height": STAND_HEIGHT})
    stationary = RewTerm(func=rewards.stationary, weight=2.0)
    position = RewTerm(func=rewards.keep_position, weight=1.5)
    heading = RewTerm(func=rewards.keep_heading, weight=-1.0)
    support = RewTerm(func=rewards.both_feet_loaded, weight=1.0)
    feet_drift = RewTerm(func=rewards.feet_drift, weight=-10.0)
    feet_lift: RewTerm | None = RewTerm(func=rewards.quiet_feet_lift, weight=-2.0, params={
        "tolerance_m": 0.005, "scale_m": 0.02, "edge_tolerance_m": 0.015, "edge_fraction": 0.25,
    })
    feet_clearance: RewTerm | None = RewTerm(
        func=rewards.feet_clearance_shortfall, weight=-1.0, params={"minimum_m": 0.03},
    )
    feet_width: RewTerm | None = None
    feet_heading = RewTerm(func=rewards.feet_heading, weight=-10.0,
                          params={"tolerance_rad": math.radians(3.0)})
    feet_slide = RewTerm(
        func=mdp.feet_slide, weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=FEET),
                "asset_cfg": SceneEntityCfg("robot", body_names=FEET)},
    )
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
    reward = StandRewardCfg()
    push = PushCfg()
    arm_motion = ArmMotionCfg()
    observe_arm_motion: bool = False
    observe_standing_state: bool = False
    stance_recovery: dict | None = DEFAULT_STANCE_RECOVERY.copy()
    min_height: float = 0.75
    max_tilt_rad: float = 0.65
    max_drift_m: float = 0.60
    foot_recovery_s: float = 2.0
    gather_pelvis_tolerance_m: float = 0.10

    def set_actor_observations(self, *, standing_state, arm_motion=False):
        if arm_motion and not standing_state:
            raise ValueError("The historical arm observation profile requires standing observations")
        self.observe_standing_state = standing_state
        self.observe_arm_motion = arm_motion
        self.robot.actor_obs_history_length = 5 if standing_state else 10

    def restore_standing_profile(self, recorded):
        self.stance_recovery = recorded.get("stance_recovery")
        # All GL-165 checkpoints written before this field used the 525-input policy.
        self.set_actor_observations(standing_state=recorded.get("observe_standing_state", True),
                                    arm_motion=recorded.get("observe_arm_motion", True))
        self.restore_push_profile(recorded)
        self.set_min_foot_clearance(recorded.get("min_foot_clearance_m", 0.0))
        gather = recorded.get("foot_gathering")
        if gather is None:
            self.reward.feet_width = None
        else:
            self.gather_pelvis_tolerance_m = gather["pelvis_tolerance_m"]
            self.reward.feet_width = RewTerm(func=rewards.feet_width_excess, weight=gather["weight"],
                                             params={k: gather[k] for k in ("maximum_m", "scale_m")})
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
                self.reward.feet_width = None
                self.set_min_foot_clearance(0.0)
                self.set_actor_observations(standing_state=True, arm_motion=widths == (525, 580))
                self.push.torso_force.final_force_n = 100.0
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
            if (expected and portable_digest(hashlib.sha256(candidate.read_bytes()).hexdigest()) == portable_digest(expected)) or (
                    not expected and len(copies) == 1 and candidate == copies[0]):
                self.arm_motion.database = str(candidate)
                return
        if expected:
            raise FileNotFoundError(f"Missing recorded GL-165 arm database with SHA256 {expected}")
        self.arm_motion.database = str(Path(__file__).parent / "datasets/arm_disturbance_overhead_xml_5ms_v5.npz")

    def restore_push_profile(self, recorded):
        self.push.mode = recorded.get("push_mode", VELOCITY_ONLY)
        if self.push.mode == MIXED_PUSH:
            self.push.torso_force.from_dict(recorded["torso_force_curriculum"])

    def set_min_foot_clearance(self, minimum_m):
        if not math.isfinite(minimum_m) or minimum_m < 0:
            raise ValueError("Minimum foot clearance must be finite and nonnegative (metres)")
        self.reward.feet_clearance = RewTerm(
            func=rewards.feet_clearance_shortfall, weight=-1.0, params={"minimum_m": minimum_m},
        ) if minimum_m > 0 else None

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = PHYSICS_DT
        self.sim.decimation = round(CONTROL_DT / PHYSICS_DT)
        if not math.isclose(self.sim.dt * self.sim.decimation, CONTROL_DT):
            raise ValueError("XML physics timestep must divide the policy control period")
        self.scene.num_envs = 4096
        self.scene.env_spacing = 2.5
        self.scene.max_episode_length_s = 20.0
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

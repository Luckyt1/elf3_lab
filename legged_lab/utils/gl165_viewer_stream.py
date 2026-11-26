"""Publish GL-165 PhysX states for the lightweight MuJoCo display process."""

import json
from pathlib import Path
import time

from legged_lab.envs.gl_165.rewards import feet_heading_error


class Gl165ViewerStream:
    def __init__(self, path, checkpoint):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.control_path = self.path.with_suffix(".control.json")
        self.checkpoint = str(checkpoint)
        self.reset_request = 0
        self.sequence = 0
        self.resets = 0
        self.last_reset = "none"
        self.last_velocity_push = None

    def controls(self):
        if not self.control_path.exists():
            return False, False
        control = json.loads(self.control_path.read_text())
        request = int(control.get("reset", 0))
        reset = request != self.reset_request
        self.reset_request = request
        return bool(control.get("paused", False)), reset

    def publish(self, env, done=None):
        if env.num_envs != 1:
            raise ValueError("The GL-165 state viewer requires exactly one environment")
        if done is not None and bool(done[0]):
            self.resets += 1
            reasons = [name for name, flags in env.failure_reasons.items() if bool(flags[0])]
            self.last_reset = "+".join(reasons) or "time limit"
        origin = env.scene.env_origins[0]
        now = time.time()
        if done is None or bool(done[0]):
            self.last_velocity_push = None
        elif bool(env.push_applied[0]):
            delta = env.push_delta_w[0].tolist()
            self.last_velocity_push = {"xy_m_s": delta[:2], "yaw_rad_s": delta[5], "timestamp": now}
        state = {
            "sequence": self.sequence,
            "timestamp": now,
            "checkpoint": self.checkpoint,
            "physics": "Isaac Sim / PhysX",
            "action_clip_enabled": env.clip_actions is not None,
            "clip_actions": env.clip_actions,
            "arm_database_sha256": env.arm_motion.sha256,
            "arm_motion_profile": env.arm_motion.metadata.get("profile", "random"),
            "arm_segment_duration_s": env.arm_motion.metadata.get("segment_duration_s", 1.0),
            "feet_heading_deg": feet_heading_error(env)[0].rad2deg().tolist(),
            "waist_angle_deg": env.robot.data.joint_pos[0, env.waist_joint_ids].rad2deg().tolist(),
            "episode_seconds": float(env.episode_length_buf[0]) * env.step_dt,
            "root_position": (env.robot.data.root_pos_w[0] - origin).tolist(),
            "root_quaternion_wxyz": env.robot.data.root_quat_w[0].tolist(),
            "joint_names": env.robot.joint_names,
            "joint_positions": env.robot.data.joint_pos[0].tolist(),
            "body_names": env.robot.body_names,
            "body_positions": (env.robot.data.body_pos_w[0] - origin).tolist(),
            "push_mode": env.cfg.push.mode,
            "last_velocity_push": self.last_velocity_push,
            "drift_m": float((env.robot.data.root_pos_w[0, :2] - env.stand_anchor_xy[0]).norm()),
            "resets": self.resets,
            "last_reset": self.last_reset,
        }
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, allow_nan=False))
        temporary.replace(self.path)
        self.sequence += 1

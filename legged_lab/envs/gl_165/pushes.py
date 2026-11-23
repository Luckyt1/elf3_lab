"""Standing perturbations scheduled independently for each GL-165 environment."""

import math

import torch

from legged_lab.assets.gl_165.disturbances import torso_force_parameters


def update_foot_plant_mask(deadline, step, recovery_steps, velocity_event, torso_force):
    """Extend each affected environment's recovery period through force release and settling."""
    until = torch.where(velocity_event, step + recovery_steps, 0)
    if torso_force is not None:
        until = torch.where(torso_force.started, step + torso_force.remaining + recovery_steps, until)
    torch.maximum(deadline, until, out=deadline)
    return step >= deadline


class TorsoForce:
    """Hold a horizontal world force at the torso COM for a whole number of physics steps."""

    def __init__(self, robot, torso_id, physics_dt):
        self.torso_id = torso_id
        self.dt = physics_dt
        count = robot.num_instances
        self.indices = torch.arange(count, device=robot.device, dtype=torch.int32)
        self.remaining = torch.zeros(count, device=robot.device, dtype=torch.long)
        self.started = torch.zeros(count, device=robot.device, dtype=torch.bool)
        self.force_w = torch.zeros(count, 3, device=robot.device)
        self.local_forces = torch.zeros(count, robot.num_bodies, 3, device=robot.device)
        # PhysX interprets these application points in the link frame. Only the
        # torso receives force; the GL-165 task keeps its local COM fixed.
        self.local_points = robot.data.com_pos_b.contiguous().view(-1, 3)

    def start(self, env_ids, magnitude, duration):
        angle = torch.rand(len(env_ids), device=self.force_w.device) * (2 * math.pi)
        self.force_w[env_ids, 0] = magnitude * angle.cos()
        self.force_w[env_ids, 1] = magnitude * angle.sin()
        self.remaining[env_ids] = max(1, round(duration / self.dt))
        self.started[env_ids] = True

    def reset(self, env_ids):
        self.remaining[env_ids] = 0
        self.force_w[env_ids] = 0.

    def apply(self, robot):
        from isaaclab.utils.math import quat_rotate_inverse

        force = self.force_w * (self.remaining > 0).unsqueeze(1)
        self.local_forces[:, self.torso_id] = quat_rotate_inverse(robot.data.body_quat_w[:, self.torso_id], force)
        robot.root_physx_view.apply_forces_and_torques_at_position(
            force_data=self.local_forces.view(-1, 3), torque_data=None, position_data=self.local_points,
            indices=self.indices, is_global=False,
        )
        self.remaining.sub_(1).clamp_(min=0)


def apply_mixed_push(env, env_ids):
    cfg = env.cfg.push.torso_force
    use_force = torch.rand(len(env_ids), device=env.device) < cfg.event_probability
    force_ids, velocity_ids = env_ids[use_force], env_ids[~use_force]
    if len(force_ids):
        env.torso_force.start(force_ids, *torso_force_parameters(cfg, env.learning_iteration))
    if len(velocity_ids):
        apply_velocity_push(env, velocity_ids)


def velocity_push_scale(cfg, simulated_seconds):
    if not cfg.enabled:
        return 0.0
    if not cfg.curriculum_enabled:
        return 1.0
    progress = min(max(simulated_seconds / cfg.curriculum_duration_s, 0.0), 1.0)
    return cfg.initial_scale + (1.0 - cfg.initial_scale) * progress


def apply_velocity_push(env, env_ids):
    # Import after Isaac Sim is running, as with the other Isaac Lab events.
    from isaaclab.envs.mdp.events import push_by_setting_velocity

    scale = env.push_scale()
    ranges = {axis: (low * scale, high * scale)
              for axis, (low, high) in env.cfg.push.velocity_ranges.items()}
    velocity_before = env.robot.data.root_vel_w[env_ids].clone()
    push_by_setting_velocity(env, env_ids, ranges)
    env.push_delta_w[env_ids] = env.robot.data.root_vel_w[env_ids] - velocity_before
    env.push_applied[env_ids] = True

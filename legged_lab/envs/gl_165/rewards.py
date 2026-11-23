"""Rewards for keeping a fixed standing pose while recovering from pushes."""

import torch


def yaw_from_quat(quat):
    # Isaac Lab quaternions are (w, x, y, z).
    w, x, y, z = quat.unbind(dim=-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def heading_error(env):
    delta = yaw_from_quat(env.robot.data.root_quat_w) - env.stand_anchor_yaw
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def torso_upright(env, std=0.25):
    q = env.robot.data.body_quat_w[:, env.torso_body_id]
    # World z component of the torso's local +z; upside-down is not rewarded.
    up_z = 1 - 2 * (q[:, 1].square() + q[:, 2].square())
    return torch.exp(-2 * (1 - up_z).clamp(min=0) / std**2)


def standing_height(env, target_height, std=0.06):
    height = env.robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.exp(-((height - target_height) / std).square())


def stationary(env, std=0.25):
    error = env.robot.data.root_lin_vel_b.square().sum(dim=-1)
    error += 0.1 * env.robot.data.root_ang_vel_b.square().sum(dim=-1)
    return torch.exp(-error / std**2)


def keep_position(env, std=0.15):
    error = (env.robot.data.root_pos_w[:, :2] - env.stand_anchor_xy).square().sum(dim=-1)
    if getattr(env, "gathering_feet", None) is not None:
        outside_tolerance = (error.sqrt() - env.cfg.gather_pelvis_tolerance_m).clamp_min(0.)
        error = torch.where(env.gathering_feet, outside_tolerance.square(), error)
    reward = torch.exp(-error / std**2)
    return reward if env.recovering is None else reward * ~env.recovering


def keep_heading(env):
    return heading_error(env).square()


def both_feet_loaded(env):
    forces = env.contact_sensor.data.net_forces_w[:, env.feet_cfg.body_ids, 2]
    loaded = (forces > 150.0).all(dim=-1).float()
    quiet = getattr(env, "feet_should_be_planted", None)
    if quiet is None and getattr(env, "gathering_feet", None) is not None:
        quiet = ~(env.recovering | env.gathering_feet)
    return loaded if quiet is None else loaded * quiet


def sole_height_bounds(env):
    """Lowest and highest XML sphere bottoms on each sole above the flat ground."""
    from isaaclab.utils.math import quat_rotate_inverse

    feet = env.robot.data.body_pos_w[:, env.feet_body_ids]
    quats = env.robot.data.body_quat_w[:, env.feet_body_ids]
    up = torch.zeros_like(feet)
    up[..., 2] = 1.
    local_up = quat_rotate_inverse(quats, up)
    offsets = (env.sole_spheres[..., :3] * local_up.unsqueeze(2)).sum(dim=-1) - env.sole_spheres[..., 3]
    origin_z = feet[..., 2] - env.scene.env_origins[:, 2:3]
    low, high = torch.aminmax(offsets, dim=-1)
    return origin_z + low, origin_z + high


def quiet_feet_lift(env, tolerance_m, scale_m, edge_tolerance_m, edge_fraction):
    """Bounded per-foot cost for lift and excess pitch or roll during quiet stance."""
    lift = ((env.sole_height_m - tolerance_m) / scale_m).clamp(0., 1.)
    span = env.sole_high_m - env.sole_height_m
    tipped = ((span - edge_tolerance_m) / scale_m).clamp(0., 1.)
    return (lift + edge_fraction * tipped).sum(dim=-1) * env.feet_should_be_planted


def feet_drift(env):
    feet_xy = env.robot.data.body_pos_w[:, env.feet_body_ids, :2]
    cost = (feet_xy - env.feet_anchor_xy).square().sum(dim=(1, 2))
    if getattr(env, "gathering_feet", None) is not None:
        cost = cost * ~env.gathering_feet
    return cost if env.recovering is None else cost * ~env.recovering


def sole_inner_gap(env):
    """Signed gap between the XML sole spheres projected onto the pelvis's horizontal left axis."""
    from isaaclab.utils.math import quat_rotate_inverse

    data = env.robot.data
    feet = data.body_pos_w[:, env.feet_body_ids]
    quats = data.body_quat_w[:, env.feet_body_ids]
    yaw = yaw_from_quat(data.root_quat_w)
    left = torch.stack((-yaw.sin(), yaw.cos(), torch.zeros_like(yaw)), dim=-1)
    lateral_width = ((feet[:, 0] - feet[:, 1]) * left).sum(dim=-1)
    local_left = quat_rotate_inverse(quats, left[:, None].expand_as(feet))
    offsets = (env.sole_spheres[..., :3] * local_left.unsqueeze(2)).sum(dim=-1)
    radii = env.sole_spheres[..., 3]
    left_inner = (offsets[:, 0] - radii[0]).amin(dim=-1)
    right_inner = (offsets[:, 1] + radii[1]).amax(dim=-1)
    return lateral_width + left_inner - right_inner


def feet_clearance_shortfall(env, minimum_m):
    return ((minimum_m - env.sole_gap_m).clamp_min(0.0) / minimum_m).square()


def feet_width_excess(env, maximum_m, scale_m):
    """Soft lateral upper bound once the pelvis has stabilized at its new position."""
    excess = (env.sole_gap_m - maximum_m).clamp_min(0.) / scale_m
    cost = torch.nn.functional.smooth_l1_loss(excess, torch.zeros_like(excess), reduction="none")
    return cost * ~env.recovering


def feet_heading_error(env):
    """Foot +x (toe direction) relative to the pelvis heading, in radians.

    Both GL-165 foot frames use +x toward the toes. Using the pelvis heading
    makes this independent of the randomized world yaw at reset.
    """
    foot_yaw = yaw_from_quat(env.robot.data.body_quat_w[:, env.feet_body_ids])
    base_yaw = yaw_from_quat(env.robot.data.root_quat_w)
    delta = foot_yaw - base_yaw[:, None]
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def feet_heading(env, tolerance_rad):
    # A small free region allows natural stance and balance corrections.
    excess = (feet_heading_error(env).abs() - tolerance_rad).clamp(min=0.0)
    return excess.square().sum(dim=-1)


def nominal_pose(env):
    return (env.robot.data.joint_pos - env.robot.data.default_joint_pos)[:, env.policy_joint_ids].square().sum(dim=-1)


def waist_zero(env):
    """Penalize actual waist roll, pitch and yaw away from joint zero (rad²)."""
    return env.robot.data.joint_pos[:, env.waist_joint_ids].square().sum(dim=-1)


def normalized_effort(env):
    return (env.robot.data.applied_torque / env.effort_limits)[:, env.policy_joint_ids].square().sum(dim=-1)

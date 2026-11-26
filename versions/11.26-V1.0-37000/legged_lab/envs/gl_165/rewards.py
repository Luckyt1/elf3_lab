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


def torso_sway(env, rate_scale_rad_s, ramp_s):
    """Penalize rotation of the chest's up axis, with a gradual return after a real push."""
    data, torso = env.robot.data, env.torso_body_id
    w, x, y, z = data.body_quat_w[:, torso].unbind(-1)
    up = torch.stack((2 * (x*z + w*y), 2 * (y*z - w*x), 1 - 2 * (x*x + y*y)), dim=-1)
    tilt_rate_sq = torch.linalg.cross(data.body_ang_vel_w[:, torso], up).square().sum(dim=-1)
    deadline = env.foot_recovery_until
    ramp = ((env.sim_step_counter - deadline) * (env.physics_dt / ramp_s)).clamp(0., 1.)
    weight_fraction = torch.where(deadline == 0, 1., ramp)
    return weight_fraction * tilt_rate_sq / (tilt_rate_sq + rate_scale_rad_s**2)


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


def foot_load_shortfall(env, minimum_n):
    """Sum each planted foot's missing fraction of the requested vertical load."""
    loads = env.contact_sensor.data.net_forces_w[:, env.feet_cfg.body_ids, 2]
    shortfall = ((minimum_n - loads) / minimum_n).clamp(0.0, 1.0)
    return shortfall.sum(dim=1) * env.feet_should_be_planted


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


def sole_inner_gap(env, env_ids=slice(None)):
    """Signed gap between the XML sole spheres projected onto the pelvis's horizontal left axis."""
    from isaaclab.utils.math import quat_rotate_inverse

    data = env.robot.data
    feet = data.body_pos_w[env_ids][:, env.feet_body_ids]
    quats = data.body_quat_w[env_ids][:, env.feet_body_ids]
    yaw = yaw_from_quat(data.root_quat_w[env_ids])
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


def quiet_sole_gap(env, minimum_m, maximum_m, scale_m):
    error = (minimum_m - env.sole_gap_m).clamp_min(0.) + (env.sole_gap_m - maximum_m).clamp_min(0.)
    return error / (error + scale_m) * env.quiet_gap_active


def signed_sole_stagger(centers_w, root_quat_w):
    """Left minus right sole centre along the pelvis's horizontal forward axis."""
    yaw = yaw_from_quat(root_quat_w)
    delta = centers_w[:, 0] - centers_w[:, 1]
    return delta[:, 0] * yaw.cos() + delta[:, 1] * yaw.sin()


def sole_fore_aft_offset(env):
    """Distance between sole centres along the pelvis's horizontal forward axis, in metres."""
    from isaaclab.utils.math import quat_apply

    data = env.robot.data
    feet = data.body_pos_w[:, env.feet_body_ids]
    offsets = quat_apply(data.body_quat_w[:, env.feet_body_ids], env.sole_centers_b.expand_as(feet))
    delta = (feet[:, 0] - feet[:, 1]) + (offsets[:, 0] - offsets[:, 1])
    yaw = yaw_from_quat(data.root_quat_w)
    return (delta[:, 0] * yaw.cos() + delta[:, 1] * yaw.sin()).abs()


def post_push_feet_stagger(env, tolerance_m, scale_m):
    excess = (env.fore_aft_offset_m - tolerance_m).clamp_min(0.) / scale_m
    cost = torch.nn.functional.smooth_l1_loss(excess, torch.zeros_like(excess), reduction="none")
    return cost * env.post_push_settled


def loaded_contact_speed(link_state, contact_data, slots, foot_load_n):
    """Ground-relative horizontal speed (m/s) of one foot's load-bearing PhysX contacts."""
    forces, points, normals, _, counts, starts = contact_data
    # Each of the nine sole spheres can contact the single flat ground collider once.
    active = slots < counts
    indices = torch.where(active, starts + slots, 0).long()
    vertical_load = (forces[indices, 0] * normals[indices, 2]).abs()
    vertical_load = torch.where(active, vertical_load, 0.0)
    lever = points[indices] - link_state[:, None, :3]
    velocity = link_state[:, None, 7:10] + torch.linalg.cross(link_state[:, None, 10:13], lever, dim=-1)
    speed = torch.linalg.vector_norm(velocity[..., :2], dim=-1)
    # Take magnitudes before averaging: opposite sliding directions at two points also count.
    weighted = torch.where(vertical_load > 0.0, vertical_load * speed, 0.0).sum(dim=1)
    load = vertical_load.sum(dim=1)
    return torch.where(load >= foot_load_n, weighted / load.clamp_min(foot_load_n), 0.0)


def loaded_sole_slide(env):
    # RewardManager integrates by step_dt. Return the mean substep speed so each metre counts once.
    return env.sole_slide_distance.sum(dim=1) / env.step_dt


def quiet_inward_slip(env, full_gap_m, fade_distance_m, speed_scale_m_s, foot_load_n, patch_tolerance_m):
    """Closing speed of the loaded sole surfaces along the pelvis's horizontal left axis."""
    from isaaclab.utils.math import quat_apply

    feet = env.robot.data.body_link_state_w[:, env.feet_body_ids]
    spheres = env.sole_spheres
    centers = spheres[None, ..., :3].expand(feet.shape[0], -1, -1, -1)
    rotations = feet[:, :, None, 3:7].expand(-1, -1, centers.shape[2], -1)
    offsets = quat_apply(rotations, centers)
    offsets[..., 2] -= spheres[..., 3]
    bottom = offsets[..., 2]
    patch = bottom <= bottom.amin(dim=-1, keepdim=True) + patch_tolerance_m
    # Average the material points nearest the ground. Include angular motion at the sole
    # so that an ankle rotating about a stationary contact is not counted as sliding.
    offset = (offsets * patch.unsqueeze(-1)).sum(dim=2) / patch.sum(dim=2, keepdim=True)
    velocity = feet[..., 7:10] + torch.linalg.cross(feet[..., 10:13], offset, dim=-1)
    relative = velocity[:, 1] - velocity[:, 0]
    yaw = yaw_from_quat(env.robot.data.root_quat_w)
    closing = (-yaw.sin() * relative[:, 0] + yaw.cos() * relative[:, 1]).clamp_min(0.)
    proximity = ((full_gap_m + fade_distance_m - env.sole_gap_m) / fade_distance_m).clamp(0., 1.)
    loads = env.contact_sensor.data.net_forces_w[:, env.feet_cfg.body_ids, 2]
    planted = env.feet_should_be_planted & (loads >= foot_load_n).all(dim=-1)
    return planted * proximity * (closing / (speed_scale_m_s + closing))


def feet_width_excess(env, maximum_m, scale_m, all_phases=False):
    """Penalize excess lateral sole gap. Historical profiles mask the recovery phase."""
    excess = (env.sole_gap_m - maximum_m).clamp_min(0.) / scale_m
    cost = torch.nn.functional.smooth_l1_loss(excess, torch.zeros_like(excess), reduction="none")
    return cost if all_phases else cost * ~env.recovering


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


def quiet_foot_stagger(env, tolerance_m, scale_m):
    excess = (env.stagger_error_m - tolerance_m).clamp_min(0.)
    return excess / (excess + scale_m) * env.quiet_gap_active


def quiet_target_curvature(env):
    """Mean squared second difference of the 15 limited waist/leg targets, in rad²."""
    valid = (env.episode_length_buf >= 3) & env.feet_should_be_planted
    return env.policy_target_curvature * valid


def update_plant_anchors(anchors, planted, xy, load, quiet, pushed, plant_load_n, release_load_n):
    """Keep each footprint through light unloading; release it on liftoff or an external push."""
    held = planted & (load >= release_load_n) & ~pushed[:, None]
    landing = ~held & (load >= plant_load_n) & (quiet & ~pushed)[:, None]
    return torch.where(landing[:, :, None], xy, anchors), held | landing


def update_quiet_stance_reference(reference, valid, xy, load, quiet, pushed, plant_load_n):
    """Keep the initial footprints across voluntary steps; relock after an external push."""
    valid = valid & ~pushed
    settled = ~valid & quiet & ~pushed & (load >= plant_load_n).all(dim=1)
    return torch.where(settled[:, None, None], xy, reference), valid | settled


def quiet_foot_slide(env, speed_scale_m_s, drift_scale_m, stance_scale_m=None):
    speed, drift = env.quiet_sole_speed, env.quiet_sole_drift
    cost = speed / (speed + speed_scale_m_s) + drift / (drift + drift_scale_m)
    if stance_scale_m is not None:
        offset = env.quiet_stance_drift
        cost = cost + 2. * offset / (offset + stance_scale_m) * env.quiet_stance_valid[:, None]
    return (cost * env.quiet_loaded_feet).mean(dim=1)

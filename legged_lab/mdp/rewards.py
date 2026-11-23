# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv


def action_rate_l2(env: BaseEnv) -> torch.Tensor:
    """Penalty action change rate.

    Compare the difference between the current action and the action at the previous moment to encourage smooth action sequences.

    Parameters:
        env: environment instance

    Return:
        torch.Tensor: action change rate penalty (sum of squares), shape [num_envs]
    """
    return torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )


def joint_vel_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocity for selected joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def flat_orientation_l2(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Punishes non-horizontal body posture.

    Check whether the body remains level (upright) by projecting the gravity vector into the body coordinate system.

    Parameters:
        env: environment instance
        asset_cfg: asset configuration (default uses robot)

    Return:
        torch.Tensor: body tilt penalty (sum of squares of gravity projection), shape [num_envs]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def is_terminated(env: BaseEnv) -> torch.Tensor:
    """Penalize early termination caused by non-timeout.

    Used to identify early termination due to a constraint violation (such as a fall) instead of the normal round timeout.

    Parameters:
        env: environment instance

    Return:
        torch.Tensor: Early termination penalty flag, shape [num_envs]
    """
    return env.reset_buf * ~env.time_out_buf


def feet_slide(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Punishing the foot for sliding on the ground.

    Calculate the horizontal speed of the foot when it touches the ground to prevent the foot from slipping.

    Parameters:
        env: environment instance
        sensor_cfg: Contact sensor configuration
        asset_cfg: asset configuration (default uses robot)

    Return:
        torch.Tensor: foot sliding penalty, shape is [num_envs]
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Detect whether the foot is in contact with the ground (contact force >1.0)
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    # Get the speed of the foot on the horizontal plane (xy)
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # Only penalizes sliding speed on contact
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def alive(env: BaseEnv) -> torch.Tensor:
    """Reward for staying alive.

    A simple constant reward that encourages the robot to not terminate early.
    Reference: DreamWaQ (HumanoidDreamWaq/legged_gym/envs/g1/g1_env.py)
    """
    return torch.ones(env.num_envs, device=env.device, dtype=torch.float)

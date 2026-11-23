"""Run a GL-165 540-input checkpoint in the packaged MuJoCo model."""

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from legged_lab.assets.gl_165.disturbances import torso_force_parameters
from legged_lab.assets.gl_165.model import (
    ARM_JOINT_NAMES, CONTROLLER_PROFILE, CONTROL_DT, MUJOCO_REFERENCE_PATH, PHYSICS_DT,
    TORSO, motion_asset_signature, portable_digest,
)
from legged_lab.envs.gl_165.arm_motion import ArmMotionDatabase
from rsl_rl.modules import ActorCritic

ARM_DATA = Path(__file__).resolve().parents[1] / "envs/gl_165/datasets/arm_disturbance_overhead_xml_5ms_v5.npz"


@torch.inference_mode()
def play(args):
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    profile = checkpoint.get("standing_profile", {})
    signature = motion_asset_signature()
    expected = {
        "profile": CONTROLLER_PROFILE, "actor_observation_frame_dim": 54, "actor_obs_history_length": 10,
        "observe_standing_state": False, "observe_arm_motion": False,
        "physics_dt_s": PHYSICS_DT, "control_dt_s": CONTROL_DT,
    }
    if any(profile.get(k) != v for k, v in expected.items()) or portable_digest(profile.get("asset_signature")) != signature:
        raise ValueError("This preview requires a GL-165 540-input checkpoint matching the supplied 5 ms model.")

    policy = ActorCritic(540, 365, 15, actor_hidden_dims=[256, 128, 128], critic_hidden_dims=[256, 128, 128],
                         activation="elu")
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    model = mujoco.MjModel.from_xml_path(str(MUJOCO_REFERENCE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("approved_standing").id)
    mujoco.mj_forward(model, data)
    base = model.body("base_link").id
    torso = model.body(TORSO).id
    joints = profile["policy_joint_names"] + ARM_JOINT_NAMES
    indices = [profile["joint_names"].index(name) for name in joints]
    qpos = [model.joint(name).qposadr[0] for name in joints]
    qvel = [model.joint(name).dofadr[0] for name in joints]
    actuators = [model.actuator(name).id for name in joints]
    nominal = np.asarray(profile["default_joint_position_rad"])[indices]
    kp = np.asarray(profile["kp"])[indices]
    kd = np.asarray(profile["kd"])[indices]
    effort = np.asarray(profile["effort_limits_nm"])[indices]
    lower, upper = np.asarray(profile["soft_position_limits_rad"])[indices].T
    scale = np.asarray(profile["action_scale_rad"])
    arms = ArmMotionDatabase(ARM_DATA, ARM_JOINT_NAMES, torch.tensor(nominal[None, 15:], dtype=torch.float32),
                             signature, 1, "cpu", CONTROL_DT)
    if portable_digest(profile["arm_database_sha256"]) != arms.sha256:
        raise ValueError("The checkpoint requires a different arm motion database.")

    action = np.zeros(15, dtype=np.float32)
    history = np.empty((10, 54), dtype=np.float32)
    angular_jac = np.empty((3, model.nv))
    command = np.zeros(3)
    force, duration = torso_force_parameters(SimpleNamespace(**profile["torso_force_curriculum"]), checkpoint["iter"])
    substeps = round(CONTROL_DT / model.opt.timestep)
    push_start = round(2.0 / model.opt.timestep)
    push_end = push_start + round(duration / model.opt.timestep)
    steps = 0
    view = nullcontext()
    if not args.headless:
        from mujoco import viewer
        view = viewer.launch_passive(model, data)
    with view as window:
        if window is not None:
            window.cam.lookat[:] = [0, 0, 0.9]
            window.cam.distance = 3.0
            window.cam.elevation = -12
            window.opt.geomgroup[3] = 0
        start = time.monotonic()
        for tick in range(math.ceil(args.duration / CONTROL_DT)):
            if window is not None and not window.is_running():
                break
            rotation = data.xmat[base].reshape(3, 3)
            mujoco.mj_jacBodyCom(model, data, None, angular_jac, base)
            frame = np.concatenate((rotation.T @ (angular_jac @ data.qvel), -rotation[2], command,
                                    data.qpos[qpos[:15]] - nominal[:15], 0.1 * data.qvel[qvel[:15]], action))
            if tick == 0:
                history[:] = np.clip(frame, -100, 100)
            else:
                history[:-1] = history[1:]
                history[-1] = np.clip(frame, -100, 100)
            action = policy.act_inference(torch.from_numpy(history.reshape(1, 540)))[0].numpy()
            target = np.concatenate((nominal[:15] + scale * action, arms.step(1.0)[0].numpy()))
            target = np.clip(target, lower, upper)
            for _ in range(substeps):
                torque = kp * (target - data.qpos[qpos]) - kd * data.qvel[qvel]
                data.ctrl[actuators] = np.clip(torque, -effort, effort)
                # Add our test force to any force from dragging the torso in the viewer.
                test_force = force if args.push and push_start <= steps < push_end else 0.
                data.xfrc_applied[torso, 1] += test_force
                mujoco.mj_step(model, data)
                data.xfrc_applied[torso, 1] -= test_force
                steps += 1
            mujoco.mj_forward(model, data)
            if not np.isfinite(data.qpos).all():
                raise RuntimeError("MuJoCo produced a non-finite pose.")
            if window is not None:
                window.sync()
                time.sleep(max(0., start + data.time - time.monotonic()))
    delivered_steps = max(0, min(steps, push_end) - push_start) if args.push else 0
    print(json.dumps({"iteration": checkpoint["iter"], "simulated_s": round(data.time, 4),
                      "base_height_m": round(data.xpos[base, 2], 4),
                      "torso_push_n": force if delivered_steps else 0.,
                      "torso_push_s": round(delivered_steps * model.opt.timestep, 4)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=30., help="Seconds of simulated time.")
    parser.add_argument("--seed", type=int, default=42, help="Arm clip sampling seed.")
    parser.add_argument("--headless", action="store_true", help="Run without a window.")
    parser.add_argument("--push", action="store_true", help="Push the torso in world +Y at 2 s using the checkpoint curriculum.")
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be positive and finite")
    play(args)

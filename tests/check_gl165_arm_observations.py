"""Verify the GL-165 observation contracts against a running Isaac scene."""

import argparse
import json
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from legged_lab.envs.gl_165.stand_cfg import Gl165StandEnvCfg
from legged_lab.envs.gl_165.stand_env import Gl165StandEnv


def check():
    cfg = Gl165StandEnvCfg()
    cfg.device, cfg.scene.num_envs = args.device, 8
    cfg.scene.seed = 42
    cfg.push.enabled = False
    env = Gl165StandEnv(cfg, args.headless)
    try:
        assert not cfg.observe_arm_motion and not cfg.observe_standing_state
        hardware_actor, hardware_critic = env.compute_current_observations()
        assert hardware_actor.shape == (8, 54) and hardware_critic.shape == (8, 73)
        assert env.noise_scale_vec.shape == (54,) and env.noise_scale_vec[39:].count_nonzero() == 0
        cfg.set_actor_observations(standing_state=True)
        base_actor, base_critic = env.compute_current_observations()
        assert base_actor.shape == (8, 62) and base_critic.shape == (8, 73)
        torch.testing.assert_close(hardware_actor, base_actor[:, :54])
        torch.testing.assert_close(hardware_critic, base_critic)
        cfg.observe_arm_motion = True
        old_actor, old_critic = env.compute_current_observations()
        assert old_actor.shape == (8, 105) and old_critic.shape == (8, 116)
        torch.testing.assert_close(torch.cat((old_actor[:, :54], old_actor[:, 97:]), 1), base_actor)
        torch.testing.assert_close(torch.cat((old_critic[:, :59], old_critic[:, 102:]), 1), base_critic)

        # Vary just the inputs of the observation builder without advancing physics.
        # Changing a leg reading below serves as a positive control.
        env.robot.data.joint_pos[:, env.arm_joint_ids] += 0.4
        env.robot.data.joint_vel[:, env.arm_joint_ids] -= 1.7
        env.arm_motion.target += 0.23
        env.arm_motion.frame.fill_(120)
        changed_legacy, _ = env.compute_current_observations()
        assert (changed_legacy[:, 54:97] != old_actor[:, 54:97]).all()
        cfg.observe_arm_motion = False
        actor, critic = env.compute_current_observations()
        assert torch.equal(actor, base_actor) and torch.equal(critic, base_critic)
        env.robot.data.joint_pos[:, env.policy_joint_ids[0]] += 0.1
        actor, critic = env.compute_current_observations()
        assert (actor[:, 9] - base_actor[:, 9]).abs().min() > .09
        assert (critic[:, 9] - base_critic[:, 9]).abs().min() > .09
        cfg.set_actor_observations(standing_state=False)
        previous = torch.linspace(-.7, .7, 15, device=env.device).expand(8, 15)
        env.action_buffer.compute(previous)
        env.recovering.fill_(True)
        actor, critic = env.compute_observations()
        assert actor.shape == (8, 540) and critic.shape == (8, 365)
        torch.testing.assert_close(actor.reshape(8, 10, 54)[:, -1, 39:], previous)
        assert critic.reshape(8, 5, 73)[:, :, -11:-9].count_nonzero() == 0
        return {"envs": env.num_envs, "actor_frame": 54, "actor_history": 10, "critic_frame": 73,
                "arm_position_velocity_target_phase_changed": True, "max_arm_only_difference": 0.0,
                "waist_leg_feedback_detected": True, "legacy_frame_comparison_passed": True}
    finally:
        env.sim.clear_all_callbacks()
        env.sim.clear_instance()


try:
    result = check()
    result["passed"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")
except Exception:
    args.output.write_text(json.dumps({"passed": False, "error": traceback.format_exc()}, indent=2) + "\n")
    traceback.print_exc()
    raise
finally:
    app.close(wait_for_replicator=False)

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

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--total_iterations", type=int, default=None,
                    help="Absolute iteration target; resume after the checkpoint's last completed iteration.")
parser.add_argument("--arm-database", type=Path, help="Override the GL-165 arm disturbance database.")
parser.add_argument("--standing-baseline", type=Path,
                    help="Start a new six-input GL-165 run using this checkpoint's recorded standing and push settings.")
parser.add_argument("--torso-force-final-duration-s", type=float,
                    help="Final duration of the GL-165 sustained torso push curriculum, in seconds.")
parser.add_argument("--min-foot-clearance-m", type=float,
                    help="GL-165 soft minimum lateral gap between the sole edges in metres (0 disables).")
parser.add_argument("--max-foot-clearance-m", type=float,
                    help="GL-165 all-phase soft maximum lateral gap between sole edges in metres (0 disables).")
parser.add_argument("--max-foot-clearance-weight", type=float,
                    help="Override the negative penalty weight for exceeding the GL-165 maximum sole gap.")
parser.add_argument("--foot-clearance-weight", type=float,
                    help="Override the negative weight of the GL-165 sole-gap shortfall penalty.")
parser.add_argument("--min-reset-foot-clearance-m", type=float,
                    help="GL-165 minimum measured sole-edge gap in the sampled start pose (0 disables).")
parser.add_argument("--quiet-inward-slip", action="store_true",
                    help="Enable the GL-165 quiet-stance inward sliding cost when continuing an older profile.")
parser.add_argument("--foot-load-shortfall", action="store_true",
                    help="Replace the GL-165 double-support bonus with the per-foot load shortfall penalty.")
parser.add_argument("--xmz-standing", action="store_true",
                    help="Apply the GL-165 quiet stance, footprint, smoothing and 120 s scenario settings.")
parser.add_argument("--no-feet-drift", action="store_true",
                    help="Remove the GL-165 footprint penalty when resuming a checkpoint trained with it.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.arm_database is not None and args_cli.task != "stand_gl165":
    parser.error("--arm-database is supported only for stand_gl165")
if args_cli.standing_baseline is not None and (args_cli.task != "stand_gl165" or args_cli.resume):
    parser.error("--standing-baseline requires a fresh stand_gl165 run")
if args_cli.task != "stand_gl165" and (
    args_cli.min_foot_clearance_m is not None or args_cli.max_foot_clearance_m is not None
    or args_cli.foot_clearance_weight is not None or args_cli.max_foot_clearance_weight is not None
    or args_cli.min_reset_foot_clearance_m is not None or args_cli.quiet_inward_slip or args_cli.no_feet_drift
    or args_cli.torso_force_final_duration_s is not None or args_cli.foot_load_shortfall or args_cli.xmz_standing
):
    parser.error("Standing foot rewards and torso-force settings are supported only for stand_gl165")
if args_cli.total_iterations is not None and args_cli.total_iterations <= 0:
    parser.error("--total_iterations must be positive")
# Start camera rendering
if "sensor" in args_cli.task:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import os
from datetime import datetime

import torch
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def train():
    runner: OnPolicyRunner | AmpOnPolicyRunner

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)
    env_class = task_registry.get_task_class(env_class_name)
    if args_cli.arm_database is not None:
        env_cfg.arm_motion.database = str(args_cli.arm_database.resolve(strict=True))

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed
    baseline = None
    if args_cli.standing_baseline is not None:
        if agent_cfg.resume:
            raise ValueError("A configuration baseline starts with a fresh model and optimizer")
        baseline = args_cli.standing_baseline.resolve(strict=True)
        env_cfg.restore_for_playback(baseline, args_cli.arm_database)
        env_cfg.set_actor_observations(standing_state=False)

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.scene.seed = seed
        agent_cfg.seed = seed

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        if env_class_name == "stand_gl165":
            recorded = json.loads((Path(resume_path).parent / "params/controller_parameters.json").read_text())
            env_cfg.restore_for_playback(resume_path, args_cli.arm_database)
    if args_cli.xmz_standing:
        env_cfg.use_xmz_standing()
    if (baseline is not None and env_cfg.quiet_foot_stance is None) or args_cli.no_feet_drift:
        env_cfg.reward.feet_drift = None
    if args_cli.torso_force_final_duration_s is not None:
        env_cfg.set_final_torso_force_duration(args_cli.torso_force_final_duration_s)
    if args_cli.min_foot_clearance_m is not None or args_cli.foot_clearance_weight is not None:
        env_cfg.set_min_foot_clearance(args_cli.min_foot_clearance_m, weight=args_cli.foot_clearance_weight)
    if args_cli.max_foot_clearance_m is not None:
        env_cfg.set_max_foot_clearance(args_cli.max_foot_clearance_m)
    if args_cli.max_foot_clearance_weight is not None:
        env_cfg.set_max_foot_clearance_weight(args_cli.max_foot_clearance_weight)
    if args_cli.min_reset_foot_clearance_m is not None:
        env_cfg.min_reset_foot_clearance_m = args_cli.min_reset_foot_clearance_m
    if args_cli.quiet_inward_slip:
        from legged_lab.envs.gl_165.stand_cfg import quiet_inward_slip_term
        env_cfg.reward.feet_inward_slip = quiet_inward_slip_term()
    if args_cli.foot_load_shortfall:
        from legged_lab.envs.gl_165.stand_cfg import foot_load_shortfall_term
        env_cfg.reward.support = foot_load_shortfall_term()
    env = env_class(env_cfg, args_cli.headless)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    runner_class: OnPolicyRunner | AmpOnPolicyRunner = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    if agent_cfg.resume:
        if args_cli.task == "stand_gl165":
            for key in ("profile", "source_xml_sha256", "asset_signature", "physics_dt_s", "control_dt_s", "push_mode", "stance_recovery", "arm_database_sha256"):
                if recorded.get(key) != env.controller_parameters.get(key):
                    raise ValueError(f"Cannot resume GL-165 checkpoint with different {key}; start a new run")
            if "velocity_push" in recorded and recorded["velocity_push"] != env.controller_parameters["velocity_push"]:
                raise ValueError("Cannot resume GL-165 checkpoint with different velocity disturbances")
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
        if args_cli.task == "stand_gl165":
            # The Tang push curriculum uses elapsed simulation time, including
            # completed rollouts when resuming a checkpoint of the same profile.
            env.sim_step_counter = (runner.current_learning_iteration + 1) * runner.num_steps_per_env * env.cfg.sim.decimation
        if args_cli.total_iterations is not None:
            # Checkpoints store the iteration whose optimizer update finished.
            runner.current_learning_iteration += 1
            if hasattr(runner.alg, "learning_rate"):
                runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]["lr"]

    iterations_to_run = agent_cfg.max_iterations
    if args_cli.total_iterations is not None:
        iterations_to_run = args_cli.total_iterations - runner.current_learning_iteration
        if iterations_to_run <= 0:
            raise ValueError("The checkpoint has already reached --total_iterations")
        agent_cfg.max_iterations = iterations_to_run
        agent_cfg.total_iterations = args_cli.total_iterations
        runner.cfg["max_iterations"] = iterations_to_run
        print(f"[INFO] Continuing at iteration {runner.current_learning_iteration}; "
              f"target {args_cli.total_iterations}; remaining {iterations_to_run}.", flush=True)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    if args_cli.task == "stand_gl165":
        import hashlib
        import shutil
        import zipfile
        physics_source = env.physics_asset.directory
        Path(log_dir, "params", "controller_parameters.json").write_text(
            json.dumps(env.controller_parameters, indent=2) + "\n"
        )
        params_dir = Path(log_dir, "params")
        arm_dir = params_dir / "arm_motion"
        arm_dir.mkdir(exist_ok=True)
        shutil.copy2(env.arm_motion.path, arm_dir / env.arm_motion.path.name)
        physics_dir = params_dir / "physical_model"
        physics_dir.mkdir(exist_ok=True)
        for path in physics_source.iterdir():
            if path.is_file() and path.suffix in (".xml", ".json", ".urdf"):
                shutil.copy2(path, physics_dir / path.name)
        if "source_urdf_copy" in env.physics_asset.parameters:
            for folder in ("source", "meshes"):
                shutil.copytree(physics_source / folder, physics_dir / folder)
        source_root = Path(__file__).resolve().parents[2]
        archive_path = params_dir / "source_snapshot.zip"
        source_files = list((source_root / "legged_lab").rglob("*.py"))
        source_files.append(source_root / "legged_lab/envs/gl_165/mujoco_velocity_push.cc")
        source_files += list((source_root / "rsl_rl/rsl_rl").rglob("*.py"))
        source_files += list((source_root / "tests").glob("*.py"))
        source_files += list((source_root / "scripts").glob("*.sh"))
        source_files += [source_root / "AGENTS.md"]
        source_files += [p for p in physics_source.iterdir() if p.is_file() and p.suffix in (".xml", ".json", ".urdf")]
        if "source_urdf_copy" in env.physics_asset.parameters:
            source_files += [p for folder in ("source", "meshes") for p in (physics_source / folder).rglob("*") if p.is_file()]
        source_files += [source_root / "legged_lab/assets/gl_165/urdf/gl_165.urdf"]
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(set(source_files)):
                archive.write(path, path.relative_to(source_root))
        (params_dir / "provenance.json").write_text(json.dumps({
            "source_snapshot_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "fresh_optimizer": not agent_cfg.resume,
            "starting_iteration": runner.current_learning_iteration,
            **({"parent_checkpoint": str(resume_path),
                "parent_checkpoint_sha256": hashlib.sha256(Path(resume_path).read_bytes()).hexdigest()}
               if agent_cfg.resume else {}),
            **({"configuration_baseline": str(baseline),
                "configuration_baseline_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest()}
               if baseline is not None else {}),
            "arm_database_sha256": env.arm_motion.sha256,
            "source_xml_sha256": env.controller_parameters["source_xml_sha256"],
            "source_urdf_sha256": env.controller_parameters["source_urdf_sha256"],
        }, indent=2) + "\n")

    runner.learn(num_learning_iterations=iterations_to_run, init_at_random_ep_len=True)
    env.sim.clear_all_callbacks()
    env.sim.clear_instance()


if __name__ == "__main__":
    train()
    simulation_app.close()

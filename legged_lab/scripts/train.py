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
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from isaaclab.app import AppLauncher

from rsl_rl.runners import OnPolicyRunner

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", choices=["stand_gl165"], default="stand_gl165", help="Standing task name.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--total_iterations", type=int, default=None,
                    help="Absolute iteration target; resume after the checkpoint's last completed iteration.")
parser.add_argument("--arm-database", type=Path, help="Override the GL-165 arm disturbance database.")
parser.add_argument("--standing-baseline", type=Path,
                    help="Start a new six-input GL-165 run using this checkpoint's recorded standing and push settings.")
parser.add_argument("--min-foot-clearance-m", type=float,
                    help="GL-165 soft minimum lateral gap between the sole edges in metres (0 disables).")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.standing_baseline is not None and args_cli.resume:
    parser.error("--standing-baseline requires a fresh stand_gl165 run")
if args_cli.total_iterations is not None and args_cli.total_iterations <= 0:
    parser.error("--total_iterations must be positive")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import os
from datetime import datetime

import torch
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.assets.gl_165.model import PHYSICS_DIR, portable_digest
from legged_lab.envs.gl_165.stand_cfg import Gl165StandAgentCfg, Gl165StandEnvCfg
from legged_lab.envs.gl_165.stand_env import Gl165StandEnv
from legged_lab.utils.cli_args import update_rsl_rl_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def train():
    env_cfg, agent_cfg = Gl165StandEnvCfg(), Gl165StandAgentCfg()
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
        recorded = json.loads((Path(resume_path).parent / "params/controller_parameters.json").read_text())
        env_cfg.restore_for_playback(resume_path, args_cli.arm_database)
    if args_cli.min_foot_clearance_m is not None:
        env_cfg.set_min_foot_clearance(args_cli.min_foot_clearance_m)
    env = Gl165StandEnv(env_cfg, args_cli.headless)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    if agent_cfg.resume:
        for key in ("profile", "source_xml_sha256", "asset_signature", "physics_dt_s", "control_dt_s", "push_mode", "stance_recovery", "arm_database_sha256"):
            old, new = recorded.get(key), env.controller_parameters.get(key)
            if key in ("asset_signature", "arm_database_sha256"):
                old, new = portable_digest(old), portable_digest(new)
            if old != new:
                raise ValueError(f"Cannot resume GL-165 checkpoint with different {key}; start a new run")
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
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
    Path(log_dir, "params", "controller_parameters.json").write_text(
        json.dumps(env.controller_parameters, indent=2) + "\n"
    )
    params_dir = Path(log_dir, "params")
    arm_dir = params_dir / "arm_motion"
    arm_dir.mkdir(exist_ok=True)
    shutil.copy2(env.arm_motion.path, arm_dir / env.arm_motion.path.name)
    physics_dir = params_dir / "physical_model"
    physics_dir.mkdir(exist_ok=True)
    for path in PHYSICS_DIR.iterdir():
        if path.is_file() and path.suffix in (".xml", ".json", ".urdf"):
            shutil.copy2(path, physics_dir / path.name)
    source_root = Path(__file__).resolve().parents[2]
    archive_path = params_dir / "source_snapshot.zip"
    source_files = list((source_root / "legged_lab").rglob("*.py"))
    source_files += list((source_root / "rsl_rl/rsl_rl").rglob("*.py"))
    source_files += list((source_root / "tests").glob("*.py"))
    source_files += list((source_root / "scripts").glob("*.sh"))
    source_files += [source_root / "README.md"]
    source_files += [p for p in PHYSICS_DIR.iterdir() if p.is_file() and p.suffix in (".xml", ".json", ".urdf")]
    source_files += [source_root / "legged_lab/assets/gl_165/urdf/gl_165.urdf"]
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(source_files)):
            archive.write(path, path.relative_to(source_root))
    (params_dir / "provenance.json").write_text(json.dumps({
        "source_snapshot_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "fresh_optimizer": not agent_cfg.resume,
        "starting_iteration": runner.current_learning_iteration,
        **({"configuration_baseline": str(baseline),
            "configuration_baseline_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest()}
           if baseline is not None else {}),
        "arm_database_sha256": env.arm_motion.sha256,
        "source_xml_sha256": env.controller_parameters["source_xml_sha256"],
    }, indent=2) + "\n")

    runner.learn(num_learning_iterations=iterations_to_run, init_at_random_ep_len=True)
    env.sim.clear_all_callbacks()
    env.sim.clear_instance()


if __name__ == "__main__":
    train()
    simulation_app.close()

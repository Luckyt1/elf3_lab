"""Start the native GL-165 ROS/MuJoCo viewer with the training run's disturbances."""

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import yaml
from legged_lab.assets.gl_165.disturbances import MIXED_PUSH, VELOCITY_ONLY, torso_force_parameters


def prepare(runtime, mod, env_file, directory, *, torso_force_n=None, arm_database=None):
    # Saved Isaac configs contain Python tuple tags. BaseLoader reads their values as data.
    env = yaml.load(env_file.read_text(), Loader=yaml.BaseLoader)
    push = env["push"]
    if push.get("mode") not in (VELOCITY_ONLY, MIXED_PUSH) or push["enabled"].lower() != "true":
        raise ValueError("Select the env.yaml from an enabled GL-165 standing disturbance run")
    if torso_force_n is not None and (push["mode"] != MIXED_PUSH or not math.isfinite(torso_force_n) or torso_force_n <= 0):
        raise ValueError("A preview torso force requires mixed disturbances and a positive finite force")
    spec = json.loads((mod / "assets/policy_interface.json").read_text())
    training_observes_arms = env.get("observe_arm_motion", "true").lower() == "true"
    training_observes_standing = env.get("observe_standing_state", "true").lower() == "true"
    if spec.get("observe_standing_state", True) != training_observes_standing:
        raise ValueError("The preview actor and training config use different standing observations")
    if spec.get("observe_arm_motion", True) != training_observes_arms:
        raise ValueError("The preview actor and the recorded training config use different arm observations")
    saved_recovery = env.get("stance_recovery")
    if isinstance(saved_recovery, dict):
        saved_recovery = {key: float(value) for key, value in saved_recovery.items()}
    else:
        saved_recovery = None
    if spec.get("stance_recovery") != saved_recovery:
        raise ValueError("The preview actor and training config use different standing references")
    dt = float(env["sim"]["dt"])
    control_dt = dt * int(env["sim"]["decimation"])
    if (dt, control_dt) != (spec["physics_dt_s"], spec["control_dt_s"]):
        raise ValueError("Preview and training timesteps differ")
    policy_share = mod.parents[1]
    if policy_share.name != "bxi_example_py_elf3" or mod.parent.name != "mods":
        raise ValueError("--mod-root must belong to the exported BXI ROS package share")

    directory.mkdir(parents=True, exist_ok=False)
    arm_override = None
    if arm_database is not None:
        import numpy as np
        arm_database = Path(arm_database).resolve(strict=True)
        folder = directory / "arm_motion"
        folder.mkdir()
        copy = folder / arm_database.name
        shutil.copy2(arm_database, copy)
        with np.load(copy, allow_pickle=False) as data, np.load(mod / "assets" / spec["arm_motion_source"], allow_pickle=False) as trained:
            report = json.loads(str(data["metadata"]))
            if (report["asset_signature"] != json.loads(str(trained["metadata"]))["asset_signature"]
                    or list(data["joint_names"]) != spec["arm_joint_order"]
                    or not math.isclose(float(data["control_dt"]), control_dt)):
                raise ValueError("Preview arm library differs in robot, joint order or control timestep")
        arm_override = dict(file=str(copy.relative_to(directory)), sha256=hashlib.sha256(copy.read_bytes()).hexdigest(),
                            segment_duration_s=report["segment_duration_s"], source=str(arm_database))
    overlay = directory / "ros"
    vendor = runtime / "bxi_ros2_pkg"
    executable_dir = overlay / "lib/mujoco"
    plugin_dir = executable_dir / "mujoco_plugin"
    plugin_dir.mkdir(parents=True)
    # Native simulate looks for plugins next to its executable. Give this session
    # its own executable directory and use the installed simulator and SDK.
    shutil.copy2(vendor / "lib/mujoco/simulation", executable_dir / "simulation")
    (executable_dir / "lib").symlink_to(vendor / "lib/mujoco/lib")
    markers = overlay / "share/ament_index/resource_index/packages"
    markers.mkdir(parents=True)
    for package, share in (("mujoco", vendor / "share/mujoco"), ("bxi_example_py_elf3", policy_share)):
        (markers / package).touch()
        (overlay / "share" / package).symlink_to(share)
    sdk = Path(subprocess.check_output([
        str(runtime / "controller_env/bin/python"), "-c",
        "import pathlib,mujoco; print(pathlib.Path(mujoco.__file__).parent)",
    ], text=True).strip())
    vendor_lib = (executable_dir / "lib/libmujoco.so").resolve()
    if not (sdk / vendor_lib.name).is_file():
        raise ValueError(f"MuJoCo SDK does not match {vendor_lib.name}")
    source = Path(__file__).resolve().parents[1] / "envs/gl_165/mujoco_velocity_push.cc"
    subprocess.run(["c++", "-std=c++17", "-O2", "-fPIC", "-shared", "-I", str(sdk / "include"),
                    str(source), "-L", str(vendor_lib.parent), "-lmujoco",
                    "-o", str(plugin_dir / "libgl165_velocity_push.so")], check=True)

    # Keep the verified policy/physics bundle intact. The simulator receives a
    # derived MJCF and the controller still discovers the original policy via ament.
    scene = directory / "scene"
    model_dir = scene / "assets/mujoco"
    model_dir.mkdir(parents=True)
    for name in ("tools", "configs"):
        (scene / name).symlink_to(mod / name)
    for source_dir, dest_dir, skip in ((mod / "assets", scene / "assets", "mujoco"),
                                      (mod / "assets/mujoco", model_dir, "gl165.xml")):
        for path in source_dir.iterdir():
            if path.name != skip:
                (dest_dir / path.name).symlink_to(path)
    tree = ET.parse(mod / "assets/mujoco/gl165.xml")
    root = tree.getroot()
    extension = root.find("extension")
    if extension is None:
        extension = ET.SubElement(root, "extension")
    plugin = ET.SubElement(extension, "plugin", plugin="gl165.tang_velocity_push")
    instance = ET.SubElement(plugin, "instance", name="standing_velocity_push")
    bounds = {"interval_s": push["interval_s"], **push["velocity_ranges"]}
    for key, values in bounds.items():
        ET.SubElement(instance, "config", key=key, value=" ".join(values))
    ET.SubElement(instance, "config", key="control_dt", value=str(control_dt))
    torso = None
    if push["mode"] == MIXED_PUSH:
        cfg = SimpleNamespace(**{k: float(v) for k, v in push["torso_force"].items()})
        iteration = int(spec["checkpoint_iteration"])
        force, duration = torso_force_parameters(cfg, iteration)
        torso = dict(iteration=iteration, force_n=force, duration_s=duration, event_probability=cfg.event_probability)
        if torso_force_n is not None:
            torso["checkpoint_force_n"] = force
            torso["force_n"] = force = torso_force_n
        for key, value in (("force_probability", cfg.event_probability), ("force_n", force), ("duration_s", duration)):
            ET.SubElement(instance, "config", key=key, value=str(value))
    tree.write(model_dir / "gl165.xml", encoding="utf-8", xml_declaration=True)
    info = {"pid": os.getpid(), "checkpoint": spec["checkpoint"], "checkpoint_training_run": spec["training_run"],
            "checkpoint_sha256": spec["checkpoint_sha256"], "policy_mod": str(mod),
            "disturbance_run_config": str(env_file), "mode": push["mode"], "scale": 1.0,
            "interval_s": list(map(float, push["interval_s"])),
            "velocity_ranges": {k: list(map(float, v)) for k, v in push["velocity_ranges"].items()},
            "model": "scene/assets/mujoco/gl165.xml", "overlay": "ros"}
    if torso:
        info["torso_force"] = torso
    info["arm_segment_duration_s"] = spec["arm_segment_duration_s"]
    if arm_override:
        info["arm_motion_override"] = arm_override
        info["checkpoint_arm_segment_duration_s"] = info["arm_segment_duration_s"]
        info["arm_segment_duration_s"] = arm_override["segment_duration_s"]
    info["observe_arm_motion"] = training_observes_arms
    info["observe_standing_state"] = training_observes_standing
    info["actor_observations"] = spec["observation_dim"]
    (directory / "session.json").write_text(json.dumps(info, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True, help="Prepared BXI ROS runtime")
    parser.add_argument("--mod-root", type=Path, required=True, help="Exported GL-165 policy Mod")
    parser.add_argument("--training-config", type=Path, required=True, help="Standing run's params/env.yaml")
    parser.add_argument("--output-dir", type=Path, help="New directory for this preview session")
    parser.add_argument("--torso-force-n", type=float, help="Override the torso force for this preview")
    parser.add_argument("--arm-database", type=Path, help="Use this validated arm library in the preview")
    args = parser.parse_args()
    runtime, mod = args.runtime.resolve(), args.mod_root.resolve()
    directory = (args.output_dir or Path("logs/stand_gl165") /
                 datetime.now().strftime("native_velocity_%Y%m%d_%H%M%S_%f")).resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        parser.error(f"Preview output directory already exists: {directory}")
    with tempfile.TemporaryDirectory(dir=directory.parent, prefix=".gl165-preview-") as temporary:
        stage = Path(temporary) / "session"
        prepare(runtime, mod, args.training_config.resolve(), stage, torso_force_n=args.torso_force_n, arm_database=args.arm_database)
        stage.rename(directory)
    overlay, scene = directory / "ros", directory / "scene"
    os.environ["GL165_PREVIEW_SESSION"] = str(directory / "session.json")
    launch = """
set -eo pipefail
source "$1/setup.bash"
source "$1/install/local_setup.bash"
export AMENT_PREFIX_PATH="$2:$AMENT_PREFIX_PATH"
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=165 PYTHONDONTWRITEBYTECODE=1
exec python -u "$(command -v ros2)" launch "$3/tools/gl165.launch.py" "mod_root:=$4"
"""
    os.execlp("bash", "bash", "-c", launch, "gl165-velocity-preview",
              str(runtime), str(overlay), str(mod), str(scene))


if __name__ == "__main__":
    main()

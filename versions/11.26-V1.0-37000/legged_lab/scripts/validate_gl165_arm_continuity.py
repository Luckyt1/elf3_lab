"""Check consecutive full-amplitude arm clips without resetting joint states."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from visualize_elf3_tang_arm_motion import ArmCollisionGuard, gl165_preview_module


def main(args):
    path = args.database.resolve()
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        clips = archive["targets"].copy()
        names = archive["joint_names"].tolist()
    robot = gl165_preview_module().load_robot_metadata()
    if metadata["asset_signature"] != robot.motion_asset_signature():
        raise ValueError("Dataset does not match the current robot physics")
    scene = path.parent / (path.stem + "_validation_scene") / "arms_scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    original_physics_dt = float(model.opt.timestep)
    if args.physics_dt is not None:
        if not math.isfinite(args.physics_dt) or args.physics_dt <= 0:
            raise ValueError("Physics timestep must be finite and positive")
        model.opt.timestep = args.physics_dt
    control_dt = float(metadata["control_dt"])
    decimation = round(control_dt / model.opt.timestep)
    if decimation < 1 or not math.isclose(decimation * model.opt.timestep, control_dt):
        raise ValueError("Physics timestep must divide the recorded control period")
    data = mujoco.MjData(model)
    qids = np.array([model.joint(n).qposadr[0] for n in names])
    vids = np.array([model.joint(n).dofadr[0] for n in names])
    aids = np.array([model.actuator(n).id for n in names])
    gains = np.array([metadata["arm_gains"][n] for n in names])
    kp, kd, effort = gains.T
    model.actuator_biastype[aids] = mujoco.mjtBias.mjBIAS_NONE
    model.actuator_biasprm[aids] = 0
    model.actuator_gainprm[aids, 0] = 1
    nominal = np.array([robot.DEFAULT_JOINT_POS[n] for n in names])
    data.qpos[qids] = nominal
    mujoco.mj_forward(model, data)
    guard = ArmCollisionGuard(model, data, qids, metadata["settings"], mujoco)
    speed_limit = np.array(metadata["urdf_joint_speed_limits_rad_s"])
    peak = np.zeros(len(names)); peak_error = np.zeros(len(names))
    order = np.random.default_rng(args.seed).choice(len(clips), min(args.clips, len(clips)), replace=False).tolist()
    failure = None
    for index, clip in enumerate(order):
        # Training starts at frame 1, then inserts frame 0 on subsequent switches.
        rows = clips[clip, 1:] if index == 0 else clips[clip]
        for target in rows:
            for _ in range(decimation):
                data.ctrl[aids] = np.clip(kp*(target-data.qpos[qids])-kd*data.qvel[vids], -effort, effort)
                mujoco.mj_step(model, data)
                gap = guard.observe()
                speed = np.abs(data.qvel[vids])
                peak = np.maximum(peak, speed)
                peak_error = np.maximum(peak_error, np.abs(target-data.qpos[qids]))
                if (gap < guard.clearance or np.any(speed > speed_limit)
                        or not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
                        or data.warning.number.any()):
                    failure = {"clip": clip, "time_s": float(data.time), "gap_m": gap,
                               "peak_speed_fraction": float((speed/speed_limit).max())}
                    break
            if failure:
                break
        if failure:
            break
        if (index+1) % 8 == 0:
            print(f"Continuous playback: {index+1}/{len(order)} clips, t={data.time:.2f} s", flush=True)
    report = {"passed": failure is None, "failure": failure, "clips": len(order), "clip_order": order,
              "database_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "source_xml_sha256": metadata["source_xml_sha256"], "pd": "explicit",
              "physics_dt_s": float(model.opt.timestep), "duration_s": float(data.time),
              "original_physics_dt_s": original_physics_dt,
              "physics_dt_override_s": args.physics_dt,
              "control_dt_s": control_dt, "decimation": decimation,
              "peak_actual_speed_rad_s": peak.tolist(), "peak_speed_fraction": float((peak/speed_limit).max()),
              "max_tracking_error_rad": peak_error.tolist(), "collision_validation": guard.report(),
              "limitation": "Fixed base/waist/legs, full-amplitude arm playback without resets; not whole-body balance"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "failure", "duration_s", "peak_speed_fraction")}), flush=True)
    if failure:
        raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clips", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42017)
    parser.add_argument("--physics-dt", type=float,
                        help="Offline timestep comparison; does not change the dataset or training configuration.")
    main(parser.parse_args())

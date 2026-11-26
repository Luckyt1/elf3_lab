"""Build closed, collision-checked GL-165 arm clips for standing disturbances."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import visualize_elf3_tang_arm_motion as preview


def overhead_clip(cfg, nominal, limits, guard, segment_steps, seed):
    """Alternate low gestures with a left, right and bilateral overhead reach."""
    rng = np.random.default_rng(seed)
    names = cfg["joint_names"]
    modes = rng.permutation(["left", "right", "both"]).tolist()
    knots = [nominal.copy()]

    def pose(mode):
        goal = nominal.copy()
        for side, sign in (("l", 1), ("r", -1)):
            raised = mode == "both" or mode == {"l": "left", "r": "right"}[side]
            values = {
                "shouder_y": rng.uniform(-2.45, -2.15) if raised else rng.uniform(-0.55, 0.15),
                "shouder_x": sign * rng.uniform(math.radians(20 if raised else 15),
                                                math.radians(38 if raised else 65)),
                "shouder_z": rng.uniform(-0.12, 0.12) if raised else rng.uniform(-0.30, 0.30),
                "elbow_y": rng.uniform(0.20, 0.55) if raised else rng.uniform(0.70, 1.45),
                "wrist_x": rng.uniform(-0.40, 0.40),
                "wrist_y": rng.uniform(-0.25, 0.25),
                "wrist_z": rng.uniform(-0.25, 0.25),
            }
            for joint, value in values.items():
                goal[names.index(f"{side}{joint}_joint")] = value
        return np.clip(goal, limits[:, 0], limits[:, 1])

    for mode in [value for reach in modes for value in ("low", reach)] + ["low"]:
        for _ in range(80):
            goal = pose(mode)
            if guard.path_is_clear(knots[-1], goal):
                knots.append(goal)
                break
            guard.rejected_goals += 1
        else:
            raise RuntimeError(f"No collision-free {mode} transition for seed {seed}")
    if not guard.path_is_clear(knots[-1], nominal):
        raise RuntimeError(f"Unsafe return to nominal for seed {seed}")
    knots.append(nominal.copy())
    rows = [nominal.copy()]
    for start, goal in zip(knots[:-1], knots[1:]):
        for frame in range(1, segment_steps + 1):
            progress = frame / segment_steps
            alpha = progress * progress * (3 - 2 * progress)
            rows.append((1 - alpha) * start + alpha * goal)
    return rows, modes


def fast_overhead_clip(rows, source_steps, names, segment_steps):
    """Retain the reaches and shorten the low-to-high shoulder swing for 0.4 s playback."""
    knots = np.asarray(rows, dtype=np.float64)[::source_steps].copy()
    for side in ("l", "r"):
        pitch = knots[:, names.index(f"{side}shouder_y_joint")]
        # A direct time compression exceeds the shoulder's recorded velocity limit.
        deepest_neighbour = np.minimum(pitch[:-2], pitch[2:])
        pitch[1:-1] = np.minimum(pitch[1:-1], deepest_neighbour + 1.50)
    # Trapezoidal velocity with 40 ms acceleration and deceleration at each end.
    t = np.arange(1, segment_steps + 1) / segment_steps
    ramp = 0.1
    phase = np.where(t < ramp, t*t/(2*ramp), np.where(
        t <= 1-ramp, t-ramp/2, 1-ramp-(1-t)**2/(2*ramp))) / (1-ramp)
    targets = knots[:-1, None] + phase[None, :, None] * np.diff(knots, axis=0)[:, None]
    return np.concatenate((knots[:1], targets.reshape(-1, knots.shape[1])))


def head_top_height(model, data):
    heights = []
    for geom in range(model.ngeom):
        if (model.body(model.geom_bodyid[geom]).name.startswith("head_")
                and model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH):
            mesh = model.geom_dataid[geom]
            start = model.mesh_vertadr[mesh]
            vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
            world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            heights.append(float(world[:, 2].max()))
    return max(heights)


def build(args):
    args.output = args.output.resolve()
    metadata = preview.gl165_preview_module().load_robot_metadata()
    cfg, pose, gains = preview.read_training_settings(
        preview.ROOT / "legged_lab/envs/elf3_tang/walk_cfg.py", "gl165")
    cfg.update(preview_motion_mode="smooth", preview_shoulder_opening_deg=65.0,
               preview_avoid_collisions=True, resample_interval_s=args.segment_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    work = args.output.parent / (args.output.stem + "_validation_scene")
    work.mkdir(exist_ok=True)
    model, data, nominal, limits, qids = preview.build_model(None, cfg, pose, gains, work, mujoco)
    # Match the deployment's externally computed torque PD during validation.
    # The Isaac training actuator remains implicit; both use the same XML inertia.
    arm_kp = np.array([gains[n][0] for n in cfg["joint_names"]])
    arm_kd = np.array([gains[n][1] for n in cfg["joint_names"]])
    arm_effort = np.array([gains[n][2] for n in cfg["joint_names"]])
    if args.pd_mode == "explicit":
        model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_NONE
        model.actuator_biasprm[:] = 0
        model.actuator_gainprm[:, 0] = 1
    guard = preview.ArmCollisionGuard(model, data, qids, cfg, mujoco)
    step_dt = cfg["physics_dt"] * cfg["decimation"]
    segment_steps = round(args.segment_seconds / step_dt)
    if segment_steps < 1 or not math.isclose(segment_steps * step_dt, args.segment_seconds, abs_tol=1e-9):
        raise ValueError("Segment duration must be a positive integer number of control steps")
    source_clips, source_report, source_record = None, None, None
    fast_overhead = args.profile == "overhead" and math.isclose(args.segment_seconds, 0.4)
    retime_source = False
    if args.source_database is not None:
        source_path = args.source_database.resolve(strict=True)
        if source_path == args.output:
            raise ValueError("Revalidation must write a separate dataset")
        with np.load(source_path, allow_pickle=False) as archive:
            source_clips = archive["targets"].copy()
            source_report = json.loads(str(archive["metadata"]))
            if archive["joint_names"].tolist() != cfg["joint_names"]:
                raise ValueError("Source dataset has different arm joint names/order")
        source_steps = int(source_report["segment_steps"])
        retime_source = source_steps != segment_steps
        if (source_clips.shape != (args.clips, 8 * source_steps + 1, len(nominal))
                or source_report["profile"] != args.profile
                or not math.isclose(source_report["control_dt"], step_dt)
                or not math.isclose(source_report["segment_duration_s"], source_steps * step_dt)
                or (retime_source and not fast_overhead)
                or not np.isfinite(source_clips).all()
                or not np.allclose(source_clips[:, [0, -1]], nominal, atol=1e-6)):
            raise ValueError("Source dataset does not match the requested timing/shape/nominal pose")
        source_record = {"path": str(source_path), "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                         "targets_unchanged": not retime_source, "asset_signature": source_report["asset_signature"],
                         "segment_duration_s": source_report["segment_duration_s"]}
    urdf = ET.parse(metadata.URDF_PATH).getroot()
    speed_limits = np.array([float(urdf.find(f"joint[@name='{name}']/limit").get("velocity"))
                             for name in cfg["joint_names"]])
    wrist_ids = [model.body(f"{side}wrist_z_link").id for side in ("l", "r")]
    head_top = head_top_height(model, data)
    clips, measurements = [], []
    reach_measurements, motion_orders = [], []
    peak_speed = np.zeros(len(nominal))
    peak_command_speed = np.zeros(len(nominal))
    started = time.monotonic()
    rejected_clips, accepted_seeds = [], []
    for attempt in range(args.clips if source_clips is not None else args.clips * 20):
        index = len(clips)
        candidate_seed = (source_report["accepted_seeds"][attempt]
                          if source_clips is not None else args.seed + attempt)
        mujoco.mj_resetData(model, data)
        data.qpos[qids], data.ctrl[:] = nominal, 0 if args.pd_mode == "explicit" else nominal
        mujoco.mj_forward(model, data)
        if source_clips is not None:
            rows = source_clips[attempt].copy()
            if retime_source:
                rows = fast_overhead_clip(rows, source_steps, cfg["joint_names"], segment_steps)
            order = source_report["overhead_order_per_clip"][attempt] if args.profile == "overhead" else []
        elif args.profile == "overhead":
            rows, order = overhead_clip(cfg, nominal, limits, guard, segment_steps, candidate_seed)
            if fast_overhead:
                rows = fast_overhead_clip(rows, segment_steps, cfg["joint_names"], segment_steps)
        else:
            generator = preview.ArmTargets(cfg, nominal, limits, 1.0, candidate_seed, guard)
            rows = [nominal.copy()]
            for _ in range(7 * segment_steps):
                rows.append(generator.step())
            start = rows[-1].copy()
            if not guard.path_is_clear(start, nominal):
                raise RuntimeError(f"Clip {index}: return to nominal is unsafe")
            for frame in range(1, segment_steps + 1):
                t = frame / segment_steps
                alpha = t * t * (3 - 2 * t)
                rows.append((1 - alpha) * start + alpha * nominal)
        rows = np.asarray(rows, dtype=np.float32)
        if np.any(rows < limits[:, 0]) or np.any(rows > limits[:, 1]):
            raise RuntimeError(f"Clip {index}: target position exceeds the joint limits")
        command_speed = np.abs(np.diff(rows, axis=0)).max(axis=0) / step_dt
        if np.any(command_speed > speed_limits):
            raise RuntimeError(f"Clip {index}: target velocity exceeds the URDF speed limits")
        # Check the curriculum as well as the full-amplitude motion. Kinematic
        # path checks cover the entire segments, not just their endpoints.
        for scale in (0.02, 0.25, 0.51, 0.75, 1.0):
            knots = nominal + scale * (rows[::segment_steps] - nominal)
            for first, second in zip(knots[:-1], knots[1:]):
                if not guard.path_is_clear(first, second):
                    raise RuntimeError(f"Clip {index}: unsafe curriculum scale {scale}")
        # Run the commanded trajectory through the currently configured arm PD dynamics.
        minimum = 1.0
        candidate_peak_speed = np.zeros(len(nominal))
        failure = None
        previous_guard_statistics = (guard.min_distance, guard.closest_pair, guard.contact_steps)
        wrist_peaks = np.zeros(2)
        raised_steps = {"left": 0, "right": 0, "both": 0}
        for target in rows[1:]:
            for _ in range(cfg["decimation"]):
                data.ctrl[:] = (np.clip(arm_kp * (target - data.qpos[qids]) - arm_kd * data.qvel[qids],
                                       -arm_effort, arm_effort) if args.pd_mode == "explicit" else target)
                mujoco.mj_step(model, data)
                gap = guard.observe()
                minimum = min(minimum, gap)
                speed = np.abs(data.qvel[qids])
                candidate_peak_speed = np.maximum(candidate_peak_speed, speed)
                heights = data.xpos[wrist_ids, 2]
                wrist_peaks = np.maximum(wrist_peaks, heights)
                raised = heights > head_top + 0.02
                raised_steps["left"] += int(raised[0] and not raised[1])
                raised_steps["right"] += int(raised[1] and not raised[0])
                raised_steps["both"] += int(raised.all())
                if (gap < guard.clearance or not np.isfinite(data.qpos).all()
                        or not np.isfinite(data.qvel).all() or data.warning.number.any()):
                    failure = f"dynamic collision/numerical check: gap={gap}"
                    break
                if np.any(speed > speed_limits):
                    offending = int(np.argmax(speed / speed_limits))
                    failure = f"{cfg['joint_names'][offending]} speed {speed[offending]:.4f} > {speed_limits[offending]}"
                    break
            if failure:
                break
        if args.profile == "overhead" and not all(raised_steps.values()):
            failure = failure or f"PD tracking did not achieve all overhead gestures: {raised_steps}"
        if failure:
            if source_clips is not None:
                raise RuntimeError(f"Source clip {index} (seed {candidate_seed}) failed revalidation: {failure}")
            rejected_clips.append({"seed": candidate_seed, "reason": failure})
            guard.min_distance, guard.closest_pair, guard.contact_steps = previous_guard_statistics
            if len(rejected_clips) <= 8 or len(rejected_clips) % 16 == 0:
                print(f"Rejected candidate seed {candidate_seed}: {failure}", flush=True)
            continue
        peak_speed = np.maximum(peak_speed, candidate_peak_speed)
        peak_command_speed = np.maximum(peak_command_speed, command_speed)
        accepted_seeds.append(candidate_seed)
        if args.profile == "overhead":
            motion_orders.append(order)
        clips.append(rows)
        measurements.append(minimum)
        reach_measurements.append({"peak_wrist_height_m": wrist_peaks.tolist(),
                                   "overhead_duration_s": {key: value * cfg["physics_dt"]
                                                           for key, value in raised_steps.items()}})
        if (index + 1) % 8 == 0 or index == 0:
            print(f"Validated {index + 1}/{args.clips} clips; min gap {min(measurements) * 1000:.2f} mm; "
                  f"elapsed {time.monotonic() - started:.1f} s", flush=True)
        if len(clips) == args.clips:
            break
    if len(clips) != args.clips:
        raise RuntimeError(f"Only {len(clips)}/{args.clips} collision/speed-safe clips after {attempt + 1} attempts")
    report = {"schema_version": 2, "purpose": "standing_arm_disturbances_not_amp", "profile": args.profile,
              "asset_signature": metadata.motion_asset_signature(), "mujoco_version": mujoco.__version__,
              "controller_profile": metadata.CONTROLLER_PROFILE, "arm_gains": gains,
              "physics_profile": metadata.PHYSICS_PARAMETERS["profile"],
              "source_xml_sha256": metadata.PHYSICS_PARAMETERS["source_xml_sha256"],
              "source_urdf_sha256": metadata.PHYSICS_PARAMETERS["source_urdf_sha256"],
              "armature_kg_m2": {name: metadata.JOINT_PHYSICS[name]["armature_kg_m2"] for name in cfg["joint_names"]},
              "validation_pd_mode": args.pd_mode, "accepted_seeds": accepted_seeds,
              "revalidated_source_database": source_record,
              "rejected_candidates": rejected_clips,
              "clips": args.clips, "clip_duration_s": 8 * args.segment_seconds, "control_dt": step_dt,
              "segment_duration_s": args.segment_seconds, "segment_steps": segment_steps,
              "motion_types": ["left_overhead", "right_overhead", "both_overhead", "low_and_sideways"]
              if args.profile == "overhead" else ["low_and_sideways"],
              "overhead_order_per_clip": motion_orders, "head_top_m": head_top,
              "overhead_measurements": reach_measurements,
              "peak_actual_joint_speed_rad_s": peak_speed.tolist(),
              "peak_target_joint_speed_rad_s": peak_command_speed.tolist(),
              "urdf_joint_speed_limits_rad_s": speed_limits.tolist(),
              "target_and_actual_speed_limits_checked": True,
              "seed": args.seed, "shoulder_opening_deg": [15, 65], "settings": cfg,
              "kinematic_curriculum_scales": [0.02, 0.25, 0.51, 0.75, 1.0],
              "dynamic_validation_scale": 1.0, "nominal_non_arm_joints": pose,
              "collision_validation": guard.report(), "minimum_gap_per_clip_m": measurements,
              "limitation": "Checked with fixed waist and legs; training enables PhysX self-contact and failure resets."}
    if fast_overhead:
        report["interpolation"] = {"kind": "trapezoidal_velocity", "ramp_s": 0.04,
                                   "shoulder_pitch_low_to_high_limit_rad": 1.50}
    temporary = args.output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, targets=np.stack(clips), joint_names=np.asarray(cfg["joint_names"]),
                        control_dt=np.asarray(step_dt), metadata=np.asarray(json.dumps(report, sort_keys=True)))
    temporary.replace(args.output)
    report["sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}: {args.clips} x {report['clip_duration_s']:.2f} s; "
          f"SHA256={report['sha256']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42000)
    parser.add_argument("--profile", choices=("random", "overhead"), default="overhead")
    parser.add_argument("--pd-mode", choices=("explicit", "native"), default="explicit")
    parser.add_argument("--segment-seconds", type=float, default=0.4,
                        help="Duration of one complete target transition, in simulation seconds.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-database", type=Path,
                        help="Revalidate the source clips, or retime their overhead waypoints for 0.4 s playback.")
    args = parser.parse_args()
    if args.clips < 1:
        parser.error("--clips must be positive")
    if not math.isfinite(args.segment_seconds) or args.segment_seconds <= 0:
        parser.error("--segment-seconds must be finite and positive")
    if args.output is None:
        name = ("arm_disturbance_overhead_xml_5ms_0p4s_v6.npz" if args.profile == "overhead" and args.segment_seconds == 0.4
                else f"arm_disturbance_{args.profile}_xml_5ms_{args.segment_seconds:g}s.npz")
        args.output = preview.ROOT / "legged_lab/envs/gl_165/datasets" / name
    build(args)

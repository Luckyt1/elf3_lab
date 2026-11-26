"""Display live GL-165 policy rollouts in MuJoCo; dynamics remain in Isaac Sim.

Run with .venv-gl165-mujoco/bin/python. MuJoCo performs forward kinematics only;
the subprocess runs the same floating-base PhysX environment used for training.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from preview_gl165_mujoco import ROOT, build_model, load_robot_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arm-database", type=Path, help="Use a specific checked arm disturbance database.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "logs/stand_gl165/mujoco_policy_live")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "state.json"
    control_path = state_path.with_suffix(".control.json")
    state_path.unlink(missing_ok=True)
    controls = {"paused": False, "reset": 0}

    def write_controls():
        temporary = control_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(controls))
        temporary.replace(control_path)

    def key_callback(key):
        if key == 32:
            controls["paused"] = not controls["paused"]
            write_controls()
        elif key in (82, 114):
            controls["reset"] += 1
            write_controls()

    write_controls()
    os.environ.setdefault("MUJOCO_GL", "glfw")
    import mujoco
    import mujoco.viewer
    import numpy as np

    sidecar = checkpoint.parent / "params/controller_parameters.json"
    if sidecar.is_file():
        profile = json.loads(sidecar.read_text())
    else:
        # The viewer venv only contains MuJoCo. Read standalone metadata with the training interpreter.
        reader = "import json, sys, torch; print(json.dumps(torch.load(sys.argv[1], map_location='cpu', weights_only=True).get('standing_profile', {})))"
        profile = json.loads(subprocess.check_output([str(ROOT / ".venv-gl165/bin/python"), "-c", reader, str(checkpoint)]))
    metadata = load_robot_metadata(profile.get("physics_profile", "gl165_xml_armature_20261117_5ms"),
                                   profile.get("nominal_stance"))
    model, data, _ = build_model(metadata, output, None, mujoco, np,
                               scene_name=f"GL-165 PPO {checkpoint.stem} - live Isaac Sim states")
    # The static pose ruler is irrelevant to a moving policy rollout.
    for geom_id in range(model.ngeom):
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").startswith("base_height_ruler"):
            model.geom_rgba[geom_id, 3] = 0.0
    root_qpos = model.jnt_qposadr[model.joint("root_free").id]
    command = [str(ROOT / "scripts/gl165.sh"), "play", "--headless", "--num_envs", "1", "--real-time",
               "--load_run", checkpoint.parent.name, "--checkpoint", checkpoint.name,
               "--state-stream", str(state_path)]
    if args.arm_database is not None:
        command += ["--arm-database", str(args.arm_database.resolve(strict=True))]
    env = os.environ.copy()
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    log_path = output / "physics.log"
    with log_path.open("w") as log:
        worker = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=subprocess.STDOUT)
        (output / "processes.json").write_text(json.dumps({
            "viewer_pid": os.getpid(), "physics_pid": worker.pid, "checkpoint": str(checkpoint),
            "physics": "Isaac Sim / PhysX", "display": "MuJoCo forward kinematics", "command": command,
        }, indent=2) + "\n")
        try:
            with mujoco.viewer.launch_passive(model, data, key_callback=key_callback,
                                             show_left_ui=False, show_right_ui=False) as viewer:
                with viewer.lock():
                    viewer.cam.lookat[:] = [0.0, 0.0, 0.85]
                    viewer.cam.distance = 3.2
                    viewer.cam.azimuth = 35
                    viewer.cam.elevation = -12
                    viewer.opt.geomgroup[0] = 0
                print(f"Viewer open: {checkpoint.name}; waiting for Isaac Sim states", flush=True)
                joint_ids = None
                sequence = -1
                last_validation = -1
                mapping_errors = []
                start = time.monotonic()
                while viewer.is_running():
                    if worker.poll() is not None:
                        raise RuntimeError(f"Physics worker exited ({worker.returncode}); see {log_path}")
                    lines = [f"GL-165 PPO | {checkpoint.name}", "Isaac Sim physics | MuJoCo display",
                             "Space: pause/resume | R: reset"]
                    if state_path.exists():
                        state = json.loads(state_path.read_text())
                        if joint_ids is None:
                            if set(state["joint_names"]) != set(metadata.DEFAULT_JOINT_POS):
                                raise RuntimeError("Joint names differ between training and the viewer")
                            joint_ids = np.array([model.jnt_qposadr[model.joint(name).id]
                                                  for name in state["joint_names"]])
                        if "last_velocity_push" in state:
                            push = state["last_velocity_push"]
                            arrow_vector = np.zeros(3)
                            push_label = "Waiting for a base velocity change"
                            if push:
                                dx, dy = push["xy_m_s"]
                                push_label = (f"Last velocity change: X {dx:+.2f}, Y {dy:+.2f} m/s | "
                                              f"yaw {push['yaw_rad_s']:+.2f} rad/s")
                                if state["timestamp"] - push["timestamp"] <= .6:
                                    arrow_vector[:2] = [dx, dy]
                        else:
                            force = np.asarray(state["force_world_n"])
                            arrow_vector = force * .008
                            push_label = f"Push {np.linalg.norm(force):.1f} N"
                        with viewer.lock():
                            data.qpos[root_qpos:root_qpos + 3] = state["root_position"]
                            data.qpos[root_qpos + 3:root_qpos + 7] = state["root_quaternion_wxyz"]
                            data.qpos[joint_ids] = state["joint_positions"]
                            if not np.isfinite(data.qpos).all():
                                raise RuntimeError("Non-finite streamed robot pose")
                            mujoco.mj_forward(model, data)
                            viewer.user_scn.ngeom = 0
                            if np.linalg.norm(arrow_vector) > 1e-6:
                                arrow = viewer.user_scn.geoms[0]
                                mujoco.mjv_initGeom(arrow, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                                                   np.zeros(3), np.eye(3).ravel(), np.array([1., .3, .05, 1.]))
                                anchor = data.xpos[model.body("base_link" if "last_velocity_push" in state else metadata.TORSO).id]
                                mujoco.mjv_connector(arrow, mujoco.mjtGeom.mjGEOM_ARROW, .025,
                                                    anchor, anchor + arrow_vector)
                                viewer.user_scn.ngeom = 1
                            # Check joint order, quaternion convention and the URDF transforms
                            # against every PhysX link, not just the root, over moving frames.
                            if state["sequence"] != last_validation and len(mapping_errors) < 50:
                                errors = [float(np.linalg.norm(data.xpos[model.body(name).id] - position))
                                          for name, position in zip(state["body_names"], state["body_positions"])]
                                mapping_errors.append(max(errors))
                                last_validation = state["sequence"]
                                if max(errors) > .005:
                                    raise RuntimeError(f"Display mapping error exceeds 5 mm: {max(errors):.6f} m")
                                (output / "mapping_validation.json").write_text(json.dumps({
                                    "frames": len(mapping_errors), "links": len(errors),
                                    "max_link_position_error_m": max(mapping_errors),
                                    "checkpoint": str(checkpoint),
                                }, indent=2) + "\n")
                        if state["sequence"] != sequence and sequence < 0:
                            print("Receiving live policy states; all link transforms verified", flush=True)
                        sequence = state["sequence"]
                        age = time.time() - state["timestamp"]
                        status = "PAUSED" if controls["paused"] else ("LIVE" if age < 1.0 else "WAITING FOR PHYSICS")
                        if "action_clip_enabled" in state:
                            lines.append("Action clipping: " + (str(state["clip_actions"]) if state["action_clip_enabled"] else "OFF"))
                        if "arm_motion_profile" in state:
                            lines.append(f"Arms: {state['arm_motion_profile']} | target transition "
                                         f"{state['arm_segment_duration_s']:.2f} s")
                        if "feet_heading_deg" in state:
                            left, right = state["feet_heading_deg"]
                            lines.append(f"Feet yaw vs pelvis: L {left:+.1f} | R {right:+.1f} deg")
                        if "waist_angle_deg" in state:
                            roll, pitch, yaw = state["waist_angle_deg"]
                            lines.append(f"Waist X/Y/Z: {roll:+.1f} / {pitch:+.1f} / {yaw:+.1f} deg")
                        lines += [f"{status} | episode {state['episode_seconds']:.2f} s",
                                  f"Base height {state['root_position'][2]:.3f} m | drift {state['drift_m']:.3f} m",
                                  push_label,
                                  f"Resets {state['resets']} | last: {state['last_reset']}"]
                    else:
                        lines.append("Loading physics and policy...")
                        if time.monotonic() - start > 120:
                            raise TimeoutError(f"No physics states received; see {log_path}")
                    viewer.set_texts((None, None, "\n".join(lines), None))
                    viewer.sync()
                    time.sleep(1 / 60)
        finally:
            worker.terminate()
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()


if __name__ == "__main__":
    main()

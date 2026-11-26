"""Preview tang's environment-generated arm targets in MuJoCo, without a policy.

Base, waist and legs are held for inspection. Targets follow the training loop's
sampling, clipping and interpolation clocks, including its mixed step counters.
Training source files are only read, never changed.

Example: .venv-gl165-mujoco/bin/python -B
  legged_lab/scripts/visualize_elf3_tang_arm_motion.py --robot gl165 --cycle-stages

At dt=0.005 and decimation=4, training samples a new goal every 5 control
steps, but interpolates using a denominator of 20. It reaches smoothstep(5/20)
= 0.15625 of each new segment before resampling. This preview preserves that
behavior; the configured amplitudes describe sampled goals, not actual swings.
"""

import argparse
import ast
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARM_JOINT_NAMES = [f"{side}_{joint}_joint" for side in ("l", "r") for joint in (
    "shoulder_y", "shoulder_x", "shoulder_z", "elbow_y", "wrist_x", "wrist_y", "wrist_z",
)]
GL165_ARM_JOINT_NAMES = [name.replace("l_", "l", 1) if name.startswith("l_")
                        else name.replace("r_", "r", 1) for name in ARM_JOINT_NAMES]
GL165_ARM_JOINT_NAMES = [name.replace("shoulder", "shouder") for name in GL165_ARM_JOINT_NAMES]


def keywords(node):
    return {item.arg: item.value for item in node.keywords}


def gl165_preview_module():
    path = ROOT / "legged_lab/scripts/preview_gl165_mujoco.py"
    spec = importlib.util.spec_from_file_location("gl165_pose_preview", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_training_settings(cfg_path, robot="tang"):
    """Read literal configuration without importing Isaac Sim."""
    tree = ast.parse(Path(cfg_path).read_text())
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    arm = {node.target.id: ast.literal_eval(node.value)
           for node in classes["ArmMotionCurriculumCfg"].body if isinstance(node, ast.AnnAssign)}
    sim = next(node.value for node in classes["Elf3WalkTangFlatEnvCfg"].body
               if isinstance(node, ast.AnnAssign) and node.target.id == "sim")
    arm["physics_dt"] = ast.literal_eval(keywords(sim)["dt"])
    arm["decimation"] = ast.literal_eval(keywords(sim)["decimation"])
    arm["rollout_steps"] = next(ast.literal_eval(node.value)
                                for node in classes["Elf3WalkTangAgentCfg"].body
                                if isinstance(node, ast.Assign)
                                and any(isinstance(t, ast.Name) and t.id == "num_steps_per_env" for t in node.targets))
    if robot == "gl165":
        metadata = gl165_preview_module().load_robot_metadata()
        arm["physics_dt"] = metadata.PHYSICS_DT
        arm["decimation"] = round(metadata.CONTROL_DT / metadata.PHYSICS_DT)
        urdf = ET.parse(metadata.URDF_PATH).getroot()
        gains = {}
        for name in GL165_ARM_JOINT_NAMES:
            group = next(value for value in metadata.JOINT_GROUPS.values() if re.fullmatch(value[0], name))
            effort = float(urdf.find(f"joint[@name='{name}']/limit").get("effort"))
            gains[name] = [group[1], group[2], effort]
        asset_tree = ast.parse((ROOT / "legged_lab/assets/gl_165/gl_165.py").read_text())
        articulation = next(node for node in ast.walk(asset_tree) if isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Name) and node.func.id == "ArticulationCfg")
        arm["soft_limit_factor"] = ast.literal_eval(keywords(articulation)["soft_joint_pos_limit_factor"])
        arm["base_position"] = (0.0, 0.0, metadata.STAND_HEIGHT)
        arm["joint_names"] = GL165_ARM_JOINT_NAMES
        arm["robot_label"] = "GL-165"
        arm["collision_group"], arm["visual_group"] = 0, 1
        arm["robot_urdf"] = str(metadata.URDF_PATH)
        arm["source_robot_directory"] = str(Path(metadata.PHYSICS_PARAMETERS["source_urdf"]).parent)
        return arm, metadata.DEFAULT_JOINT_POS.copy(), gains
    asset_tree = ast.parse((ROOT / "legged_lab/assets/elf3_lite/elf3.py").read_text())
    asset = next(node.value for node in asset_tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "ELF3LITE_CFG" for t in node.targets))
    args = keywords(asset)
    pose = ast.literal_eval(keywords(args["init_state"])["joint_pos"])
    for node in tree.body:
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == "ELF3_TANG_STRAIGHT_JOINT_POS.update"):
            pose.update(ast.literal_eval(node.value.args[0]))
    straight = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "ELF3_TANG_STRAIGHT_CFG" for t in node.targets))
    arm["base_position"] = ast.literal_eval(keywords(keywords(straight)["init_state"])["pos"])
    arm["soft_limit_factor"] = ast.literal_eval(args["soft_joint_pos_limit_factor"])
    gains = {}
    for actuator in args["actuators"].values:
        values = keywords(actuator)
        maps = [ast.literal_eval(values[key]) for key in ("stiffness", "damping", "effort_limit_sim")]
        for name in ARM_JOINT_NAMES:
            if any(re.fullmatch(pattern, name) for pattern in ast.literal_eval(values["joint_names_expr"])):
                gains[name] = [next(value for pattern, value in mapping.items() if re.fullmatch(pattern, name))
                               for mapping in maps]
    if len(gains) != 14 or len(arm["stand_amplitudes"]) != 14:
        raise ValueError("Expected the current tang configuration with 14 arm joints")
    arm["joint_names"] = ARM_JOINT_NAMES
    arm["robot_label"] = "TANG"
    arm["collision_group"], arm["visual_group"] = 3, 2
    return arm, pose, gains


class ArmCollisionGuard:
    """Check original URDF collision hulls, including the interpolated path.

    Only direct joint mounts and GL-165's nested wrist assembly are excluded.
    Collision-enabled compilation builds MuJoCo's convex graphs, so distance
    queries remain fast even with the original high-resolution visual meshes.
    """

    clearance = 0.010
    stop_clearance = 0.005

    def __init__(self, model, data, qpos_ids, cfg, mujoco):
        self.model, self.data, self.qpos_ids, self.mujoco = model, data, qpos_ids, mujoco
        self.scratch = mujoco.MjData(model)
        arm_bodies = {int(model.jnt_bodyid[model.joint(name).id]) for name in cfg["joint_names"]}
        excluded = {frozenset(pair) for pair in cfg["collision_excluded_body_pairs"]}
        geoms = np.flatnonzero((model.geom_group == cfg["collision_group"]) & (model.geom_bodyid > 0))
        self.pairs = []
        for first, second in itertools.combinations(geoms.tolist(), 2):
            body1, body2 = int(model.geom_bodyid[first]), int(model.geom_bodyid[second])
            names = frozenset((model.body(body1).name, model.body(body2).name))
            if (body1 in arm_bodies or body2 in arm_bodies) and names not in excluded:
                self.pairs.append((first, second))
        self.pair_set = {tuple(sorted(pair)) for pair in self.pairs}
        self.rejected_goals = self.held_segments = self.safety_stops = self.contact_steps = 0
        self.min_distance = math.inf
        self.closest_pair = None
        if self.distance(data)[0] < self.clearance:
            raise ValueError("The initial arm pose does not meet the preview collision clearance")

    def distance(self, data):
        minimum, closest = 0.05, None
        for first, second in self.pairs:
            distance = self.mujoco.mj_geomDistance(self.model, data, first, second, minimum, None)
            if distance < minimum:
                minimum, closest = distance, (first, second)
        return minimum, closest

    def path_is_clear(self, start, goal):
        # Check both the commanded path and a path from the actual PD pose.
        for origin in (start, self.data.qpos[self.qpos_ids]):
            steps = max(1, math.ceil(float(np.max(np.abs(goal - origin))) / 0.02))
            for alpha in np.linspace(0, 1, steps + 1):
                self.scratch.qpos[self.qpos_ids] = origin + alpha * (goal - origin)
                self.mujoco.mj_kinematics(self.model, self.scratch)
                if self.distance(self.scratch)[0] < self.clearance:
                    return False
        return True

    def observe(self):
        # mj_step advances qpos after its geometry calculation; refresh geometry
        # to check the new state, not the beginning of the physics step.
        self.mujoco.mj_kinematics(self.model, self.data)
        distance, pair = self.distance(self.data)
        if distance < self.min_distance:
            self.min_distance, self.closest_pair = distance, pair
        if any(contact.dist <= 0 and tuple(sorted((int(contact.geom1), int(contact.geom2)))) in self.pair_set
               for contact in self.data.contact):
            self.contact_steps += 1
        return distance

    def report(self):
        return {"checked_pairs": len(self.pairs), "target_clearance_m": self.clearance,
                "stop_clearance_m": self.stop_clearance, "path_max_joint_step_rad": 0.02,
                "minimum_actual_distance_m": self.min_distance,
                "closest_body_pair": [self.model.body(self.model.geom_bodyid[g]).name for g in self.closest_pair]
                if self.closest_pair else None,
                "rejected_goals": self.rejected_goals, "held_segments": self.held_segments,
                "safety_stops": self.safety_stops, "contact_steps": self.contact_steps}


class ArmTargets:
    """Original tang timing, or complete transitions for preview comparison."""

    def __init__(self, cfg, nominal, limits, scale, seed, guard=None):
        self.cfg, self.nominal, self.limits = cfg, nominal, limits
        self.guard = guard
        self.scale = scale
        self.rng = np.random.default_rng(seed)
        self.mode = cfg.get("preview_motion_mode", "training")
        if self.mode == "training":
            self.interval = max(1, int(cfg["resample_interval_s"] / cfg["physics_dt"]))
            self.counter_increment = cfg["decimation"]
        else:
            control_dt = cfg["physics_dt"] * cfg["decimation"]
            self.interval = max(1, math.ceil(cfg["resample_interval_s"] / control_dt))
            self.counter_increment = 1
        self.current = nominal.copy()
        self.start = nominal.copy()
        self.goal = nominal.copy()
        self.transition_step = 0
        self.sample_step = 0

    def step(self):
        if self.sample_step % self.interval == 0:
            self.start = self.current.copy()
            for _ in range(40 if self.guard else 1):
                offset = self.rng.uniform(-1.0, 1.0, len(self.nominal))
                goal = self.nominal + offset * np.asarray(self.cfg["stand_amplitudes"]) * self.scale
                opening = self.cfg.get("preview_shoulder_opening_deg")
                if opening is not None:
                    for side, sign in (("l", 1), ("r", -1)):
                        index = self.cfg["joint_names"].index(f"{side}shouder_x_joint")
                        angle = math.radians(15 + (opening - 15) * (offset[index] + 1) / 2)
                        goal[index] = self.nominal[index] + self.scale * (sign * angle - self.nominal[index])
                goal = np.clip(goal, self.limits[:, 0], self.limits[:, 1])
                if self.guard is None or self.guard.path_is_clear(self.start, goal):
                    self.goal = goal
                    break
                self.guard.rejected_goals += 1
            else:
                self.goal = self.start.copy()
                self.guard.held_segments += 1
            self.transition_step = 0
        progress = min(1.0, (self.transition_step + 1) / self.interval)
        alpha = progress * progress * (3.0 - 2.0 * progress)
        self.current = (1.0 - alpha) * self.start + alpha * self.goal
        self.transition_step = min(self.transition_step + 1, self.interval)
        self.sample_step += self.counter_increment
        return self.current.copy()


def build_model(model_path, cfg, pose, gains, output, mujoco):
    if cfg["robot_label"] == "GL-165":
        preview = gl165_preview_module()
        metadata = preview.load_robot_metadata()
        original, initial, report = preview.build_model(metadata, output, None, mujoco, np)
        cfg["base_position"] = (0.0, 0.0, report["base_height_m"])
        xml = ET.parse(report["scene"]).getroot()
        xml.find("option").set("integrator", "implicitfast")
        xml.find("option").set("timestep", str(cfg["physics_dt"]))
        # Collision avoidance is a preview override, never a training edit.
        for geom in xml.findall("./worldbody/body//geom"):
            enabled = cfg.get("preview_avoid_collisions") and geom.get("group", "0") == "0"
            geom.set("contype", "1" if enabled else "0")
            geom.set("conaffinity", "1" if enabled else "0")
        if cfg.get("preview_avoid_collisions"):
            contact = ET.SubElement(xml, "contact")
            excluded = []
            for name in cfg["joint_names"]:
                body = int(original.jnt_bodyid[original.joint(name).id])
                excluded.append([original.body(int(original.body_parentid[body])).name, original.body(body).name])
            # The two wrist axes share a mounting volume in the supplied URDF.
            excluded += [[f"{side}smallarm_x_link", f"{side}wrist_y_link"] for side in ("l", "r")]
            for first, second in excluded:
                ET.SubElement(contact, "exclude", body1=first, body2=second)
            cfg["collision_excluded_body_pairs"] = excluded
        return make_arm_fixture(xml, original, initial, cfg, pose, gains, output, mujoco)
    source = Path(model_path).resolve()
    xml = ET.parse(source).getroot()
    compiler = xml.find("compiler")
    meshdir = (source.parent / compiler.get("meshdir", ".")).resolve()
    if not meshdir.is_dir():
        meshdir = source.parent.parent / "urdf/meshes"
    compiler.set("meshdir", str(meshdir))
    # Use a flat inspection scene instead of the original stair-course include.
    for include in xml.findall("include"):
        xml.remove(include)
    ET.SubElement(xml, "option", timestep=str(cfg["physics_dt"]), integrator="implicitfast", gravity="0 0 -9.81")
    base = xml.find("./worldbody/body")
    base.set("pos", " ".join(map(str, cfg["base_position"])))
    ET.SubElement(xml.find("worldbody"), "geom", name="floor", type="plane", size="3 3 0.1",
                  rgba="0.78 0.82 0.87 1", contype="0", conaffinity="0")
    original = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    initial = mujoco.MjData(original)
    for name, angle in pose.items():
        initial.qpos[original.jnt_qposadr[original.joint(name).id]] = angle
    mujoco.mj_forward(original, initial)
    visual = ET.SubElement(xml, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960", fovy="35")
    ET.SubElement(visual, "headlight", ambient="0.4 0.4 0.4", diffuse="0.7 0.7 0.7")
    ET.SubElement(xml.find("asset"), "texture", name="preview_sky", type="skybox", builtin="gradient",
                  rgb1="0.80 0.86 0.93", rgb2="0.96 0.98 1", width="512", height="3072")
    return make_arm_fixture(xml, original, initial, cfg, pose, gains, output, mujoco)


def make_arm_fixture(xml, original, initial, cfg, pose, gains, output, mujoco):
    arm_names = cfg["joint_names"]
    # Freeze non-arm joints at their nominal transforms, retaining the full mesh.
    for body in xml.findall(".//body"):
        joints = body.findall("joint") + body.findall("freejoint")
        if joints and any(joint.get("name") not in arm_names for joint in joints):
            body_id = original.body(body.get("name")).id
            parent = original.body_parentid[body_id]
            rotation = initial.xmat[parent].reshape(3, 3).T
            pos = rotation @ (initial.xpos[body_id] - initial.xpos[parent])
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, (rotation @ initial.xmat[body_id].reshape(3, 3)).ravel())
            body.set("pos", " ".join(map(str, pos)))
            body.set("quat", " ".join(map(str, quat)))
            for joint in joints:
                body.remove(joint)
    for section in ("actuator", "sensor", "keyframe"):
        for element in xml.findall(section):
            xml.remove(element)
    actuators = ET.SubElement(xml, "actuator")
    for name in arm_names:
        kp, kd, effort = gains[name]
        ET.SubElement(actuators, "position", name=name, joint=name, kp=str(kp), kv=str(kd),
                      forcelimited="true", forcerange=f"{-effort} {effort}")
    xml.set("model", f"{cfg['robot_label']} - tang arm disturbance - NO POLICY - fixed base")
    ET.indent(xml)
    scene_path = output / "arms_scene.xml"
    scene_path.write_text(ET.tostring(xml, encoding="unicode") + "\n")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    ids = np.array([model.joint(name).id for name in arm_names])
    nominal = np.array([pose[name] for name in arm_names])
    data.qpos[model.jnt_qposadr[ids]] = nominal
    data.ctrl[:] = nominal
    mujoco.mj_forward(model, data)
    for body_id in range(1, model.nbody):
        old_id = original.body(model.body(body_id).name).id
        np.testing.assert_allclose(data.xpos[body_id], initial.xpos[old_id], atol=1e-7)
        np.testing.assert_allclose(data.xmat[body_id], initial.xmat[old_id], atol=1e-7)
    if model.nv != 14 or model.nu != 14:
        raise ValueError("The inspection fixture must leave only the 14 arm joints moving")
    limits = model.jnt_range[ids].copy()
    margin = (limits[:, 1] - limits[:, 0]) * (1.0 - cfg["soft_limit_factor"]) / 2
    limits[:, 0] += margin
    limits[:, 1] -= margin
    return model, data, nominal, limits, model.jnt_qposadr[ids]


def configure_camera(camera):
    camera.lookat[:] = [0.03, 0, 1.05]
    camera.distance = 2.4
    camera.azimuth = 30
    camera.elevation = -10


def run(args):
    os.environ.setdefault("MUJOCO_GL", "egl" if args.headless else "glfw")
    import mujoco

    cfg, pose, gains = read_training_settings(args.cfg, args.robot)
    arm_names = cfg["joint_names"]
    label = cfg["robot_label"]
    training_interval = cfg["resample_interval_s"]
    cfg["preview_motion_mode"] = args.motion_mode
    cfg["preview_shoulder_opening_deg"] = args.shoulder_opening_deg
    cfg["preview_avoid_collisions"] = args.avoid_collisions
    if args.motion_mode == "smooth":
        cfg["resample_interval_s"] = 1.0
    if args.dt is not None:
        cfg["physics_dt"] = args.dt
    if args.resample_interval is not None:
        cfg["resample_interval_s"] = args.resample_interval
    scale = args.scale if args.scale is not None else 1.0
    if args.iteration is not None:
        elapsed = args.iteration - cfg["start_iteration"]
        progress = min(1.0, max(0, elapsed) * cfg["rollout_steps"] * cfg["physics_dt"]
                       * cfg["decimation"] / cfg["curriculum_duration_s"])
        scale = cfg["initial_scale"] + (1 - cfg["initial_scale"]) * progress if elapsed >= 0 else 0.0
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model, data, nominal, limits, qpos_ids = build_model(args.model, cfg, pose, gains, output, mujoco)
    guard = ArmCollisionGuard(model, data, qpos_ids, cfg, mujoco) if args.avoid_collisions else None
    backup = mujoco.MjData(model) if guard else None
    targets = ArmTargets(cfg, nominal, limits, scale, args.seed, guard)
    step_dt = cfg["physics_dt"] * cfg["decimation"]
    stages = [cfg["initial_scale"], (1 + cfg["initial_scale"]) / 2, 1.0]
    controls = {"scale": scale, "paused": False, "safety_stop": False, "mode": args.motion_mode,
                "interval": cfg["resample_interval_s"]}
    report = {"mode": "tang_arm_randomization_without_policy", "base_waist_legs_fixed": True,
              "source_config": str(Path(args.cfg).resolve()), "settings": cfg,
              "robot": label, "initial_scale": scale, "joint_order": arm_names, "actuator_kp_kd_effort": gains,
              "timing_note": ("Original tang timing: interpolation counts control steps; resampling counts physics steps."
                              if args.motion_mode == "training" else
                              "Preview comparison only: complete interpolation in control steps; training is unchanged.")}
    (output / "preview_config.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"MuJoCo: {label}, no policy, mode={args.motion_mode}. Scale={scale:.1%}, target interval={cfg['resample_interval_s']} s.", flush=True)
    print("Base/waist/legs held. Keys 1/2/3: scale; 4: tang timing; 5: smooth preview; Space: pause.", flush=True)
    if guard:
        print(f"Collision avoidance: {len(guard.pairs)} body pairs; path clearance 10 mm; stop at 5 mm.", flush=True)

    def key_callback(key):
        if key in (49, 50, 51):
            controls["scale"] = stages[key - 49]
            args.cycle_stages = False
        elif key == 32:
            controls["paused"] = not controls["paused"]
        elif key in (52, 53):
            controls["mode"] = "training" if key == 52 else "smooth"
            controls["interval"] = training_interval if key == 52 else 1.0

    viewer = renderer = None
    frames, samples = [], []
    warnings = np.zeros(mujoco.mjtWarning.mjNWARNING, dtype=int)
    options = mujoco.MjvOption()
    options.geomgroup[cfg["collision_group"]] = 0
    camera = mujoco.MjvCamera()
    configure_camera(camera)
    if not args.headless:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback,
                                              show_left_ui=False, show_right_ui=False)
        with viewer.lock():
            configure_camera(viewer.cam)
            viewer.opt.geomgroup[cfg["collision_group"]] = 0
        print("Viewer ready on the desktop.", flush=True)
    if args.gif:
        from PIL import Image, ImageDraw, ImageFont
        renderer = mujoco.Renderer(model, height=600, width=720)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    next_render = 0.0
    elapsed = 0.0
    try:
        while (args.duration == 0 or elapsed < args.duration - 1e-9) and (viewer is None or viewer.is_running()):
            wall_start = time.perf_counter()
            if controls["paused"] or controls["safety_stop"]:
                viewer.sync()
                time.sleep(0.02)
                continue
            stage_scale = stages[int(elapsed / args.stage_seconds) % 3] if args.cycle_stages else controls["scale"]
            if (stage_scale != targets.scale or controls["mode"] != targets.mode
                    or controls["interval"] != cfg["resample_interval_s"]):
                cfg["preview_motion_mode"] = controls["mode"]
                cfg["resample_interval_s"] = controls["interval"]
                targets = ArmTargets(cfg, nominal, limits, stage_scale, args.seed, guard)
                mujoco.mj_resetData(model, data)
                data.qpos[qpos_ids] = nominal
                data.ctrl[:] = nominal
                mujoco.mj_forward(model, data)
                print(f"Preview stage: {stage_scale:.1%}, mode={targets.mode}, interval={cfg['resample_interval_s']} s", flush=True)
            data.ctrl[:] = targets.step()
            for _ in range(cfg["decimation"]):
                if guard:
                    mujoco.mj_copyData(backup, model, data)
                mujoco.mj_step(model, data)
                if guard and guard.observe() < guard.stop_clearance:
                    guard.safety_stops += 1
                    mujoco.mj_copyData(data, model, backup)
                    controls["safety_stop"] = True
                    print("SAFETY STOP: arm clearance below 5 mm; restored the preceding safe state.", flush=True)
                    break
            elapsed += step_dt
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("Non-finite MuJoCo state")
            warnings = np.maximum(warnings, data.warning.number)
            if args.duration > 0:
                samples.append([elapsed, stage_scale, *data.ctrl, *data.qpos[qpos_ids]])
            if args.print_targets and targets.transition_step == 1:
                print(f"t={elapsed:.2f} scale={stage_scale:.2f} left goals={targets.goal[:4]}", flush=True)
            collision_text = (f"Avoid collisions | gap {guard.min_distance * 1000:.1f} mm min"
                              if guard else "Base / waist / legs fixed")
            if controls["safety_stop"]:
                collision_text = "SAFETY STOP: motion frozen; restart the preview"
            if viewer is not None:
                # Numeric keys also toggle MuJoCo geometry groups. Keep meshes
                # visible when using 1/2/3 as curriculum shortcuts.
                with viewer.lock():
                    viewer.opt.geomgroup[2] = 1
                    viewer.opt.geomgroup[cfg["visual_group"]] = 1
                    viewer.opt.geomgroup[cfg["collision_group"]] = 0
                viewer.set_texts((None, None,
                                  f"{label} | no policy | arm scale {stage_scale:.0%}\n"
                                  f"Mode: {'original tang timing' if targets.mode == 'training' else 'smooth PREVIEW ONLY'}\n"
                                  f"Goal interval {cfg['resample_interval_s']:.2f} s\n"
                                  f"{collision_text}\n"
                                  "1/2/3: scale   4: tang timing   5: smooth   Space: pause", ""))
                viewer.sync()
            if renderer is not None and elapsed + 1e-9 >= next_render:
                renderer.update_scene(data, camera=camera, scene_option=options)
                frame = Image.new("RGB", (720, 685), "#eef3f8")
                frame.paste(Image.fromarray(renderer.render()), (0, 85))
                draw = ImageDraw.Draw(frame)
                draw.text((16, 8), f"{label} | arm disturbance {stage_scale:.0%} | no policy", font=font, fill="#20344b")
                timing_label = "original tang timing" if targets.mode == "training" else "smooth PREVIEW ONLY"
                draw.text((16, 33), f"Targets every {cfg['resample_interval_s']:.2f} s | {timing_label}", font=font, fill="#20344b")
                draw.text((16, 58), collision_text, font=font, fill="#41556b")
                if not frames:
                    frame.save(output / "preview.png")
                frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
                next_render += 0.04
            if viewer is not None:
                time.sleep(max(0.0, step_dt - (time.perf_counter() - wall_start)))
            elif controls["safety_stop"]:
                break
    finally:
        if viewer is not None:
            viewer.close()
        if renderer is not None:
            renderer.close()
    if frames:
        frames[0].save(args.gif, save_all=True, append_images=frames[1:], duration=40, loop=0, optimize=False)
        print(f"Saved animation: {args.gif}", flush=True)
    if samples:
        values = np.asarray(samples)
        header = "time_s,scale," + ",".join(f"target_{name}" for name in arm_names)
        header += "," + ",".join(f"actual_{name}" for name in arm_names)
        np.savetxt(output / "arm_motion.csv", values, delimiter=",", header=header, comments="")
        report["seconds"] = elapsed
        report["finite"] = True
        report["max_target_offset_rad"] = dict(zip(arm_names, np.max(np.abs(values[:, 2:16] - nominal), axis=0).tolist()))
        report["mujoco_warning_counts"] = warnings.tolist()
        if guard:
            report["collision_avoidance"] = guard.report()
        (output / "preview_result.json").write_text(json.dumps(report, indent=2) + "\n")
    if controls["safety_stop"] and args.headless:
        raise RuntimeError("Preview stopped early because of insufficient arm clearance; see preview_result.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("tang", "gl165"), default="tang",
                        help="Robot to preview; GL-165 uses its prepared URDF, pose and PD gains.")
    parser.add_argument("--model", type=Path, help="Override the tang MJCF; GL-165 uses its prepared URDF.")
    parser.add_argument("--cfg", type=Path, default=ROOT / "legged_lab/envs/elf3_tang/walk_cfg.py")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run; zero keeps the viewer open.")
    parser.add_argument("--dt", type=float, help="Override training physics timestep for comparison.")
    parser.add_argument("--resample-interval", type=float, help="Override training target sampling interval.")
    parser.add_argument("--motion-mode", choices=("training", "smooth"), default="training",
                        help="Original tang timing, or full transitions every 1 s for preview comparison only.")
    parser.add_argument("--shoulder-opening-deg", type=float,
                        help="GL-165 preview only: sample outward shoulder angles from 15 degrees to this maximum.")
    parser.add_argument("--avoid-collisions", action="store_true",
                        help="GL-165 preview only: enable self-collisions, reject unsafe paths, monitor actual clearance.")
    scaling = parser.add_mutually_exclusive_group()
    scaling.add_argument("--scale", type=float, help="Fixed curriculum scale; default is 1.0.")
    scaling.add_argument("--iteration", type=int, help="Use the curriculum scale at this PPO iteration.")
    parser.add_argument("--cycle-stages", action="store_true", help="Cycle initial, halfway and full curriculum scale.")
    parser.add_argument("--stage-seconds", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--gif", type=Path, help="Save a finite run as an animated GIF.")
    parser.add_argument("--print-targets", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.robot == "gl165" and args.model is not None:
        parser.error("--model overrides the tang MJCF only; GL-165 is loaded from its prepared URDF")
    if args.robot != "gl165" and (args.avoid_collisions or args.shoulder_opening_deg is not None):
        parser.error("--avoid-collisions and --shoulder-opening-deg currently support GL-165 only")
    if args.shoulder_opening_deg is not None and not 15 <= args.shoulder_opening_deg <= 90:
        parser.error("--shoulder-opening-deg must be between 15 and 90 degrees")
    if args.model is None:
        args.model = ROOT / "legged_lab/assets/elf3_lite/xml/elf3.xml"
    if args.output_dir is None:
        args.output_dir = ROOT / ("logs/stand_gl165/mujoco_arm_preview" if args.robot == "gl165" else "logs/tang_arm_preview")
    for name in ("dt", "resample_interval", "stage_seconds"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error("--duration must be finite and non-negative")
    if args.scale is not None and (not math.isfinite(args.scale) or not 0 <= args.scale <= 1):
        parser.error("--scale must be between 0 and 1")
    if (args.headless or args.gif) and args.duration <= 0:
        parser.error("--headless and --gif require a positive --duration")
    if args.gif:
        args.gif = args.gif.resolve()
        args.gif.parent.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()

"""Render and inspect the configured GL-165 pose in MuJoCo, without advancing physics.

Run with .venv-gl165-mujoco/bin/python. Add --viewer for an interactive window.
The viewer calls mj_forward, not mj_step: this is a pose inspection, not a balance test.
MuJoCo API: https://mujoco.readthedocs.io/en/stable/python.html
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import struct
import time
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]


def load_robot_metadata(physics_profile=None, nominal_stance=None):
    # Import the shared pose without starting Isaac Sim or importing legged_lab.envs.
    path = ROOT / "legged_lab/assets/gl_165/model.py"
    spec = importlib.util.spec_from_file_location("gl165_pose_metadata", path)
    metadata = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metadata)
    if physics_profile is not None:
        physics = metadata.load_physics(physics_profile, nominal_stance)
        metadata.DEFAULT_PHYSICS = physics
        metadata.PHYSICS_DIR = physics.directory
        metadata.PHYSICS_PARAMETERS_PATH = physics.parameters_path
        metadata.MUJOCO_REFERENCE_PATH = physics.directory / physics.parameters["prepared_mujoco_xml"]
        metadata.DEFAULT_JOINT_POS = physics.default_joint_pos
        metadata.URDF_PATH = physics.urdf
        metadata.PHYSICS_PARAMETERS = physics.parameters
        metadata.JOINT_PHYSICS = physics.parameters["joint_parameters"]
        metadata.PHYSICS_DT = physics.parameters["physics_dt_s"]
        metadata.STAND_HEIGHT = physics.parameters["stand_height_m"]
    return metadata


def import_urdf(metadata, output, mujoco, np):
    urdf = ET.parse(metadata.URDF_PATH).getroot()
    mesh_paths = {}
    for mesh in urdf.iter("mesh"):
        source = (metadata.URDF_PATH.parent / mesh.get("filename")).resolve()
        if source not in mesh_paths:
            target = source
            data = source.read_bytes()
            count = struct.unpack_from("<I", data, 80)[0]
            if source.suffix.lower() == ".stl" and count > 200_000:
                # MuJoCo's STL decoder limits face count. Preserve every triangle in OBJ.
                target = output / "meshes" / f"{source.stem}.obj"
                if not target.exists() or target.stat().st_mtime < source.stat().st_mtime:
                    if len(data) != 84 + count * 50:
                        raise ValueError(f"Expected a binary STL: {source}")
                    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
                    vertices = np.frombuffer(data, dtype=dtype, count=count, offset=84)["vertices"].reshape(-1, 3)
                    unique, indices = np.unique(vertices, axis=0, return_inverse=True)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("w") as stream:
                        stream.write("# Original STL triangles; no decimation or scaling.\n")
                        np.savetxt(stream, unique, fmt="v %.9g %.9g %.9g")
                        np.savetxt(stream, indices.reshape(-1, 3) + 1, fmt="f %d %d %d")
            mesh_paths[source] = target
        mesh.set("filename", str(mesh_paths[source]))
    extension = ET.SubElement(urdf, "mujoco")
    ET.SubElement(extension, "compiler", strippath="false", discardvisual="false", fusestatic="false")
    imported = mujoco.MjModel.from_xml_string(ET.tostring(urdf, encoding="unicode"))
    imported_path = output / "imported.xml"
    mujoco.mj_saveLastXML(str(imported_path), imported)
    xml = ET.parse(imported_path).getroot()
    for joint in xml.findall("worldbody//joint"):
        values = getattr(metadata, "JOINT_PHYSICS", {}).get(joint.get("name"))
        if values is not None:
            joint.set("armature", str(values["armature_kg_m2"]))
    return xml


def add_scene(xml):
    xml.set("model", "GL-165 nominal pose - static preview")
    ET.SubElement(xml, "option", timestep="0.005", gravity="0 0 -9.81")
    visual = ET.SubElement(xml, "visual")
    ET.SubElement(visual, "global", offwidth="1600", offheight="1200", fovy="35")
    ET.SubElement(visual, "headlight", ambient="0.35 0.35 0.35", diffuse="0.55 0.55 0.55", specular="0.2 0.2 0.2")
    ET.SubElement(visual, "quality", shadowsize="4096", offsamples="4")
    asset = xml.find("asset")
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient", rgb1="0.83 0.88 0.94",
                  rgb2="0.96 0.97 0.99", width="512", height="3072")
    ET.SubElement(asset, "texture", name="ground_grid", type="2d", builtin="checker", rgb1="0.86 0.88 0.90",
                  rgb2="0.77 0.80 0.83", width="512", height="512")
    ET.SubElement(asset, "material", name="ground_material", texture="ground_grid", texrepeat="6 6",
                  texuniform="true", reflectance="0", shininess="0.05")
    world = xml.find("worldbody")
    base = world.find("body[@name='base_link']")
    if base is None:
        raise ValueError("URDF import did not preserve base_link")
    base.insert(0, ET.Element("freejoint", name="root_free"))
    ET.SubElement(world, "geom", name="ground", type="plane", size="3 3 0.1", group="2",
                  material="ground_material", friction="0.8 0.005 0.0001")
    ET.SubElement(world, "light", name="key_light", pos="2 -2 4", dir="-1 1 -2", directional="true",
                  diffuse="0.7 0.7 0.7", specular="0.2 0.2 0.2", castshadow="true")
    ET.SubElement(world, "light", name="fill_light", pos="-2 2 3", dir="1 -1 -1", directional="true",
                  diffuse="0.35 0.35 0.35", castshadow="false")


def sole_heights(model, data, feet, mujoco, np):
    heights = {}
    for name in feet:
        body_id = model.body(name).id
        minimum = float("inf")
        for geom_id in np.flatnonzero((model.geom_bodyid == body_id) & (model.geom_group == 0)):
            if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE:
                minimum = min(minimum, float(data.geom_xpos[geom_id, 2] - model.geom_size[geom_id, 0]))
                continue
            if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh_id = model.geom_dataid[geom_id]
            start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
            vertices = model.mesh_vert[start:start + count]
            world = vertices @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
            minimum = min(minimum, float(world[:, 2].min()))
        if not np.isfinite(minimum):
            raise ValueError(f"No collision geometry for foot {name}")
        heights[name] = minimum
    return heights


def lower_with_fixed_feet(model, data, metadata, lower_by, mujoco, np):
    """Bend both legs while preserving the original world poses of both feet."""
    targets = {
        name: (data.xpos[model.body(name).id].copy(), data.xmat[model.body(name).id].reshape(3, 3).copy())
        for name in metadata.FEET
    }
    root_addr = model.jnt_qposadr[model.joint("root_free").id]
    data.qpos[root_addr + 2] -= lower_by
    for side, foot in zip(("l", "r"), metadata.FEET):
        names = [f"{side}{suffix}" for suffix in (
            "hip_y_joint", "hip_x_joint", "hip_z_joint", "knee_y_joint", "ankle_y_joint", "ankle_x_joint",
        )]
        joint_ids = np.array([model.joint(name).id for name in names])
        qpos_ids = model.jnt_qposadr[joint_ids]
        dof_ids = model.jnt_dofadr[joint_ids]
        limits = model.jnt_range[joint_ids]
        margin = (limits[:, 1] - limits[:, 0]) * 0.025
        target_position, target_rotation = targets[foot]
        body_id = model.body(foot).id
        jac_position = np.zeros((3, model.nv))
        jac_rotation = np.zeros((3, model.nv))
        for _ in range(120):
            mujoco.mj_forward(model, data)
            rotation_error = target_rotation @ data.xmat[body_id].reshape(3, 3).T
            angular_error = 0.5 * np.array([
                rotation_error[2, 1] - rotation_error[1, 2],
                rotation_error[0, 2] - rotation_error[2, 0],
                rotation_error[1, 0] - rotation_error[0, 1],
            ])
            error = np.concatenate((target_position - data.xpos[body_id], angular_error))
            if np.linalg.norm(error) < 1e-9:
                break
            mujoco.mj_jacBody(model, data, jac_position, jac_rotation, body_id)
            jacobian = np.vstack((jac_position[:, dof_ids], jac_rotation[:, dof_ids]))
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-8 * np.eye(6), error)
            data.qpos[qpos_ids] = np.clip(
                data.qpos[qpos_ids] + np.clip(delta, -0.12, 0.12),
                limits[:, 0] + margin, limits[:, 1] - margin,
            )
        else:
            raise ValueError(f"Cannot lower by {lower_by:.3f} m with fixed feet and soft joint limits")
    mujoco.mj_forward(model, data)
    for foot, (position, rotation) in targets.items():
        body_id = model.body(foot).id
        if (np.linalg.norm(data.xpos[body_id] - position) > 1e-7
                or np.linalg.norm(data.xmat[body_id].reshape(3, 3) - rotation) > 1e-7):
            raise ValueError(f"Lowered pose changed the position or orientation of {foot}")


def build_model(metadata, output, base_height, mujoco, np, lower_by=0.0, scene_name=None):
    xml = import_urdf(metadata, output, mujoco, np)
    add_scene(xml)
    xml.find("option").set("timestep", str(metadata.PHYSICS_DT))
    xml.find("option").set("integrator", metadata.PHYSICS_PARAMETERS["mujoco_integrator"])
    if scene_name:
        xml.set("model", scene_name)
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    data = mujoco.MjData(model)
    joint_count = len(metadata.DEFAULT_JOINT_POS)
    if model.nq != 7 + joint_count or model.nv != 6 + joint_count:
        raise ValueError("Expected a floating base and the configured active joints")
    root_addr = model.jnt_qposadr[model.joint("root_free").id]
    data.qpos[root_addr:root_addr + 7] = [0, 0, metadata.STAND_HEIGHT, 1, 0, 0, 0]
    for name, angle in metadata.DEFAULT_JOINT_POS.items():
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = angle
    mujoco.mj_forward(model, data)
    if base_height is None:
        data.qpos[root_addr + 2] -= min(sole_heights(model, data, metadata.FEET, mujoco, np).values())
    else:
        data.qpos[root_addr + 2] = base_height
    mujoco.mj_forward(model, data)
    original_height = float(data.xpos[model.body("base_link").id, 2])
    if lower_by:
        lower_with_fixed_feet(model, data, metadata, lower_by, mujoco, np)
    qpos = data.qpos.copy()
    height = float(data.xpos[model.body("base_link").id, 2])
    # A thin blue height ruler next to the robot, from ground to base_link origin.
    world = xml.find("worldbody")
    segments = [(0.40, 0.44, 0, 0.40, 0.44, height),
                (0.34, 0.44, 0.002, 0.46, 0.44, 0.002),
                (0.34, 0.44, height, 0.46, 0.44, height)]
    for index, segment in enumerate(segments):
        ET.SubElement(world, "geom", name=f"base_height_ruler_{index}", type="capsule", size="0.0025",
                      fromto=" ".join(map(str, segment)), rgba="0.10 0.45 0.75 1", group="2", contype="0", conaffinity="0")
    keyframes = ET.SubElement(xml, "keyframe")
    ET.SubElement(keyframes, "key", name="nominal_standing", qpos=" ".join(f"{value:.12g}" for value in qpos))
    ET.indent(xml)
    scene_path = output / "gl165_nominal_scene.xml"
    scene_path.write_text(ET.tostring(xml, encoding="unicode") + "\n")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    feet = sole_heights(model, data, metadata.FEET, mujoco, np)
    if base_height is None and max(abs(value) for value in feet.values()) > 1e-5:
        raise ValueError(f"Soles are not aligned to the ground: {feet}")
    report = {"mujoco_version": mujoco.__version__, "mode": "static_pose_without_physics_steps",
              "source_urdf": str(metadata.URDF_PATH), "scene": str(scene_path), "floating_base": True,
              "num_joints": joint_count, "base_height_m": height, "training_target_height_m": metadata.STAND_HEIGHT,
              "original_base_height_m": original_height, "lowered_by_m": lower_by,
              "sole_lowest_z_m": feet,
              "joint_positions_rad": {
                  name: float(data.qpos[model.jnt_qposadr[model.joint(name).id]])
                  for name in metadata.DEFAULT_JOINT_POS
              }}
    (output / "pose_measurements.json").write_text(json.dumps(report, indent=2) + "\n")
    return model, data, report


def camera(mujoco, azimuth, elevation=-5, distance=3.15):
    result = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(result)
    result.lookat[:] = [0.015, 0, 0.83]
    result.distance = distance
    result.azimuth = azimuth
    result.elevation = elevation
    return result


def render_views(model, data, report, output, mujoco):
    from PIL import Image, ImageDraw, ImageFont

    options = mujoco.MjvOption()
    options.geomgroup[0] = 0  # Hide collision duplicates; retain original URDF visual meshes.
    width, height = 640, 840
    views = [("FRONT", 0, -4), ("SIDE", 90, -4), ("THREE-QUARTER", 35, -12)]
    canvas = Image.new("RGB", (width * len(views), height + 142), "#f5f7fa")
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    title_font = ImageFont.truetype(font_path, 26)
    body_font = ImageFont.truetype(font_path, 22)
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for index, (label, azimuth, elevation) in enumerate(views):
            renderer.update_scene(data, camera=camera(mujoco, azimuth, elevation), scene_option=options)
            frame = Image.fromarray(renderer.render())
            frame.save(output / f"{label.lower().replace('-', '_')}.png")
            canvas.paste(frame, (index * width, 48))
            draw.text((index * width + 24, 11), label, font=title_font, fill="#20344b")
    pose = report["joint_positions_rad"]
    prefix = "GL-165" + (f"  |  Lowered {100 * report['lowered_by_m']:.1f} cm" if report["lowered_by_m"] else "")
    draw.text((24, height + 61),
              f"{prefix}  |  Hip pitch {pose['lhip_y_joint']:.3f} rad  |  Knee {pose['lknee_y_joint']:.3f} rad"
              f"  |  Ankle pitch {pose['lankle_y_joint']:.3f} rad",
              font=body_font, fill="#20344b")
    draw.text((24, height + 97), f"MuJoCo static pose  |  Blue ruler: base_link height {report['base_height_m']:.6f} m  |  No balance controller",
              font=body_font, fill="#41556b")
    canvas.save(output / "nominal_pose_views.png")


def show_viewer(model, data, mujoco):
    import mujoco.viewer

    fixed_pose = data.qpos.copy()
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [0.015, 0, 0.83]
            viewer.cam.distance = 3.15
            viewer.cam.azimuth = 35
            viewer.cam.elevation = -12
            viewer.opt.geomgroup[0] = 0
        print("Viewer ready: static nominal pose. Drag to orbit; scroll to zoom. Close window to exit.", flush=True)
        while viewer.is_running():
            with viewer.lock():
                data.qpos[:] = fixed_pose
                data.qvel[:] = 0
                mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1 / 30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", action="store_true", help="Open a static, interactive MuJoCo viewer.")
    parser.add_argument("--no-render", action="store_true", help="Skip PNG rendering when opening the viewer.")
    height_options = parser.add_mutually_exclusive_group()
    height_options.add_argument("--base-height", type=float, help="Explicit base height; default aligns both soles to the floor.")
    height_options.add_argument("--lower-by", type=float, default=0.0,
                                help="Lower the base by this many meters, bending the legs while holding both feet in place.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "logs/stand_gl165/mujoco_pose")
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "glfw" if args.viewer else "egl")
    import mujoco
    import numpy as np

    if not np.isfinite(args.lower_by) or args.lower_by < 0:
        parser.error("--lower-by must be a finite, non-negative distance in meters")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model, data, report = build_model(load_robot_metadata(), output, args.base_height, mujoco, np, args.lower_by)
    print(json.dumps(report, indent=2), flush=True)
    if not args.no_render:
        render_views(model, data, report, output, mujoco)
        print(f"Rendered: {output / 'nominal_pose_views.png'}", flush=True)
    if args.viewer:
        show_viewer(model, data, mujoco)


if __name__ == "__main__":
    main()

"""Prepare a named GL-165 physics profile from the user-supplied deployment MJCF.

The original URDF and source XML remain immutable. Mesh triangles come from the
original URDF; the XML supplies armature, fixed joints and contact geometry.
Run with a Python environment containing MuJoCo and NumPy.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_name(name):
    if name.startswith(("l_", "r_")):
        return (name[0] + name[2:]).replace("shoulder", "shouder")
    return name


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def floats(value):
    return np.fromstring(value, sep=" ")


def rpy_matrix(value):
    r, p, y = floats(value)
    cr, cp, cy = np.cos([r, p, y]); sr, sp, sy = np.sin([r, p, y])
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


def prepare_cad_urdf(source, destination, previous_urdf):
    """Keep the CAD meshes and inertia, with the established training link names."""
    from prepare_gl165 import prepare as prepare_floating_base

    output = prepare_floating_base(source, destination / "source")
    shutil.copy2(source, output.parent / source.name)
    robot = ET.parse(output)
    previous = {j.get("name"): j for j in ET.parse(previous_urdf).findall("joint")}
    joints = {j.get("name"): j for j in robot.findall("joint")}
    if joints.keys() != previous.keys():
        raise ValueError("The new CAD package has a different joint set")
    links = {"base_link": "base_link"}
    links.update({j.find("child").get("link"): previous[name].find("child").get("link")
                  for name, j in joints.items()})
    for name, j in joints.items():
        for tag in ("parent", "child"):
            link = links[j.find(tag).get("link")]
            if link != previous[name].find(tag).get("link"):
                raise ValueError(f"CAD joint topology changed at {name}")
            j.find(tag).set("link", link)
    for link in robot.findall("link"):
        link.set("name", links[link.get("name")])
    robot.write(output, encoding="utf-8", xml_declaration=True)
    return output, links


def prepare(source, destination, physics_dt=None, timestep_reason=None, robot_urdf=None):
    source, destination = source.resolve(), destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    original_urdf = ROOT / "legged_lab/assets/gl_165/urdf/gl_165.urdf"
    if robot_urdf is not None:
        robot_urdf = robot_urdf.resolve(strict=True)
        original_urdf, link_names = prepare_cad_urdf(robot_urdf, destination, original_urdf)
    urdf = ET.parse(original_urdf).getroot()
    xml = ET.parse(source).getroot()
    if xml.find("compiler").get("angle") != "radian":
        raise ValueError("Expected explicit radians in the source XML")
    source_physics_dt = float(xml.find("option").get("timestep"))
    selected_dt = source_physics_dt if physics_dt is None else physics_dt
    if not math.isfinite(selected_dt) or selected_dt <= 0:
        raise ValueError("Physics timestep must be finite and positive")
    decimation = round(.02 / selected_dt)
    if decimation < 1 or not math.isclose(decimation * selected_dt, .02):
        raise ValueError("Physics timestep must divide the 20 ms policy period")
    timestep_override = None
    profile = "gl165_xml_armature_20261117"
    if robot_urdf is not None:
        profile = "gl165_urdf_261123_2"
    if not math.isclose(selected_dt, source_physics_dt):
        if not timestep_reason:
            raise ValueError("Record the reason for overriding the source XML timestep")
        timestep_override = {"source_s": source_physics_dt, "effective_s": selected_dt,
                             "reason": timestep_reason}
        profile += f"_{selected_dt * 1000:g}ms"
        xml.find("option").set("timestep", str(selected_dt))
    bodies = {b.get("name"): b for b in xml.findall("worldbody//body")}
    links = {b.get("name"): b for b in urdf.findall("link")}
    if bodies.keys() != links.keys():
        raise ValueError("XML and URDF body names differ")
    joints = {j.get("name"): j for j in urdf.findall("joint")}
    parents = {child.get("name"): parent.get("name") for parent in bodies.values() for child in parent.findall("body")}
    rows, fixed = {}, []
    for body_name, body in bodies.items():
        inertia = body.find("inertial")
        original = links[body_name].find("inertial")
        tensor = original.find("inertia")
        expected = [float(tensor.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
        if robot_urdf is not None:
            if not np.allclose(floats(original.find("origin").get("rpy")), 0):
                raise ValueError(f"Rotated CAD inertia frame requires tensor rotation: {body_name}")
            inertia.set("fullinertia", " ".join(map(str, expected)))
            inertia.set("mass", original.find("mass").get("value"))
            inertia.set("pos", original.find("origin").get("xyz"))
        if not np.allclose(floats(inertia.get("fullinertia")), expected, atol=1e-12, rtol=1e-9):
            raise ValueError(f"Inertia tensor mismatch: {body_name}")
        if not math.isclose(float(inertia.get("mass")), float(original.find("mass").get("value")), rel_tol=1e-9):
            raise ValueError(f"Mass mismatch: {body_name}")
        if not np.allclose(floats(inertia.get("pos")), floats(original.find("origin").get("xyz")), atol=1e-12):
            raise ValueError(f"COM mismatch: {body_name}")
        incoming = next((j for j in joints.values() if j.find("child").get("link") == body_name), None)
        if incoming is None:
            continue
        origin = incoming.find("origin")
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, floats(body.get("quat", "1 0 0 0")))
        if (parents[body_name] != incoming.find("parent").get("link")
                or not np.allclose(floats(body.get("pos", "0 0 0")), floats(origin.get("xyz")), atol=1e-10)
                or not np.allclose(matrix.reshape(3, 3), rpy_matrix(origin.get("rpy")), atol=1e-9)):
            raise ValueError(f"Joint frame mismatch: {body_name}")
        if robot_urdf is not None:
            body.set("pos", origin.get("xyz"))
            rotation = np.empty(4)
            mujoco.mju_mat2Quat(rotation, rpy_matrix(origin.get("rpy")).ravel())
            body.set("quat", " ".join(map(str, rotation)))
        name = incoming.get("name")
        joint = body.find("joint")
        if joint is None:
            if name not in ("head_z_joint", "head_y_joint"):
                raise ValueError(f"Unexpected fixed joint: {name}")
            fixed.append(name)
            incoming.set("type", "fixed")
            for tag in ("axis", "limit", "dynamics"):
                for child in incoming.findall(tag):
                    incoming.remove(child)
            continue
        if training_name(joint.get("name")) != name:
            raise ValueError(f"Joint/body mapping mismatch: {name}")
        axis = floats(incoming.find("axis").get("xyz"))
        if not np.allclose(floats(joint.get("axis")), axis, atol=1e-10):
            raise ValueError(f"Joint axis mismatch: {name}")
        if robot_urdf is not None:
            joint.set("axis", incoming.find("axis").get("xyz"))
        armature = float(joint.get("armature"))
        if not math.isfinite(armature) or armature <= 0:
            raise ValueError(f"Missing positive armature: {name}")
        if float(joint.get("damping", "0")) != 0:
            raise ValueError("Nonzero passive damping needs a separate PhysX mapping")
        motor = xml.find(f"actuator/motor[@joint='{joint.get('name')}']")
        if motor is None or float(motor.get("gear")) != 1:
            raise ValueError(f"Expected unit-gear torque motor: {name}")
        ranges = [floats(joint.get("actuatorfrcrange")), floats(motor.get("ctrlrange")), floats(motor.get("forcerange"))]
        if any(not np.allclose(r, ranges[0]) for r in ranges) or ranges[0][0] != -ranges[0][1]:
            raise ValueError(f"Inconsistent torque limits: {name}")
        limit = incoming.find("limit")
        original_limit = dict(limit.attrib)
        lower, upper = floats(joint.get("range"))
        limit.set("lower", str(lower)); limit.set("upper", str(upper))
        limit.set("effort", str(ranges[0][1]))
        rows[name] = {
            "xml_joint": joint.get("name"), "armature_kg_m2": armature,
            "effort_limit_nm": float(ranges[0][1]), "position_limits_rad": [lower, upper],
            "velocity_limit_rad_s": float(limit.get("velocity")),
            "velocity_limit_source": "original GL-165 URDF; MJCF has no joint speed cap",
            "original_urdf_limits": original_limit,
        }
    if len(rows) != 29 or set(fixed) != {"head_z_joint", "head_y_joint"}:
        raise ValueError("Expected 29 active joints and two fixed head joints")

    # Preserve every explicitly authored sole contact in the supplied XML.
    foot_contacts = {}
    for name in ("lfoot_x_link", "rfoot_x_link"):
        for collision in links[name].findall("collision"):
            links[name].remove(collision)
        geoms = [g for g in bodies[name].findall("geom") if g.get("class") == "collision"]
        if len(geoms) != 9 or any(g.get("type") != "sphere" for g in geoms):
            raise ValueError(f"Expected nine spherical contacts on {name}")
        foot_contacts[name] = []
        for geom in geoms:
            collision = ET.SubElement(links[name], "collision", name=geom.get("name"))
            ET.SubElement(collision, "origin", xyz=geom.get("pos"), rpy="0 0 0")
            ET.SubElement(ET.SubElement(collision, "geometry"), "sphere", radius=geom.get("size"))
            foot_contacts[name].append({"position_m": floats(geom.get("pos")).tolist(), "radius_m": float(geom.get("size"))})
    for mesh in urdf.iter("mesh"):
        path = (original_urdf.parent / mesh.get("filename")).resolve()
        mesh.set("filename", os.path.relpath(path, destination))
    generated_urdf = destination / "gl_165_xml_physics.urdf"
    ET.indent(urdf)
    generated_urdf.write_text(ET.tostring(urdf, encoding="unicode") + "\n")

    # Reuse the existing lossless STL->OBJ reader for MuJoCo's STL size limit.
    # This also gives the exact original mesh filenames for the new MJCF.
    preview = load_file("gl165_xml_pose", ROOT / "legged_lab/scripts/preview_gl165_mujoco.py")
    metadata = preview.load_robot_metadata()
    baseline = SimpleNamespace(**vars(metadata))
    baseline.URDF_PATH = original_urdf
    baseline.ASSET_DIR = original_urdf.parent.parent
    imported = preview.import_urdf(baseline, destination, mujoco, np)
    imported_meshes = {m.get("name"): m.get("file") for m in imported.findall("asset/mesh")}
    mesh_for_body = {}
    for body in imported.findall("worldbody//body"):
        meshes = {g.get("mesh") for g in body.findall("geom") if g.get("type") == "mesh"}
        if len(meshes) == 1:
            mesh_for_body[body.get("name")] = imported_meshes[meshes.pop()]
    asset = xml.find("asset")
    for mesh in asset.findall("mesh"):
        asset.remove(mesh)
    for body in bodies.values():
        # The XML splits the torso visual into two files. The supplied original
        # STL already contains both parts, so draw that whole mesh only once.
        seen_visual = False
        for geom in list(body.findall("geom")):
            if geom.get("type") != "mesh":
                continue
            visual = geom.get("contype") == "0"
            if visual and seen_visual:
                body.remove(geom)
                continue
            seen_visual |= visual
            geom.set("mesh", body.get("name") + "_original")
        if body.get("name") in mesh_for_body:
            path = Path(mesh_for_body[body.get("name")])
            if not path.is_absolute():
                path = (destination / path).resolve()
            ET.SubElement(asset, "mesh", name=body.get("name") + "_original",
                          file=os.path.relpath(path, destination))
    xml.find("compiler").set("meshdir", ".")
    for joint in xml.findall("worldbody//joint"):
        joint.set("name", training_name(joint.get("name")))
    for motor in xml.findall("actuator/motor"):
        motor.set("joint", training_name(motor.get("joint")))
        motor.set("name", motor.get("joint"))
    output_xml = destination / "gl165_training_reference.xml"
    ET.indent(xml)
    output_xml.write_text(ET.tostring(xml, encoding="unicode") + "\n")
    model = mujoco.MjModel.from_xml_path(str(output_xml))
    data = mujoco.MjData(model)
    baseline_height = getattr(metadata, "LEGACY_STAND_HEIGHT", metadata.STAND_HEIGHT)
    data.qpos[:7] = [0, 0, baseline_height, 1, 0, 0, 0]
    for name in rows:
        data.qpos[model.joint(name).qposadr[0]] = metadata.ORIGINAL_JOINT_POS[name]
    mujoco.mj_forward(model, data)
    sole_bottoms = [float(data.geom_xpos[model.geom(g.get("name")).id, 2] - float(g.get("size")))
                    for name in foot_contacts for g in bodies[name].findall("geom") if g.get("type") == "sphere"]
    stand_height = baseline_height - min(sole_bottoms)
    data.qpos[2] = stand_height + .003
    ET.SubElement(xml.find("keyframe"), "key", name="approved_standing", qpos=" ".join(map(str, data.qpos)))
    ET.indent(xml)
    output_xml.write_text(ET.tostring(xml, encoding="unicode") + "\n")
    model = mujoco.MjModel.from_xml_path(str(output_xml))
    for name, row in rows.items():
        if not math.isclose(float(model.dof_armature[model.joint(name).dofadr[0]]), row["armature_kg_m2"], rel_tol=1e-12):
            raise ValueError(f"Compiled armature mismatch: {name}")
    preserved_source = destination / "source_deployment.xml"
    if preserved_source.exists():
        if sha256(preserved_source) != sha256(source):
            raise ValueError("Use a new profile directory for a different source XML")
    else:
        shutil.copy2(source, preserved_source)
    report = {
        "profile": profile, "source_xml": str(source), "source_xml_sha256": sha256(source),
        "source_urdf": str(original_urdf), "source_urdf_sha256": sha256(original_urdf),
        "prepared_urdf": generated_urdf.name, "prepared_urdf_sha256": sha256(generated_urdf),
        "prepared_mujoco_xml": output_xml.name, "prepared_mujoco_xml_sha256": sha256(output_xml),
        "joint_parameters": rows, "fixed_joints": fixed, "foot_contacts": foot_contacts,
        "physics_dt_s": float(xml.find("option").get("timestep")), "control_dt_s": .02,
        "source_physics_dt_s": source_physics_dt, "timestep_override": timestep_override,
        "mujoco_integrator": xml.find("option").get("integrator"), "isaac_actuator": "ImplicitActuatorCfg",
        "stand_height_m": stand_height, "original_stand_height_m": baseline_height,
        "standing_pose_source": "previous user-approved feet-forward crouched pose; sole height recomputed for XML spheres",
        "nested_collision_pairs": [[e.get("body1"), e.get("body2")] for e in xml.findall("contact/exclude")],
        "original_body_masses_com_inertias_verified": len(bodies),
        "mass_kg": float(model.body_mass.sum()), "active_joints": model.nu,
        "gains_source": "previously approved ELF3LITE_CFG; XML has no PD gains or action scales",
        "solver_note": "MuJoCo integrator/iterations have no direct PhysX equivalence; Isaac keeps its existing solver settings",
        "friction_note": "XML nominal friction 1; retain previously approved training friction randomization",
        "imu_note": "XML site imu_in_torso is attached to base_link; retain existing pelvis observations",
    }
    if robot_urdf is not None:
        report.update(source_urdf=str(robot_urdf), source_urdf_sha256=sha256(robot_urdf),
                      source_urdf_copy=f"source/urdf/{robot_urdf.name}", cad_link_to_training_link=link_names,
                      mesh_directory="source/meshes",
                      source_mesh_sha256={p.name: sha256(p) for p in sorted((destination / "source/meshes").iterdir())})
    (destination / "physics_parameters.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("profile", "source_xml_sha256", "active_joints", "mass_kg", "stand_height_m", "physics_dt_s", "fixed_joints")}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--physics-dt", type=float,
                        help="Explicit override of the source XML timestep; source remains unchanged.")
    parser.add_argument("--timestep-reason", help="Record the user instruction authorizing a timestep override.")
    parser.add_argument("--robot-urdf", type=Path, help="Update the robot geometry and inertia from the 261123_2 CAD package.")
    args = parser.parse_args()
    prepare(args.source, args.destination, args.physics_dt, args.timestep_reason, args.robot_urdf)

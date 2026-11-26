"""Audit GL-165 gains/scales against ELF3 training and deployment source by name.

Read source literals without importing ROS, starting hardware, or importing Isaac.
The report distinguishes controller matches from task-specific pose differences.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEPLOYMENT_ROOT = Path("/home/cheng/bxi_controller_ros2")


def assigned(tree, name):
    return next(node.value for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
                and any(isinstance(target, ast.Name) and target.id == name
                        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])))


def keywords(node):
    return {item.arg: item.value for item in node.keywords}


def elf3_name(name):
    if name.startswith(("l", "r")):
        return (name[0] + "_" + name[1:]).replace("shouder", "shoulder")
    return name


def read_references(deployment_root=DEFAULT_DEPLOYMENT_ROOT):
    package = deployment_root / "src/bxi_example_py_elf3"
    files = {
        "training_gains": ROOT / "legged_lab/assets/elf3_lite/elf3.py",
        "training_scales": ROOT / "legged_lab/envs/elf3_tang/walk_cfg.py",
        "deployment_body": package / "bxi_example_py_elf3/policies/joints.py",
        "deployment_head": package / "bxi_example_py_elf3/bxi_example_demo.py",
        "deployment_runtime": package / "config/elf3_state_machine.yaml",
        "deployment_policy": package / "bxi_example_py_elf3/policies/amp.py",
    }
    trees = {key: ast.parse(path.read_text()) for key, path in files.items() if path.suffix == ".py"}
    asset = keywords(assigned(trees["training_gains"], "ELF3LITE_CFG"))
    training = {}
    for actuator in asset["actuators"].values:
        params = keywords(actuator)
        for key in ("stiffness", "damping", "effort_limit_sim", "velocity_limit_sim"):
            training.setdefault(key, {}).update(ast.literal_eval(params[key]))
    deploy = {
        row[0]: dict(zip(("default_position", "kp", "kd", "action_scale"), row[1:]))
        for row in ast.literal_eval(assigned(trees["deployment_body"], "ELF3_ISAAC_PARAMETERS").args[1])
    }
    order = ast.literal_eval(assigned(trees["deployment_body"], "ELF3_POLICY_JOINTS").args[0])[:15]
    cls = next(node for node in trees["training_scales"].body
               if isinstance(node, ast.ClassDef) and node.name == "Elf3WalkTangFlatEnvCfg")
    scales = ast.literal_eval(keywords(assigned(cls, "robot"))["action_scale"])
    if len(scales) != len(order):
        raise ValueError("ELF3 policy action scale width changed")
    scale_by_name = dict(zip(order, scales))
    heads = assigned(trees["deployment_head"], "ELF3_COMMAND_DEFAULTS").args[0]
    for name, call in zip(heads.keys, heads.values):
        params = {key: ast.literal_eval(value) for key, value in keywords(call).items()}
        deploy[ast.literal_eval(name)] = {
            "default_position": params["position"], "kp": params["kp"], "kd": params["kd"],
            "action_scale": None,
        }
    return files, training, scale_by_name, deploy


def build_report(deployment_root=DEFAULT_DEPLOYMENT_ROOT):
    spec = importlib.util.spec_from_file_location("gl165_model_audit", ROOT / "legged_lab/assets/gl_165/model.py")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    files, training, scales, deployment = read_references(deployment_root)
    limits = {joint.get("name"): joint.find("limit").attrib
              for joint in ET.parse(model.URDF_PATH).getroot().findall("joint") if joint.get("type") == "revolute"}
    rows, errors = [], []
    for name, default in model.DEFAULT_JOINT_POS.items():
        reference_name = elf3_name(name)
        kp, kd, scale = model.joint_control_parameters(name)
        deploy = deployment[reference_name]
        row = {"gl165_joint": name, "elf3_joint": reference_name,
               "kp": kp, "kd": kd, "policy_action_scale": scale,
               "gl165_default_position": default, "deployment_default_position": deploy["default_position"],
               "default_position_matches": abs(default - deploy["default_position"]) < 1e-9,
               "gl165_urdf_limits": limits[name], "deployment_reference": deploy,
               "xml_physics": model.JOINT_PHYSICS[name]}
        if (kp, kd) != (deploy["kp"], deploy["kd"]):
            errors.append(f"{name}: PD differs from deployment")
        if not name.startswith("head_"):
            resolved = {}
            for field, patterns in training.items():
                values = [value for pattern, value in patterns.items() if re.fullmatch(pattern, reference_name)]
                if len(values) != 1:
                    raise ValueError(f"Ambiguous ELF3 source for {reference_name}/{field}: {values}")
                resolved[field] = values[0]
            row["elf3_training_reference"] = resolved
            if (kp, kd) != (resolved["stiffness"], resolved["damping"]):
                errors.append(f"{name}: PD differs from ELF3LITE_CFG")
        if re.fullmatch(model.POLICY_JOINT_PATTERN, name):
            if scale != scales[reference_name] or scale != deploy["action_scale"]:
                errors.append(f"{name}: policy action scale differs from ELF3/deployment")
        elif scale is not None:
            errors.append(f"{name}: non-policy joint must have no policy scale")
        rows.append(row)
    report = {
        "controller_profile": model.CONTROLLER_PROFILE,
        "sources": {key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for key, path in files.items()},
        "matched_pd_joints": len(rows) if not errors else None,
        "matched_policy_action_scales": len(scales) if not errors else None,
        "errors": errors, "joints": rows,
        "scope": "ELF3LITE_CFG and built-in AMP deployment policy, plus platform head defaults; "
                 "other deployment states may use their own model metadata or override gains.",
        "head_control": "fixed at zero according to the user-supplied XML, without head actuators",
        "physics_parameters": model.PHYSICS_PARAMETERS,
        "deployment_adapter_required": True,
        "deployment_differences": [
            "The six-input GL-165 actor uses 54 values x 10 frames. Earlier standing actors use 310 or 525 inputs.",
            "GL-165 retains its approved feet-forward crouched nominal pose, not the ELF3 walking pose.",
            "GL-165 joint names/order must be mapped explicitly, including shouder vs shoulder spelling.",
            "GL-165 joint velocity observation scale is 0.1; existing AMP deployment uses 1.0.",
            "Historical 310/525-input actors require base velocity, displacement, heading and height. Only 525-input actors include arm state, targets and phase.",
            "Training clamps joint targets to 95% of GL-165 URDF limits; the AMP Python target path has no such clamp.",
            "GL-165 XML supplies effort/position limits and armature; original URDF supplies velocity limits.",
            "Hardware-side motor limits and sensor availability are not established by this source audit.",
        ],
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-root", type=Path, default=DEFAULT_DEPLOYMENT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.deployment_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if report["errors"]:
        raise SystemExit("\n".join(report["errors"]))
    print(f"Verified {report['matched_pd_joints']} PD joints and {report['matched_policy_action_scales']} "
          f"policy scales against training + deployment source. Report: {args.output}")

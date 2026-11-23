"""GL-165 joint groups and nominal standing pose (radians, SI units)."""

from pathlib import Path
import hashlib
import json
import re

ASSET_DIR = Path(__file__).resolve().parent
ORIGINAL_URDF_PATH = ASSET_DIR / "urdf" / "gl_165.urdf"
PHYSICS_DIR = ASSET_DIR / "deployment_xml_20261117_5ms"
PHYSICS_PARAMETERS_PATH = PHYSICS_DIR / "physics_parameters.json"
PHYSICS_PARAMETERS = json.loads(PHYSICS_PARAMETERS_PATH.read_text())
URDF_PATH = PHYSICS_DIR / PHYSICS_PARAMETERS["prepared_urdf"]
MUJOCO_REFERENCE_PATH = PHYSICS_DIR / PHYSICS_PARAMETERS["prepared_mujoco_xml"]
PHYSICS_DT = PHYSICS_PARAMETERS["physics_dt_s"]
CONTROL_DT = PHYSICS_PARAMETERS["control_dt_s"]
JOINT_PHYSICS = PHYSICS_PARAMETERS["joint_parameters"]
FIXED_JOINT_POS = {name: 0.0 for name in PHYSICS_PARAMETERS["fixed_joints"]}
FEET = ["lfoot_x_link", "rfoot_x_link"]
TORSO = "body_y_link"
ARM_JOINT_NAMES = [f"{side}{joint}_joint" for side in ("l", "r") for joint in (
    "shouder_y", "shouder_x", "shouder_z", "elbow_y", "wrist_x", "wrist_y", "wrist_z",
)]
POLICY_JOINT_PATTERN = "waist_.*|[lr](hip_.*|knee_y|ankle_.*)_joint"
# Non-adjacent CAD parts that share a multi-axis joint mounting volume.
NESTED_COLLISION_PAIRS = PHYSICS_PARAMETERS["nested_collision_pairs"]
# Approved crouched pose: 3 cm below the original nominal stance, with both feet fixed.
# URDF FK and MuJoCo agree on the base height when the soles touch the floor.
LEGACY_STAND_HEIGHT = 0.987285
# Keep the approved joint pose; align the XML's spherical sole contacts to ground.
STAND_HEIGHT = PHYSICS_PARAMETERS["stand_height_m"]
# Spawn slightly above contact; this clearance is not part of the reward target.
SPAWN_HEIGHT = STAND_HEIGHT + 0.003

# User-confirmed sources: ELF3LITE_CFG in assets/elf3_lite/elf3.py for
# 29 gains, elf3_tang/walk_cfg.py for the 15 waist/leg action scales.
# The deployment's policies/joints.py agrees with both after mapping names.
# Head gains: bxi_controller_ros2/src/bxi_example_py_elf3/
# bxi_example_py_elf3/bxi_example_demo.py::ELF3_COMMAND_DEFAULTS.
CONTROLLER_PROFILE = "elf3_gains_gl165_xml_armature_fixed_head_20261117_5ms"
# (joint name regex, Kp, Kd, policy action scale). None means no policy action:
# Arms follow radian-valued disturbance targets. The XML fixes both head joints.
# XML supplies effort/position limits and armature; URDF supplies speed limits.
JOINT_GROUPS = {
    "hip_pitch_roll": ("[lr]hip_[xy]_joint", 176.421, 11.231, 0.213),
    "hip_yaw": ("[lr]hip_z_joint", 54.224, 3.452, 0.231),
    "knees": ("[lr]knee_y_joint", 176.421, 11.231, 0.213),
    "ankle_pitch": ("[lr]ankle_y_joint", 33.493, 2.132, 0.373),
    "ankle_roll": ("[lr]ankle_x_joint", 21.771, 1.386, 0.230),
    "waist_yaw": ("waist_z_joint", 176.421, 11.231, 0.213),
    "waist_pitch": ("waist_y_joint", 108.448, 6.904, 0.231),
    "waist_roll": ("waist_x_joint", 162.672, 10.356, 0.154),
    # 'shouder' is the spelling in the supplied URDF.
    "shoulder_pitch_roll": ("[lr]shouder_[xy]_joint", 54.224, 3.452, None),
    "shoulder_yaw": ("[lr]shouder_z_joint", 16.747, 1.066, None),
    "elbows": ("[lr]elbow_y_joint", 54.224, 3.452, None),
    "wrists": ("[lr]wrist_[xyz]_joint", 16.747, 1.066, None),
}


def joint_control_parameters(name):
    """Resolve gains/scales by name; never silently fall back to a scalar."""
    matches = [values[1:] for values in JOINT_GROUPS.values() if re.fullmatch(values[0], name)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one control parameter group for {name}: {matches}")
    return matches[0]

DEFAULT_JOINT_POS = {"waist_z_joint": 0.0, "waist_x_joint": 0.0, "waist_y_joint": 0.0}
for side, sign in (("l", 1.0), ("r", -1.0)):
    DEFAULT_JOINT_POS.update({
        f"{side}hip_y_joint": -0.315364011971,
        f"{side}hip_x_joint": -sign * 0.00517333597433,
        f"{side}hip_z_joint": -sign * 0.0560915128826,
        f"{side}knee_y_joint": 0.597529351667,
        # Compensate the tilted hip axis and preserve each foot's world orientation.
        f"{side}ankle_y_joint": -0.300928949319,
        f"{side}ankle_x_joint": -sign * 0.00278352160397,
        f"{side}shouder_y_joint": 0.0,
        f"{side}shouder_x_joint": sign * 0.10,
        f"{side}shouder_z_joint": 0.0,
        f"{side}elbow_y_joint": 1.10,
        f"{side}wrist_x_joint": 0.0,
        f"{side}wrist_z_joint": 0.0,
        f"{side}wrist_y_joint": 0.0,
    })


def motion_asset_signature():
    """Invalidate motion data after geometry, pose, actuator or contact changes."""
    digest = hashlib.sha256()
    digest.update(URDF_PATH.read_bytes())
    digest.update(PHYSICS_PARAMETERS_PATH.read_bytes())
    for path in sorted((ASSET_DIR / "meshes").iterdir()):
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    digest.update(json.dumps([DEFAULT_JOINT_POS, JOINT_GROUPS, NESTED_COLLISION_PAIRS], sort_keys=True).encode())
    return digest.hexdigest()


# Exact digest pairs before and after removing workstation paths from the metadata.
# Both identities refer to the same physics, joint order, timing and target arrays.
_PORTABLE_DIGESTS = {
    "92461d2dc797c2a6ac9c658f2d7bb85b092075065cf570903f7a78c468a4ed45": "fe2ddb08d5dba86d59231e2419643748d222462ebaf6a2b9d42f0d5dba250722",
    "219a78557cdca817c1335e6012741086dbdc39d42a6217f7afbb771ec91636e6": "540d16852dc4ba47ee43585be46174527bd905531f8ca247734d94356797ebdd",
}


def portable_digest(digest):
    return _PORTABLE_DIGESTS.get(digest, digest)
